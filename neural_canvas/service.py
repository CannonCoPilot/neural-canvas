"""Nexus/Pulse integration: the watched folder feeds the Pulse kanban, the kanban drives the runs.

    watch_dir/  ──new image──▶  Pulse task (runner:neural-canvas, staging:wait)   [Inbox column]
                                   │  claimed atomically (conditional-update)
                                   ▼
                     active:running + docent:<stage> labels, progress notes      [Active column]
                                   │
               all gates pass ─────┴───── a gate fails / error
                     ▼                               ▼
     active:done + completed:done, closed      blocked:yes + reason:docent-<stage>  [Blocked]
                                                task stays OPEN, alert on the message bus

The ``runner:neural-canvas`` label tells Alfred's pipeline-watcher the task is owned by this
deterministic runner (no Claude stage/evaluate/execute/diagnose). Re-running a blocked task is a
board action: the standard ``v2-reset-to-staging`` transition puts it back to staging:wait.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import requests

from .config import Config
from .dedup import ArtworkRegistry, phash
from .deliver import GooglePhotos
from .indexer import ArtistIndexer
from .pipeline import IMAGE_EXTS, Docent, RunStopped, ingest, load_image

log = logging.getLogger(__name__)
ACTOR = "neural-canvas"
# Pulse fills any pipeline dimension a new task leaves out with its default, and staging's default is
# "wait" (= ready to run). A record-only task must therefore say every dimension is finished. The
# completed dimension has no "duplicate" value; outcome: is a free label and survives the close.
FINISHED = ["staging:done", "evaluated:done", "queued:done", "active:done", "completed:done", "blocked:no"]
RETRY_S = 600          # a file whose task could not be created is retried after this long


class Pulse:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.s = requests.Session()
        self._lock = threading.Lock()          # one session, shared by the worker threads

    def _r(self, method: str, path: str, **kw):
        with self._lock:
            r = self.s.request(method, f"{self.url}{path}", timeout=30, **kw)
        if r.status_code == 409:
            return None
        if r.status_code >= 400:     # Pulse explains a rejection in the body; keep it in the error
            raise requests.HTTPError(f"{r.status_code} {method} {path}: {r.text[:300]}", response=r)
        return r.json() if r.content else {}

    def create(self, title: str, description: str, labels: list[str], metadata: dict) -> dict:
        t = self._r("POST", "/tasks", json={"title": title, "description": description, "priority": "medium",
                                           "labels": labels, "metadata": metadata, "actor": ACTOR})
        if not t or not t.get("id"):
            raise requests.HTTPError(f"Pulse created no task for {title!r}")
        return t

    def list(self, label: str, status: str | None = None) -> list[dict]:
        params = {"label": label, "limit": 500, **({"status": status} if status else {})}
        return (self._r("GET", "/tasks", params=params) or {}).get("tasks", [])

    def get(self, task_id: str) -> dict:
        return self._r("GET", f"/tasks/{task_id}")

    def claim(self, task_id: str, runner_label: str) -> bool:
        # Pulse consumes every precondition label in a claim, so the ownership label must never be one
        # (claiming on it strips it and hands the task to Alfred's Claude pipeline). Pulse also does not
        # replace a dimension's old value here, so the defaults are removed explicitly.
        res = self._r("POST", f"/tasks/{task_id}/conditional-update", json={
            "preconditions": [{"label_value": "staging:wait"}],
            "set_labels": ["staging:done", "evaluated:done", "queued:done", "active:running", "blocked:no"],
            "remove_labels": ["staging:wait", "evaluated:no", "queued:no", "active:no"],
            "status": "in_progress", "actor": ACTOR})
        if res is None:
            return False
        if runner_label not in (res.get("labels") or []):    # belt and braces: re-assert ownership
            self.labels(task_id, add=[runner_label])
        return True

    def labels(self, task_id: str, add: list[str] = (), remove: list[str] = ()):
        for lbl in remove:
            try:
                self._r("DELETE", f"/tasks/{task_id}/labels/{lbl}", params={"actor": ACTOR})
            except requests.HTTPError:
                pass
        if add:
            self._r("POST", f"/tasks/{task_id}/labels", json={"labels": list(add), "actor": ACTOR})

    def note(self, task_id: str, text: str, metadata: dict | None = None, status: str | None = None):
        body = {"append_notes": f"[{time.strftime('%H:%M:%S')}] {text}", "actor": ACTOR}
        if metadata is not None:
            body["metadata"] = metadata
        if status:
            body["status"] = status
        self._r("PATCH", f"/tasks/{task_id}", json=body)

    def close(self, task_id: str, reason: str):
        self._r("POST", f"/tasks/{task_id}/close", json={"reason": reason, "actor": ACTOR})

    def alert(self, severity: str, text: str, data: dict):
        try:
            self._r("POST", "/messages", json={"event_type": "docent_blocked", "source": "neural-canvas",
                                               "actor": ACTOR, "severity": severity,
                                               "data": {"summary": text, **data}})
        except requests.RequestException as e:
            log.error("alert could not be posted: %s", e)


class Service:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        p = cfg["pulse"]
        self.pulse = Pulse(p["url"])
        self.runner_label, self.project_label = p["runner_label"], p["project_label"]
        self.watch = cfg.path("watch_dir")
        self.ingested = self.watch / "ingested"
        self.ingested.mkdir(exist_ok=True)
        self.duplicates = self.watch / "duplicates"
        self.duplicates.mkdir(exist_ok=True)
        self.state = cfg.path("state_dir")
        self.known_path = self.state / "known.json"
        self.known: dict = json.loads(self.known_path.read_text()) if self.known_path.exists() else {}
        self._sizes: dict[str, int] = {}
        self._failed: dict[tuple[str, int], float] = {}     # (name, size) -> when its task create failed
        self.registry = ArtworkRegistry(cfg.path("state_dir"))
        self.photos = GooglePhotos(cfg["google_photos"], cfg.path("state_dir"))
        self._auth_alerted = 0.0
        self.workers = max(1, int(p.get("workers", 1)))
        self._pool = ThreadPoolExecutor(self.workers, thread_name_prefix="worker")
        self._running: dict[str, Future] = {}      # task id -> its run
        self._local = threading.local()
        # first sight: an identified work's artist is indexed in the background, never during a run
        self.indexer = ArtistIndexer(self.state, lambda: Docent(self.cfg))
        self._stop = threading.Event()

    @property
    def docent(self) -> Docent:
        """This worker thread's own Docent: a run keeps its working state on the models' objects."""
        d = getattr(self._local, "docent", None)
        if d is None:
            d = self._local.docent = Docent(self.cfg)
        return d

    # ── watcher: folder → tasks ────────────────────────────────────────────
    def scan(self) -> list[str]:
        """Queue every stable new image. A file leaves the inbox only once its task exists, so a Pulse
        failure leaves it where it is (retried after RETRY_S) instead of moving it with no task."""
        created = []
        self.registry.reload()
        for f in sorted(self.watch.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS or f.name.startswith("."):
                continue
            size = f.stat().st_size
            if self._sizes.get(f.name) != size:     # still being written: wait for a stable size
                self._sizes[f.name] = size
                continue
            key = (f.name, size)
            if time.time() - self._failed.get(key, 0.0) < RETRY_S:
                continue
            try:
                tid = self._intake(f)
            except requests.RequestException as e:
                first = key not in self._failed
                self._failed[key] = time.time()
                log.error("could not queue %s, left in the inbox (retry in %ds): %s", f.name, RETRY_S, e)
                if first:
                    self.pulse.alert("warning", f"neural-canvas could not queue {f.name}",
                                     {"file": f.name, "error": str(e)[:300]})
                continue
            self._sizes.pop(f.name, None)
            self._failed.pop(key, None)
            if tid:
                created.append(tid)
        return created

    def _intake(self, f: Path) -> str | None:
        try:
            ph = phash(load_image(f))
        except Exception as e:          # unreadable: queue it anyway, the run will block and say why
            log.warning("could not hash %s: %s", f.name, e)
            ph = None
        twin = self.registry.near(ph) if ph is not None and not f.name.startswith("requeue-") else None
        if twin:
            dest = self.duplicates / f.name
            dest = dest if not dest.exists() else self.duplicates / f"{int(time.time())}-{f.name}"
            like = twin[1].get("original_name") or twin[1].get("title", "")
            t = self.pulse.create(
                title=f"Docent: duplicate set aside — {f.name[:60]}",
                description=(f"{f.name} is {twin[2]} bits (of 64) from run {twin[0]} ({like}) by perceptual hash, "
                             f"so it was not queued. It was moved to {dest}.\n\nIf it is a different work, move it "
                             "back into the watch folder with its name prefixed 'requeue-' and it will be queued."),
                labels=[self.runner_label, self.project_label, "agent:aifred", "type:docent-run",
                        "outcome:duplicate", *FINISHED],
                metadata={"original_name": f.name, "duplicate_of": twin[0], "hamming": twin[2], "path": str(dest)})
            shutil.move(str(f), dest)
            self.pulse.close(t["id"], f"duplicate of run {twin[0][:8]} ({twin[2]} bits)")
            log.info("duplicate set aside: %s is %d bits from run %s (%s)", f.name, twin[2], twin[0][:8], like)
            return None
        rid, src = ingest(self.cfg, f)
        dest = self.ingested / f.name
        dest = dest if not dest.exists() else self.ingested / f"{rid[:8]}-{f.name}"
        if rid in self.known:
            shutil.move(str(f), dest)
            log.info("already known %s (%s) — not re-queued", f.name, self.known[rid])
            return None
        t = self.pulse.create(
            title=f"Docent: new artwork {rid[:8]}",
            description=(f"New image detected in the neural-canvas watch folder.\n\nRun id: {rid}\n"
                         f"Original file: {f.name}\nWork dir: {src.parent}\n\n"
                         "Owned by the neural-canvas runner (local models). Stages: identify → interpret → "
                         "compose → upscale → label → deliver."),
            labels=[self.runner_label, self.project_label, "agent:aifred", "type:docent-run",
                    "staging:wait", "blocked:no"],
            metadata={"run_id": rid, "source_path": str(src), "original_name": f.name})
        self.known[rid] = t["id"]
        self.known_path.write_text(json.dumps(self.known, indent=1))
        self.registry.record(rid, phash=ph, status="queued", original_name=f.name)
        shutil.move(str(f), dest)
        log.info("queued %s as %s", f.name, t["id"])
        return t["id"]

    # ── runner: tasks → runs ──────────────────────────────────────────────
    def requeue_stranded(self) -> list[str]:
        """Tasks left active:running by a service that stopped mid-run go back to the queue. Only one
        service runs at a time (service.lock), so at startup no run of ours is really in progress."""
        out = []
        for t in self.pulse.list(self.runner_label):
            labels = t.get("labels", [])
            if t.get("status") != "closed" and "active:running" in labels:
                self.pulse.labels(t["id"], add=["staging:wait", "evaluated:no", "queued:no", "active:no"],
                                  remove=["active:running", "staging:done", "evaluated:done", "queued:done",
                                          *[l for l in labels if l.startswith("docent:")]])
                self.pulse.note(t["id"], "re-queued: the service stopped during this run", status="open")
                out.append(t["id"])
        if out:
            log.warning("re-queued %d task(s) stranded by the last stop: %s", len(out), ", ".join(out))
        return out

    def run_ready(self) -> int:
        """Start ready tasks on free workers; returns how many were started. Runs go on in the
        background while the watcher keeps queuing new files."""
        for tid, fut in list(self._running.items()):
            if fut.done():
                del self._running[tid]
        free = self.workers - len(self._running)
        if free <= 0:
            return 0
        n = 0
        ready = [t for t in self.pulse.list(self.runner_label)
                 if t.get("status") != "closed" and "staging:wait" in t.get("labels", [])
                 and (t.get("metadata") or {}).get("run_id")     # record-only tasks carry no run
                 and t["id"] not in self._running]
        for i, t in enumerate(ready):
            if n >= free:
                break
            # checked before EVERY claim: a run takes minutes, the queue days, and a sign-in can lapse
            # midway. Every run would then end blocked at deliver, so leave the queue untouched and say so.
            why = self.photos.check() if self.cfg["google_photos"].get("enabled", True) else ""
            if why:
                log.error("runner paused, %d tasks waiting: %s", len(ready) - i, why)
                if time.time() - self._auth_alerted > 3600:
                    self.pulse.alert("critical", f"neural-canvas paused: {why}", {"waiting": len(ready) - i})
                    self._auth_alerted = time.time()
                return n
            if not self.pulse.claim(t["id"], self.runner_label):
                continue
            self._running[t["id"]] = self._pool.submit(self._run_guarded, t["id"])
            n += 1
        return n

    def _run_guarded(self, tid: str):
        """A worker's run. An error escaping the run blocks its task and says so, never strands it."""
        try:
            self.run_task(self.pulse.get(tid))
        except Exception as e:
            log.exception("run of task %s failed", tid)
            try:
                self.pulse.labels(tid, add=["blocked:yes", "reason:docent-error", "active:no"],
                                  remove=["active:running"])
                self.pulse.note(tid, f"BLOCKED: runner error {type(e).__name__}: {str(e)[:500]}", status="open")
                self.pulse.alert("warning", f"neural-canvas task {tid} failed in the runner", {"task_id": tid,
                                 "error": f"{type(e).__name__}: {str(e)[:300]}"})
            except requests.RequestException:
                log.error("task %s could not be marked blocked (Pulse unreachable)", tid)

    def run_task(self, task: dict):
        tid, meta = task["id"], task.get("metadata") or {}
        rid, src = meta["run_id"], Path(meta["source_path"])
        threading.current_thread().name = f"run-{rid[:8]}"
        self.pulse.labels(tid, remove=[l for l in task.get("labels", []) if l.startswith(("docent:", "reason:"))])
        self.pulse.note(tid, "claimed by neural-canvas runner")
        cur = {"stage": None}

        def on_stage(stage):
            # a task closed on the board is a decision: the run ends before its next stage (Sir, 2026-11-05)
            try:
                closed = (self.pulse.get(tid) or {}).get("status") == "closed"
            except requests.RequestException:
                closed = False                  # Pulse unreachable: carry on; the next stage asks again
            if closed:
                raise RunStopped("task closed on the board")
            if cur["stage"]:
                self.pulse.labels(tid, remove=[f"docent:{cur['stage']}"])
            cur["stage"] = stage
            self.pulse.labels(tid, add=[f"docent:{stage}"])
            self.pulse.note(tid, f"stage: {stage}")

        t0 = time.time()
        res = self.docent.run(rid, src, resume=True, on_stage=on_stage)
        rec = json.loads(Path(res.record_path).read_text()) if res.record_path else {}
        ident = rec.get("stages", {}).get("identify", {})
        summary = {"run_id": rid, "status": res.status, "record": res.record_path, "frame": res.frame_path,
                   "artist": ident.get("artist", ""), "title": ident.get("title", ""), "date": ident.get("date", ""),
                   "minutes": round((time.time() - t0) / 60, 1)}
        if res.status == "stopped":         # the board closed it: record why, change nothing else
            log.info("run %s stopped before %s: task %s was closed on the board", rid[:8], res.stage, tid)
            self.pulse.note(tid, f"STOPPED before {res.stage}: the task was closed on the board; nothing delivered")
            return
        if cur["stage"]:
            self.pulse.labels(tid, remove=[f"docent:{cur['stage']}"])
        if res.status == "complete":
            ruled = rec.get("stages", {}).get("catalogue", {}).get("facts", {}).get("artist") or summary["artist"]
            self.indexer.enqueue(ruled)        # "have I indexed this artist yet?" (no-op when already done)
            self.pulse.labels(tid, add=["active:done", "completed:done"], remove=["active:running"])
            self.pulse.note(tid, f"COMPLETE: {summary['artist']} — {summary['title']} ({summary['date']}) "
                                 f"→ {res.frame_path}", metadata={**meta, "result": summary})
            self.pulse.close(tid, f"delivered: {summary['artist']} — {summary['title']}")
        elif res.status == "duplicate":
            self.pulse.labels(tid, add=["active:done", "completed:done", "outcome:duplicate"],
                              remove=["active:running"])
            self.pulse.note(tid, f"DUPLICATE: {'; '.join(res.reasons)}", metadata={**meta, "result": summary})
            self.pulse.close(tid, "duplicate of a delivered artwork")
        else:
            reason = f"reason:docent-{res.stage or 'error'}"
            self.pulse.labels(tid, add=["blocked:yes", reason, "active:no"], remove=["active:running"])
            self.pulse.note(tid, f"BLOCKED at {res.stage}: {'; '.join(res.reasons)[:800]}",
                            metadata={**meta, "result": {**summary, "reasons": res.reasons}}, status="open")
            self.pulse.alert("warning", f"neural-canvas run {rid[:8]} blocked at {res.stage}",
                             {"task_id": tid, "reasons": res.reasons[:5]})

    # ── daemon ────────────────────────────────────────────────────────────
    def serve(self, once: bool = False):
        lock = open(self.state / "service.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another neural-canvas service is running")
        lock.write(str(os.getpid()))
        lock.flush()
        log.info("watching %s; Pulse %s; %d worker(s)", self.watch, self.cfg["pulse"]["url"], self.workers)
        try:
            self.requeue_stranded()
        except requests.RequestException as e:
            log.error("Pulse unreachable, stranded tasks not checked: %s", e)
        if not once:
            threading.Thread(target=self.indexer.run, args=(self._stop,), name="indexer", daemon=True).start()
        while True:
            try:
                self.scan()
                if once:              # a single pass still needs the stable-size second look
                    time.sleep(1)
                    self.scan()
                self.run_ready()
            except requests.RequestException as e:
                log.error("Pulse unreachable: %s", e)
            if once:
                for fut in list(self._running.values()):
                    fut.result()
                return
            time.sleep(self.cfg["pulse"]["poll_s"])
