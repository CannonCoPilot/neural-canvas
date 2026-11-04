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
import time
from pathlib import Path

import requests

from .config import Config
from .dedup import ArtworkRegistry, phash
from .pipeline import IMAGE_EXTS, Docent, ingest, load_image

log = logging.getLogger(__name__)
ACTOR = "neural-canvas"


class Pulse:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.s = requests.Session()

    def _r(self, method: str, path: str, **kw):
        r = self.s.request(method, f"{self.url}{path}", timeout=30, **kw)
        if r.status_code == 409:
            return None
        r.raise_for_status()
        return r.json() if r.content else {}

    def create(self, title: str, description: str, labels: list[str], metadata: dict) -> dict:
        return self._r("POST", "/tasks", json={"title": title, "description": description, "priority": "medium",
                                              "labels": labels, "metadata": metadata, "actor": ACTOR})

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
        self.registry = ArtworkRegistry(cfg.path("state_dir"))
        self._docent: Docent | None = None

    @property
    def docent(self) -> Docent:
        if self._docent is None:
            self._docent = Docent(self.cfg)
        return self._docent

    # ── watcher: folder → tasks ────────────────────────────────────────────
    def scan(self) -> list[str]:
        created = []
        self.registry.reload()
        for f in sorted(self.watch.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS or f.name.startswith("."):
                continue
            size = f.stat().st_size
            if self._sizes.get(f.name) != size:     # still being written: wait for a stable size
                self._sizes[f.name] = size
                continue
            self._sizes.pop(f.name, None)
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
                            "completed:duplicate", "blocked:no"],
                    metadata={"original_name": f.name, "duplicate_of": twin[0], "hamming": twin[2], "path": str(dest)})
                shutil.move(str(f), dest)
                self.pulse.close(t["id"], f"duplicate of run {twin[0][:8]} ({twin[2]} bits)")
                log.info("duplicate set aside: %s is %d bits from run %s (%s)", f.name, twin[2], twin[0][:8], like)
                continue
            rid, src = ingest(self.cfg, f)
            dest = self.ingested / f.name
            shutil.move(str(f), dest if not dest.exists() else self.ingested / f"{rid[:8]}-{f.name}")
            if rid in self.known:
                log.info("already known %s (%s) — not re-queued", f.name, self.known[rid])
                continue
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
            self.registry.record(rid, phash=ph, status="queued", original_name=f.name)
            self.known_path.write_text(json.dumps(self.known, indent=1))
            log.info("queued %s as %s", f.name, t["id"])
            created.append(t["id"])
        return created

    # ── runner: tasks → runs ──────────────────────────────────────────────
    def run_ready(self) -> int:
        n = 0
        for t in self.pulse.list(self.runner_label):
            if t.get("status") == "closed" or "staging:wait" not in t.get("labels", []):
                continue
            if not self.pulse.claim(t["id"], self.runner_label):
                continue
            self.run_task(self.pulse.get(t["id"]))
            n += 1
        return n

    def run_task(self, task: dict):
        tid, meta = task["id"], task.get("metadata") or {}
        rid, src = meta["run_id"], Path(meta["source_path"])
        self.pulse.labels(tid, remove=[l for l in task.get("labels", []) if l.startswith(("docent:", "reason:"))])
        self.pulse.note(tid, "claimed by neural-canvas runner")
        cur = {"stage": None}

        def on_stage(stage):
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
        if cur["stage"]:
            self.pulse.labels(tid, remove=[f"docent:{cur['stage']}"])
        if res.status == "complete":
            self.pulse.labels(tid, add=["active:done", "completed:done"], remove=["active:running"])
            self.pulse.note(tid, f"COMPLETE: {summary['artist']} — {summary['title']} ({summary['date']}) "
                                 f"→ {res.frame_path}", metadata={**meta, "result": summary})
            self.pulse.close(tid, f"delivered: {summary['artist']} — {summary['title']}")
        elif res.status == "duplicate":
            self.pulse.labels(tid, add=["active:done", "completed:duplicate"], remove=["active:running"])
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
        log.info("watching %s; Pulse %s", self.watch, self.cfg["pulse"]["url"])
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
                return
            time.sleep(self.cfg["pulse"]["poll_s"])
