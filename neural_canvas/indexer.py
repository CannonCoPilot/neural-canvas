"""First sight: an artist's catalogues are embedded after a work by them is identified (Sir, 2026-11-05).

A run never sweeps a catalogue cold: embedding every catalogued work by an artist takes minutes at
Wikimedia's pace (about one thumbnail a second), and identification needs only what is already cached.
Once a run has identified a work and the catalogue stage has ruled its artist, the service asks "have I
indexed this artist?" and, if not, queues them here. A background thread then embeds the artist's
Wikidata catalogue, NGA works and Artvee catalogue, so the next work by that artist matches from cache.

The queue and the indexed list live on disk (var/state/index_queue.json, var/state/indexed.json), so a
restart loses nothing; the 500-painter build (scripts/build-top-painters) uses the same index_artist().
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class ArtistIndexer:
    def __init__(self, state_dir: Path, docent_factory):
        self.queue_path = Path(state_dir) / "index_queue.json"
        self.done_path = Path(state_dir) / "indexed.json"
        self._docent_factory = docent_factory
        self._docent = None
        self._tlock = threading.Lock()
        self._lockfile = Path(state_dir) / "indexed.lock"

    @contextlib.contextmanager
    def _lock(self):
        """One writer at a time, across threads and processes (the service and `build-index` share the files)."""
        with self._tlock, open(self._lockfile, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def _load(self, path: Path, default):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return default

    def _save(self, path: Path, data) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        tmp.replace(path)

    def indexed(self, artist: str) -> bool:
        return artist.strip().lower() in self._load(self.done_path, {})

    def enqueue(self, artist: str) -> bool:
        """Queue an identified work's artist unless already indexed or queued. True when newly queued."""
        a = (artist or "").strip()
        if not a:
            return False
        with self._lock():
            q = self._load(self.queue_path, [])
            if self.indexed(a) or a.lower() in {x.lower() for x in q}:
                return False
            q.append(a)
            self._save(self.queue_path, q)
        log.info("first sight of %s: queued for indexing (%d waiting)", a, len(q))
        return True

    def _pop(self) -> str | None:
        with self._lock():
            q = self._load(self.queue_path, [])
            return q[0] if q else None

    def _finish(self, artist: str, result: dict) -> None:
        with self._lock():
            q = [x for x in self._load(self.queue_path, []) if x.lower() != artist.lower()]
            self._save(self.queue_path, q)
            done = self._load(self.done_path, {})
            done[artist.lower()] = {"artist": artist, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **result}
            self._save(self.done_path, done)

    def mark(self, artist: str, result: dict) -> None:
        """Record an artist indexed outside the queue (the `build-index` command)."""
        self._finish(artist, result)

    @staticmethod
    def index_artist(docent, artist: str, qid: str = "") -> dict:
        """Embed every catalogued picture of the artist: Wikidata (all works with an image), the NGA's
        works, the Artvee catalogue. Each source is cached on disk; a failure in one is recorded, not fatal."""
        from .catalogue import Catalogue
        from .nga import NGA
        idf, out = docent.identifier, {}
        t0 = time.time()
        try:
            qid = qid or idf.wiki.find_artist(artist)
            works = idf._oeuvre_vectors(qid, cold=True)[0] if qid else []
            out["wikidata"] = len(works)
        except Exception as e:
            out["wikidata_error"] = f"{type(e).__name__}: {e}"[:200]
        try:
            cat = Catalogue(idf, docent.cfg.thresholds)
            path = docent.cfg["sources"].get("nga_db") or ""
            nga = NGA(Path(path).expanduser()) if path else None
            rows = [r for r in nga.works_by(artist) if r.get("iiif")] if nga is not None and nga.available else []
            if rows:
                cat._nga_vectors(artist, rows[:800], cold=True)
            out["nga"] = len(rows)
        except Exception as e:
            out["nga_error"] = f"{type(e).__name__}: {e}"[:200]
        try:
            out["artvee"] = len(idf.artvee.nearest(artist, np.zeros(512, dtype=np.float32), k=10 ** 6, cold=True))
        except Exception as e:
            out["artvee_error"] = f"{type(e).__name__}: {e}"[:200]
        out["seconds"] = round(time.time() - t0, 1)
        return out

    def run(self, stop: threading.Event, idle_s: float = 30.0) -> None:
        """The background loop: one artist at a time, forever; never raises."""
        while not stop.is_set():
            artist = self._pop()
            if artist is None:
                stop.wait(idle_s)
                continue
            try:
                if self._docent is None:
                    self._docent = self._docent_factory()
                log.info("indexing %s", artist)
                result = self.index_artist(self._docent, artist)
                log.info("indexed %s: %s", artist, result)
            except Exception as e:                       # the artist is dropped from the queue, with the reason
                log.exception("indexing %s failed", artist)
                result = {"error": f"{type(e).__name__}: {e}"[:300]}
            self._finish(artist, result)


TOP_PAINTERS_QUERY = """PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/>
SELECT ?a (COUNT(DISTINCT ?w) AS ?n) WHERE { ?w wdt:P31 wd:Q3305213; wdt:P18 ?i; wdt:P170 ?a. }
GROUP BY ?a ORDER BY DESC(?n) LIMIT %d"""
NOT_A_PAINTER = {"Q4233718", "Q24238356"}        # "anonymous", "unknown": creators without a catalogue


def build_top_painters(cfg, top: int = 500) -> int:
    """Embed the catalogues of the `top` painters with the most catalogued paintings that have an image
    (Sir, 2026-11-05: about 117,600 paintings for 500 painters, roughly a day and a half at Wikimedia's
    pace). Most catalogued first; resumable (indexed artists are skipped); shares the indexed list with the
    service's first-sight indexer."""
    import requests
    from .pipeline import Docent
    ua = cfg["sources"]["user_agent"]
    listing = Path(cfg.path("state_dir")) / f"top_painters_{top}.json"
    if listing.exists():
        painters = json.loads(listing.read_text())
    else:      # QLever answers this whole-Wikidata count in seconds; the public Wikidata endpoint times out
        r = requests.get("https://qlever.dev/api/wikidata", params={"query": TOP_PAINTERS_QUERY % (top + 20)},
                         headers={"User-Agent": ua, "Accept": "application/sparql-results+json"}, timeout=180)
        r.raise_for_status()
        rows = [(b["a"]["value"].rsplit("/", 1)[-1], int(b["n"]["value"])) for b in r.json()["results"]["bindings"]]
        rows = [(q, n) for q, n in rows if q.startswith("Q") and q not in NOT_A_PAINTER][:top]
        d = Docent(cfg)
        labels = {}
        for i in range(0, len(rows), 50):
            ents = d.identifier.wiki.entities([q for q, _ in rows[i:i + 50]])
            labels.update({q: d.identifier.wiki.label(e) for q, e in ents.items()})
        painters = [{"qid": q, "artist": labels.get(q) or q, "paintings": n} for q, n in rows]
        listing.write_text(json.dumps(painters, indent=1, ensure_ascii=False))
    ix = ArtistIndexer(cfg.path("state_dir"), lambda: None)
    d = Docent(cfg)
    todo = [p for p in painters if not ix.indexed(p["artist"])]
    log.info("build-index: %d of %d painters to index", len(todo), len(painters))
    t0 = time.time()
    for k, p in enumerate(todo, 1):
        res = ArtistIndexer.index_artist(d, p["artist"], p["qid"])
        ix.mark(p["artist"], {"qid": p["qid"], **res})
        el = time.time() - t0
        log.info("build-index %d/%d %s: %s (%.1f h elapsed)", k, len(todo), p["artist"], res, el / 3600)
    log.info("build-index done: %d painters", len(painters))
    return 0
