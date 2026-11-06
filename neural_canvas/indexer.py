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
        self._lock = threading.Lock()

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
        with self._lock:
            q = self._load(self.queue_path, [])
            if self.indexed(a) or a.lower() in {x.lower() for x in q}:
                return False
            q.append(a)
            self._save(self.queue_path, q)
        log.info("first sight of %s: queued for indexing (%d waiting)", a, len(q))
        return True

    def _pop(self) -> str | None:
        with self._lock:
            q = self._load(self.queue_path, [])
            return q[0] if q else None

    def _finish(self, artist: str, result: dict) -> None:
        with self._lock:
            q = [x for x in self._load(self.queue_path, []) if x.lower() != artist.lower()]
            self._save(self.queue_path, q)
            done = self._load(self.done_path, {})
            done[artist.lower()] = {"artist": artist, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **result}
            self._save(self.done_path, done)

    @staticmethod
    def index_artist(docent, artist: str) -> dict:
        """Embed every catalogued picture of the artist: Wikidata (all works with an image), the NGA's
        works, the Artvee catalogue. Each source is cached on disk; a failure in one is recorded, not fatal."""
        from .catalogue import Catalogue
        from .nga import NGA
        idf, out = docent.identifier, {}
        t0 = time.time()
        try:
            qid = idf.wiki.find_artist(artist)
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
