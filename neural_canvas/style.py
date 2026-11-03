"""Style shortlist: which painters does this picture *look like*?

An artist-candidate generator independent of the VLM: CLIP image embedding against CLIP text
embeddings of "a painting by <name>" for every painter with ≥30 catalogued paintings on Wikidata.
It is strong exactly where the VLM is weak (less famous painters with a distinctive manner) and is
only ever a candidate source — every candidate still has to pass the two-instrument match.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
ROSTER_Q = ("SELECT ?a (COUNT(?w) AS ?n) WHERE { ?w wdt:P31 wd:Q3305213; wdt:P18 ?img; wdt:P170 ?a. } "
            "GROUP BY ?a HAVING (COUNT(?w) >= 30)")
MAX_AGE_S = 90 * 86400


class StyleShortlist:
    def __init__(self, wiki, embedder, state_dir: Path):
        self.wiki, self.emb = wiki, embedder
        self.path_json = state_dir / "painter-roster.json"
        self.path_vec = state_dir / "painter-roster.npy"
        self.names: list[str] = []
        self.vecs: np.ndarray | None = None

    def _build(self):
        from .references import SPARQL
        log.info("building painter roster from Wikidata")
        r = self.wiki.s.get(SPARQL, params={"query": ROSTER_Q, "format": "json"}, timeout=120)
        r.raise_for_status()
        ids = [b["a"]["value"].rsplit("/", 1)[-1] for b in r.json()["results"]["bindings"]]
        names = {}
        for i in range(0, len(ids), 50):
            for q, e in self.wiki.entities(ids[i:i + 50]).items():
                if self.wiki.label(e):
                    names[q] = self.wiki.label(e)
        self.names = sorted(set(names.values()))
        self.vecs = self.emb.embed_text([f"a painting by {n}" for n in self.names])
        self.path_json.write_text(json.dumps(self.names))
        np.save(self.path_vec, self.vecs)

    def _load(self):
        if self.vecs is not None:
            return
        fresh = self.path_vec.exists() and time.time() - self.path_vec.stat().st_mtime < MAX_AGE_S
        if fresh:
            self.names = json.loads(self.path_json.read_text())
            self.vecs = np.load(self.path_vec)
        else:
            self._build()

    def top(self, img_vec: np.ndarray, k: int = 4) -> list[tuple[str, float]]:
        self._load()
        sc = self.vecs @ img_vec
        order = np.argsort(-sc)[:k]
        return [(self.names[i], round(float(sc[i]), 4)) for i in order]
