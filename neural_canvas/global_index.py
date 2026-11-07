"""One searchable index over every picture already embedded: first look's local arm (Sir, 2026-11-06).

Sources: the Wikidata catalogue sweeps (state/oeuvre/<artist qid>.npz), the Artvee artist catalogues
(state/artvee/<slug>.npz + .json). The index grows as the first-sight indexer and the top-500 build embed
more artists; ``rebuild`` folds them in. The Wikidata caches hold ids only, so each artist's listing (title,
image file) is fetched once from SPARQL and kept under state/global_index/listings/, which makes a rebuild
cost only the artists added since the last one.

Searching is one matrix product (31k works: about 2 s with fetching the closest few). A reference from a
Wikidata row carries its item id, so a confirmed match reads its facts from Wikidata directly.
"""
from __future__ import annotations

import fcntl
import glob
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from .references import file_thumb
from .websearch import WebReference

log = logging.getLogger(__name__)


_SHARED: dict[str, tuple] = {}     # path -> (mtime, vecs, meta): one copy per process (110k works = 225 MB)
_SHARED_LOCK = threading.Lock()


class GlobalIndex:
    def __init__(self, state_dir: Path):
        self.state = state_dir
        self.dir = state_dir / "global_index"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def npz(self) -> Path:
        return self.dir / "cached.npz"

    def _load(self):
        if not self.npz.exists():
            return None, None
        m = self.npz.stat().st_mtime
        key = str(self.npz)
        with _SHARED_LOCK:
            have = _SHARED.get(key, (0.0, None, None))
            if have[0] != m:                       # reloaded when a rebuild replaces the file
                vecs = np.load(self.npz)["vecs"]
                meta = json.load(open(self.dir / "cached.json"))
                if len(meta) != len(vecs):         # caught between a rebuild's two renames: keep the last pair
                    return have[1], have[2]
                have = _SHARED[key] = (m, vecs, meta)
            return have[1], have[2]

    def __len__(self) -> int:
        _, meta = self._load()
        return len(meta or [])

    def nearest(self, vec: np.ndarray, k: int = 10) -> list[WebReference]:
        vecs, meta = self._load()
        if vecs is None:
            raise RuntimeError(f"no global index at {self.npz}: run `neural-canvas build-global-index`")
        out = []
        for i in np.argsort(-(vecs @ vec))[:k]:
            m = meta[int(i)]
            out.append(WebReference(f"{m['title']} — {m['artist']}", hash(str(m["id"])) & 0x7FFFFFFFFFFF, m["image"],
                                    m.get("page", ""), m["image"],
                                    {"source": f"cached_index:{m['src']}", "artist": m["artist"], "title": m["title"],
                                     **({"wikidata": m["id"]} if m["src"] == "wikidata" else {})}))
        return out

    # ── building ──────────────────────────────────────────────────────────
    def _listing(self, wiki, aq: str) -> dict | None:
        f = self.dir / "listings" / f"{aq}.json"
        if f.exists():
            return json.loads(f.read_text())
        try:
            side = self.state / "oeuvre" / f"{aq}.json"      # the sweep keeps its listing beside the vectors
            works = json.loads(side.read_text())["works"] if side.exists() else wiki.oeuvre(aq)
            d = {"artist": wiki.label(wiki.entities([aq]).get(aq, {})),
                 "works": {w["qid"]: {"label": w["label"], "file": w["file"]} for w in works}}
        except Exception as e:
            log.warning("global index: listing for %s failed: %s", aq, e)
            return None
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(d))
        return d

    def rebuild(self, wiki) -> int:
        """Rewrite the index from every per-artist cache; one builder at a time across processes."""
        with open(self.dir / "build.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            t0 = time.time()
            vecs, meta = [], []
            for f in sorted(glob.glob(str(self.state / "oeuvre" / "*.npz"))):
                aq = Path(f).stem
                lst = self._listing(wiki, aq)
                if not lst:
                    continue
                z = np.load(f)
                for q, v in zip(z["qids"].tolist(), z["vecs"]):
                    w = lst["works"].get(q)
                    if w:
                        vecs.append(v)
                        meta.append({"src": "wikidata", "id": q, "artist": lst["artist"], "title": w["label"],
                                     "image": file_thumb(w["file"], 500), "page": f"https://www.wikidata.org/wiki/{q}"})
            n = len(meta)
            for f in sorted(glob.glob(str(self.state / "artvee" / "*.npz"))):
                js = Path(f).with_suffix(".json")
                if not js.exists():
                    continue
                by = {w["thumb_url"]: w for w in json.load(open(js))}
                z = np.load(f)
                for u, v in zip(z["urls"].tolist(), z["vecs"]):
                    w = by.get(u)
                    if w:
                        vecs.append(v)
                        meta.append({"src": "artvee", "id": u, "artist": Path(f).stem.replace("-", " "),
                                     "title": w.get("file_title", ""), "image": w.get("image_url") or u,
                                     "page": w.get("page_url", "")})
            if not vecs:
                raise RuntimeError("global index: no embedded catalogues found")
            tmp = self.dir / "cached.tmp.npz"
            np.savez(tmp, vecs=np.stack(vecs).astype(np.float32))
            (self.dir / "cached.tmp.json").write_text(json.dumps(meta))
            (self.dir / "cached.tmp.json").replace(self.dir / "cached.json")
            tmp.replace(self.npz)                  # the npz last: its mtime is what readers watch
            log.info("global index rebuilt: %d works (%d Wikidata, %d Artvee) in %.0f s", len(meta), n, len(meta) - n,
                     time.time() - t0)
            return len(meta)


def maybe_rebuild(state_dir: Path, wiki, min_gap_s: float = 6 * 3600) -> int | None:
    """Fold newly embedded artists into the index at most every ``min_gap_s`` (the indexer's hook)."""
    gi = GlobalIndex(state_dir)
    if gi.npz.exists() and time.time() - gi.npz.stat().st_mtime < min_gap_s:
        return None
    try:
        return gi.rebuild(wiki)
    except Exception as e:           # the old index keeps serving; the failure is in the log
        log.warning("global index rebuild failed: %s", e)
        return None
