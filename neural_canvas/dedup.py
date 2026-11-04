"""Duplicate artworks: each painting is processed and delivered once.

Two checks, at two moments:

* at intake, by picture — a 64-bit DCT perceptual hash of every image the folder has ever queued.
  A resized, recompressed or re-saved copy lands within a few bits of the original, so it is set
  aside in ``duplicates/`` instead of becoming a second run.
* after identification, by identity — two different photographs of the same painting hash far
  apart, but they confirm to the same artist and title. If that work is already in the album, the
  run ends as ``duplicate`` rather than delivering it twice.

The registry is ``state_dir/artworks.json``; the runner is sequential, so a copy queued behind its
twin is checked only after the twin has delivered.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

PHASH_NEAR = 10          # bits of 64. Measured on the 578-image collection (2026-11-04): copies of one
                         # image 0-10, different paintings 16 and up (most 18-24)
# titles that name a kind of picture rather than one picture
GENERIC_TITLE_WORDS = {"untitled", "composition", "no", "nr", "number", "study", "sketch", "self", "portrait",
                       "still", "life", "landscape", "seascape", "figure", "figures", "abstract", "abstraction",
                       "painting", "drawing", "nude", "head", "woman", "man", "girl", "boy", "interior", "flowers",
                       "improvisation", "series", "plate", "pl", "fig", "opus", "op", "i", "ii", "iii", "iv", "v"}
_N = 32
_DCT = np.cos(np.pi * (2 * np.arange(_N)[None, :] + 1) * np.arange(_N)[:, None] / (2 * _N))


def phash(img: Image.Image) -> int:
    """Classic pHash: 32x32 grey → 2-D DCT → the 8x8 lowest frequencies against their median."""
    g = np.asarray(img.convert("L").resize((_N, _N), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (_DCT @ g @ _DCT.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])          # the DC term is brightness, not structure
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def work_key(artist: str, title: str) -> str:
    """Surname + title words, case/accents/punctuation/articles folded: 'Andrew Wyeth' / 'The Carry'
    and 'A. Wyeth' / 'Carry' are the same work."""
    def fold(s):
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
        return [w for w in re.findall(r"[a-z0-9]+", s) if w not in {"the", "a", "an", "of", "le", "la", "les", "de"}]
    surname, words = (fold(artist) or [""])[-1], fold(title)
    if not surname or all(w in GENERIC_TITLE_WORDS or w.isdigit() for w in words):
        return ""      # 'Untitled', 'Composition No. 7', 'Self-Portrait': many works share these
    return f"{surname}|{' '.join(words)}"


class ArtworkRegistry:
    def __init__(self, state_dir: Path):
        self.path = state_dir / "artworks.json"
        self.works: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.works, indent=1, ensure_ascii=False))
        os.replace(tmp, self.path)

    def reload(self):
        if self.path.exists():
            self.works = json.loads(self.path.read_text())

    def near(self, ph: int, exclude: str = "") -> tuple[str, dict, int] | None:
        """The closest registered image within PHASH_NEAR bits, if any."""
        best = None
        for rid, w in self.works.items():
            if rid != exclude and w.get("phash") is not None:
                d = hamming(ph, int(w["phash"]))
                if d <= PHASH_NEAR and (best is None or d < best[2]):
                    best = (rid, w, d)
        return best

    def delivered_twin(self, artist: str, title: str, exclude: str = "") -> tuple[str, dict] | None:
        key = work_key(artist, title)
        for rid, w in self.works.items():
            if key and rid != exclude and w.get("status") == "delivered" and w.get("key") == key:
                return rid, w
        return None

    def record(self, rid: str, **fields):
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._record(rid, **fields)

    def _record(self, rid: str, **fields):
        self.reload()
        w = self.works.setdefault(rid, {"added": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        if w.get("status") == "delivered":
            fields.pop("status", None)          # a resumed run never un-delivers a work
        w.update({k: v for k, v in fields.items() if v is not None})
        if "phash" in fields and fields["phash"] is not None:
            w["phash"] = str(fields["phash"])      # 64-bit ints survive JSON as strings
        if fields.get("artist") or fields.get("title"):
            w["key"] = work_key(w.get("artist", ""), w.get("title", ""))
        self._save()
