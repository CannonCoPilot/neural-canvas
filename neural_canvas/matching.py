"""Two independent "is this the same artwork?" instruments.

* ``Embedder.similarity`` — CLIP image-image cosine. Robust to colour/scale/photo differences,
  but two paintings of the same subject in the same style can also score high.
* ``geometric_inliers`` — ORB keypoints + RANSAC homography inlier count. Only the *same* physical
  painting yields a large consistent planar mapping; style twins do not.

Identification requires agreement of both (or an overwhelming embedding score), so a shared
weakness of one instrument cannot confirm an identity on its own.
"""
from __future__ import annotations

import io
import logging
import os
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")   # weights are cached; no network round-trips per run
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        ims = [i.convert("RGB") for i in images]
        return self.model.encode(ims, normalize_embeddings=True, convert_to_numpy=True)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return b @ a


def _gray(img: Image.Image, side: int = 900) -> np.ndarray:
    im = img.convert("L")
    im.thumbnail((side, side), Image.LANCZOS)
    return np.asarray(im)


def geometric_inliers(a: Image.Image, b: Image.Image) -> int:
    ga, gb = _gray(a), _gray(b)
    orb = cv2.ORB_create(nfeatures=3000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < 8:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return int(mask.sum()) if mask is not None else 0


_CACHE: dict[str, Image.Image] = {}
_last_by_host: dict[str, float] = {}
WIKIMEDIA_INTERVAL_S = 0.12   # serial and paced: Wikimedia answers bursts with HTTP 429


def fetch_image(url: str, user_agent: str) -> Image.Image | None:
    """Fetch a reference image into memory (never written to disk). Successes are cached; failures are
    not, so a transient error is retried next time. Wikimedia hosts are paced and 429s honour Retry-After."""
    if url in _CACHE:
        return _CACHE[url]
    host = urlparse(url).netloc
    wm = host.endswith("wikimedia.org") or host.endswith("wikipedia.org")
    for attempt in range(4):
        if wm:
            wait = WIKIMEDIA_INTERVAL_S - (time.time() - _last_by_host.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            _last_by_host[host] = time.time()
        try:
            r = requests.get(url, headers={"User-Agent": user_agent}, timeout=12)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(30.0, float(r.headers.get("Retry-After") or 2 ** (attempt + 1))))
                continue
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            if len(_CACHE) > 600:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[url] = im
            return im
        except Exception as e:  # a dead reference is just a non-match; logged, not hidden
            log.warning("reference fetch failed %s: %s", url[:120], e)
            return None
    log.warning("reference fetch gave up after rate limiting: %s", url[:120])
    return None
