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


def homography(a: Image.Image, b: Image.Image) -> tuple[np.ndarray | None, int, float]:
    """ORB + RANSAC: (H mapping b's working-size pixels onto a's, inlier count, a's working scale).
    The working copies are at most 900 px on a side; ``scale`` converts a's working pixels to its own."""
    ga, gb = _gray(a), _gray(b)
    scale = a.width / ga.shape[1]
    orb = cv2.ORB_create(nfeatures=3000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0, scale
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    if len(good) < 8:
        return None, 0, scale
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 6.0)       # b -> a
    return H, (int(mask.sum()) if mask is not None else 0), scale


def geometric_inliers(a: Image.Image, b: Image.Image) -> int:
    return homography(a, b)[1]


_CACHE: dict[str, Image.Image] = {}
_last_by_host: dict[str, float] = {}
WIKIMEDIA_INTERVAL_S = 0.12   # serial and paced: Wikimedia answers bursts with HTTP 429
# seconds between fetches per host family. The Art Institute allows 60 a minute; a burst past that is met
# with a Cloudflare challenge (HTTP 403, cf-mitigated) on every image for some minutes (2026-11-05).
HOST_INTERVAL_S = {"wikimedia.org": WIKIMEDIA_INTERVAL_S, "wikipedia.org": WIKIMEDIA_INTERVAL_S, "artic.edu": 1.0}


def _interval(host: str) -> float:
    return next((v for k, v in HOST_INTERVAL_S.items() if host == k or host.endswith("." + k)), 0.0)


def fetch_image(url: str, user_agent: str) -> Image.Image | None:
    """Fetch a reference image into memory (never written to disk). Successes are cached; failures are
    not, so a transient error is retried next time. Rate-limited hosts are paced (HOST_INTERVAL_S); a 429,
    a 5xx or a rate challenge backs off (Retry-After when given) and tries again."""
    if url in _CACHE:
        return _CACHE[url]
    host = urlparse(url).netloc
    gap = _interval(host)
    for attempt in range(4):
        if gap:
            wait = gap - (time.time() - _last_by_host.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            _last_by_host[host] = time.time()
        try:
            r = requests.get(url, headers={"User-Agent": user_agent}, timeout=12)
            challenged = r.status_code == 403 and r.headers.get("cf-mitigated") == "challenge"
            if r.status_code == 429 or r.status_code >= 500 or challenged:
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
