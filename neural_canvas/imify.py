"""Imify — the free, self-hosted reverse-image arm (github.com/Mr-DS-ML-85/Imify).

Imify uploads the picture itself to Yandex and Bing visual search (and SearXNG's reverse engines),
then checks every hit with perceptual hashes. That finds work that text search cannot reach: an
obscure or contemporary painting whose captions use none of the words we would guess.

It runs as a loopback-only local service (``~/Claude/Staging/verified/imify/run-imify.sh``), started
in parallel with the other tiers at the start of identification. Imify's own verdict is only a lead:
each hit becomes a ``WebReference`` candidate and must pass the same two-instrument match as every
other source before it confirms anything.
"""
from __future__ import annotations

import io
import logging
import re
from urllib.parse import unquote, urlparse

import requests
from PIL import Image

from .websearch import WebReference

log = logging.getLogger(__name__)

# Imify's verdict ladder: copies first, then "visually similar" leads, then unverified engine hits
_RANK = {"exact": 0, "near": 1, "modified": 2, "changed": 3, "similar": 4, "unverified": 5}


def _slug_title(url: str) -> str:
    """A page whose title Imify could not read is often named by its URL: 'gallery/tomioka-soichiro/'."""
    parts = [p for p in urlparse(url).path.split("/") if p and not re.fullmatch(r"[\d-]+|\w{1,3}", p)]
    return re.sub(r"[-_+]+", " ", unquote(" / ".join(parts[-2:]))).strip()


class Imify:
    def __init__(self, cfg: dict):
        self.url = cfg["url"].rstrip("/")
        self.timeout_s = cfg.get("timeout_s", 90)
        self.num_results = cfg.get("num_results", 40)

    def search_image(self, img: Image.Image) -> list[WebReference]:
        im = img.convert("RGB")
        im.thumbnail((2048, 2048))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        r = requests.post(f"{self.url}/v1/search", files={"file": ("query.jpg", buf.getvalue(), "image/jpeg")},
                          data={"num_results": str(self.num_results), "include_web": "true", "learn": "false"},
                          timeout=self.timeout_s)
        r.raise_for_status()
        d = r.json()
        if d.get("degraded"):
            log.info("imify degraded providers: %s", d["degraded"])
        rows = [x for x in d.get("results", []) if x.get("url", "").startswith("http")]
        rows.sort(key=lambda x: (_RANK.get(x.get("match_type"), 9), -(x.get("score") or 0)))
        out = []
        for x in rows:
            page = x.get("page_url") or ""
            title = x.get("title") or ""
            if not title or title.startswith("http"):
                title = _slug_title(page)
            out.append(WebReference(file_title=title, page_id=hash(x["url"]) & 0x7FFFFFFFFFFF, thumb_url=x["url"],
                                    page_url=page, image_url=x["url"],
                                    meta={"source": "imify", "engine": x.get("engine", ""),
                                          "imify_match": x.get("match_type", ""), "imify_score": x.get("score"),
                                          "source_urls": x.get("source_urls", [])[:5]}))
        log.info("imify: %d candidates (%s) in %.1fs, degraded %s", len(out),
                 ", ".join(f"{x.get('match_type')}" for x in rows[:6]), d.get("took_ms", 0) / 1000, d.get("degraded"))
        return out
