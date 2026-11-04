"""Web image search — the reference tier for works Commons cannot host (in-copyright, contemporary).

Backed by two free text-to-image engines queried in parallel and merged: DuckDuckGo image search via
``ddgs`` (no key), and the local SearXNG that ships with Imify (bing/google/duckduckgo/yahoo/brave
image engines). The sanctioned alternative is Google's
Custom Search JSON API, which the estate holds a key for but which is not enabled on its project
(HTTP 403 "This project does not have the access to Custom Search JSON API").
Results are only ever *candidates*: a web image becomes a reference solely by passing the same
two-instrument match as a Commons file.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


@dataclass
class WebReference:
    file_title: str          # page title / caption
    page_id: int             # stable hash of the image URL (dedup key shared with Commons refs)
    thumb_url: str
    page_url: str
    image_url: str
    meta: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return urlparse(self.page_url or self.image_url).netloc.removeprefix("www.")


class WebImageSearch:
    def __init__(self, min_interval_s: float = 1.5, searxng_url: str = ""):
        self.min_interval_s = min_interval_s
        self.searxng_url = searxng_url.rstrip("/")
        self._last = 0.0

    def search(self, query: str, limit: int = 12) -> list[WebReference]:
        """Both engines at once; their hits interleaved, duplicates (same image URL) dropped.
        One engine failing is logged and the other's results stand; both failing raises."""
        if not self.searxng_url:
            return self._ddg(query, limit)
        with ThreadPoolExecutor(2) as ex:
            futs = {"ddg": ex.submit(self._ddg, query, limit), "searxng": ex.submit(self._searx, query, limit)}
        res, errs = {}, {}
        for name, f in futs.items():
            try:
                res[name] = f.result()
            except Exception as e:
                errs[name] = e
                log.warning("%s image search failed for %r: %s", name, query, e)
        if not res:
            raise RuntimeError(f"web image search unavailable for query {query!r}: {errs}")
        out, seen = [], set()
        for group in zip(*[r + [None] * (limit - len(r)) for r in res.values()]):
            for r in group:
                if r is not None and r.image_url not in seen:
                    seen.add(r.image_url)
                    out.append(r)
        return out

    def _searx(self, query: str, limit: int) -> list[WebReference]:
        r = requests.get(f"{self.searxng_url}/search", params={"q": query, "categories": "images", "format": "json"},
                         timeout=30)
        r.raise_for_status()
        out = []
        for x in r.json().get("results", [])[:limit]:
            img = x.get("img_src") or ""
            if not img.startswith("http"):
                continue
            out.append(WebReference(file_title=x.get("title", ""), page_id=hash(img) & 0x7FFFFFFFFFFF,
                                    thumb_url=x.get("thumbnail_src") or img, page_url=x.get("url", ""),
                                    image_url=img, meta={"source": "searxng", "engine": x.get("engine", "")}))
        return out

    def _ddg(self, query: str, limit: int) -> list[WebReference]:
        from ddgs import DDGS
        wait = self.min_interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                rows = DDGS().images(query, max_results=limit)
                break
            except Exception as e:   # rate limit / transient; retried, then surfaced
                log.warning("web image search failed (%s), attempt %d", e, attempt + 1)
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"web image search unavailable for query {query!r}")
        self._last = time.time()
        out = []
        for r in rows:
            img = r.get("image") or ""
            if not img:
                continue
            out.append(WebReference(file_title=r.get("title", ""), page_id=hash(img) & 0x7FFFFFFFFFFF,
                                    thumb_url=r.get("thumbnail") or img, page_url=r.get("url", ""),
                                    image_url=img, meta={"source": "ddg", "engine": r.get("source", "")}))
        return out
