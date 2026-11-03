"""Web image search — the reference tier for works Commons cannot host (in-copyright, contemporary).

Backed by DuckDuckGo image search via ``ddgs`` (no key). The sanctioned alternative is Google's
Custom Search JSON API, which the estate holds a key for but which is not enabled on its project
(HTTP 403 "This project does not have the access to Custom Search JSON API").
Results are only ever *candidates*: a web image becomes a reference solely by passing the same
two-instrument match as a Commons file.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

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
    def __init__(self, min_interval_s: float = 1.5):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def search(self, query: str, limit: int = 12) -> list[WebReference]:
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
                                    image_url=img, meta={"source": r.get("source", "")}))
        return out
