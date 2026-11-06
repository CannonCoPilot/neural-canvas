"""Web image search — the reference tier for works Commons cannot host (in-copyright, contemporary).

Backed by two free text-to-image engines queried in parallel and merged: DuckDuckGo image search via
``ddgs`` (no key), and the local SearXNG that ships with Imify (bing/google/duckduckgo/yahoo/brave
image engines). The sanctioned alternative is Google's
Custom Search JSON API, which the estate holds a key for but which is not enabled on its project
(HTTP 403 "This project does not have the access to Custom Search JSON API").
Results are only ever *candidates*: a web image becomes a reference solely by passing the same
two-instrument match as a Commons file.

Text (page) search, ``pages()``, also asks keyed engines when their keys are in the credentials store
(2026-11-05): Serper (Google's results) and Brave's own Search API. Google's Custom Search JSON API is
closed to new customers and ends for all on 2027-01-01, so it is not used.
"""
from __future__ import annotations

import logging
import re
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
    def __init__(self, min_interval_s: float = 1.5, searxng_url: str = "", text_search: dict | None = None):
        self.min_interval_s = min_interval_s
        self.searxng_url = searxng_url.rstrip("/")
        self._last = 0.0
        self.text_search = text_search or {}      # {"credentials_path": ..., "providers": {name: key path}}
        self._keys: dict[str, str] = {}

    def _key(self, provider: str) -> str:
        """A keyed provider's key from the credentials store, read once; "" when absent."""
        if provider not in self._keys:
            from .vision_web import _read_key
            path = (self.text_search.get("providers") or {}).get(provider, "")
            try:
                self._keys[provider] = _read_key(self.text_search["credentials_path"], path) if path else ""
            except (KeyError, OSError):
                self._keys[provider] = ""
        return self._keys[provider]

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

    def pages(self, query: str, limit: int = 20) -> list[tuple[str, str]]:
        """Ordinary web search (not images): (url, title) of each result page, from every available
        engine at once, interleaved and deduplicated. Collection pages that no image search returns are
        often the first text hit. One engine failing is logged; the others stand."""
        engines = {"searxng": self._searx_pages} if self.searxng_url else {}
        engines.update({p: getattr(self, f"_{p}_pages") for p in ("serper", "brave") if self._key(p)})
        if not engines:
            return []
        with ThreadPoolExecutor(len(engines)) as ex:
            futs = {name: ex.submit(fn, query, limit) for name, fn in engines.items()}
        lists = []
        for name, f in futs.items():
            try:
                lists.append(f.result())
            except Exception as e:
                msg = str(e)
                for k in self._keys.values():
                    msg = msg.replace(k, "***") if k else msg
                log.warning("text search %s failed for %r: %s", name, query, msg[:200])
        out, seen = [], set()
        for row in zip(*[l + [None] * (limit - len(l)) for l in lists]) if lists else []:
            for hit in row:
                if hit and hit[0] not in seen:
                    seen.add(hit[0])
                    out.append(hit)
        return out[:limit * 2]

    # ── SerpAPI: Google's results on a free plan of 250 searches a month, for unresolved works only ──
    SERPAPI_RESERVE = 10        # searches always left untouched on the plan

    def serpapi_left(self) -> int:
        """Searches left on the SerpAPI plan, from its own account endpoint (which costs no search)."""
        key = self._key("serpapi")
        if not key:
            return 0
        try:
            r = requests.get("https://serpapi.com/account.json", params={"api_key": key}, timeout=20)
            r.raise_for_status()
            return int(r.json().get("total_searches_left") or 0)
        except Exception as e:
            log.warning("serpapi account check failed: %s", str(e).replace(key, "***")[:200])
            return 0

    def _serpapi(self, params: dict) -> dict:
        left = self.serpapi_left()
        if left <= self.SERPAPI_RESERVE:
            raise RuntimeError(f"SerpAPI monthly searches spent ({left} left, {self.SERPAPI_RESERVE} kept in reserve)")
        key = self._key("serpapi")
        try:
            r = requests.get("https://serpapi.com/search.json", params={**params, "api_key": key}, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise RuntimeError(str(e).replace(key, "***")) from None

    def serpapi_pages(self, query: str, limit: int = 20) -> list[tuple[str, str]]:
        d = self._serpapi({"engine": "google", "q": query, "num": min(limit, 20)})
        return [(x["link"], x.get("title", "")) for x in d.get("organic_results", []) if x.get("link", "").startswith("http")]

    def serpapi_images(self, query: str, limit: int = 30) -> list[WebReference]:
        d = self._serpapi({"engine": "google_images", "q": query})
        out = []
        for x in d.get("images_results", [])[:limit]:
            img = x.get("original") or ""
            if img.startswith("http"):
                out.append(WebReference(x.get("title", ""), hash(img) & 0x7FFFFFFFFFFF, x.get("thumbnail") or img,
                                        x.get("link", ""), img, {"source": "serpapi"}))
        return out

    @staticmethod
    def temp_public_url(img) -> str:
        """A copy of the input at a public address for about an hour, for Google Lens, which takes images by URL
        only. Hosts tried in Sir's order (2026-11-05): litterbox.catbox.moe (deletes after 1 h), then
        tmpfiles.org (deletes after 60 min)."""
        import io
        im = img.convert("RGB").copy()
        im.thumbnail((1600, 1600))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        data = buf.getvalue()
        errors = []
        try:
            r = requests.post("https://litterbox.catbox.moe/resources/internals/api.php",
                              data={"reqtype": "fileupload", "time": "1h"},
                              files={"fileToUpload": ("work.jpg", data, "image/jpeg")}, timeout=120)
            r.raise_for_status()
            url = r.text.strip()
            if url.startswith("https://"):
                return url
            errors.append(f"litterbox: {url[:120]}")
        except requests.RequestException as e:
            errors.append(f"litterbox: {e}"[:160])
        try:
            r = requests.post("https://tmpfiles.org/api/v1/upload",
                              files={"file": ("work.jpg", data, "image/jpeg")}, timeout=120)
            r.raise_for_status()
            page = (r.json().get("data") or {}).get("url", "").replace("http://", "https://", 1)
            # the API returns the viewer page; the image is served at a signed /dl/ link (about 5 min) found on it
            m = re.search(r'href="(https://tmpfiles\.org/dl/[^"]+)"', requests.get(page, timeout=30).text) if page else None
            if m:
                return m.group(1)
            errors.append(f"tmpfiles: no direct link on {page[:120]}")
        except (requests.RequestException, ValueError) as e:
            errors.append(f"tmpfiles: {e}"[:160])
        raise RuntimeError("no temporary host took the upload: " + "; ".join(errors))

    def serpapi_lens(self, image_url: str, limit: int = 40) -> list[WebReference]:
        """Google Lens visual matches through SerpAPI (one search from the free plan)."""
        d = self._serpapi({"engine": "google_lens", "url": image_url})
        out = []
        for x in d.get("visual_matches", [])[:limit]:
            img = x.get("image") or x.get("thumbnail") or ""
            if img.startswith("http"):
                out.append(WebReference(x.get("title", ""), hash(img) & 0x7FFFFFFFFFFF, x.get("thumbnail") or img,
                                        x.get("link", ""), img, {"source": "google_lens", "site": x.get("source", "")}))
        return out

    def _searx_pages(self, query: str, limit: int) -> list[tuple[str, str]]:
        r = requests.get(f"{self.searxng_url}/search", params={"q": query, "format": "json"}, timeout=30)
        r.raise_for_status()
        return [(x["url"], x.get("title", "")) for x in r.json().get("results", [])[:limit] if x.get("url", "").startswith("http")]

    def _serper_pages(self, query: str, limit: int) -> list[tuple[str, str]]:
        r = requests.post("https://google.serper.dev/search", json={"q": query, "num": min(limit, 20)},
                          headers={"X-API-KEY": self._key("serper")}, timeout=30)
        r.raise_for_status()
        return [(x["link"], x.get("title", "")) for x in r.json().get("organic", []) if x.get("link", "").startswith("http")]

    def _brave_pages(self, query: str, limit: int) -> list[tuple[str, str]]:
        r = requests.get("https://api.search.brave.com/res/v1/web/search", params={"q": query, "count": min(limit, 20)},
                         headers={"X-Subscription-Token": self._key("brave"), "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        return [(x["url"], x.get("title", "")) for x in (r.json().get("web") or {}).get("results", [])
                if x.get("url", "").startswith("http")]

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
