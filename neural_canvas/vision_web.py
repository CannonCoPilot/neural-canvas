"""Google Cloud Vision web detection — the billed reverse-image FALLBACK.

A true reverse-image search: it sends the picture itself and returns the pages and images on the web
that carry it, plus Google's best-guess label and the entities those pages are about. The free tiers
(Commons, the Wikidata oeuvre sweep, DuckDuckGo text search, Imify) run first; this one runs only
when they end a round without a confirmation, or when a near-miss shows the work is out there but
no free copy matched on both instruments.

Like every other source, its results are *candidates*: a page becomes a reference only by passing
the same two-instrument match. Calls past the first 1,000 a month are billed (~$3.50/1000), and nothing
may ever spill into paid (Sir, 2026-11-06): calls are counted in the shared credit ledger and stop at 90% of
the free 1,000 (the pool's 10% floor). ``vision_web.enabled: false`` turns the tier off.

The API key never lives in code or config: config names the credentials file and the key's dotted
path inside it, and the key is read at call time.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
from pathlib import Path

import requests
import yaml
from PIL import Image

from .websearch import WebReference

log = logging.getLogger(__name__)

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def _read_key(credentials_path: str, key_path: str) -> str:
    """``credentials.yaml`` is multi-document YAML: look the dotted path up in every document."""
    parts = key_path.split(".")
    with open(Path(credentials_path).expanduser()) as f:
        for doc in yaml.safe_load_all(f):
            node = doc
            for p in parts:
                node = node.get(p) if isinstance(node, dict) else None
            if isinstance(node, str) and node.strip():
                return node.strip()
    raise KeyError(f"no credential at {key_path!r} in {credentials_path}")


def _scrub(text: str, key: str) -> str:
    return re.sub(r"key=[^&\s'\"]+", "key=<redacted>", text.replace(key, "<redacted>"))


class VisionWebDetection:
    free_per_month = 1000

    def __init__(self, cfg: dict, cache_dir: Path, ledger=None):
        self.cfg = cfg
        self.calls = 0          # billed calls this process has made (copied into each run record)
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger    # lens_pool.Ledger; None = no limit kept (tests)
        if ledger is not None:
            import time
            e = ledger.get("vision")
            if e.get("month") != time.strftime("%Y-%m"):   # first sight of this month: count the replies on disk
                start = time.mktime(time.strptime(time.strftime("%Y-%m-01"), "%Y-%m-%d"))
                n = sum(1 for f in cache_dir.glob("*.json") if f.stat().st_mtime >= start)
                ledger.update("vision", month=time.strftime("%Y-%m"), used_month=n)

    def used_this_month(self) -> int:
        import time
        e = self.ledger.get("vision") if self.ledger is not None else {}
        return e.get("used_month", 0) if e.get("month") == time.strftime("%Y-%m") else 0

    def cached(self, img: Image.Image) -> bool:
        return self._cache(img).exists()

    def _cache(self, img: Image.Image) -> Path:
        return self.cache_dir / f"{hashlib.sha256(img.convert('RGB').tobytes()).hexdigest()[:24]}.json"

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled", True))

    def detect(self, img: Image.Image) -> dict:
        """One billed call per distinct image, ever: the response is kept on disk by pixel hash, so a
        rerun (even --fresh) reads it back instead of paying again."""
        im = img.convert("RGB")
        cache = self._cache(im)
        if cache.exists():
            return json.loads(cache.read_text())
        if self.ledger is not None:          # the floor check and the charge are one locked step
            from .lens_pool import Provider
            if not self.ledger.reserve(Provider("vision", lambda w: [], self.free_per_month, "monthly")):
                raise RuntimeError(f"Vision web detection: {self.used_this_month()} of the {self.free_per_month} free calls "
                                   f"this month are spent down to the 10% floor; nothing may spill into paid")
        im.thumbnail((self.cfg.get("max_side", 1600),) * 2)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        body = {"requests": [{"image": {"content": base64.b64encode(buf.getvalue()).decode()},
                              "features": [{"type": "WEB_DETECTION", "maxResults": self.cfg.get("max_results", 30)}],
                              "imageContext": {"webDetectionParams": {"includeGeoResults": False}}}]}
        api_key = _read_key(self.cfg["credentials_path"], self.cfg["key_path"])
        self.calls += 1
        # the key goes in a header: requests' connection errors quote the full URL, query string included
        try:
            r = requests.post(ENDPOINT, headers={"X-goog-api-key": api_key}, json=body,
                              timeout=self.cfg.get("timeout_s", 60))
        except requests.RequestException as e:
            raise RuntimeError(f"Vision web detection unreachable: {_scrub(str(e), api_key)}") from None
        if r.status_code != 200:
            # the body names the remedy (billing disabled, API not enabled)
            raise RuntimeError(f"Vision web detection HTTP {r.status_code}: {_scrub(r.text[:300], api_key)}")
        resp = r.json()["responses"][0]
        if "error" in resp:
            raise RuntimeError(f"Vision web detection error: {resp['error']}")
        web = resp.get("webDetection", {})
        cache.write_text(json.dumps(web))
        log.info("vision web detection: best guess %s, %d pages, %d full / %d partial matches",
                 [g.get("label") for g in web.get("bestGuessLabels", [])], len(web.get("pagesWithMatchingImages", [])),
                 len(web.get("fullMatchingImages", [])), len(web.get("partialMatchingImages", [])))
        return web

    @staticmethod
    def references(web: dict) -> list[WebReference]:
        """Pages that carry the image (their own copy, titled by the page) first, then bare matching
        images. Visually-similar images are left out: they are by definition other pictures."""
        out, seen = [], set()

        def add(img_url: str, page_url: str, title: str, kind: str):
            if not img_url or img_url in seen:
                return
            seen.add(img_url)
            out.append(WebReference(file_title=title, page_id=hash(img_url) & 0x7FFFFFFFFFFF, thumb_url=img_url,
                                    page_url=page_url, image_url=img_url, meta={"source": "vision_web", "match": kind}))

        for p in web.get("pagesWithMatchingImages", []):
            title = p.get("pageTitle", "")
            for kind in ("fullMatchingImages", "partialMatchingImages"):
                for m in p.get(kind, [])[:2]:
                    add(m.get("url", ""), p.get("url", ""), title, kind)
        for kind in ("fullMatchingImages", "partialMatchingImages"):
            for m in web.get(kind, []):
                add(m.get("url", ""), "", "", kind)
        return out

    @staticmethod
    def hints(web: dict) -> dict:
        """What the matching pages say the image is: Google's best-guess label and the top web
        entities. They name candidates for the next round; they never confirm anything."""
        ents = [e["description"] for e in sorted(web.get("webEntities", []), key=lambda e: -e.get("score", 0))
                if e.get("description")]
        return {"best_guess": [g["label"] for g in web.get("bestGuessLabels", []) if g.get("label")],
                "entities": ents[:10],
                "page_titles": [p.get("pageTitle", "") for p in web.get("pagesWithMatchingImages", [])[:10]]}
