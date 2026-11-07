"""Free reverse-image search, direct to Bing and Yandex: the picture itself is uploaded, so an obscure or
contemporary work is found even when no caption would ever be guessed.

Replaces the local Imify service (Sir, 2026-11-06): its filter dropped good Yandex matches and its Bing and
SearXNG engines were broken. Both flows here are what the engines' own web pages do today:

* **Bing** (two steps): GET /images for cookies; POST /images/kblob (multipart ``imageBin`` = base64 JPEG)
  -> ``{"redirectUrl": "/search?q=<best guess>&bcid=<blob id>..."}``; GET /images<redirectUrl>. Every result
  tile carries an HTML-escaped JSON record ``m="{murl, purl, t, turl, ...}"``. The ``q`` is Bing's own caption
  for the picture: a lead in its own right, returned beside the references.
* **Yandex** (two steps): POST the JPEG bytes to /images-apphost/image-download -> ``{"cbir_id", "sizes"}``;
  GET /images/search?rpt=imageview&cbir_id=...&cbir_page=sites. The page carries its whole state as JSON in a
  ``data-state`` attribute: ``cbirSites.sites`` (page + original image + title), ``cbirOtherSizes`` (the same
  picture at other sizes), ``cbirSimilar.thumbs`` (look-alikes) and ``cbirTags`` (Yandex's captions). That
  state is read directly; Imify scraped the markup with regexes and emitted page links as image URLs.

Sir's rules (2026-11-06): an error, a redirect to a homepage, a captcha page or an empty result page is a
FAILURE and raises ``RuntimeError`` with a short reason; ``[]`` means only that the engine answered a genuine
results page for this upload and it held no matches. How each engine's genuine page is told apart is
documented on its parser. Calls to the same engine are spaced at least ``PACE_S`` apart (thread-safe).

Every reference is only a candidate: it must pass the same two-instrument match as every other source.
"""
from __future__ import annotations

import base64
import html
import io
import json
import logging
import re
import threading
import time
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urlsplit

import requests
from PIL import Image

from .websearch import WebReference

log = logging.getLogger(__name__)

PACE_S = 2.0          # minimum gap between two calls to the same engine (Sir, 2026-11-06: pace politely)
BING = "https://www.bing.com"
YANDEX = "https://yandex.com"
# a bot wall, by URL or by page marker (none of these appear on a real results page: checked 2026-11-06)
_BING_WALL = ("b_captcha", "turingCaptcha", "challenges.cloudflare.com/turnstile")
_YANDEX_WALL = ("showcaptcha", "SmartCaptcha", "CheckboxCaptcha", "AdvancedCaptcha")


def _jpeg(img: Image.Image, side: int, quality: int) -> bytes:
    im = img.convert("RGB").copy()
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _abs(u: str) -> str:
    """Yandex serves its thumbnails protocol-relative ('//avatars.mds.yandex.net/i?id=...')."""
    u = html.unescape(u or "").strip()
    return "https:" + u if u.startswith("//") else u


def _clean(u: str) -> str:
    """Yandex appends utm_source=yandexsmartcamera to every result link; stripped so dedup sees one URL."""
    p = urlsplit(_abs(u))
    if "utm_" not in p.query:
        return p.geturl()
    q = urlencode([(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not k.lower().startswith("utm_")])
    return p._replace(query=q).geturl()


def _site(u: str) -> str:
    return urlsplit(u).netloc.lower().removeprefix("www.")


def _slug(u: str) -> str:
    """A title for a bare image URL, from its file name: '.../fredericleighton_juliet.jpg' -> 'fredericleighton juliet'."""
    name = unquote(urlsplit(u).path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
    return re.sub(r"[-_+\s]+", " ", name).strip()


def _ref(title: str, page: str, image: str, thumb: str, source: str, **meta) -> WebReference:
    return WebReference(title, hash(image) & 0x7FFFFFFFFFFF, thumb or image, page, image,
                        {"source": source, "site": _site(page or image), **meta})


def _title(page: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", page, re.S | re.I)
    return html.unescape(m.group(1)).strip() if m else ""


# ── Bing ──────────────────────────────────────────────────────────────────────────────────────────────────
def parse_bing(page: str, url: str, bcid: str = "") -> list[WebReference]:
    """References from a Bing visual-search results page.

    A genuine results page is: final URL path ``/images/search`` carrying our upload's ``bcid`` (Bing answers a
    failed upload with a redirect to the Images homepage, whose trending tiles carry the same ``m=`` records and
    once fooled Imify), no bot-wall marker, and Bing's result grid (``id="mmComponent_images_``). On such a page
    zero tiles is a genuine no-match and returns ``[]``; a page without the grid is a failure."""
    parts = urlsplit(url)
    if "captcha" in url.lower() or "/challenge" in parts.path or any(k in page for k in _BING_WALL):
        raise RuntimeError("bing: captcha / bot challenge")
    if parts.path.rstrip("/") != "/images/search":
        raise RuntimeError(f"bing: redirected to {parts.path or '/'} ({_title(page)[:40]!r}), not a results page")
    if bcid and parse_qs(parts.query).get("bcid", [""])[0] != bcid:
        raise RuntimeError("bing: results page is not for this upload (bcid missing or changed)")
    if 'id="mmComponent_images_' not in page:
        raise RuntimeError(f"bing: no result grid on the page ({len(page)} bytes)")
    out, seen = [], set()
    for m in re.finditer(r'\bm="(\{[^"]+\})"', page):
        try:
            j = json.loads(html.unescape(m.group(1)))
        except ValueError:
            continue
        img = j.get("murl") or ""
        if not img.startswith("http") or img in seen:
            continue
        seen.add(img)
        #  /  are Bing's hit-highlight brackets around query words
        title = html.unescape(re.sub("[]", "", j.get("t") or j.get("desc") or "")).strip()
        out.append(_ref(title or _slug(img), j.get("purl") or "", img, j.get("turl") or "", "bing"))
    return out


# ── Yandex ────────────────────────────────────────────────────────────────────────────────────────────────
def _yandex_state(page: str) -> dict:
    """The page's own state: the ``data-state`` JSON that holds ``initialState`` (one of several on the page)."""
    for m in re.finditer(r'data-state="([^"]*)"', page):
        if "initialState" not in m.group(1):
            continue
        try:
            s = json.loads(html.unescape(m.group(1)))
        except ValueError:
            continue
        if isinstance(s.get("initialState"), dict):
            return s["initialState"]
    return {}


def parse_yandex(page: str, url: str, cbir_id: str = "") -> tuple[list[WebReference], list[str]]:
    """(references, Yandex's caption tags) from a Yandex imageview results page.

    A genuine results page is: final URL path ``/images/search``, no captcha (Yandex redirects a suspected bot
    to /showcaptcha), and a page state whose ``cbirPreview.cbirId`` is our upload's id and not
    ``imageExpired``, with the ``cbirSites`` block present. A shell whose cbirId is empty or someone else's is
    the "upload never registered" failure Imify returned as "no results". On a genuine page with no sites, no
    other sizes and no look-alikes the answer is ``[]``.

    Order: sites (page + original + title), then other sizes of the same picture, then look-alikes. Titles are
    kept as Yandex gives them, Cyrillic included. Filters, each because the item can never be a reference:
    * a URL under ``get-images-cbir/`` is our own upload echoed back (it would 'match' itself);
    * a URL on ``yastatic.net`` is Yandex's UI art (logos, sprites).
    Yandex's own thumbnails (avatars.mds.yandex.net/i?id=...) are KEPT as ``thumb_url``: they are copies of the
    matched picture, and Imify's blanket 'yandex.net/' filter is what dropped good matches."""
    parts = urlsplit(url)
    if "captcha" in url.lower() or any(k in page for k in _YANDEX_WALL):
        raise RuntimeError("yandex: captcha")
    if not parts.path.startswith("/images/search"):
        raise RuntimeError(f"yandex: redirected to {parts.path or '/'} ({_title(page)[:40]!r}), not a results page")
    s = _yandex_state(page)
    prev = s.get("cbirPreview") or {}
    if not s or "cbirSites" not in s or not prev:
        raise RuntimeError(f"yandex: no results state on the page ({len(page)} bytes)")
    if not prev.get("cbirId") or (cbir_id and prev["cbirId"] != cbir_id):
        raise RuntimeError("yandex: empty shell, the upload was not registered (cbirId missing or changed)")
    if prev.get("imageExpired"):
        raise RuntimeError("yandex: uploaded image expired before the search")

    out, seen = [], set()

    def add(title: str, page_url: str, img: str, thumb: str, kind: str) -> None:
        img = _clean(img)
        if not img.startswith("http") or img in seen:
            return
        if "get-images-cbir/" in img or "yastatic.net" in img:
            return
        seen.add(img)
        out.append(_ref((title or "").strip() or _slug(img), _clean(page_url), img, _abs(thumb), "yandex", kind=kind))

    for x in (s.get("cbirSites") or {}).get("sites") or []:
        add(x.get("title", ""), x.get("url", ""), (x.get("originalImage") or {}).get("url", ""),
            (x.get("thumb") or {}).get("url", ""), "site")
    for group in ("large_dups", "medium_dups", "small_dups"):   # largest first: the best copy to check
        for x in ((s.get("cbirOtherSizes") or {}).get("items") or {}).get(group) or []:
            add("", "", x.get("url", ""), "", "size")
    for x in (s.get("cbirSimilar") or {}).get("thumbs") or []:
        orig = parse_qs(urlsplit(x.get("linkUrl", "")).query).get("img_url", [""])[0]
        add(x.get("title", ""), "", orig or x.get("imageUrl", ""), x.get("imageUrl", ""), "similar")
    tags = [t["text"] for t in (s.get("cbirTags") or {}).get("tags") or [] if t.get("text")]
    return out, tags


# ── client ────────────────────────────────────────────────────────────────────────────────────────────────
class ReverseImage:
    def __init__(self, user_agent: str, timeout_s: float = 45):
        self.ua, self.timeout_s = user_agent, timeout_s
        self._locks = {"bing": threading.Lock(), "yandex": threading.Lock()}
        self._last = {"bing": 0.0, "yandex": 0.0}

    def _session(self) -> requests.Session:
        s = requests.Session()     # fresh cookies per search: one upload never rides another's session
        s.headers.update({"User-Agent": self.ua, "Accept-Language": "en-US,en;q=0.9"})
        return s

    def _paced(self, engine: str, fn):
        """Run fn with this engine's lock held, starting at least PACE_S after the previous call ended."""
        with self._locks[engine]:
            wait = self._last[engine] + PACE_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                return fn()
            except requests.RequestException as e:
                raise RuntimeError(f"{engine}: {type(e).__name__}: {e}"[:240]) from e
            finally:
                self._last[engine] = time.monotonic()

    def bing(self, img: Image.Image) -> tuple[list[WebReference], str]:
        """(references, Bing's best-guess caption). Raises RuntimeError on any failure (see parse_bing)."""
        def run():
            t0, s = time.time(), self._session()
            with s:
                s.get(f"{BING}/images", timeout=self.timeout_s).raise_for_status()
                r = s.post(f"{BING}/images/kblob", timeout=self.timeout_s,
                           files={"imageBin": (None, base64.b64encode(_jpeg(img, 800, 85)).decode())})
                r.raise_for_status()
                try:
                    red = r.json().get("redirectUrl") or ""
                except ValueError:
                    red = ""
                bcid = parse_qs(urlsplit(red).query).get("bcid", [""])[0]
                if not bcid:
                    raise RuntimeError(f"bing: upload not accepted (kblob answered {r.text[:120]!r})")
                caption = parse_qs(urlsplit(red).query).get("q", [""])[0].strip()
                p = s.get(f"{BING}/images{red}", timeout=self.timeout_s)
                p.raise_for_status()
            refs = parse_bing(p.text, p.url, bcid)
            log.info("bing reverse: %d references, caption %r, %.1fs", len(refs), caption, time.time() - t0)
            return refs, caption
        return self._paced("bing", run)

    def yandex(self, img: Image.Image, tags: list[str] | None = None) -> list[WebReference]:
        """References for the picture. Pass a list as ``tags`` to receive Yandex's caption tags (leads, often
        Cyrillic). Raises RuntimeError on any failure (see parse_yandex)."""
        def run():
            t0, s = time.time(), self._session()
            with s:
                up = s.post(f"{YANDEX}/images-apphost/image-download?cbird=37&images_avatars_size=preview"
                            "&images_avatars_namespace=images-cbir", data=_jpeg(img, 1600, 90), timeout=self.timeout_s,
                            headers={"Content-Type": "image/jpeg", "Origin": YANDEX, "Accept": "application/json",
                                     "Referer": f"{YANDEX}/images/search"})
                up.raise_for_status()
                try:
                    j = up.json()
                except ValueError:
                    j = {}
                cid = j.get("cbir_id") or ""
                orig = ((j.get("sizes") or {}).get("orig") or {}).get("path") or ""
                if not cid or not orig:
                    raise RuntimeError(f"yandex: upload not registered (image-download answered {up.text[:120]!r})")
                p = s.get(f"{YANDEX}/images/search", timeout=self.timeout_s,
                          params={"rpt": "imageview", "url": orig, "cbir_id": cid, "cbir_page": "sites", "family": "0",
                                  "pornowhitelist": "1", "ipnd": "1"})
                p.raise_for_status()
            refs, tg = parse_yandex(p.text, p.url, cid)
            if tags is not None:
                tags[:] = tg
            log.info("yandex reverse: %d references (%s), tags %s, %.1fs", len(refs),
                     ", ".join(f"{k} {sum(r.meta['kind'] == k for r in refs)}" for k in ("site", "size", "similar")),
                     tg[:3], time.time() - t0)
            return refs
        return self._paced("yandex", run)
