"""Before the reading: a larger copy of the work when the source is too small for the display (Sir, 2026-11-10).

A source that would need more than the single-pass upscale limit to fill the display (Monet's *Le Pavé de
Chailly* came in at 850 px and needed 4.52x) is replaced by a larger copy of the same work, when one is found:

1. **known** — pictures the run already verified: the catalogue's records (a Commons thumbnail is traced to its
   file and taken at up to 3840 px) and the identify ledger's verified copies that are picture files or
   Commons file pages;
2. **Commons** — a file search on artist and title (free; Commons reports every file's size);
3. **Google's image index** — through Serper, which reports each picture's size (Serper's own 10% floor).

A candidate replaces the source only when it is large enough to need no more than ``max_upscale``, keeps the
source's framing (aspect within ``ASPECT_TOL`` after its own borders are trimmed), and matches the source on
both instruments at the identification bar (CLIP and geometric inliers). Candidates are fetched one at a time,
in memory, capped in bytes; only the chosen one is written (``work/source-large.jpg``). When none qualifies the
source stays as it was and the upscale stage blocks, as before: nothing is accepted below the bar.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

import requests
from PIL import Image

from .borders import trim_borders
from .matching import geometric_inliers

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; neural-canvas/1.0; +https://github.com/CannonCoPilot/neural-canvas)"
ASPECT_TOL = 0.03           # a copy whose trimmed aspect differs more is framed differently (a detail, a mount)
MAX_BYTES = 40_000_000      # one picture, held in memory
MAX_FETCHES = 10            # candidates downloaded per work, one at a time
COMMONS_WIDTH = 3840        # a standard Commons thumbnail step: ample for the display, never a 100 MP original
IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp|tiff?)(\?|$)", re.I)


@dataclass
class Candidate:
    url: str
    origin: str                 # known | commons | google_index
    width: int = 0              # 0 = unknown until fetched
    height: int = 0
    note: str = ""


@dataclass
class Search:
    needed: float               # scale the current source needs
    tried: list = field(default_factory=list)
    chosen: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"needed_scale": round(self.needed, 3), "tried": self.tried, "chosen": self.chosen}


def needed_scale(size: tuple[int, int], display: tuple[int, int]) -> float:
    """The upscale a full-bleed window of this picture needs: its short side fills the display."""
    return max(display[0] / size[0], display[1] / size[1])


def _commons_file(url: str) -> str:
    """'File:Name.jpg' from a Commons file page, an upload URL or a thumbnail URL; '' otherwise."""
    u = urlparse(url)
    if u.netloc.endswith("commons.wikimedia.org") and "/wiki/File:" in u.path:
        return "File:" + unquote(u.path.split("/wiki/File:", 1)[1]).replace("_", " ")
    if u.netloc == "upload.wikimedia.org":
        parts = unquote(u.path).split("/")
        if "thumb" in parts:                       # .../thumb/a/ab/Name.jpg/960px-Name.jpg
            i = parts.index("thumb")
            return "File:" + parts[i + 3].replace("_", " ") if len(parts) > i + 3 else ""
        return "File:" + parts[-1].replace("_", " ") if parts[-1] else ""
    return ""


class LargerSource:
    def __init__(self, wiki, web, embedder, thresholds: dict):
        self.wiki, self.web, self.emb, self.t = wiki, web, embedder, thresholds
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA

    # ── candidates ────────────────────────────────────────────────────────
    def _commons(self, files: list[str], origin: str) -> list[Candidate]:
        """Commons files at up to COMMONS_WIDTH, with their real sizes."""
        out = []
        for i in range(0, len(files), 20):
            try:
                d = self.wiki._get(self.wiki.commons_api, {"action": "query", "titles": "|".join(files[i:i + 20]),
                                                           "prop": "imageinfo", "iiprop": "url|size",
                                                           "iiurlwidth": COMMONS_WIDTH})
            except Exception as e:
                log.warning("larger source: commons lookup failed: %s", e)
                continue
            for p in d.get("query", {}).get("pages", []):
                ii = (p.get("imageinfo") or [{}])[0]
                url = ii.get("thumburl") or ii.get("url", "")
                w = ii.get("thumbwidth") or ii.get("width", 0)          # the size of the picture fetched
                h = ii.get("thumbheight") or ii.get("height", 0)
                if url and w and h:
                    out.append(Candidate(url, origin, w, h, p.get("title", "")))
        return out

    def known(self, st: dict) -> list[Candidate]:
        urls = []
        for r in (st.get("catalogue") or {}).get("sources") or []:
            urls += [r.get("image_url", ""), *(r.get("images") or [])]
        for it in ((st.get("identify") or {}).get("evidence") or {}).get("ledger") or []:
            if it.get("kind") == "copy" and it.get("verified"):
                urls.append(it.get("url", ""))
        urls = [u for u in dict.fromkeys(urls) if u and u.startswith("http")]
        files = list(dict.fromkeys(f for u in urls if (f := _commons_file(u))))
        out = self._commons(files, "known") if files else []
        out += [Candidate(u, "known") for u in urls if not _commons_file(u) and IMAGE_EXT.search(urlparse(u).path)]
        return out

    def searched(self, artist: str, title: str) -> list[Candidate]:
        q = f"{artist} {title}".strip()
        out = []
        try:
            refs = self.wiki.search_files(q, 12)
            out += self._commons([r.file_title for r in refs], "commons")
        except Exception as e:
            log.warning("larger source: commons search failed for %r: %s", q, e)
        if self.web is not None and hasattr(self.web, "index_images"):
            try:
                for r in self.web.index_images(q, 20):
                    m = r.meta
                    if str(m.get("original", "")).startswith("http"):
                        out.append(Candidate(m["original"], "google_index", int(m.get("width") or 0),
                                             int(m.get("height") or 0), m.get("site", "")))
            except Exception as e:
                log.warning("larger source: google image index failed for %r: %s", q, e)
        return out

    # ── checks ────────────────────────────────────────────────────────────
    def _fetch(self, url: str) -> Image.Image | None:
        try:
            with self.s.get(url, timeout=60, stream=True) as r:
                r.raise_for_status()
                if int(r.headers.get("Content-Length") or 0) > MAX_BYTES:
                    return None
                buf = io.BytesIO()
                for chunk in r.iter_content(1 << 16):
                    buf.write(chunk)
                    if buf.tell() > MAX_BYTES:
                        return None
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        except Exception as e:
            log.warning("larger source: fetch failed %s: %s", url[:120], e)
            return None

    def _same_work(self, src: Image.Image, src_vec, im: Image.Image) -> tuple[bool, float, int, str]:
        a, b = src.width / src.height, im.width / im.height
        if abs(b - a) / a > ASPECT_TOL:
            return False, 0.0, 0, f"framing differs (aspect {b:.3f} vs {a:.3f})"
        sim = float(self.emb.similarity(src_vec, self.emb.embed([im]))[0])
        if sim < self.t["id_clip_min"]:
            return False, round(sim, 4), 0, f"CLIP {sim:.3f} below {self.t['id_clip_min']}"
        inl = geometric_inliers(src, im)
        if inl < self.t["id_inliers_min"]:
            return False, round(sim, 4), inl, f"{inl} inliers below {self.t['id_inliers_min']}"
        return True, round(sim, 4), inl, ""

    # ── the search ────────────────────────────────────────────────────────
    def find(self, src: Image.Image, st: dict, artist: str, title: str, display: tuple[int, int],
             max_scale: float) -> tuple[Image.Image | None, Search]:
        """The largest verified copy that needs no more than ``max_scale``, trimmed of its own borders; or None."""
        rep = Search(needed_scale(src.size, display))
        src_vec = self.emb.embed([src])[0]
        fetched = 0
        for stage in ("known", "searched"):
            cands = self.known(st) if stage == "known" else self.searched(artist, title)
            # sizes known: the big enough ones, largest first; sizes unknown: after them, in the order found
            sized = sorted((c for c in cands if c.width and needed_scale((c.width, c.height), display) <= max_scale),
                           key=lambda c: -c.width * c.height)
            for c in sized + [c for c in cands if not c.width]:
                if fetched >= MAX_FETCHES:
                    break
                if any(t["url"] == c.url for t in rep.tried):
                    continue
                fetched += 1
                row = {"url": c.url, "origin": c.origin, "note": c.note[:80]}
                rep.tried.append(row)
                im = self._fetch(c.url)
                if im is None:
                    row["verdict"] = "would not load"
                    continue
                im, _ = trim_borders(im)
                row["size"] = list(im.size)
                if needed_scale(im.size, display) > max_scale:
                    row["verdict"] = f"too small ({needed_scale(im.size, display):.2f}x)"
                    continue
                ok, sim, inl, why = self._same_work(src, src_vec, im)
                row.update(clip=sim, inliers=inl)
                if not ok:
                    row["verdict"] = why
                    continue
                row["verdict"] = "chosen"
                rep.chosen = dict(row)
                log.info("larger source: %s %s (%s) replaces the %s source; CLIP %.3f, %d inliers", c.origin,
                         im.size, c.url[:100], src.size, sim, inl)
                return im, rep
        log.warning("larger source: none of %d candidates qualifies (source %s needs %.2fx)", len(rep.tried),
                    src.size, rep.needed)
        return None, rep
