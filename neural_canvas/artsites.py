"""Artvee and Arthur.io: two large picture collections, searched for the picture itself (Sir, 2026-11-05).

Artvee is a credible source (Sir, 2026-11-05): its public-domain scans carry title, date and the artist's
nationality and life, and the catalogue stage counts it as one vote. Arthur.io is contributor pages, "a
systematised Pinterest": its captions are LEADS that the catalogue stage searches curated sources with.

* Artvee: its search and artist pages are read with artvee-scraper's own card parsers. An artist's
  whole Artvee catalogue is listed once, its thumbnails embedded once, both cached under
  ``state_dir/artvee``; later runs rank it by embedding for free.
* Arthur.io: its pages answer automated requests with a Cloudflare challenge (2026-11-05), which is not
  worked around. Its image server answers, and search engines have indexed its pictures with their
  captions, so its pictures are reached through image search kept to the site.

Both return ``WebReference`` objects, so identification scores them like any web hit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from bs4 import BeautifulSoup

from .matching import fetch_image, save_npz
from .websearch import WebReference

log = logging.getLogger(__name__)
ARTVEE = "https://artvee.com"
ARTVEE_UA = "neural-canvas/1.0"      # Artvee refuses agents that carry a contact URL (HTTP 403)
ARTVEE_MAX_PAGES = 40                # 30 works a page: 1,200 works is more than any artist there
LIST_TTL_S = 30 * 86400


def slug(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def _pid(url: str) -> int:
    return int(hashlib.sha1(url.encode()).hexdigest()[:12], 16)


class Artvee:
    def __init__(self, state_dir: Path, embedder=None, interval_s: float = 1.0):
        self.dir = Path(state_dir) / "artvee"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.emb, self.interval_s = embedder, interval_s
        self.s = requests.Session()
        self.s.headers["User-Agent"] = ARTVEE_UA
        self._last = 0.0

    def _get(self, url: str) -> str:
        wait = self.interval_s - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        r = self.s.get(url, timeout=30)
        if r.status_code == 404:
            return ""
        r.raise_for_status()
        return r.text

    @staticmethod
    def parse(page_html: str) -> list[WebReference]:
        """Every artwork card on a listing page, read by artvee-scraper's parsers."""
        from artvee_scraper.artvee_client import ArtveeClient
        from artvee_scraper.artwork import CategoryType
        out = []
        soup = BeautifulSoup(page_html, "html.parser")
        for card in soup.find_all("div", {"class": "product-element-bottom"}):
            try:
                m = ArtveeClient._parse_artist_metadata(card, CategoryType.ABSTRACT)   # category is unused here
                im = ArtveeClient._parse_image_metadata(card)
            except Exception as e:
                log.debug("artvee card skipped: %s", e)
                continue
            if not im.source_url:
                continue
            thumb = im.source_url.replace("/sdl/", "/ft/").replace("sdl.jpg", ".jpg")   # 500 px
            date = "" if m.date == "n.d." else m.date
            caption = f"{m.title}{f' ({date})' if date else ''} by {m.artist}"
            out.append(WebReference(caption, _pid(thumb), thumb, m.url, thumb,
                                    {"title": m.title, "date": date, "artist": m.artist, "origin": m.origin or "",
                                     "full": im.source_url, "source": "artvee"}))
        return out

    def search(self, query: str) -> list[WebReference]:
        return self.parse(self._get(f"{ARTVEE}/main/?s={quote(query)}"))

    def artist_pages(self, artist: str) -> list[str]:
        """Artvee's artist page slugs for this artist, found by searching the surname rather than guessing
        the address: names are spelt variously ("Frederick" for frederic-remington) and one artist can be
        filed under two pages (james-mcneill-whistler, james-abbott-mcneill-whistler)."""
        sur = slug(artist).rsplit("-", 1)[-1]
        if len(sur) < 3:
            return [slug(artist)]
        page = self._get(f"{ARTVEE}/main/?s={quote(sur)}")
        found = re.findall(r'href="https://artvee\.com/artist/([a-z0-9-]+)/"', page)
        slugs = [x for x in dict.fromkeys(found) if x.split("-")[-1] == sur]   # not remington-schuyler
        return list(dict.fromkeys([slug(artist)] + slugs))[:4]

    def artist_works(self, artist: str) -> list[WebReference]:
        """The artist's whole Artvee catalogue (every page Artvee files the artist under), listed once
        and cached."""
        path = self.dir / f"{slug(artist)}.json"
        if path.exists() and time.time() - path.stat().st_mtime < LIST_TTL_S:
            return [WebReference(**d) for d in json.loads(path.read_text())]
        works: list[WebReference] = []
        for page_slug in self.artist_pages(artist):
            for n in range(1, ARTVEE_MAX_PAGES + 1):
                url = f"{ARTVEE}/artist/{page_slug}/" + (f"page/{n}/" if n > 1 else "")
                batch = self.parse(self._get(url))
                if not batch:
                    break
                works += batch
        seen: set[str] = set()
        works = [w for w in works if not (w.page_url in seen or seen.add(w.page_url))]
        log.info("artvee: %d works listed for %s", len(works), artist)
        if works:                             # an empty listing may be a misspelling or an outage: not kept
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps([w.__dict__ for w in works]))
            tmp.replace(path)
        return works

    def nearest(self, artist: str, img_vec: np.ndarray, k: int = 6, cold: bool = False) -> list[WebReference]:
        """The artist's Artvee works closest to the input by embedding. A run ranks only the thumbnails
        already embedded; the background indexer (cold=True) embeds the rest."""
        works = self.artist_works(artist)
        if not works or self.emb is None:
            return works[:k]
        cache = self.dir / f"{slug(artist)}.npz"
        have: dict[str, np.ndarray] = {}
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            have = dict(zip(z["urls"].tolist(), z["vecs"]))
        todo = [w for w in works if w.thumb_url not in have] if cold else []
        if not cold and not have:
            return []
        if todo:
            log.info("artvee: embedding %d thumbnails for %s", len(todo), artist)
            ok = [(w, im) for w in todo for im in [fetch_image(w.thumb_url, ARTVEE_UA)] if im is not None]
            for i in range(0, len(ok), 64):
                chunk = ok[i:i + 64]
                for (w, _), v in zip(chunk, self.emb.embed([im for _, im in chunk])):
                    have[w.thumb_url] = v
            if have:
                save_npz(cache, urls=np.array(list(have)), vecs=np.stack(list(have.values())))
        ranked = [w for w in works if w.thumb_url in have]
        if not ranked:
            return []
        sims = np.stack([have[w.thumb_url] for w in ranked]) @ img_vec
        return [ranked[int(i)] for i in np.argsort(-sims)[:k]]


# Catalogues raisonnés that answer automated visitors with a bot challenge (pages and images alike). They are
# read through Google's index: Google's copy of the picture and the title it shows, never the site itself.
# Such a record is identity-verified on Google's copy and votes on the title only (Sir, 2026-11-05).
WALLED_CATALOGUES = {
    "wyeth": ["collections.brandywine.org"],     # N. C. Wyeth catalogue raisonné (Brandywine Museum of Art)
}


def index_title(caption: str) -> str:
    """The work's title from an indexed page title: "Crows in Winter – Works – eMuseum" -> "Crows in Winter"."""
    return re.split(r"\s+[–—|]\s+|\s+-\s+", caption or "", maxsplit=1)[0].strip()


class WalledCatalogue:
    """An artist's walled catalogue raisonné through Google's image index (see WALLED_CATALOGUES)."""

    def __init__(self, web):
        self.web = web

    @staticmethod
    def sites(artist: str) -> list[str]:
        sur = (artist or "").split()[-1:] or [""]
        return WALLED_CATALOGUES.get(sur[0].lower(), [])

    def search(self, artist: str, words: tuple[str, ...] | list[str] = (), limit: int = 20) -> list[WebReference]:
        out, seen = [], set()
        # the surname only: Google needs every word, and a catalogue prints its own form of the name
        # ("N. C. Wyeth", never "Newell Convers Wyeth")
        sur = (artist or "").split()[-1:] or [""]
        for site in self.sites(artist):
            for q in list(dict.fromkeys([f"site:{site} {w} {sur[0]}" for w in words if w][:3] + [f"site:{site} {sur[0]}"])):
                try:
                    hits = self.web.index_images(q, limit)
                except Exception as e:
                    log.info("walled catalogue search %r failed: %s", q, e)
                    continue
                for r in hits:
                    if site in (r.meta.get("site") or "") and r.image_url not in seen:
                        seen.add(r.image_url)
                        r.meta.update(source="walled_catalogue", catalogue=site, title_from_index=index_title(r.file_title))
                        out.append(r)
        return out


class Arthur:
    """Arthur.io through image search kept to the site: its pictures and the captions engines indexed."""
    IMG = re.compile(r"^https://arthur\.io/img/art/[^?#]+/(thumb|small|medium|large)/([^/?#]+)\.(jpg|jpeg|png|webp)", re.I)

    def __init__(self, web):
        self.web = web

    def search(self, artist: str, words: tuple[str, ...] | list[str] = (), limit: int = 40) -> list[WebReference]:
        queries = [f"site:arthur.io {artist} {w}" for w in words if w][:3] + [f"site:arthur.io {artist}"]
        out, seen = [], set()
        for q in queries:
            try:
                hits = self.web.search(q, limit)
            except Exception as e:
                log.info("arthur search %r failed: %s", q, e)
                continue
            for r in hits:
                m = self.IMG.match(r.image_url or "")
                if not m:
                    continue
                large = r.image_url.replace(f"/{m.group(1)}/", "/large/")
                if large in seen:
                    continue
                seen.add(large)
                # the file name is "<artist-slug>--<title-slug>": the title survives when the caption does not
                name = m.group(2).split("--", 1)[-1].replace("-", " ").strip()
                page = r.page_url if "arthur.io/art/" in (r.page_url or "") else ""
                out.append(WebReference(r.file_title or name, _pid(large), large.replace("/large/", "/thumb/"), page, large,
                                        {"title_from_file": name, "source": "arthur"}))
        return out
