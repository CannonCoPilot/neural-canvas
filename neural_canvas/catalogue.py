"""Stage — rule on the artwork's facts from CURATED sources only (Sir's rule, 2026-11-05).

Identification found the picture and gathered leads: an artist, candidate titles, captions, file names,
a title written on the work. None of that is printed. Here each lead becomes a SEARCH of curated
catalogues, and a fact is RULED only when two or more independent curated sources state it:

* sources (each one vote): Wikidata (the catalogue item of the work), the Art Institute of Chicago,
  the Metropolitan Museum, the Cleveland Museum of Art, and any museum or institute page on which
  identification itself verified the picture (``sources.classify`` == "curated").
* a source speaks about THIS work only when it is the same picture: its image is matched by the same
  two instruments identification uses (CLIP + ORB/RANSAC), or it is linked from a Wikidata item that
  was (a museum ID on the item, e.g. P3634 Met, P4610 Art Institute).
* title and artist must each be ruled, or the run BLOCKS here with the sources it consulted. Every other
  field (date, medium, collection, movement, the artist's life and nationality) is printed only when
  ruled; otherwise it is left off the card and listed in ``unruled``.
* a title written legibly on the work that differs from the ruled title is shown after it in single
  quotes: Poisonous Berries 'Giftbeeren'.

Unavailable on 2026-11-05 (remediable, not silently skipped): Getty ULAN (server-side TLS failure),
Rijksmuseum (API retired, 410), Harvard Art Museums / Europeana / RKD / Smithsonian (free key or
registration needed), Norman Rockwell Museum (no programmatic access).
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from .matching import fetch_image, geometric_inliers
from .references import file_thumb
from .sources import classify, institutional, non_title

log = logging.getLogger(__name__)
UA = "neural-canvas/1.0 (+https://github.com/CannonCoPilot/neural-canvas)"
LANGS = "en|de|fr|nl|it|es|ja|da|sv|no|nb|fi|cs|ru|pl|pt"
UNAVAILABLE = {
    "Getty ULAN": "server-side TLS failure on vocab.getty.edu (2026-11-05)",
    "Rijksmuseum": "collection API retired (HTTP 410)",
    "Harvard Art Museums": "free API key needed (docs.harvardartmuseums.org)",
    "Europeana": "free API key needed (europeana.eu/en/apis)",
    "RKD": "free API registration needed (rkd.nl)",
    "Smithsonian Open Access": "free api.data.gov key needed (DEMO_KEY is throttled)",
    "Norman Rockwell Museum": "no programmatic access (blocks automated requests)",
}
STOP = {"the", "and", "with", "von", "van", "der", "die", "das", "les", "des", "del", "della", "of", "a", "an",
        "le", "la", "de", "du", "in", "on", "at", "en", "et", "und", "im", "am", "au", "aux"}
DEMONYM = {"american": "united states", "french": "france", "german": "germany", "british": "united kingdom",
           "english": "united kingdom", "scottish": "united kingdom", "dutch": "netherlands", "japanese": "japan",
           "norwegian": "norway", "danish": "denmark", "belgian": "belgium", "russian": "russia",
           "swiss": "switzerland", "austrian": "austria", "spanish": "spain", "italian": "italy", "czech": "czech",
           "swedish": "sweden", "finnish": "finland", "irish": "ireland", "canadian": "canada", "polish": "poland",
           "hungarian": "hungary", "mexican": "mexico", "australian": "australia", "chinese": "china"}
MATERIALS = ("oil", "acrylic", "tempera", "gouache", "watercolor", "watercolour", "pastel", "charcoal", "ink",
             "graphite", "pencil", "woodblock", "woodcut", "lithograph", "etching", "engraving", "canvas", "panel",
             "paper", "board", "cardboard", "wood", "linen", "cloth", "silk", "copper", "fresco", "print")


def fold(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", html.unescape(s or ""))          # punctuation (en dashes too) -> space first
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def _toks(s: str) -> set[str]:
    return {w for w in fold(s).split() if len(w) > 2 and w not in STOP}


def same_title(a: str, b: str) -> bool:
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return False
    if fa == fb or (min(len(fa), len(fb)) >= 6 and (fa in fb or fb in fa)):
        return True
    ta, tb = _toks(a), _toks(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.6


def _surname(name: str) -> str:
    w = fold(name).split()
    return w[-1] if w else ""


def _years(s: str) -> tuple[int, int] | None:
    """The span a date string covers: '1886–1894' -> (1886, 1894); '1860s' -> (1860, 1869); '19th century' -> (1800, 1899)."""
    t = fold(s)
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th) century\b", t)
    if m:
        c = int(m.group(1))
        return (c - 1) * 100, (c - 1) * 100 + 99
    ys = [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})(s?)\b", t) for y in [y[0] if isinstance(y, tuple) else y]]
    if not ys:
        return None
    decade = re.search(r"\b(1[0-9]{3}|20[0-9]{2})s\b", t)
    lo, hi = min(ys), max(ys)
    if decade and lo == hi:
        hi = lo + 9
    return lo, hi


def same_date(a: str, b: str) -> bool:
    ya, yb = _years(a), _years(b)
    return bool(ya and yb) and ya[0] <= yb[1] and yb[0] <= ya[1]


def _materials(s: str) -> set[str]:
    t = fold(s).replace("watercolour", "watercolor")
    return {m.replace("watercolour", "watercolor") for m in MATERIALS if re.search(rf"\b{m}", t)}


def same_medium(a: str, b: str) -> bool:
    ma, mb = _materials(a), _materials(b)
    paints = {"oil", "acrylic", "tempera", "gouache", "watercolor", "pastel", "charcoal", "ink", "graphite", "pencil",
              "woodblock", "woodcut", "lithograph", "etching", "engraving"}
    pa, pb = ma & paints, mb & paints
    if pa and pb and not pa & pb:
        return False                                  # oil versus watercolour: different
    return bool(ma & mb)


def same_collection(a: str, b: str) -> bool:
    drop = {"museum", "art", "arts", "gallery", "national", "collection", "fine", "institute", "the", "of"}
    ta, tb = _toks(a) - drop, _toks(b) - drop
    return bool(ta and tb) and (ta <= tb or tb <= ta or len(ta & tb) / len(ta | tb) >= 0.5)


JUNK = re.compile(r"just a moment|page not found|not found|404|403|access denied|forbidden|service status|"
                  r"log ?in|logg p|sign in|anmelden|captcha|attention required|error|unavailable|cookies?", re.I)
_SAME_INSTITUTION = {"artic": "artic.edu", "met": "metmuseum.org", "cleveland": "clevelandart.org",
                     "wikidata": "wikidata.org"}


def institution(url_or_source: str) -> str:
    """One vote per institution: subdomains and the API of a museum are the museum (images.nga.gov =
    nga.gov; the Art Institute's API = artic.edu)."""
    s = _SAME_INSTITUTION.get(url_or_source, url_or_source)
    host = (urlparse(s).netloc or s).lower().removeprefix("www.")
    parts = host.split(".")
    n = 3 if len(parts) >= 3 and parts[-2] in ("co", "ac", "gov", "org", "com", "edu") and len(parts[-1]) == 2 else 2
    return ".".join(parts[-n:])


@dataclass
class Record:
    source: str                  # one vote per source: "wikidata", "artic", "met", "cleveland", or a curated host
    url: str
    title: str
    titles: list = field(default_factory=list)       # every other title the source gives (languages, aliases)
    artist: str = ""
    artist_bio: str = ""         # "French, 1840–1926" as the source prints it
    date: str = ""
    medium: str = ""
    collection: str = ""
    movement: str = ""
    image_url: str = ""
    verified_by: str = ""        # "image" (both instruments) or "link from <source>"
    clip: float = 0.0
    inliers: int = 0
    artist_born: str = ""
    artist_died: str = ""
    artist_nationality: str = ""
    text: str = ""               # an institution page's own text (folded): it supports what it states

    def states(self, value: str) -> bool:
        """An institution page supports a fact its text states (all of the value's significant words)."""
        if not self.text or not value:
            return False
        words = _toks(value) or set(fold(value).split())
        return bool(words) and all(w in self.text for w in words)

    def all_titles(self) -> list[str]:
        return [t for t in [self.title, *self.titles] if t]


class Catalogue:
    def __init__(self, identifier, thresholds: dict):
        self.idf, self.t = identifier, thresholds
        self.wiki, self.emb = identifier.wiki, identifier.emb
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.consulted: list[str] = []

    # ── is this record's picture the input? ───────────────────────────────
    def _verify(self, img: Image.Image, img_vec, url: str) -> tuple[bool, float, int]:
        im = fetch_image(url, UA) if url else None
        if im is None:
            return False, 0.0, 0
        sim = float(self.emb.similarity(img_vec, self.emb.embed([im]))[0])
        if sim < self.t["id_clip_min"] - 0.08:
            return False, round(sim, 4), 0
        inl = geometric_inliers(img, im)
        return sim >= self.t["id_clip_min"] and inl >= self.t["id_inliers_min"], round(sim, 4), inl

    def _get(self, url: str, **params) -> dict:
        try:
            r = self.s.get(url, params=params, timeout=20)
            if r.status_code != 200:
                log.info("catalogue %s -> HTTP %s", url[:80], r.status_code)
                return {}
            return r.json()
        except Exception as e:
            log.info("catalogue %s failed: %s", url[:80], e)
            return {}

    # ── Wikidata ───────────────────────────────────────────────────────────
    def _wikidata_items(self, img, img_vec, artist: str, ident: dict) -> list[str]:
        """Catalogue items of THIS picture: the identification's own item, else the artist's catalogued
        works ranked by embedding and verified by geometry."""
        if ident.get("wikidata"):
            return [ident["wikidata"]]
        qid = self.wiki.find_artist(artist) if artist else None
        if not qid:
            return []
        works, vecs = self.idf._oeuvre_vectors(qid)
        if not works:
            return []
        out = []
        for i in np.argsort(-(vecs @ img_vec))[:4]:
            w = works[int(i)]
            ok, sim, inl = self._verify(img, img_vec, file_thumb(w["file"], 500))
            if ok:
                out.append(w["qid"])
        return out

    def _wikidata_record(self, qid: str) -> tuple[Record, dict]:
        d = self.wiki._get(self.wiki.wikidata_api, {"action": "wbgetentities", "ids": qid,
                                                    "props": "labels|aliases|claims", "languages": LANGS})
        ent = (d.get("entities") or {}).get(qid, {})
        labels = ent.get("labels") or {}
        en = (labels.get("en") or {}).get("value", "")
        others = [v["value"] for k, v in labels.items() if k != "en"]
        others += [a["value"] for lst in (ent.get("aliases") or {}).values() for a in lst]
        f = self.wiki.artwork_facts(qid)
        from .references import Wikimedia
        p135 = Wikimedia.claim_ids(ent, "P135")
        mov = self.wiki.label(self.wiki.entities(p135[:1]).get(p135[0], {})) if p135 else ""
        imgs = (ent.get("claims") or {}).get("P18", [])
        image_url = file_thumb(imgs[0]["mainsnak"]["datavalue"]["value"], 800) if imgs else ""
        rec = Record("wikidata", f"https://www.wikidata.org/wiki/{qid}", en or (others[0] if others else ""),
                     list(dict.fromkeys(others)), f.get("artist", ""), "", f.get("date", ""), f.get("medium", ""),
                     f.get("collection", ""), mov, image_url, "image",
                     artist_born=f.get("artist_born", ""), artist_died=f.get("artist_died", ""),
                     artist_nationality=f.get("artist_nationality", ""))
        ids = {p: [c["mainsnak"]["datavalue"]["value"] for c in (ent.get("claims") or {}).get(p, [])
                   if c.get("mainsnak", {}).get("datavalue")] for p in ("P3634", "P4610", "P973")}
        ids["pages"] = self._linked_pages(ent, imgs[0]["mainsnak"]["datavalue"]["value"] if imgs else "")
        return rec, ids

    _formatters: dict = {}

    def _linked_pages(self, ent: dict, image_file: str) -> list[str]:
        """Institution pages the catalogue item itself points at: every museum/collection ID on it
        (built with the property's own formatter URL, P1630), "described at URL" (P973), and the source
        credited on its Commons image."""
        claims = ent.get("claims") or {}
        ext = {p: [c["mainsnak"]["datavalue"]["value"] for c in cs if c.get("mainsnak", {}).get("datatype") == "external-id"
                   and c["mainsnak"].get("datavalue")] for p, cs in claims.items()}
        ext = {p: v for p, v in ext.items() if v and p not in ("P3634", "P4610")}
        need = [p for p in ext if p not in self._formatters]
        for i in range(0, len(need), 50):
            d = self.wiki._get(self.wiki.wikidata_api, {"action": "wbgetentities", "ids": "|".join(need[i:i + 50]), "props": "claims"})
            for pid, pe in (d.get("entities") or {}).items():
                fs = [c["mainsnak"]["datavalue"]["value"] for c in (pe.get("claims") or {}).get("P1630", []) if c["mainsnak"].get("datavalue")]
                self._formatters[pid] = fs[0] if fs else ""
        urls = [self._formatters[p].replace("$1", str(v[0])) for p, v in ext.items() if self._formatters.get(p)]
        urls += [c["mainsnak"]["datavalue"]["value"] for c in claims.get("P973", []) if c["mainsnak"].get("datavalue")]
        if image_file:
            try:
                ii = self._get("https://commons.wikimedia.org/w/api.php", action="query", titles="File:" + image_file,
                               prop="imageinfo", iiprop="extmetadata", format="json")
                meta = next(iter(ii["query"]["pages"].values()))["imageinfo"][0]["extmetadata"]
                for k in ("Credit", "ImageDescription"):
                    urls += re.findall(r'href="(https?://[^"]+)"', (meta.get(k) or {}).get("value", ""))
            except Exception:
                pass
        out, hosts = [], set()
        for u in urls:
            u = html.unescape(u)
            h = urlparse(u).netloc.removeprefix("www.")
            if h and h not in hosts and "wiki" not in h and institutional(u) and "search" not in u.lower():
                hosts.add(h)
                out.append(u)
        return out[:6]

    # ── museums ────────────────────────────────────────────────────────────
    def _artic(self, rid: int | str, verified_by: str = "image") -> Record | None:
        d = self._get(f"https://api.artic.edu/api/v1/artworks/{rid}",
                      fields="id,title,alt_titles,artist_title,artist_display,date_display,medium_display,style_title,image_id")
        x = d.get("data")
        if not x:
            return None
        return self._artic_rec(x, verified_by)

    @staticmethod
    def _artic_rec(x: dict, verified_by: str) -> Record:
        bio = (x.get("artist_display") or "").split("\n", 1)[-1]
        img = f"https://www.artic.edu/iiif/2/{x['image_id']}/full/600,/0/default.jpg" if x.get("image_id") else ""
        return Record("artic", f"https://www.artic.edu/artworks/{x['id']}", x.get("title", ""), x.get("alt_titles") or [],
                      x.get("artist_title") or "", bio, x.get("date_display") or "", x.get("medium_display") or "",
                      "Art Institute of Chicago", x.get("style_title") or "", img, verified_by)

    def _artic_search(self, img, img_vec, queries: list[str], surname: str) -> list[Record]:
        out, seen = [], set()
        for q in queries:
            d = self._get("https://api.artic.edu/api/v1/artworks/search", q=q, limit=10,
                          fields="id,title,alt_titles,artist_title,artist_display,date_display,medium_display,style_title,image_id")
            for x in d.get("data") or []:
                if x["id"] in seen or surname not in fold(x.get("artist_title") or x.get("artist_display") or ""):
                    continue
                seen.add(x["id"])
                rec = self._artic_rec(x, "image")
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url)
                if ok:
                    return [rec]
        return out

    def _met(self, oid: str | int, verified_by: str) -> Record | None:
        d = self._get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}")
        if not d.get("title"):
            return None
        return Record("met", d.get("objectURL") or f"https://www.metmuseum.org/art/collection/search/{oid}", d["title"], [],
                      d.get("artistDisplayName", ""), d.get("artistDisplayBio", ""), d.get("objectDate", ""), d.get("medium", ""),
                      "The Metropolitan Museum of Art", "", d.get("primaryImageSmall") or "", verified_by)

    def _met_search(self, img, img_vec, queries: list[str], surname: str) -> list[Record]:
        for q in queries[:3]:
            d = self._get("https://collectionapi.metmuseum.org/public/collection/v1.1/search", q=q)
            for oid in (d.get("objectIDs") or [])[:5]:
                rec = self._met(oid, "image")
                if not rec or surname not in fold(rec.artist) or not rec.image_url:
                    continue                      # the Met's records rarely carry an image now: unverifiable
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url)
                if ok:
                    return [rec]
        return []

    def _cleveland_search(self, img, img_vec, queries: list[str], surname: str) -> list[Record]:
        for q in queries:
            d = self._get("https://openaccess-api.clevelandart.org/api/artworks/", q=q, limit=8)
            for x in d.get("data") or []:
                cr = (x.get("creators") or [{}])[0]
                if surname not in fold(cr.get("description") or ""):
                    continue
                url = ((x.get("images") or {}).get("web") or {}).get("url") or ""
                ok, sim, inl = self._verify(img, img_vec, url)
                if ok:
                    bio = re.sub(r"^[^(]*\(|\)\s*$", "", cr.get("description") or "")
                    return [Record("cleveland", x.get("url") or "", x.get("title", ""), [], cr.get("description", "").split(" (")[0],
                                   bio, x.get("creation_date") or "", x.get("technique") or "", "Cleveland Museum of Art",
                                   "", url, "image", sim, inl)]
        return []

    def _curated_page(self, url: str, verified_by: str, image_url: str = "", title_hint: str = "") -> Record | None:
        """A museum or institute page: its title (<meta og:title> or <title>) and its text. A page that is
        not a record (an error, a login wall, a bot check, a status page) is no source at all."""
        try:
            r = self.s.get(url, timeout=20)
        except Exception as e:
            log.info("curated page %s failed: %s", url[:80], e)
            return None
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return None
        m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', r.text) or \
            re.search(r"<title[^>]*>([^<]+)</title>", r.text)
        title = html.unescape(m.group(1)).strip() if m else title_hint
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text[:400_000], flags=re.S | re.I)
        text = fold(html.unescape(re.sub(r"<[^>]+>", " ", body)))
        if JUNK.search(title) or len(text) < 400:
            return None
        title = re.split(r"\s+[|–—]\s+|\s+-\s+", title)[0].strip()
        rec = Record(institution(url), url, "" if non_title(title) else title, [], verified_by=verified_by,
                     image_url=image_url, text=text)
        return rec

    def _curated_web(self, img, img_vec, queries: list[str], have: set[str]) -> list[Record]:
        """Museum and institute pages without an API (Zentrum Paul Klee, the Freer, a foundation's
        catalogue): the web image search, kept to curated domains, each picture verified."""
        web = getattr(self.idf, "web", None)
        if web is None:
            return []
        out, seen = [], set(have)
        for q in queries[:6]:
            try:
                refs = web.search(q, 25)
            except Exception as e:
                log.info("curated web search %r failed: %s", q, e)
                continue
            for r in refs:
                host = r.domain
                if host in seen or classify(r.page_url) != "curated":
                    continue
                ok, sim, inl = self._verify(img, img_vec, r.image_url or r.thumb_url)
                if not ok:
                    continue
                rec = self._curated_page(r.page_url, "image", r.image_url, "" if non_title(r.file_title) else r.file_title)
                if rec:
                    rec.clip, rec.inliers = sim, inl
                    out.append(rec)
                    seen.add(host)
            if len(out) >= 3:
                break
        return out

    # ── gather + rule ─────────────────────────────────────────────────────
    def gather(self, img: Image.Image, ident: dict, leads: list[str]) -> list[Record]:
        img_vec = self.emb.embed([img])[0]
        artist = ident.get("artist", "")
        surname = _surname(artist)
        titles = [t for t in leads if t]
        queries = list(dict.fromkeys([f"{artist} {t}" for t in titles[:5]] + [artist]))
        recs: list[Record] = []
        self.consulted = ["Wikidata", "Art Institute of Chicago", "Metropolitan Museum", "Cleveland Museum of Art"]
        for qid in self._wikidata_items(img, img_vec, artist, ident):
            rec, ids = self._wikidata_record(qid)
            recs.append(rec)
            for v in ids.get("P4610", [])[:1]:
                r = self._artic(v, f"link from Wikidata {qid}")
                if r:
                    recs.append(r)
            for v in ids.get("P3634", [])[:1]:
                r = self._met(v, f"link from Wikidata {qid}")
                if r:
                    recs.append(r)
            n = 0
            for u in ids.get("pages", []):
                if n >= 3:
                    break
                r = self._curated_page(u, f"link from Wikidata {qid}")
                if r and r.source not in {x.source for x in recs}:
                    if same_collection(rec.collection, r.source.replace(".", " ").replace("-", " ")):
                        r.collection = rec.collection        # the holding institution's own site
                    recs.append(r)
                    n += 1
        have = {r.source for r in recs}
        if "artic" not in have:
            recs += self._artic_search(img, img_vec, queries, surname)
        if "met" not in have:
            recs += self._met_search(img, img_vec, queries, surname)
        recs += self._cleveland_search(img, img_vec, queries, surname)
        self.consulted.append("museum and institute pages found by web search")
        recs += self._curated_web(img, img_vec, queries, {r.source for r in recs})
        # museum and institute pages on which identification itself verified the picture
        for rnd in ident.get("rounds", []):
            for m in rnd.get("confirmed_by", []):
                if classify(m.get("page_url", "")) == "curated" and "wikidata.org" not in m.get("page_url", ""):
                    r = self._curated_page(m["page_url"], "image", m.get("thumb_url", ""))
                    if r:
                        r.clip, r.inliers = m.get("clip", 0.0), m.get("inliers", 0)
                        recs.append(r)
        by_source: dict[str, Record] = {}
        for r in recs:
            key = institution(r.source)
            if key in by_source:                       # one vote per institution; keep the richer record
                if not by_source[key].title and r.title:
                    by_source[key] = r
                continue
            by_source[key] = r
        return list(by_source.values())

    def rule(self, recs: list[Record], ident: dict) -> dict:
        """Facts stated by 2+ independent curated sources; the rest is listed as unruled."""
        pref = sorted(recs, key=lambda r: (r.source == "wikidata", r.verified_by != "image"))   # holding museums first
        out = {"facts": {}, "support": {}, "unruled": {}, "sources": [asdict(r) for r in recs],
               "consulted": self.consulted, "unavailable": UNAVAILABLE, "failures": []}

        def pick(name, values, agree):
            for r, v in values:
                sup = sorted({r2.source for r2, v2 in values if agree(v, v2)} |
                             {r2.source for r2 in recs if r2.text and r2.states(v)})
                if len(sup) >= 2:
                    out["facts"][name], out["support"][name] = v, sup
                    return v
            out["unruled"][name] = [f"{r.source}: {v}" for r, v in values] or ["no curated source states it"]
            return ""

        # title: any of a source's titles (languages, aliases) may agree; the display is the preferred source's
        tvals = [(r, r.title) for r in pref if r.title]
        title = ""
        for r, t in tvals:
            sup = sorted({r2.source for r2 in recs if any(same_title(t, t2) for t2 in r2.all_titles()) or r2.states(t)})
            if len(sup) >= 2:
                wd = next((x for x in recs if x.source == "wikidata"), None)
                title = wd.title if (wd and "wikidata" in sup and wd.title) else t   # English, when Wikidata agrees
                out["facts"]["title"], out["support"]["title"] = title, sup
                break
        if not title:
            out["unruled"]["title"] = [f"{r.source}: {t}" for r, t in tvals] or ["no curated source names the work"]
        surname = _surname(ident.get("artist", ""))
        avals = [(r, r.artist) for r in pref if r.artist and _surname(r.artist) == surname]
        artist = pick("artist", avals, lambda a, b: _surname(a) == _surname(b))
        if not artist and avals:      # pages name the artist in their text, not in a field
            sup = sorted({avals[0][0].source} | {r.source for r in recs if r.text and surname and surname in r.text.split()})
            if len(sup) >= 2:
                artist = avals[0][1]
                out["facts"]["artist"], out["support"]["artist"] = artist, sup
                out["unruled"].pop("artist", None)
        if artist:            # Wikidata's form of the name when it is among the agreeing sources
            wd = next((r.artist for r in recs if r.source == "wikidata" and r.artist), "")
            out["facts"]["artist"] = wd or artist
        pick("date", [(r, r.date) for r in pref if _years(r.date)], same_date)
        pick("medium", [(r, r.medium) for r in pref if _materials(r.medium)], same_medium)
        pick("collection", [(r, r.collection) for r in pref if r.collection], same_collection)
        pick("movement", [(r, r.movement) for r in pref if r.movement], lambda a, b: fold(a) == fold(b))
        # the artist's life: Wikidata's years against a museum's printed biography
        wd = next((r for r in recs if r.source == "wikidata" and r.artist_born), None)
        bios = [r for r in recs if r.artist_bio and r.source != "wikidata"]
        if wd:
            agree = [r.source for r in bios if wd.artist_born in r.artist_bio and (not wd.artist_died or wd.artist_died in r.artist_bio)]
            agree += [r.source for r in recs if r.text and wd.artist_born and wd.artist_born in r.text
                      and (not wd.artist_died or wd.artist_died in r.text) and r.source not in agree]
            if agree:
                out["facts"]["artist_born"], out["facts"]["artist_died"] = wd.artist_born, wd.artist_died
                out["support"]["artist_life"] = ["wikidata", *agree]
                nat = fold(wd.artist_nationality)
                said = [r.source for r in bios if any(dem in fold(r.artist_bio) and DEMONYM[dem] in nat for dem in DEMONYM)]
                if nat and said:
                    out["facts"]["artist_nationality"] = wd.artist_nationality
                    out["support"]["artist_nationality"] = ["wikidata", *said]
                else:
                    out["unruled"]["artist_nationality"] = [f"wikidata: {wd.artist_nationality}"] + [f"{r.source}: {r.artist_bio}" for r in bios]
            else:
                out["unruled"]["artist_life"] = [f"wikidata: {wd.artist_born}-{wd.artist_died}"] + [f"{r.source}: {r.artist_bio}" for r in bios]
        else:
            out["unruled"]["artist_life"] = [f"{r.source}: {r.artist_bio}" for r in bios] or ["no curated source"]
        # a title written legibly on the work, when it differs from the ruled one, is shown beside it
        insc = (ident.get("inscribed_title") or "").strip()
        if title and insc and not same_title(insc, title):
            out["facts"]["inscribed_title"] = insc
        if not title:
            out["failures"].append("title not ruled: " + "; ".join(out["unruled"]["title"])[:300])
        if not artist:
            out["failures"].append("artist not ruled: " + "; ".join(out["unruled"].get("artist", ["no curated source"]))[:300])
        if out["failures"]:
            out["failures"].append(f"{len(recs)} curated source(s) matched the picture ({', '.join(r.source for r in recs) or 'none'}); "
                                   "2 are needed. Unavailable: " + "; ".join(f"{k} ({v})" for k, v in UNAVAILABLE.items()))
        return out


def leads(ident: dict) -> list[str]:
    """Search text from identification: titles, inscriptions, the matched copies' captions and file names.
    Non-titles (board names, shop wording) are kept as search text with the furniture removed."""
    out = [ident.get("inscribed_title", ""), ident.get("title", "")]
    for line in ident.get("evidence", {}).get("web_references", []) or []:
        m = re.match(r"file: (.*?) \| page: (.*?) \| site:", line)
        if m:
            out += [m.group(2), m.group(1)]
    clean = []
    for t in out:
        t = re.sub(r"\b(\d+\+?\s+|ideas?|pin on|for sale|prints?|poster|paintings?|art print|wall art|fine art america|pinterest)\b", " ", t, flags=re.I)
        t = re.split(r"\s[|–—]\s", t)[0]                         # "Title | Site" -> "Title"
        t = re.sub(r"\b(?=\w*\d)(?=\w*[a-z])\w{3,}\b", " ", t)   # hash fragments mix letters and digits
        t = re.sub(r"\(\s*\)", " ", t)
        t = re.sub(r"\s+", " ", t).strip(" -|,.")
        if len(t) >= 4 and not re.fullmatch(r"[\d\s]+", t):
            clean.append(t)
    return list(dict.fromkeys(clean))
