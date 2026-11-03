"""Open-collection reference lookups: Wikimedia Commons (images) and Wikidata (catalogue facts).

Requests are serial and identify themselves with a User-Agent, per Wikimedia API etiquette.
"""
from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass, field
from urllib.parse import quote

import requests

log = logging.getLogger(__name__)

THUMB_WIDTH = 500   # a standard Commons thumbnail step; arbitrary widths are throttled
SPARQL = "https://query.wikidata.org/sparql"
ART_OCCUPATION = re.compile(r"paint|artist|illustrat|printmak|engrav|sculpt|draughts|woodcut")


def file_thumb(filename: str, width: int) -> str:
    """Direct upload.wikimedia.org thumbnail URL (the CDN path, not the rate-limited Special:FilePath redirect)."""
    import hashlib
    name = filename.replace(" ", "_")
    h = hashlib.md5(name.encode()).hexdigest()
    ext = name.rsplit(".", 1)[-1].lower()
    suffix = {"tif": ".jpg", "tiff": ".jpg", "svg": ".png", "pdf": ".jpg", "webp": ".png"}.get(ext, "")
    return (f"https://upload.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{quote(name)}/"
            f"{width}px-{quote(name)}{suffix}")


@dataclass
class Reference:
    file_title: str
    page_id: int
    thumb_url: str
    page_url: str
    width: int
    height: int
    meta: dict = field(default_factory=dict)   # cleaned extmetadata


def _clean(v: str | None) -> str:
    if not v:
        return ""
    v = re.sub(r"<[^>]+>", " ", v)
    return re.sub(r"\s+", " ", html.unescape(v)).strip()


class Wikimedia:
    def __init__(self, commons_api: str, wikidata_api: str, user_agent: str):
        self.commons_api, self.wikidata_api = commons_api, wikidata_api
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self._last = 0.0

    def _get(self, url: str, params: dict) -> dict:
        wait = 0.2 - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            r = self.s.get(url, params={**params, "format": "json", "formatversion": 2}, timeout=30)
            self._last = time.time()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return {}

    # ── Commons ────────────────────────────────────────────────────────────
    def search_files(self, query: str, limit: int = 12) -> list[Reference]:
        data = self._get(self.commons_api, {
            "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
            "iiprop": "url|size|extmetadata", "iiurlwidth": THUMB_WIDTH,
            "iiextmetadatafilter": "ObjectName|Artist|DateTimeOriginal|ImageDescription|Credit",
        })
        pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
        out = []
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            if not ii.get("thumburl") or min(ii.get("width", 0), ii.get("height", 0)) < 300:
                continue
            meta = {k: _clean(v.get("value")) for k, v in (ii.get("extmetadata") or {}).items()}
            out.append(Reference(p["title"], p["pageid"], ii["thumburl"], ii.get("descriptionurl", ""),
                                 ii["width"], ii["height"], meta))
        return out

    def depicted_item(self, page_id: int) -> str | None:
        """Wikidata item a Commons file is a digital representation of (structured data P6243)."""
        data = self._get(self.commons_api, {"action": "wbgetentities", "ids": f"M{page_id}"})
        ent = (data.get("entities") or {}).get(f"M{page_id}", {})
        for prop in ("P6243", "P180"):   # digital representation of, then depicts
            for st in (ent.get("statements") or {}).get(prop, []):
                v = st.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                if isinstance(v, dict) and v.get("id"):
                    return v["id"]
        return None

    # ── Wikidata ───────────────────────────────────────────────────────────
    def entities(self, ids: list[str]) -> dict:
        if not ids:
            return {}
        data = self._get(self.wikidata_api, {"action": "wbgetentities", "ids": "|".join(ids[:50]),
                                             "props": "labels|claims|descriptions", "languages": "en"})
        return data.get("entities") or {}

    @staticmethod
    def label(ent: dict) -> str:
        return ((ent.get("labels") or {}).get("en") or {}).get("value", "")

    @staticmethod
    def claim_ids(ent: dict, prop: str) -> list[str]:
        out = []
        for c in (ent.get("claims") or {}).get(prop, []):
            v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(v, dict) and v.get("id"):
                out.append(v["id"])
        return out

    @staticmethod
    def claim_year(ent: dict, prop: str) -> str:
        for c in (ent.get("claims") or {}).get(prop, []):
            v = c.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(v, dict) and v.get("time"):
                m = re.match(r"[+-]?(\d{1,4})", v["time"])
                if m:
                    return str(int(m.group(1)))
        return ""

    def find_artist(self, name: str) -> str | None:
        """Wikidata QID of the visual artist best matching ``name`` (description must name an art occupation)."""
        data = self._get(self.wikidata_api, {"action": "wbsearchentities", "search": name, "language": "en",
                                             "type": "item", "limit": 7})
        for e in data.get("search", []):
            if ART_OCCUPATION.search((e.get("description") or "").lower()):
                return e["id"]
        return None

    def oeuvre(self, artist_qid: str) -> list[dict]:
        """Every catalogued work by the artist that has an image: [{qid, label, file}]."""
        q = (f"SELECT ?w ?wLabel ?img WHERE {{ ?w wdt:P170 wd:{artist_qid}; wdt:P18 ?img. "
             f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }}')
        for attempt in range(3):
            r = self.s.get(SPARQL, params={"query": q, "format": "json"}, timeout=90)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        seen, out = set(), []
        for b in r.json()["results"]["bindings"]:
            qid = b["w"]["value"].rsplit("/", 1)[-1]
            if qid in seen:
                continue
            seen.add(qid)
            fname = b["img"]["value"].rsplit("/", 1)[-1]
            from urllib.parse import unquote
            out.append({"qid": qid, "label": b.get("wLabel", {}).get("value", ""), "file": unquote(fname)})
        return out

    def artwork_facts(self, qid: str) -> dict:
        """Catalogue facts for a painting item and its creator. Empty dict if not an artwork."""
        ent = self.entities([qid]).get(qid, {})
        creators = self.claim_ids(ent, "P170")
        if not creators:
            return {}
        materials = self.claim_ids(ent, "P186")
        collections = self.claim_ids(ent, "P195")
        creator_id = creators[0]
        rel = self.entities([creator_id] + materials[:3] + collections[:1])
        creator = rel.get(creator_id, {})
        nat_ids = self.claim_ids(creator, "P27")
        nat = self.entities(nat_ids[:1]).get(nat_ids[0], {}) if nat_ids else {}
        mats = [self.label(rel[m]) for m in materials[:3] if m in rel]
        return {
            "wikidata": qid,
            "title": self.label(ent),
            "date": self.claim_year(ent, "P571"),
            "artist": self.label(creator),
            "artist_wikidata": creator_id,
            "artist_born": self.claim_year(creator, "P569"),
            "artist_died": self.claim_year(creator, "P570"),
            "artist_nationality": self.label(nat),
            "medium": _medium(mats),
            "collection": self.label(rel.get(collections[0], {})) if collections else "",
        }


def _medium(mats: list[str]) -> str:
    mats = [m for m in mats if m]
    if not mats:
        return ""
    paints = [m for m in mats if any(w in m.lower() for w in ("paint", "oil", "acrylic", "tempera",
                                                             "watercolor", "watercolour", "gouache", "ink"))]
    supports = [m for m in mats if m not in paints]
    if paints and supports:
        return f"{paints[0].capitalize()} on {supports[0]}"
    return ", ".join(mats).capitalize()
