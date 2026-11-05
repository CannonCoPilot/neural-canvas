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

Unavailable on 2026-11-05 (remediable, not silently skipped): see ``UNAVAILABLE``. Each entry names
the wall, so the run's failure record says what would open it.
"""
from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from itertools import chain
from urllib.parse import quote, urljoin, urlparse

import numpy as np
import requests
from PIL import Image

from .matching import fetch_image, geometric_inliers
from .references import file_thumb
from .sources import classify, institutional, non_title, rulable, supporting

log = logging.getLogger(__name__)
UA = "neural-canvas/1.0 (+https://github.com/CannonCoPilot/neural-canvas)"
LANGS = "en|de|fr|nl|it|es|ja|da|sv|no|nb|fi|cs|ru|pl|pt"
UNAVAILABLE = {
    "Getty ULAN": "server-side TLS failure on vocab.getty.edu (2026-11-05)",
    "Norman Rockwell Museum": "collection.nrm.org answers every automated request with a bot challenge (2026-11-05)",
}
# Smithsonian art units. The API ignores its fq filter; a fielded unit_code term inside q filters (2026-11-05).
SI_UNITS = "SAAM OR NPG OR HMSG OR NMAA OR CHNDM OR NMAfA"   # NMAA = Freer|Sackler (FSG matches nothing)
ARTIC_FIELDS = "id,title,alt_titles,artist_title,artist_display,date_display,medium_display,style_title,image_id"
LUX = "https://lux.collections.yale.edu"     # Yale: Linked Art records, Yale-provided metadata CC0
TRANSLATE_PROMPT = """Artwork titles, possibly in German, French, Dutch, Italian, Spanish or another language:
{titles}
For each, give the language and the plain English title a museum would print (Giftbeeren -> Poisonous Berries).
Translate only; never add words that are not in the title. Reply JSON:
{{"titles": [{{"original": "...", "language": "...", "english": "..."}}]}}"""
# Catalogues raisonnés and the institutions that hold or published an artist's work, searched with site:
# (Sir, 2026-11-05). Each page found is read and must show the same picture. Adding an artist is a line here.
CATALOGUE_SITES = {
    "whistler": ["whistlerpaintings.gla.ac.uk"],              # University of Glasgow catalogue raisonné
    "remington": ["remington.centerofthewest.org"],           # Buffalo Bill Center of the West catalogue raisonné
    "rockwell": ["prints.nrm.org", "saturdayeveningpost.com"],  # Norman Rockwell Museum; the Post's own cover archive
    "wyeth": ["crystalbridges.org", "wyethfoundation.org", "brandywine.org", "farnsworthmuseum.org"],
    "klee": ["zpk.org"],
}
LABELS = ("Artist", "Date", "Collection", "Accession Number", "Medium", "Support", "Size", "Signature", "Inscription",
          "Dimensions", "Credit Line", "Catalogue Number", "Provenance")
# Getty AAT ids the Rijksmuseum's Linked Art uses in place of labels
AAT_ENGLISH, AAT_PREFERRED, AAT_MATERIALS = "300388277", "300404670", "300435429"
RKD_ARTISTS = "https://api.rkd.triply.cc/datasets/rkd/RKD-SDO-Knowledge-Graph/services/Virtuoso/sparql"
RKD_WORKS = "https://api.rkd.triply.cc/datasets/rkd/RKD-Knowledge-Graph/services/SPARQL/sparql"
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


PAINTS = {"oil", "acrylic", "tempera", "gouache", "watercolor", "pastel", "charcoal", "ink", "graphite", "pencil",
          "woodblock", "woodcut", "lithograph", "etching", "engraving"}


def medium_clash(a: str, b: str) -> bool:
    """Both name the paint and no paint is shared: oil versus pastel."""
    pa, pb = _materials(a) & PAINTS, _materials(b) & PAINTS
    return bool(pa and pb and not pa & pb)


def same_medium(a: str, b: str) -> bool:
    if medium_clash(a, b):
        return False                                  # oil versus watercolour: different
    return bool(_materials(a) & _materials(b))


def same_collection(a: str, b: str) -> bool:
    drop = {"museum", "art", "arts", "gallery", "national", "collection", "fine", "institute", "the", "of"}
    ta, tb = _toks(a) - drop, _toks(b) - drop
    return bool(ta and tb) and (ta <= tb or tb <= ta or len(ta & tb) / len(ta | tb) >= 0.5)


JUNK = re.compile(r"just a moment|page not found|not found|404|403|access denied|forbidden|service status|"
                  r"log ?in|logg p|sign in|anmelden|captcha|attention required|error|unavailable|cookies?", re.I)
EVENT = re.compile(r"\b(events?|talks?|webinars?|lectures?|exhibitions?/upcoming|shop|store|tickets?|blog|news|press)\b|"
                   r"\bplate \d+\b", re.I)
_SAME_INSTITUTION = {"artic": "artic.edu", "met": "metmuseum.org", "cleveland": "clevelandart.org",
                     "wikidata": "wikidata.org"}


def institution(url_or_source: str) -> str:
    """One vote per institution: subdomains and the API of a museum are the museum (images.nga.gov =
    nga.gov; the Art Institute's API = artic.edu)."""
    s = _SAME_INSTITUTION.get(url_or_source, url_or_source)
    if ":" in s and "://" not in s:
        return s                                   # an aggregator's provider key ("europeana:<museum>")
    host = (urlparse(s).netloc or s).lower().removeprefix("www.")
    parts = host.split(".")
    n = 3 if len(parts) >= 3 and parts[-2] in ("co", "ac", "gov", "org", "com", "edu") and len(parts[-1]) == 2 else 2
    reg = parts[-n:]
    return reg[0] if len(reg) > 1 else host        # the name without its country domain: meisterdrucke.us = .uk


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
    authority: bool = False      # an artist authority record: rules the artist's life, never the work
    support: bool = False        # an auction house or the press: corroborates, never counts
    images: list = field(default_factory=list)       # the page's other pictures, tried when image_url is not the work
    overrule: bool = False       # Wikidata and what it links, a museum's own API, a catalogue raisonné (Sir, 2026-11-05)

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
        self.cfg = getattr(identifier, "cfg", None)
        self._apikeys: dict | None = None
        self.wiki, self.emb = identifier.wiki, identifier.emb
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.s.headers["AIC-User-Agent"] = UA       # the Art Institute asks callers to name themselves
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

    def keys(self) -> dict:
        """API keys from the secrets file ("site: key" lines); never logged."""
        if self._apikeys is None:
            self._apikeys = {}
            path = (self.cfg["sources"].get("api_keys_path") if self.cfg else "") or ""
            try:
                for line in open(path, encoding="utf-8"):
                    name, sep, val = line.strip().partition(":")
                    if sep and val.strip() and not name.startswith("#"):
                        self._apikeys[name.strip().lower()] = val.strip()
            except OSError:
                pass
        return self._apikeys

    def _keyed(self, url: str, key: str, **params) -> dict:
        try:
            r = self.s.get(url, params=params, timeout=20)
            if r.status_code != 200:
                log.info("catalogue %s -> HTTP %s", url[:60], r.status_code)
                return {}
            return r.json()
        except Exception as e:      # the key travels in the query string: never let it reach a log
            log.info("catalogue %s failed: %s", url[:60], str(e).replace(key, "***"))
            return {}

    def _sparql(self, endpoint: str, query: str) -> list[dict]:
        try:
            r = self.s.post(endpoint, data={"query": query}, timeout=60,
                            headers={"Accept": "application/sparql-results+json"})
            r.raise_for_status()
            return [{k: v["value"] for k, v in b.items()} for b in r.json()["results"]["bindings"]]
        except Exception as e:
            log.info("sparql %s failed: %s", endpoint[:60], e)
            return []

    def _rkd_artist(self, rkd_id: str) -> Record | None:
        """RKDartists (RKD, Netherlands Institute for Art History): name, life, nationality. An authority."""
        rows = self._sparql(RKD_ARTISTS, f"""PREFIX sdo: <https://schema.org/>
            SELECT ?p ?o WHERE {{ <https://data.rkd.nl/artists/{int(rkd_id)}> ?p ?o .
              VALUES ?p {{ sdo:name sdo:birthDate sdo:deathDate sdo:nationality }} }}""")
        if not rows:
            return None
        v = {}
        for r in rows:
            v.setdefault(r["p"].rsplit("/", 1)[-1], []).append(r["o"])
        name = (v.get("name") or [""])[0]
        if "," in name:
            last, first = name.split(",", 1)
            name = f"{first.strip()} {last.strip()}"
        yr = lambda k: (re.findall(r"\d{4}", (v.get(k) or [""])[0]) or [""])[0]
        nat = next((n for n in v.get("nationality", []) if not n.startswith("http") and n[:1].isupper()
                    and fold(n) in DEMONYM), "")
        born, died = yr("birthDate"), yr("deathDate")
        return Record("rkd.nl", f"https://rkd.nl/artists/{rkd_id}", "", [], name,
                      f"{nat}, {born}–{died}".strip(", –"), verified_by="Wikidata P650", authority=True,
                      artist_born=born, artist_died=died, artist_nationality=nat)

    def _rkd_work(self, rkd_img: str, verified_by: str) -> Record | None:
        """RKDimages: the work's titles (several languages) and production date."""
        rows = self._sparql(RKD_WORKS, f"""PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
            SELECT ?title ?begin ?end WHERE {{ BIND(<https://data.rkd.nl/images/{int(rkd_img)}> AS ?w)
              OPTIONAL {{ ?w crm:P1_is_identified_by ?a . ?a a crm:E33_E41_Linguistic_Appellation ; crm:P190_has_symbolic_content ?title }}
              OPTIONAL {{ ?w crm:P108i_was_produced_by ?p . ?p crm:P4_has_time-span ?ts .
                         ?ts crm:P82a_begin_of_the_begin ?begin ; crm:P82b_end_of_the_end ?end }} }} LIMIT 20""")
        titles = list(dict.fromkeys(r["title"] for r in rows if r.get("title")))
        if not titles:
            return None
        b = next((r for r in rows if r.get("begin")), {})
        lo, hi = (re.findall(r"\d{4}", b.get("begin", "")) or [""])[0], (re.findall(r"\d{4}", b.get("end", "")) or [""])[0]
        date = lo if lo == hi else (f"{lo}–{hi}" if lo and hi else lo or hi)
        return Record("rkd.nl", f"https://rkd.nl/images/{rkd_img}", titles[0], titles[1:], date=date, verified_by=verified_by)

    def _wikipedia(self, page_title: str, verified_by: str, img=None, img_vec=None, lang: str = "en") -> Record | None:
        d = self._get(f"https://{lang}.wikipedia.org/w/api.php", action="query", titles=page_title, prop="extracts|pageimages",
                      explaintext=1, piprop="original", redirects=1, format="json")
        page = next(iter(((d.get("query") or {}).get("pages") or {}).values()), {})
        if not page or "missing" in page or not page.get("extract"):
            return None
        image = (page.get("original") or {}).get("source", "")
        clip = inl = 0
        if img is not None:                       # reached by search: its picture must be this one
            ok, clip, inl = self._verify(img, img_vec, image)
            if not ok:
                return None
        return Record("wikipedia.org", f"https://{lang}.wikipedia.org/wiki/{quote(page['title'].replace(' ', '_'))}",
                      page["title"], [], image_url=image, verified_by=verified_by, clip=clip, inliers=inl,
                      text=fold(page["extract"][:60000]))

    def _commons(self, file: str, verified_by: str) -> Record | None:
        d = self._get("https://commons.wikimedia.org/w/api.php", action="query", titles="File:" + file, prop="imageinfo",
                      iiprop="extmetadata|url", iiurlwidth=800, format="json")
        page = next(iter(((d.get("query") or {}).get("pages") or {}).values()), {})
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        def val(k):   # Commons fields carry template residue ("date QS:P571,+1890"): strip it
            v = html.unescape(re.sub(r"<[^>]+>", " ", str((meta.get(k) or {}).get("value", ""))))
            return re.sub(r"\s+", " ", re.sub(r"\b(?:title|date)?\s*QS:\S+(?:\s+\S*\+\S+)?", " ", v)).strip()
        title = re.sub(r"\s+", " ", val("ObjectName"))
        if not (title or meta):
            return None
        text = fold(" ".join(val(k) for k in meta))
        return Record("commons.wikimedia.org", f"https://commons.wikimedia.org/wiki/File:{quote(file)}",
                      "" if non_title(title) else title, [], re.sub(r"\s+", " ", val("Artist"))[:120], "",
                      val("DateTimeOriginal")[:40], "", "", "", info.get("thumburl", ""), verified_by, text=text)

    def _europeana(self, img, img_vec, queries: list[str], surname: str) -> list[Record]:
        key = self.keys().get("europeana")
        if not key:
            return []
        self.consulted.append("Europeana")
        for q in queries[:4]:
            d = self._keyed("https://api.europeana.eu/record/v2/search.json", key, wskey=key, query=q, rows=12,
                            media="true", qf="TYPE:IMAGE")
            for x in d.get("items") or []:
                creator = " ".join(x.get("dcCreator") or [])
                if surname not in fold(creator):
                    continue
                url = (x.get("edmIsShownBy") or x.get("edmPreview") or [""])[0]
                ok, sim, inl = self._verify(img, img_vec, url)
                if ok:
                    prov = (x.get("dataProvider") or ["Europeana"])[0]
                    return [Record(f"europeana:{fold(prov)}", x.get("guid", ""), (x.get("title") or [""])[0], [], creator,
                                   "", (x.get("year") or [""])[0], "", prov, "", url, "image", sim, inl)]
        return []

    def _harvard(self, img, img_vec, queries: list[str], surname: str, titles: list[str] = ()) -> list[Record]:
        key = self.keys().get("harvard")
        if not key:
            return []
        self.consulted.append("Harvard Art Museums")
        url = "https://api.harvardartmuseums.org/object"
        # person + title together is a fielded AND (either alone does not filter); free text q is the fallback
        fielded = (self._keyed(url, key, apikey=key, person=surname, title=t, size=10) for t in titles[:4] if surname)
        plain = (self._keyed(url, key, apikey=key, q=q, size=10, hasimage=1) for q in queries[:4])
        for d in chain(fielded, plain):
            for x in d.get("records") or []:
                people = " ".join(p.get("name", "") for p in x.get("people") or [])
                if surname not in fold(people) or not x.get("primaryimageurl"):
                    continue
                ok, sim, inl = self._verify(img, img_vec, x["primaryimageurl"] + "?height=800")
                if ok:
                    bio = next((p.get("displaydate", "") for p in x.get("people") or [] if surname in fold(p.get("name", ""))), "")
                    former = [t.get("title", "") for t in x.get("titles") or [] if t.get("title") and t.get("title") != x.get("title")]
                    return [Record("harvardartmuseums.org", x.get("url", ""), x.get("title", ""), former, people, bio,
                                   x.get("dated", ""), x.get("medium") or "", "Harvard Art Museums", x.get("style") or "",
                                   x["primaryimageurl"], "image", sim, inl)]
        return []

    def _smithsonian(self, img, img_vec, queries: list[str], surname: str) -> list[Record]:
        """Smithsonian Open Access: the art museums (American Art, Portrait Gallery, Hirshhorn, Freer|Sackler,
        Cooper Hewitt, African Art). Images come from its own image server at 800 px."""
        key = self.keys().get("smithsonian")
        if not key:
            return []
        self.consulted.append("Smithsonian Open Access")
        for q in queries[:4]:
            q = re.sub(r"[^\w\s'’]", " ", q)          # its query parser reads : ( ) " - as syntax
            d = self._keyed("https://api.si.edu/openaccess/api/v1.0/search", key, api_key=key,
                            q=f"unit_code:({SI_UNITS}) AND {q}", rows=10)
            for x in (d.get("response") or {}).get("rows") or []:
                c = x.get("content") or {}
                ft, dn = c.get("freetext") or {}, c.get("descriptiveNonRepeating") or {}
                def first(k, label=None):
                    return next((e.get("content", "") for e in ft.get(k) or [] if label in (None, e.get("label"))), "")
                maker = first("name", "Artist") or first("name")
                media = [m["idsId"] for m in (dn.get("online_media") or {}).get("media") or []
                         if m.get("type") == "Images" and m.get("idsId")]
                if surname not in fold(maker) or not media:
                    continue
                url = f"https://ids.si.edu/ids/deliveryService?id={media[0]}&max=800"
                ok, sim, inl = self._verify(img, img_vec, url)
                if ok:
                    artist, _, bio = maker.partition(", ")         # "Winslow Homer, born Boston, MA 1836-died ..."
                    title = x.get("title") or (dn.get("title") or {}).get("content", "")
                    notes = " ".join(e.get("content", "") for e in ft.get("notes") or [])
                    return [Record("si", dn.get("record_link") or dn.get("guid", ""), title, [], artist, bio,
                                   first("date"), first("physicalDescription", "Medium"),
                                   dn.get("data_source") or "Smithsonian Institution", "", url, "image", sim, inl,
                                   text=fold(f"{title} {maker} {notes}"))]
        return []

    @staticmethod
    def _lux_rec(x: dict) -> Record | None:
        """A Yale Linked Art object record: title, maker, display date, medium, owner, image."""
        if not x.get("id"):
            return None
        def cls(e):
            return {c.get("_label", "") for c in e.get("classified_as") or []}
        title = next((i.get("content", "") for i in x.get("identified_by") or []
                      if i.get("type") == "Name" and "Primary Title" in cls(i)), x.get("_label", ""))
        others = [i.get("content", "") for i in x.get("identified_by") or []
                  if i.get("type") == "Name" and "Primary Title" not in cls(i) and i.get("content") != title]
        prod = x.get("produced_by") or {}
        makers = [a.get("_label", "") for part in [prod, *(prod.get("part") or [])] for a in part.get("carried_out_by") or []]
        maker = re.sub(r"^[A-Z][a-z ]+:\s*", "", makers[0]) if makers else ""   # "Artist: Name (American, 1834–1903)"
        artist, _, bio = maker.partition(" (")
        date = next((i.get("content", "") for i in (prod.get("timespan") or {}).get("identified_by") or []), "")
        refs = x.get("referred_to_by") or []
        medium = next((r.get("content", "") for r in refs if "Medium" in cls(r)), "")
        owner = next((o.get("_label", "") for o in x.get("current_owner") or []), "")
        image = next((a.get("id", "") for rep in x.get("representation") or [] for d in rep.get("digitally_shown_by") or []
                      for a in d.get("access_point") or []), "")
        text = fold(" ".join([title, maker, *(str(r.get("content", "")) for r in refs)]))
        return Record("yale", x["id"], title, others, artist, bio.rstrip(")"), date, medium, owner, "", image, "image",
                      text=text)

    def _yale(self, img, img_vec, queries: list[str], surname: str) -> list[Record]:
        """Yale (LUX): the Art Gallery, the Center for British Art and the libraries. Yale-provided metadata is
        CC0 (LUX Open Access page, read 2026-11-05); one lookup per work, never a harvest."""
        self.consulted.append("Yale University (LUX)")
        seen: set[str] = set()
        for q in queries[:3]:
            d = self._get(f"{LUX}/api/search/item", q=json.dumps({"text": q}))
            for stub in (d.get("orderedItems") or [])[:6]:
                u = stub.get("id", "")
                if not u.startswith(LUX) or u in seen:
                    continue
                seen.add(u)
                rec = self._lux_rec(self._get(u))
                if not rec or surname not in fold(rec.artist) or not rec.image_url:
                    continue
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url)
                if ok:
                    return [rec]
        return []

    @staticmethod
    def _rijks_rec(x: dict, surname: str = "") -> Record | None:
        """A Rijksmuseum Linked Art object: its labels are Getty AAT ids, not text."""
        if not x.get("id"):
            return None
        def ids(e):
            return {c.get("id", "").rsplit("/", 1)[-1] for c in e.get("classified_as") or []}
        def en(e):
            return any(l.get("id", "").endswith(AAT_ENGLISH) for l in e.get("language") or [])
        names = [n for n in x.get("identified_by") or [] if n.get("type") == "Name" and n.get("content")]
        pref = [n for n in names if AAT_PREFERRED in ids(n)] or names
        title = next((n["content"] for n in pref if en(n)), pref[0]["content"] if pref else "")
        prod = x.get("produced_by") or {}
        def name(maker):          # notation is a list of language forms, or a single one
            notes = maker.get("notation") or []
            notes = [notes] if isinstance(notes, dict) else notes
            return next((n.get("@value", "") for n in notes if n.get("@language") == "en"), notes[0].get("@value", "") if notes else "")
        makers = [name(a) for part in [prod, *(prod.get("part") or [])] for a in part.get("carried_out_by") or []]
        artist = next((m for m in makers if surname and surname in fold(m)), makers[0] if makers else "")
        stamps = (prod.get("timespan") or {}).get("identified_by") or []
        date = next((t.get("content", "") for t in stamps if en(t)), stamps[0].get("content", "") if stamps else "")
        refs = x.get("referred_to_by") or []
        medium = next((r.get("content", "") for r in refs if AAT_MATERIALS in ids(r) and en(r)), "")
        image = next((a.get("id", "") for s in x.get("subject_of") or [] for d in s.get("digitally_carried_by") or []
                      for a in d.get("access_point") or [] if "iiif" in a.get("id", "")), "")
        image = image.replace("/full/max/", "/full/800,/")
        text = fold(" ".join([*(n["content"] for n in names), *makers, *(str(r.get("content", "")) for r in refs)]))
        others = list(dict.fromkeys(n["content"] for n in names if n["content"] != title))
        return Record("rijksmuseum.nl", x["id"], title, others, artist, "",
                      date, medium, "Rijksmuseum", "", image, "image", text=text)

    def _rijksmuseum(self, img, img_vec, titles: list[str], artist: str, surname: str) -> list[Record]:
        """Rijksmuseum data services (keyless Linked Art; the old key API is retired): artist AND title,
        then the artist's works with images."""
        if not artist:
            return []
        self.consulted.append("Rijksmuseum")
        seen: set[str] = set()
        searches = [{"creator": artist, "title": t} for t in titles[:4]] + [{"creator": artist, "imageAvailable": "true"}]
        for params in searches:
            d = self._get("https://data.rijksmuseum.nl/search/collection", **params)
            for stub in (d.get("orderedItems") or [])[:6]:
                u = stub.get("id", "")
                if u in seen:
                    continue
                seen.add(u)
                rec = self._rijks_rec(self._get(u), surname)
                if not rec or surname not in fold(rec.artist) or not rec.image_url:
                    continue
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url)
                if ok:
                    return [rec]
        return []

    def _artist_living(self, artist: str) -> tuple[bool, str]:
        """(living, artist QID). An artist no catalogue knows, or one with no recorded death, may be living."""
        aq = self.wiki.find_artist(artist) if artist else None
        if not aq:
            return True, ""
        ent = self.wiki.entities([aq]).get(aq, {})
        return not self.wiki.claim_year(ent, "P570"), aq

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
                                                    "props": "labels|aliases|claims|sitelinks", "languages": LANGS})
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
                   if c.get("mainsnak", {}).get("datavalue")] for p in ("P3634", "P4610", "P973", "P350")}
        ids["enwiki"] = ((ent.get("sitelinks") or {}).get("enwiki") or {}).get("title", "")
        ids["image_file"] = imgs[0]["mainsnak"]["datavalue"]["value"] if imgs else ""
        ids["artist_qid"] = f.get("artist_wikidata", "")
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
        ext = {p: v for p, v in ext.items() if v and p not in ("P3634", "P4610", "P350", "P650")}
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
        d = self._get(f"https://api.artic.edu/api/v1/artworks/{rid}", fields=ARTIC_FIELDS)
        x = d.get("data")
        if not x:
            return None
        return self._artic_rec(x, verified_by)

    @staticmethod
    def _artic_rec(x: dict, verified_by: str) -> Record:
        bio = (x.get("artist_display") or "").split("\n", 1)[-1]
        # 843 px is the Art Institute's standard derivative; other widths are refused (HTTP 403)
        img = f"https://www.artic.edu/iiif/2/{x['image_id']}/full/843,/0/default.jpg" if x.get("image_id") else ""
        return Record("artic", f"https://www.artic.edu/artworks/{x['id']}", x.get("title", ""), x.get("alt_titles") or [],
                      x.get("artist_title") or "", bio, x.get("date_display") or "", x.get("medium_display") or "",
                      "Art Institute of Chicago", x.get("style_title") or "", img, verified_by)

    def _artic_post(self, artist: str, title: str) -> dict:
        """The Art Institute's fielded search: artist AND title must both match (plain q ranks the
        artist's other works first)."""
        try:
            r = self.s.post("https://api.artic.edu/api/v1/artworks/search", timeout=20, json={
                "size": 5, "fields": ARTIC_FIELDS.split(","),
                "query": {"bool": {"must": [{"match": {"artist_title": artist}}, {"match": {"title": title}}]}}})
            return r.json() if r.status_code == 200 else {}
        except Exception as e:
            log.info("artic fielded search failed: %s", e)
            return {}

    def _artic_search(self, img, img_vec, queries: list[str], surname: str,
                      artist: str = "", titles: tuple[str, ...] | list[str] = ()) -> list[Record]:
        out, seen = [], set()
        precise = [self._artic_post(artist, t) for t in titles[:4]] if artist else []
        plain = (self._get("https://api.artic.edu/api/v1/artworks/search", q=q, limit=10, fields=ARTIC_FIELDS)
                 for q in queries)
        for d in chain(precise, plain):          # plain queries run only if the fielded ones find nothing
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

    def _met_search(self, img, img_vec, queries: list[str], surname: str, artist: str = "",
                    titles: list[str] = ()) -> list[Record]:
        api = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
        # the Met honours one field flag per call: artist-only AND title-only, intersected here
        both = []
        by_artist = set(self._get(api, q=artist, artistOrCulture="true").get("objectIDs") or []) if artist and titles else set()
        for t in titles[:3] if by_artist else []:
            ids = [i for i in self._get(api, q=t, title="true").get("objectIDs") or [] if i in by_artist]
            if ids:
                both.append(ids)
        plain = ((self._get(api, q=q).get("objectIDs") or []) for q in queries[:3])
        for ids in chain(both, plain):
            for oid in ids[:5]:
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
        raw = r.text[:400_000]
        # what a script-rendered page still says: its meta tags, its schema.org data, and its own address
        meta = " ".join(re.findall(r'<meta[^>]+content=["\']([^"\']{3,400})["\']', raw))
        ld = " ".join(re.findall(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', raw, flags=re.S | re.I))
        body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        path = re.sub(r"[-_/.]+", " ", urlparse(url).path)
        text = fold(" ".join([html.unescape(re.sub(r"<[^>]+>", " ", body)), html.unescape(meta), ld, path]))
        if JUNK.search(title) or len(text) < 400:
            return None
        if EVENT.search(title) or EVENT.search(urlparse(url).path):
            return None                                   # an events or shop page shows the picture, it is no record of it
        title = re.split(r"\s+[|–—]\s+|\s+-\s+", title)[0].strip()
        og = re.search(r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)', raw) or \
             re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', raw)
        imgs = ([html.unescape(og.group(1))] if og else []) + [
            html.unescape(i) for i in re.findall(r'<img[^>]+?(?:data-src|src)=["\']([^"\']+)', raw)
            if not re.search(r"logo|icon|\.svg|banner|sprite|pixel|gravatar|avatar|/graphics/", i, re.I)]
        imgs = list(dict.fromkeys(urljoin(url, i) for i in imgs if not i.startswith("data:")))[:8]
        image_url = image_url or (imgs[0] if imgs else "")
        plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body)))
        f = labelled(plain)
        m = re.search(r"\b[A-Z]{1,4}\.\s?\d{3,5}[a-z]?\s+(.{3,160}?)\s+Artist:", plain)   # "M.0807 The Riva ... Artist:"
        if m:
            title = m.group(1).strip()
        title = re.sub(r"\s+by\s+[A-Z][\w.'’ -]{2,60}$", "", title).strip()                # "Airborne by Andrew Wyeth"
        rec = Record(institution(url), url, "" if non_title(title) else title, [], f.get("Artist", ""), "",
                     f.get("Date", ""), f.get("Medium", ""), f.get("Collection", ""), "", image_url, verified_by,
                     text=text, images=imgs)
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
                support = supporting(r.page_url)
                if host in seen or not (support or rulable(r.page_url, getattr(self, "_artist", ""), getattr(self, "_living", False))):
                    continue
                ok, sim, inl = self._verify(img, img_vec, r.image_url or r.thumb_url)
                if not ok:
                    continue
                rec = self._curated_page(r.page_url, "image", r.image_url, "" if non_title(r.file_title) else r.file_title)
                if rec:
                    rec.clip, rec.inliers, rec.support = sim, inl, support
                    out.append(rec)
                    seen.add(host)
            if sum(not x.support for x in out) >= 3:
                break
        return out

    def _translations(self, titles: list[str]) -> list[str]:
        """English forms of foreign titles (Giftbeeren -> Poisonous Berries). Searched beside the original,
        never instead of it: museums and catalogues keep the original wording too."""
        vlm = getattr(self.idf, "vlm", None)
        todo = [t for t in titles[:5] if re.search(r"[^\W\d_]{3,}", t or "")]   # the model names each language
        if vlm is None or not todo:
            return []
        try:
            d = vlm.ask_json(TRANSLATE_PROMPT.format(titles="\n".join(f"- {t}" for t in todo)), required=("titles",),
                             num_predict=512)
        except Exception as e:
            log.info("title translation failed: %s", e)
            return []
        out = []
        for x in d.get("titles") or []:
            en, orig = str(x.get("english", "")).strip(), str(x.get("original", "")).strip()
            if en and orig and fold(en) != fold(orig) and str(x.get("language", "")).lower() not in ("en", "english"):
                out.append(en)
        have = {fold(t) for t in titles}
        return [t for t in dict.fromkeys(out) if fold(t) not in have]

    def _artvee(self, img, img_vec, titles: list[str], artist: str, surname: str) -> list[Record]:
        """Artvee (credible, Sir 2026-11-05): its search by each title form, then the artist's whole Artvee
        catalogue ranked by embedding. The picture must match on both instruments."""
        av = getattr(self.idf, "artvee", None)
        if av is None:
            return []
        self.consulted.append("Artvee")
        seen: set[str] = set()
        for refs in chain((av.search(t) for t in titles[:6] if len(t) >= 4), [av.nearest(artist, img_vec, 6)] if artist else []):
            for r in refs:
                m = r.meta
                if r.page_url in seen or (surname and surname not in fold(m.get("artist", ""))):
                    continue
                seen.add(r.page_url)
                ok, sim, inl = self._verify(img, img_vec, r.thumb_url)
                if ok:
                    return [Record("artvee.com", r.page_url, m.get("title", ""), [], m.get("artist", ""), m.get("origin", ""),
                                   m.get("date", ""), "", "", "", r.thumb_url, "image", sim, inl,
                                   text=fold(f"{m.get('title', '')} {m.get('date', '')} {m.get('artist', '')} {m.get('origin', '')}"))]
        return []

    def _web_text(self, img, img_vec, queries: list[str], have: set[str], search=None,
                  label: str = "web search (text)") -> list[Record]:
        """Ordinary web search, as a person would run it: credible and supporting pages among the results,
        each read and verified by its own preview image."""
        web = getattr(self.idf, "web", None)
        search = search or (getattr(web, "pages", None) if web is not None else None)
        if search is None:
            return []
        self.consulted.append(label)
        out, seen = [], set(have)
        for q in queries[:4]:
            try:
                hits = search(q, 20)
            except Exception as e:
                log.info("text web search %r failed: %s", q, e)
                continue
            for url, _ in hits:
                host = institution(url)
                support = supporting(url)
                if host in seen or not (support or rulable(url, getattr(self, "_artist", ""), getattr(self, "_living", False))):
                    continue
                rec = self._curated_page(url, "image")
                if not rec or not (rec.image_url or rec.images):
                    continue
                if self._verify_page(img, img_vec, rec):
                    rec.support = support
                    out.append(rec)
                    seen.add(host)
            if sum(not x.support for x in out) >= 2:
                break
        return out

    def fallback(self, img: Image.Image, ident: dict, recs: list[Record], leads_: list[str]) -> list[Record]:
        """For a work the ruling could not settle, and only then: Google's results through SerpAPI (a free
        plan of 250 searches a month), at most 3 searches, read and image-verified like any other page."""
        web = getattr(self.idf, "web", None)
        if web is None or not hasattr(web, "serpapi_pages"):
            return []
        artist = ident.get("artist", "")
        titles = [t for t in leads_ if t][:2]
        queries = list(dict.fromkeys([f"{artist} {t}" for t in titles] + [f"{artist} painting"]))[:3]
        img_vec = self.emb.embed([img])[0]
        have = {institution(r.source) for r in recs}
        self._artist = artist
        return self._safe("serpapi", lambda: self._web_text(img, img_vec, queries, have, web.serpapi_pages,
                                                            "Google via SerpAPI (fallback for an unresolved work)"))

    def _verify_page(self, img, img_vec, rec: Record) -> bool:
        """A page shows the work when any of its pictures matches on both instruments (its preview image is
        often a site banner)."""
        for u in list(dict.fromkeys([rec.image_url, *rec.images]))[:6]:
            if not u:
                continue
            ok, sim, inl = self._verify(img, img_vec, u)
            if ok:
                rec.image_url, rec.clip, rec.inliers = u, sim, inl
                return True
        return False

    def _registry(self, img, img_vec, artist: str, titles: list[str], have: set[str]) -> list[Record]:
        """The artist's catalogue raisonné and holding institutions (CATALOGUE_SITES), by site: search."""
        web = getattr(self.idf, "web", None)
        sites = CATALOGUE_SITES.get(_surname(artist), [])
        if web is None or not hasattr(web, "pages") or not sites:
            return []
        out = []
        for site in sites:
            if institution(f"https://{site}") in have:
                continue
            self.consulted.append(site)
            done = False
            # a title one catalogue gives is searched at the next (the Post files Rockwell's cover as "Wet Paint")
            named = [re.sub(r"\s*\([^)]*\)|,?\s*\d{4}.*$", "", r.title).strip() for r in out if r.title]
            for q in list(dict.fromkeys([f"site:{site} {t}" for t in named + titles[:2] if t] + [f"site:{site} {artist}"])):
                try:
                    hits = web.pages(q, 8)
                except Exception as e:
                    log.info("catalogue site search %r failed: %s", q, e)
                    continue
                urls = []
                for u, _ in hits:
                    if site.split("/")[0] not in urlparse(u).netloc:
                        continue
                    mid = re.search(r"[?&]mid=(m\d+)", u) if "whistlerpaintings" in u else None
                    urls.append(f"https://www.whistlerpaintings.gla.ac.uk/catalogue/display/?mid={mid.group(1)}" if mid else u)
                for u in list(dict.fromkeys(urls))[:4]:
                    rec = self._curated_page(u, "image")
                    if rec and self._verify_page(img, img_vec, rec):
                        out.append(rec)
                        done = True
                        break
                if done:
                    break
        return out

    def _nga(self, img, img_vec, titles: list[str], artist: str) -> list[Record]:
        """National Gallery of Art, from its open data (nga.gov itself is walled): the artist's NGA works whose
        titles match a lead first, then all of them ranked by embedding; images from its IIIF server."""
        from .nga import NGA
        path = (self.cfg["sources"].get("nga_db") if self.cfg else "") or ""
        nga = NGA(Path(path).expanduser()) if path else None
        if not artist or nga is None or not nga.available:
            return []
        self.consulted.append("National Gallery of Art (open data)")
        rows = [r for r in nga.works_by(artist) if r.get("iiif")]
        if not rows:
            return []
        titled = [r for r in rows if any(same_title(t, r["title"]) for t in titles)]
        rest = [r for r in rows if r not in titled]
        if rest and len(rest) <= 800:          # the artist's other NGA works by embedding (thumbnails cached)
            vecs = self._nga_vectors(artist, rest)
            if vecs is not None:
                rest = [rest[int(i)] for i in np.argsort(-(vecs @ img_vec))]
        for r in (titled[:6] + rest[:6]):
            ok, sim, inl = self._verify(img, img_vec, NGA.image(r))
            if ok:
                return [Record("nga", f"https://www.nga.gov/collection/art-object-page.{r['objectid']}.html", r["title"], [],
                               r["artist"], ", ".join(x for x in (r["nationality"], r["artist_dates"]) if x),
                               r["displaydate"] or "", r["medium"] or "", "National Gallery of Art", "", NGA.image(r),
                               "image", sim, inl,
                               text=fold(" ".join(str(r[k] or "") for k in ("title", "attribution", "inscription", "medium",
                                                                           "displaydate", "accessionnum"))))]
        return []

    def _nga_vectors(self, artist: str, rows: list[dict]):
        """Embeddings of an artist's NGA thumbnails, cached by object id."""
        state = Path(self.cfg.path("state_dir")) / "nga" if self.cfg else None
        if state is None:
            return None
        state.mkdir(parents=True, exist_ok=True)
        cache = state / f"{re.sub(r'[^a-z0-9]+', '-', fold(artist))}.npz"
        have: dict[int, np.ndarray] = {}
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            have = dict(zip(z["ids"].tolist(), z["vecs"]))
        from .nga import NGA
        todo = [r for r in rows if r["objectid"] not in have]
        if todo:
            log.info("nga: embedding %d thumbnails for %s", len(todo), artist)
            ok = [(r, im) for r in todo for im in [fetch_image(NGA.image(r, 250), UA)] if im is not None]
            for i in range(0, len(ok), 64):
                chunk = ok[i:i + 64]
                for (r, _), v in zip(chunk, self.emb.embed([im for _, im in chunk])):
                    have[r["objectid"]] = v
            if have:
                np.savez(cache, ids=np.array(list(have)), vecs=np.stack(list(have.values())))
        if any(r["objectid"] not in have for r in rows):
            return None if not have else np.stack([have.get(r["objectid"], np.zeros(len(next(iter(have.values())))))
                                                    for r in rows])
        return np.stack([have[r["objectid"]] for r in rows])

    @staticmethod
    def _over(recs: list) -> list:
        """Mark records of the OVERRULE tier: they vote among themselves; lower-tier disagreement is discarded."""
        for r in recs:
            r.overrule = True
        return recs

    def _safe(self, name: str, fn) -> list:
        """A source that fails is recorded and skipped; it never takes the run down with it."""
        try:
            return fn() or []
        except Exception as e:
            log.warning("catalogue source %s failed: %s", name, e)
            self.consulted.append(f"{name} (failed: {type(e).__name__})")
            return []

    # ── gather + rule ─────────────────────────────────────────────────────
    def gather(self, img: Image.Image, ident: dict, leads: list[str]) -> list[Record]:
        img_vec = self.emb.embed([img])[0]
        artist = ident.get("artist", "")
        surname = _surname(artist)
        titles = [t for t in leads if t]
        titles = list(dict.fromkeys(titles[:5] + self._safe("translate", lambda: self._translations(titles)) + titles[5:]))
        queries = list(dict.fromkeys([f"{artist} {t}" for t in titles[:6]] + [artist]))
        recs: list[Record] = []
        self.consulted = ["Wikidata", "Art Institute of Chicago", "Metropolitan Museum", "Cleveland Museum of Art"]
        living, artist_qid = self._artist_living(artist)
        over = self._over
        for qid in self._wikidata_items(img, img_vec, artist, ident):
            rec, ids = self._wikidata_record(qid)
            recs.append(rec)
            artist_qid = ids.get("artist_qid") or artist_qid
            for v in ids.get("P350", [])[:1]:
                r = self._rkd_work(v, f"link from Wikidata {qid}")
                if r:
                    recs.append(r)
            if ids.get("enwiki"):
                r = self._wikipedia(ids["enwiki"], f"link from Wikidata {qid}")
                if r:
                    recs.append(r)
            if ids.get("image_file"):
                r = self._commons(ids["image_file"], f"Wikidata {qid} image")
                if r:
                    recs.append(r)
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
        over(recs)                      # the Wikidata item and every record it links to
        have = {r.source for r in recs}
        if "artic" not in have:
            recs += over(self._safe("artic_search", lambda: self._artic_search(img, img_vec, queries, surname, artist, titles)))
        if "met" not in have:
            recs += over(self._safe("met_search", lambda: self._met_search(img, img_vec, queries, surname, artist, titles)))
        recs += over(self._safe("cleveland_search", lambda: self._cleveland_search(img, img_vec, queries, surname)))
        recs += over(self._safe("europeana", lambda: self._europeana(img, img_vec, queries, surname)))
        recs += over(self._safe("harvard", lambda: self._harvard(img, img_vec, queries, surname, titles)))
        if "si" not in {institution(r.source) for r in recs}:
            recs += over(self._safe("smithsonian", lambda: self._smithsonian(img, img_vec, queries, surname)))
        if "yale" not in {institution(r.source) for r in recs}:
            recs += over(self._safe("yale", lambda: self._yale(img, img_vec, queries, surname)))
        if "rijksmuseum" not in {institution(r.source) for r in recs}:
            recs += over(self._safe("rijksmuseum", lambda: self._rijksmuseum(img, img_vec, titles, artist, surname)))
        if "wikipedia.org" not in {r.source for r in recs}:
            for q in queries[:2]:
                d = self._get("https://en.wikipedia.org/w/api.php", action="query", list="search", srsearch=q, srlimit=3, format="json")
                hit = next((r for t in (d.get("query") or {}).get("search", [])
                            for r in [self._wikipedia(t["title"], "image", img, img_vec)] if r), None)
                if hit:
                    recs.append(hit)
                    break
        self.consulted += ["RKD", "Wikipedia", "Wikimedia Commons", "museum, foundation and artist pages found by web search"]
        self._living, self._artist = living, artist
        recs += over(self._safe("nga", lambda: self._nga(img, img_vec, titles, artist)))
        recs += over(self._safe("catalogue_sites", lambda: self._registry(img, img_vec, artist, titles,
                                                                  {institution(r.source) for r in recs})))
        recs += self._safe("artvee", lambda: self._artvee(img, img_vec, titles, artist, surname))
        recs += self._safe("curated_web", lambda: self._curated_web(img, img_vec, queries, {r.source for r in recs}))
        recs += self._safe("web_text", lambda: self._web_text(img, img_vec, queries, {institution(r.source) for r in recs}))
        # the artist's authority record (RKDartists via Wikidata P650): rules life and nationality only
        if artist_qid:
            ent = self.wiki.entities([artist_qid]).get(artist_qid, {})
            from .references import Wikimedia
            for c in (ent.get("claims") or {}).get("P650", [])[:1]:
                r = self._rkd_artist(c["mainsnak"]["datavalue"]["value"])
                if r:
                    recs.append(r)
        # museum and institute pages on which identification itself verified the picture
        for rnd in ident.get("rounds", []):
            for m in rnd.get("confirmed_by", []):
                if rulable(m.get("page_url", ""), artist, living) and "wikidata.org" not in m.get("page_url", ""):
                    r = self._curated_page(m["page_url"], "image", m.get("thumb_url", ""))
                    if r:
                        r.clip, r.inliers = m.get("clip", 0.0), m.get("inliers", 0)
                        recs.append(r)
        by_source: dict[str, Record] = {}
        for r in recs:
            key = institution(r.source) + (":authority" if r.authority else "")
            if key in by_source:                       # one vote per institution; keep the richer record
                if not by_source[key].title and r.title:
                    by_source[key] = r
                continue
            by_source[key] = r
        return list(by_source.values())

    def rule(self, recs: list[Record], ident: dict) -> dict:
        """Facts stated by 2+ independent curated sources; the rest is listed as unruled."""
        authorities = [r for r in recs if r.authority]
        corroborating = [r for r in recs if r.support]
        recs = [r for r in recs if not r.authority and not r.support]
        # Sir, 2026-11-05 (revised): signatures and inscriptions are supporting data and search leads, never
        # a vote. A legible inscribed title is listed as corroboration beside the ruling.
        insc = (ident.get("inscribed_title") or "").strip()
        if inscription_is_title(insc, ident.get("artist", "")):
            corroborating.append(Record("inscription", "", insc, verified_by="written on the work", support=True))
        def rank(r):     # what the card prints: the holding museum's wording, then Wikidata, references last
            if r.source == "inscription":
                return 3     # the inscription is shown after a catalogue's title, in quotes, when they differ
            if r.source in ("wikipedia.org", "commons.wikimedia.org", "artvee.com") or "meisterdrucke" in r.source \
                    or any(d in r.source for d in ("wikiart", "artchive", "wga.hu", "britannica", "oxfordartonline", "jstor")):
                return 2
            return 1 if r.source in ("wikidata", "rkd.nl") else 0
        pref = sorted(recs, key=lambda r: (rank(r), r.verified_by != "image"))
        out = {"facts": {}, "support": {}, "unruled": {}, "sources": [{k: v for k, v in asdict(r).items() if k not in ("text", "images")} for r in recs],
               "consulted": self.consulted, "unavailable": UNAVAILABLE, "failures": [],
               "corroborating": [{"source": r.source, "url": r.url, "title": r.title} for r in corroborating],
               "discarded": {}}
        over = {r.source for r in recs if r.overrule}

        def choose(name, values, support_of):
            """Sir, 2026-11-05: the overrule tier (Wikidata and its links, museum APIs, catalogues raisonnés)
            votes among itself. When it states a fact, only its values can be ruled; lower-tier sources may
            add support by agreeing, and their disagreeing values are discarded. Most overrule support wins."""
            top = [(r, v) for r, v in values if r.overrule]
            best = None
            for r, v in top or values:
                sup = support_of(r, v)
                key = (len(over & set(sup)), len(sup))
                if len(sup) >= 2 and (best is None or key > best[0]):
                    best = (key, v, sup)
            if not best:
                return None
            if top:
                gone = [f"{r.source}: {v}" for r, v in values if not r.overrule and r.source not in best[2]]
                if gone:
                    out["discarded"][name] = gone
            return best[1], best[2]

        def pick(name, values, agree):
            got = choose(name, values, lambda r, v: sorted({r2.source for r2, v2 in values if agree(v, v2)} |
                                                           {r2.source for r2 in recs if r2.text and r2.states(v)}))
            if got:
                out["facts"][name], out["support"][name] = got
                return got[0]
            out["unruled"][name] = [f"{r.source}: {v}" for r, v in values] or ["no curated source states it"]
            return ""

        # title: any of a source's titles (languages, aliases) may agree; the display is the preferred source's
        tvals = [(r, r.title) for r in pref if r.title]
        title = ""
        # a near-same title in another medium is another work (Whistler's Riva pastel, the Salute oil)
        got = choose("title", tvals, lambda r, t: sorted({r2.source for r2 in recs if not medium_clash(r.medium, r2.medium)
                                                          and (any(same_title(t, t2) for t2 in r2.all_titles()) or r2.states(t))}))
        if got:
            t, sup = got
            wd = next((x for x in recs if x.source == "wikidata"), None)
            title = wd.title if (wd and "wikidata" in sup and wd.title) else t   # English, when Wikidata agrees
            out["facts"]["title"], out["support"]["title"] = title, sup
        if not title:
            out["unruled"]["title"] = [f"{r.source}: {t}" for r, t in tvals] or ["no curated source names the work"]
        surname = _surname(ident.get("artist", ""))
        avals = [(r, r.artist) for r in pref if r.artist and _surname(r.artist) == surname]
        artist = pick("artist", avals, lambda a, b: _surname(a) == _surname(b))
        if not artist:                # pages name the artist in their text, not in a field
            name = avals[0][1] if avals else ident.get("artist", "")
            sup = sorted(({avals[0][0].source} if avals else set()) | {r.source for r in recs if r.text and r.states(name)})
            if name and len(sup) >= 2:
                artist = name
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
        bios = [r for r in recs + authorities if r.artist_bio and r.source != "wikidata"]
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
                                   "each fact needs 2 that state it. Unavailable: " + "; ".join(f"{k} ({v})" for k, v in UNAVAILABLE.items()))
        return out


def labelled(text: str) -> dict:
    """"Label: value" fields as catalogue pages print them (Date: 1880 Collection: Yale ... Medium: ...)."""
    alt = "|".join(re.escape(l) for l in LABELS)
    return {m.group(1): m.group(2).strip(" ;,.")[:200]
            for m in re.finditer(rf"\b({alt}):\s*(.+?)(?=\s+(?:{alt}):|$)", text[:20000])}


def inscription_is_title(text: str, artist: str) -> bool:
    """A legible inscription that names the work: not a signature, initials, a date or a number."""
    t = fold(text)
    if not t or re.fullmatch(r"(?:[a-z]\s*){1,3}", t.replace(" ", "")):
        return False                                    # initials ("PK", "J. M. W.")
    names = set(fold(artist).split())
    rest = [w for w in t.split() if w not in names and not w.isdigit()]
    return len("".join(rest)) >= 3                      # something besides the artist's name and numbers


def leads(ident: dict) -> list[str]:
    """Search text from identification: titles, inscriptions, the matched copies' captions and file names.
    Non-titles (board names, shop wording) are kept as search text with the furniture removed."""
    out = [ident.get("inscribed_title", ""), ident.get("title", "")]
    for line in ident.get("evidence", {}).get("web_references", []) or []:
        m = re.match(r"file: (.*?) \| page: (.*?) \| site: (\S*)", line)
        if m:
            page, site = m.group(2), fold(m.group(3).split(".")[-2] if "." in m.group(3) else m.group(3))
            tail = re.search(r"\s[-–—]\s([^-–—]{2,40})$", page)
            if tail and site and site in fold(tail.group(1)).replace(" ", ""):
                page = page[:tail.start()]                       # "Giftbeeren - Artvee" -> "Giftbeeren"
            out += [page, m.group(1)]
    clean = []
    for t in out:
        t = re.sub(r"\b(\d+\+?\s+|ideas?|pin on|for sale|prints?|poster|paintings?|art print|wall art|fine art america|pinterest)\b", " ", t, flags=re.I)
        t = re.split(r"\s[|–—]\s", t)[0]                         # "Title | Site" -> "Title"
        t = re.sub(r"\s*\((?:c\.\s*)?\d{4}[^)]*\)|\s+by\s+[^|]*$", " ", t)   # "Title (1920) by Artist" -> "Title"
        t = re.sub(r"\b(?=\w*\d)(?=\w*[a-z])\w{3,}\b", " ", t)   # hash fragments mix letters and digits
        t = re.sub(r"\(\s*\)", " ", t)
        t = re.sub(r"\s+", " ", t).strip(" -|,.")
        if len(t) >= 4 and not re.fullmatch(r"[\d\s]+", t):
            clean.append(t)
    return list(dict.fromkeys(clean))
