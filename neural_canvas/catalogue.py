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

import functools
import html
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from itertools import chain
from urllib.parse import quote, urljoin, urlparse

import numpy as np
import requests
from PIL import Image

from .artsites import WalledCatalogue
from .matching import fetch_image, footprint, geometric_inliers, save_npz
from .references import file_thumb
from .sources import classify, institutional, non_title, rulable, supporting

log = logging.getLogger(__name__)
UA = "neural-canvas/1.0 (+https://github.com/CannonCoPilot/neural-canvas)"
LANGS = "en|mul|de|fr|nl|it|es|ja|da|sv|no|nb|fi|cs|ru|pl|pt"
UNAVAILABLE = {
    "Getty ULAN": "server-side TLS failure on vocab.getty.edu (2026-11-05)",
    "Norman Rockwell Museum": "collection.nrm.org answers every automated request with a bot challenge (2026-11-05)",
}
# Smithsonian art units. The API ignores its fq filter; a fielded unit_code term inside q filters (2026-11-05).
# With it, q is a strict AND of every word: "unit_code:(..) AND James McNeill Whistler The Riva Sunset Red and Gold"
# matches nothing, so the title goes in as an OR of its words beside the surname (2026-11-06).
SI_UNITS = "SAAM OR NPG OR HMSG OR NMAA OR CHNDM OR NMAfA"   # NMAA = Freer|Sackler (FSG matches nothing)
LUCENE_WORDS = {"and", "or", "not", "to"}
MET_API = "https://collectionapi.metmuseum.org/public/collection"
MET_IDS_MAX = 5000      # v1.1 pages at 500 at most; an artist or title past this is read to here and logged
ARTIC_FIELDS = "id,title,alt_titles,artist_title,artist_display,date_display,medium_display,style_title,image_id,thumbnail"
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
KID = "https://samlinger.slks.dk"
_KID_NATION = {"dansk": "Danish", "norsk": "Norwegian", "svensk": "Swedish", "tysk": "German", "fransk": "French",
               "hollandsk": "Dutch", "nederlandsk": "Dutch", "engelsk": "British", "britisk": "British", "italiensk": "Italian",
               "amerikansk": "American", "russisk": "Russian", "finsk": "Finnish", "islandsk": "Icelandic", "belgisk": "Belgian",
               "østrigsk": "Austrian", "schweizisk": "Swiss", "spansk": "Spanish", "polsk": "Polish"}
# KID records techniques and materials in Danish; the card prints English ("Olie på lærred" -> "Oil on canvas")
_KID_WORDS = {"olie": "oil", "lærred": "canvas", "træ": "wood", "egetræ": "oak", "papir": "paper", "karton": "cardboard",
              "pap": "board", "kobber": "copper", "akvarel": "watercolour", "gouache": "gouache", "tempera": "tempera",
              "pastel": "pastel", "kul": "charcoal", "blyant": "pencil", "tusch": "ink", "blæk": "ink", "radering": "etching",
              "kobberstik": "engraving", "træsnit": "woodcut", "litografi": "lithograph", "tørnål": "drypoint",
              "fresko": "fresco", "plade": "panel", "på": "on", "og": "and", "med": "with", "lærred,": "canvas,"}


# a study, sketch, copy or replica of the work is another object, though its picture matches: Bloch's Gethsemane at
# the Glyptotek is "Forarbejde til alterbilledet i St. Hans Kirke", 1879; the altarpiece is 1880 (2026-11-10)
KID_VERSION = re.compile(r"\b(forarbejde|skitse|studie|kopi|efter|replik|replika|udkast|variant|sketch|study|copy after|replica)\b",
                         re.I)


# when a movement began (its first works or manifesto): an artist's movement never labels their earlier work
MOVEMENT_START = {"impressionism": 1867, "post impressionism": 1886, "neo impressionism": 1886, "pointillism": 1886,
                  "les nabis": 1888, "nabis": 1888, "symbolism": 1880, "art nouveau": 1890, "vienna secession": 1897,
                  "fauvism": 1905, "expressionism": 1905, "cubism": 1907, "futurism": 1909, "de stijl": 1917,
                  "dada": 1916, "bauhaus": 1919, "art deco": 1920, "surrealism": 1924, "belgian surrealism": 1926,
                  "precisionism": 1915, "regionalism": 1928, "american regionalism": 1928, "abstract expressionism": 1943,
                  "pop art": 1955, "minimalism": 1960, "photorealism": 1965, "hyperrealism": 1970, "magic realism": 1925}


def clean_movement(v: str) -> str:
    """One movement, as a card prints it: WikiArt lists several ("Art Nouveau (Modern) , Symbolism") and adds its own
    parenthetical; the first, without it (Mucha, 2026-11-10)."""
    first = re.split(r"\s*[,;/]\s*", (v or "").strip())[0]
    first = re.sub(r"\s*\([^)]*\)", "", first).strip(" .")
    # a movement is a proper noun: "American realism" and "Hudson River school" print as American Realism and
    # Hudson River School (Wikidata's labels are sentence case, 2026-11-12)
    small = {"of", "and", "the", "de", "des", "la", "le", "les", "du", "in", "der", "die"}
    words = first.split()
    return " ".join("-".join(p[:1].upper() + p[1:] for p in w.split("-")) if i == 0 or w.lower() not in small else w
                    for i, w in enumerate(words))


def clean_collection(v: str) -> str:
    """A holder's name as a card prints it: no acronym in brackets and no postal address after it. "Los Angeles
    County Museum of Art (LACMA), Los Angeles, CA, US" prints "Los Angeles County Museum of Art" (2026-11-12);
    a comma inside the name stays ("Cooper Hewitt, Smithsonian Design Museum")."""
    v = re.sub(r"\s*\([A-Z][A-Za-z&.\s]{1,12}\)", "", (v or "").strip())
    parts = [p.strip() for p in v.split(",")]
    # a trailing address: two or more parts after the name, the last a country or state code ("CA", "US", "USA")
    if len(parts) >= 3 and re.fullmatch(r"[A-Z]{2,3}", parts[-1]):
        v = parts[0]
    return v.strip(" ,")


def kid_medium(s: str) -> str:
    """Kunstindeks Danmark's Danish technique in English, word by word; English entries pass through."""
    words = s.strip().split()
    out = " ".join(_KID_WORDS.get(w.lower(), w) for w in words)
    return out[:1].upper() + out[1:] if out else ""


LABELS = ("Artist", "Date", "Collection", "Accession Number", "Medium", "Support", "Size", "Signature", "Inscription",
          "Dimensions", "Credit Line", "Catalogue Number", "Provenance",
          # WikiArt and art references print these (2026-11-10: "Date: 1905 - 1907 Style: Art Nouveau Genre: ...
          # Media: oil, canvas Location: ..." ran on as one date, and its medium and location were never read)
          "Style", "Genre", "Media", "Location", "Current location", "Movement", "Period", "Technique", "Materials")
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
           "hungarian": "hungary", "mexican": "mexico", "australian": "australia", "chinese": "china",
           "portuguese": "portugal", "greek": "greece", "ukrainian": "ukraine", "brazilian": "brazil",
           "argentine": "argentina", "korean": "korea", "indian": "india", "israeli": "israel", "turkish": "turkey",
           "romanian": "romania", "croatian": "croatia", "estonian": "estonia", "latvian": "latvia",
           "lithuanian": "lithuania", "slovak": "slovakia", "slovenian": "slovenia", "icelandic": "iceland",
           "cuban": "cuba", "chilean": "chile", "colombian": "colombia", "peruvian": "peru", "egyptian": "egypt",
           "flemish": "flanders", "netherlandish": "southern netherlands"}
# Historical states and regions, read before modern countries and matched whole (longest first): a 15th-century
# painter of Bruges is Flemish, never "Dutch" (Wikidata gives van Eyck "Southern Netherlands": Sir, 2026-11-05)
HISTORICAL_DEMONYM = {"southern netherlands": "Flemish", "county of flanders": "Flemish", "flanders": "Flemish",
                      "spanish netherlands": "Flemish", "austrian netherlands": "Flemish",
                      "burgundian netherlands": "Netherlandish", "habsburg netherlands": "Netherlandish",
                      "dutch republic": "Dutch", "republic of the seven united netherlands": "Dutch",
                      "russian empire": "Russian", "tsardom of russia": "Russian", "soviet union": "Russian",
                      "kingdom of prussia": "German", "kingdom of bavaria": "German", "kingdom of saxony": "German",
                      "german empire": "German", "austrian empire": "Austrian", "austria-hungary": "Austrian",
                      "kingdom of france": "French", "kingdom of great britain": "British",
                      "kingdom of england": "English", "kingdom of scotland": "Scottish",
                      "republic of venice": "Venetian", "kingdom of the netherlands": "Dutch"}
# terms that name one nationality: a source saying either supports the other
DEMONYM_ALIASES = [{"flemish", "netherlandish"}]
# the adjective a card prints for a country (Wikidata gives countries; labels print demonyms: Sir, 2026-11-05)
COUNTRY_DEMONYM: dict = {}
for _dem, _country in DEMONYM.items():
    COUNTRY_DEMONYM.setdefault(_country, _dem.capitalize())     # "united kingdom" -> British (listed first)


def _country_demonym(label: str) -> str:
    """The demonym for a Wikidata country or historical state: historical names first, then a modern country
    matched as whole words ("Southern Netherlands" is Flemish, never Dutch)."""
    f = fold(label)
    for name in sorted(HISTORICAL_DEMONYM, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", f):
            return HISTORICAL_DEMONYM[name]
    for country, dem in COUNTRY_DEMONYM.items():
        if country not in ("flanders", "southern netherlands") and re.search(rf"\b{re.escape(country)}\b", f):
            return dem
    return ""


def _alias(demonym: str) -> set[str]:
    d = demonym.lower()
    return next((g for g in DEMONYM_ALIASES if d in g), {d})
NATIONALITY_PROMPT = """Artist: {artist}
What these sources say of the artist's nationality:
{statements}

What nationality would a museum wall label print for this artist?
- Answer with English adjectives (demonyms), never a country name: "American", not "United States".
- When the sources agree on one nationality, give that one.
- When the artist held two nationalities, or was born in one country and became a citizen of another, and people
  commonly describe them by both, join the two with a hyphen, the first or birth nationality first:
  "Swiss-French" (Felix Vallotton), "Dutch-American" (Willem de Kooning), "Russian-French" (Marc Chagall).
- Prefer the term museums use for the artist's own time and place, in its plain form: a 15th-century painter
  of Bruges is "Flemish" ("Netherlandish" means the same and is the clumsier word). Never project a modern
  state back onto an artist who lived before it existed: van Eyck is not "Dutch" or "Belgian".
- Use only nationalities the sources state (a historical region counts: "Southern Netherlands" is Flemish).
  Never add one.
Reply JSON: {{"label": "...", "reason": "one sentence"}}"""
MATERIALS = ("oil", "acrylic", "tempera", "gouache", "watercolor", "watercolour", "pastel", "charcoal", "ink",
             "graphite", "pencil", "woodblock", "woodcut", "lithograph", "etching", "engraving", "canvas", "panel",
             "paper", "board", "cardboard", "wood", "linen", "cloth", "silk", "copper", "fresco", "print",
             # print media the guard dropped as "names no material": Moran's "chromolithograph", Earle's "serigraph"
             # (2026-11-12)
             "chromolithograph", "serigraph", "screenprint", "aquatint", "mezzotint", "linocut", "drypoint",
             "monotype", "masonite", "vellum", "parchment", "enamel", "egg tempera")


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


def same_person(a: str, b: str) -> bool:
    """Two forms of one painter's name: the same surname and a shared given name or initial ("Carl Bloch" and
    "Carl Heinrich Bloch", "N. C. Wyeth" and "Newell Convers Wyeth", "Edgar Degas" and "Hilaire Germain Edgar
    Degas"); "Andrew Wyeth" is not "Newell Convers Wyeth"."""
    x, y = fold(a).split(), fold(b).split()
    if not x or not y or x[-1] != y[-1]:
        return False
    gx, gy = x[:-1], y[:-1]
    if not gx or not gy:
        return True
    return bool(set(gx) & set(gy)) or bool({w[0] for w in gx if len(w) == 1} & {w[0] for w in gy}) \
        or bool({w[0] for w in gy if len(w) == 1} & {w[0] for w in gx})


def _surname(name: str) -> str:
    w = fold(name).split()
    return w[-1] if w else ""


def _one(v) -> dict:
    """A Linked Art node that may come as an object or as a list of them (the Rijksmuseum gives a timespan
    both ways: Bouguereau's records crashed the adapter, 2026-11-06)."""
    if isinstance(v, list):
        v = next((x for x in v if isinstance(x, dict)), {})
    return v if isinstance(v, dict) else {}


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


def display_title(t: str, artist: str = "") -> str:
    """A title as a card prints it: a Commons file name ("File:Alexej von Jawlensky Landschaft mit orangener Wolke
    1909.jpg", ruled as the title on the withheld bench, 2026-11-06) loses its prefix, extension, the artist's
    name and the year."""
    t = (t or "").strip()
    if not re.match(r"(?i)file:", t) and not re.search(r"\.(jpe?g|png|tiff?|gif|webp)$", t, re.I):
        return t
    t = re.sub(r"(?i)^file:", "", t)
    t = re.sub(r"\.(jpe?g|png|tiff?|gif|webp)$", "", t, flags=re.I).replace("_", " ")
    t = re.sub(r"\((?:[^()]*museum|[^()]*collection|[^()]*gallery)[^()]*\)", " ", t, flags=re.I)   # holder in brackets
    def plain(x: str) -> str:
        return unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
    names = {x for x in re.split(r"[\s\-.,]+", plain(artist)) if len(x) > 1}
    keep = [w for w in t.split() if not ((parts := [x for x in re.split(r"[\-.]+", plain(w).strip(",;:()")) if x])
                                          and all(x in names for x in parts))]
    t = re.sub(r"(?i)\s+by$", "", " ".join(keep).strip(" ,;:-–—"))
    t = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b", " ", t)
    t = re.sub(r"\s*[,;:\-–—]\s*(?=[,;:\-–—]|$)", " ", t)
    return re.sub(r"\s+", " ", t).strip(" ,;:-–—")


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


class SourceError(RuntimeError):
    """A source that could not answer, or whose candidate records could not be compared with the picture. An
    empty reply is an error, never a no-match (Sir, 2026-11-06): the run's consulted list and the benchmark show it."""


def source(name: str):
    """A museum adapter's no-match is believed only when every candidate record it found was compared with the
    picture. A record served without an image, or an image that would not load, leaves the answer unknown: the
    adapter raises SourceError naming them instead of returning [] (Sir, 2026-11-06; every museum step of the
    2026-11-05 benchmark returned 0 with no error, AIC's images were behind a rate challenge)."""
    def wrap(fn):
        @functools.wraps(fn)
        def run(self, *a, **k):
            outer, self._unchecked = getattr(self, "_unchecked", None), []
            try:
                out = fn(self, *a, **k)
                if not out and self._unchecked:
                    lost = list(dict.fromkeys(self._unchecked))
                    raise SourceError(f"{name}: no record matched the picture, and {len(lost)} candidate(s) could not be "
                                      f"compared with it: " + "; ".join(lost[:5])[:500])
                return out
            finally:
                self._unchecked = outer
        return run
    return wrap


def _copy_after(title: str, surname: str) -> bool:
    """A reproduction names its model: "Chromolithograph after Jan van Eyck" (Wellcome), "Am Attersee nach Gustav
    Klimt" (MAK). Its picture matches; its date, medium and holder are the copy's, never the work's (2026-11-06)."""
    return bool(surname) and bool(re.search(rf"\b(?:after|nach|naar|apres)\s+(?:\w+\s+){{0,3}}{re.escape(surname)}\b",
                                            fold(title)))


def _title_words(title: str, drop: str = "") -> list[str]:
    """A title's significant words, safe inside a Lucene query (no operators, no punctuation)."""
    skip = set(fold(drop).split()) | LUCENE_WORDS
    return [w for w in dict.fromkeys(fold(title).split()) if len(w) > 2 and w not in STOP and w not in skip][:8]


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
    artist_nationalities: list = field(default_factory=list)   # every citizenship the source records
    text: str = ""               # an institution page's own text (folded): it supports what it states
    authority: bool = False      # an artist authority record: rules the artist's life, never the work
    support: bool = False        # an auction house or the press: corroborates, never counts
    images: list = field(default_factory=list)       # the page's other pictures, tried when image_url is not the work
    overrule: bool = False       # Wikidata and what it links, a museum's own API, a catalogue raisonné (Sir, 2026-11-05)
    wikidata: str = ""           # the source's own link to the work's Wikidata item (a join key)
    echo: list = field(default_factory=list)          # fields this record only repeats from that item (Commons)
    article_movement: str = ""   # the movement on the artist's Wikipedia infobox, as printed ("Regionalist")

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
        self._unchecked: list[str] | None = None     # inside a @source adapter: candidates never compared
        self._named_items: set[str] = set()          # Wikidata items taken by creator + unique title

    # ── is this record's picture the input? ───────────────────────────────
    def _verify(self, img: Image.Image, img_vec, url: str, what: str = "") -> tuple[bool, float, int]:
        im = fetch_image(url, UA) if url else None
        if im is None:
            if getattr(self, "_unchecked", None) is not None:     # unknown, not a no-match (Sir, 2026-11-06)
                self._unchecked.append(f"{what or url} ({'image would not load' if url else 'no image served'})")
            return False, 0.0, 0
        sim = float(self.emb.similarity(img_vec, self.emb.embed([im]))[0])
        if sim < self.t["id_clip_min"] - 0.08:
            return False, round(sim, 4), 0
        inl = geometric_inliers(img, im)
        if sim < self.t["id_clip_min"] and inl >= self.t["id_inliers_min"]:
            # the input may be a detail of the record's picture (a crop shows less than the whole sheet):
            # the same bar, on the part of the record's picture the input shows (Redon, NGA 66514, 2026-11-05)
            part = footprint(im, img)
            if part is not None:
                sim = max(sim, float(self.emb.similarity(img_vec, self.emb.embed([part]))[0]))
        return sim >= self.t["id_clip_min"] and inl >= self.t["id_inliers_min"], round(sim, 4), inl

    def _get(self, url: str, **params) -> dict:
        try:
            r = self.s.get(url, params=params, timeout=20)
            if r.status_code != 200:            # a linked lookup, not a search: logged where it shows (2026-11-06)
                log.warning("catalogue %s -> HTTP %s", url[:80], r.status_code)
                return {}
            return r.json()
        except Exception as e:
            log.warning("catalogue %s failed: %s", url[:80], e)
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

    def _key(self, name: str, label: str) -> str:
        key = self.keys().get(name)
        if not key:      # a missing key is a wall to open, never a quiet skip (Sir, 2026-11-06)
            path = (self.cfg["sources"].get("api_keys_path") if self.cfg else "") or "(sources.api_keys_path unset)"
            raise SourceError(f"{label}: no '{name}' key in {path}")
        return key

    def _api(self, url: str, expect: str, method: str = "GET", body: dict | None = None, gone: tuple = (),
             **params) -> dict:
        """A museum API's JSON reply. Unreachable, refused (HTTP != 200 after two paced retries on 429/5xx), not
        JSON, or without the field every reply of that API carries: SourceError, never {} (Sir, 2026-11-06: an
        empty reply is an error, never a no-match). Keys travel in query strings: they never reach a message."""
        secrets = [v for v in self.keys().values() if len(v) > 6]
        def clean(s: str) -> str:
            for k in secrets:
                s = s.replace(k, "***")
            return s
        err = ""
        for attempt in range(3):
            try:
                r = self.s.request(method, url, params=params or None, json=body, timeout=30)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    try:
                        d = r.json()
                    except ValueError:
                        raise SourceError(clean(f"{url[:80]}: HTTP 200 but not JSON ({r.headers.get('content-type', '')})"))
                    if isinstance(d, dict) and expect in d:
                        return d
                    raise SourceError(clean(f"{url[:80]}: reply without '{expect}': {str(d)[:160]}"))
                if r.status_code in gone:          # the caller's "no such record" (a withdrawn object): {}
                    log.warning("catalogue %s -> HTTP %s: no such record", url[:80], r.status_code)
                    return {}
                err = f"HTTP {r.status_code} {r.text[:160]}"
                if r.status_code != 429 and r.status_code < 500:
                    break
                wait = r.headers.get("Retry-After", "")
                time.sleep(min(float(wait), 30) if wait.isdigit() else 2.0 * (attempt + 1))
                continue
            time.sleep(2.0 * (attempt + 1))
        raise SourceError(clean(f"{method} {url[:80]}: {err}"))

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
        box = self._infobox(page["title"], lang)
        return Record("wikipedia.org", f"https://{lang}.wikipedia.org/wiki/{quote(page['title'].replace(' ', '_'))}",
                      page["title"], [], box.get("artist", ""), "", box.get("year", ""), box.get("medium", ""),
                      box.get("location", ""), box.get("movement", ""), image_url=image, verified_by=verified_by,
                      clip=clip, inliers=inl, text=fold(page["extract"][:60000]))

    def _infobox(self, page_title: str, lang: str = "en") -> dict:
        """An article's artwork infobox as {label: value} (lower-case labels: year, medium, location, movement);
        the extract alone drops it, so medium and location were never read from Wikipedia (2026-11-10)."""
        d = self._get(f"https://{lang}.wikipedia.org/w/api.php", action="parse", page=page_title, prop="text",
                      section=0, redirects=1, format="json", formatversion=2)
        t = ((d.get("parse") or {}).get("text")) or ""
        m = re.search(r'<table class="infobox.*?</table>', t, re.S)
        if not m:
            return {}
        box = re.sub(r"<style[^>]*>.*?</style>|<sup[^>]*>.*?</sup>", " ", m.group(0), flags=re.S)
        out = {}
        for th, td in re.findall(r'<th[^>]*class="infobox-label"[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>', box, re.S):
            k = re.sub(r"<[^>]+>", "", th).strip().lower()
            v = re.sub(r"\s+", " ", html.unescape(re.sub(r"<br\s*/?>|</li>", ", ", re.sub(r"<(?!br|/li)[^>]+>", "", td)))).strip(" ,")
            if k and v:
                out.setdefault(k, v[:200])
        return out

    def _commons(self, file: str, verified_by: str) -> Record | None:
        d = self._get("https://commons.wikimedia.org/w/api.php", action="query", titles="File:" + file, prop="imageinfo",
                      iiprop="extmetadata|url", iiurlwidth=960, format="json")
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
        art = self._commons_artwork(file)
        local = self._commons_local(file)
        # a field the file page leaves blank is filled from the Wikidata item it names: an echo, not a witness
        echo = [k for k, params in (("date", ("date",)), ("medium", ("medium", "technique")),
                                    ("collection", ("institution", "object location", "collection")))
                if art.get("item") and not any(p in local for p in params)]
        return Record("commons.wikimedia.org", f"https://commons.wikimedia.org/wiki/File:{quote(file)}",
                      "" if non_title(title) else title, [], re.sub(r"\s+", " ", val("Artist"))[:120], "",
                      art.get("date") or val("DateTimeOriginal")[:40], art.get("art_medium", ""),
                      art.get("art_gallery", ""), "", info.get("thumburl", ""), verified_by, text=text,
                      wikidata=art.get("item", ""), echo=echo)

    def _commons_local(self, file: str) -> set[str]:
        """The Artwork template's parameters the file page sets itself (non-empty in its wikitext)."""
        d = self._get("https://commons.wikimedia.org/w/api.php", action="query", prop="revisions", rvprop="content",
                      rvslots="main", titles="File:" + file, format="json", formatversion=2)
        pages = (d.get("query") or {}).get("pages") or [{}]
        t = (((pages[0].get("revisions") or [{}])[0].get("slots") or {}).get("main") or {}).get("content", "")
        return {m.group(1).strip().lower() for m in re.finditer(r"\|\s*([A-Za-z ]+?)\s*=\s*(\S[^\n|]*)", t)}

    def _commons_artwork(self, file: str) -> dict:
        """The file page's Artwork template as {field: value} (date, art_medium, art_gallery, ...), plus ``item``:
        the Wikidata item it names. extmetadata carries no medium or location (2026-11-10)."""
        d = self._get("https://commons.wikimedia.org/w/api.php", action="parse", page="File:" + file, prop="text",
                      format="json", formatversion=2)
        t = (((d.get("parse") or {}).get("text")) or "").replace("&#95;", "_")
        t = re.sub(r'<div style="display: ?none;?">.*?</div>|<style[^>]*>.*?</style>', " ", t, flags=re.S)
        out = {}
        for k, v in re.findall(r'<td id="fileinfotpl_([a-z_]+)"[^>]*>.*?</td>\s*<td[^>]*>(.*?)</td>', t, re.S):
            v = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", v))).strip()
            if v:
                out.setdefault(k, v[:200])
        m = re.search(r"\bQ\d{2,}\b", out.get("art_authority", ""))
        out["item"] = m.group(0) if m else ""
        return out

    @source("Europeana")
    def _europeana(self, img, img_vec, queries: list[str], surname: str, titles: list[str] = ()) -> list[Record]:
        """Europeana's search ANDs every word, so a whole "artist title" phrase rarely matches: the creator field
        AND any of the title's words first, then the phrases (2026-11-06)."""
        key = self._key("europeana", "Europeana")
        self.consulted.append("Europeana")
        fielded = [f"who:{surname} AND ({' OR '.join(ws)})" for t in (titles or queries)[:4]
                   for ws in [_title_words(t, surname)] if ws and surname]
        seen: set[str] = set()
        for q in list(dict.fromkeys(fielded + list(queries[:4]))):
            d = self._api("https://api.europeana.eu/record/v2/search.json", "success", wskey=key, query=q, rows=12,
                          media="true", qf="TYPE:IMAGE")
            for x in d.get("items") or []:
                # creators come as names, agent URIs or both; a record naming only URIs is judged by its picture
                names = [c for c in [*(x.get("dcCreator") or []), *((x.get("dcCreatorLangAware") or {}).get("en") or [])]
                         if not c.startswith("http")]
                creator = " ".join(dict.fromkeys(names))
                title = (x.get("title") or [""])[0]
                desc = x.get("dcDescription") or ""                     # titles come cut short ("after Jan van Ey"):
                desc = desc[0] if isinstance(desc, list) and desc else str(desc)   # the description names the model
                if x.get("id") in seen or (names and surname not in fold(creator)) or _copy_after(f"{title} {desc[:300]}", surname):
                    continue
                seen.add(x.get("id"))
                # the provider's image, else Europeana's own thumbnail of it (the Belvedere answers 403 to us; its
                # thumbnail loads: 2026-11-06)
                shown, preview = (x.get("edmIsShownBy") or [""])[0], (x.get("edmPreview") or [""])[0]
                url = next((u for u in (shown, preview) if u and fetch_image(u, UA) is not None), "")
                if not url:
                    self._unchecked.append(f"Europeana {x.get('id', '')} (image would not load)")
                    continue
                ok, sim, inl = self._verify(img, img_vec, url, f"Europeana {x.get('id', '')}")
                if ok:
                    prov = (x.get("dataProvider") or ["Europeana"])[0]
                    others = [t for ts in (x.get("dcTitleLangAware") or {}).values() for t in ts if t != title]
                    return [Record(f"europeana:{fold(prov)}", x.get("guid", ""), title, list(dict.fromkeys(others)), creator,
                                   "", (x.get("year") or [""])[0], "", prov, "", url, "image", sim, inl)]
        return []

    @source("Harvard Art Museums")
    def _harvard(self, img, img_vec, queries: list[str], surname: str, titles: list[str] = ()) -> list[Record]:
        key = self._key("harvard", "Harvard Art Museums")
        self.consulted.append("Harvard Art Museums")
        url = "https://api.harvardartmuseums.org/object"
        # person + title together is a fielded AND (either alone does not filter); free text q is the fallback
        # the API parses every field as Lucene: a quote, colon or bracket in a title is a syntax error
        # ("token_mgr_error", withheld bench 2026-11-06), so those characters go
        def lq(t: str) -> str:
            return re.sub(r'[+\-!(){}\[\]^"~*?:\\/&|]', " ", t).strip()
        fielded = (self._api(url, "records", apikey=key, person=lq(surname), title=lq(t), size=10) for t in titles[:4] if surname)
        plain = (self._api(url, "records", apikey=key, q=lq(q), size=10, hasimage=1) for q in queries[:4])
        seen: set = set()
        for d in chain(fielded, plain):
            for x in d.get("records") or []:
                people = " ".join(p.get("name", "") for p in x.get("people") or [])
                if x.get("id") in seen or surname not in fold(people):
                    continue
                seen.add(x.get("id"))
                if not x.get("primaryimageurl"):        # a title match served without a picture cannot be checked
                    if any(same_title(t, x.get("title", "")) for t in titles):
                        self._unchecked.append(f"Harvard {x.get('id')} '{x.get('title', '')}' (no image served)")
                    continue
                ok, sim, inl = self._verify(img, img_vec, x["primaryimageurl"] + "?height=800", f"Harvard {x.get('id')}")
                if ok:
                    bio = next((p.get("displaydate", "") for p in x.get("people") or [] if surname in fold(p.get("name", ""))), "")
                    former = [t.get("title", "") for t in x.get("titles") or [] if t.get("title") and t.get("title") != x.get("title")]
                    return [Record("harvardartmuseums.org", x.get("url", ""), x.get("title", ""), former, people, bio,
                                   x.get("dated", ""), x.get("medium") or "", "Harvard Art Museums", x.get("style") or "",
                                   x["primaryimageurl"], "image", sim, inl)]
        return []

    @source("Smithsonian Open Access")
    def _smithsonian(self, img, img_vec, queries: list[str], surname: str, titles: list[str] = ()) -> list[Record]:
        """Smithsonian Open Access: the art museums (American Art, Portrait Gallery, Hirshhorn, Freer|Sackler,
        Cooper Hewitt, African Art). Images come from its own image server at 800 px. The unit filter makes q a
        strict AND, so: surname AND any word of each title, then the surname alone (SI_UNITS, 2026-11-06)."""
        key = self._key("smithsonian", "Smithsonian Open Access")
        self.consulted.append("Smithsonian Open Access")
        sur = fold(surname).replace(" ", "")
        qs = [f"{sur} AND ({' OR '.join(ws)})" for t in (titles or queries)[:4] for ws in [_title_words(t, surname)] if ws]
        seen: set = set()
        for q in list(dict.fromkeys(qs + [sur]))[:5] if sur else []:
            d = self._api("https://api.si.edu/openaccess/api/v1.0/search", "response", api_key=key,
                          q=f"unit_code:({SI_UNITS}) AND {q}", rows=10)
            for x in (d.get("response") or {}).get("rows") or []:
                if x.get("id") in seen:
                    continue
                seen.add(x.get("id"))
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
                ok, sim, inl = self._verify(img, img_vec, url, f"Smithsonian {x.get('id', '')}")
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
        prod = _one(x.get("produced_by"))
        makers = [a.get("_label", "") for part in [prod, *(prod.get("part") or [])] for a in part.get("carried_out_by") or []]
        maker = re.sub(r"^[A-Z][a-z ]+:\s*", "", makers[0]) if makers else ""   # "Artist: Name (American, 1834–1903)"
        artist, _, bio = maker.partition(" (")
        date = next((i.get("content", "") for i in _one(prod.get("timespan")).get("identified_by") or []), "")
        refs = x.get("referred_to_by") or []
        medium = next((r.get("content", "") for r in refs if "Medium" in cls(r)), "")
        owner = next((o.get("_label", "") for o in x.get("current_owner") or []), "")
        image = next((a.get("id", "") for rep in x.get("representation") or [] for d in rep.get("digitally_shown_by") or []
                      for a in d.get("access_point") or []), "")
        text = fold(" ".join([title, maker, *(str(r.get("content", "")) for r in refs)]))
        return Record("yale", x["id"], title, others, artist, bio.rstrip(")"), date, medium, owner, "", image, "image",
                      text=text)

    @source("Kunstindeks Danmark")
    def _kid(self, img, img_vec, artist: str, titles: list[str] = ()) -> list[Record]:
        """Kunstindeks Danmark (samlinger.slks.dk): the national register of the works in Danish state and
        state-subsidised museums (Sir, 2026-11-10: Bloch's paintings at Frederiksborg, the Glyptotek, St. Hans).
        Its JSON API serves the search the site itself runs; works are filtered by the creator as KID writes the
        name ("Bloch, Carl"), searched by title, and each candidate's reproduction is verified on both instruments."""
        self.consulted.append("Kunstindeks Danmark")
        name = self._kid_name(artist)
        if not name:
            return []
        seen: set[str] = set()
        cands: list[dict] = []
        for q in list(dict.fromkeys([t for t in titles[:6] if t] + [""])):
            for page in ((1,) if q else (1, 2, 3)):
                d = self._kid_get("es_artworks", creatorsNames=name, hasReproductions="true", q=q, page=page)
                for m in d.get("hydra:member") or []:
                    if m.get("id") not in seen and m.get("reproductions"):
                        seen.add(m["id"])
                        cands.append(m)
            if len(cands) >= 40:
                break
        # the titles' own matches first, then paintings, then the rest
        cands.sort(key=lambda m: (not any(same_title(t, m.get("title", "")) for t in titles if t),
                                  "Maleri" not in (m.get("objectNames") or []), m.get("objectTemplate") != "ART"))
        for m in cands[:14]:
            if KID_VERSION.search(f"{m.get('title', '')} {m.get('description', '')}"):
                continue
            url = KID + m["reproductions"][0]
            ok, sim, inl = self._verify(img, img_vec, url, f"Kunstindeks Danmark {m['id']} '{m.get('title', '')}'")
            if ok:
                full = self._kid_get(f"es_artworks/{m['id']}") or m
                if KID_VERSION.search(f"{full.get('title', '')} {full.get('description', '')}"):
                    continue          # the search hit carries no description; the record does
                rec = self._kid_rec(full)
                rec.image_url, rec.clip, rec.inliers = url, sim, inl
                return [rec]
        return []

    def _kid_get(self, path: str, **params) -> dict:
        try:
            r = self.s.get(f"{KID}/api/{path}", params={k: v for k, v in params.items() if v not in ("", None)}, timeout=30)
            if r.status_code != 200:
                log.warning("Kunstindeks Danmark %s -> HTTP %s", path[:60], r.status_code)
                return {}
            return r.json()
        except Exception as e:
            log.warning("Kunstindeks Danmark %s failed: %s", path[:60], e)
            return {}

    def _kid_name(self, artist: str) -> str:
        """The creator as KID files them, "Surname, Given names", tried from the fullest form down; '' if absent."""
        parts = artist.split()
        if len(parts) < 2:
            return ""
        forms = []
        for i in range(len(parts) - 1, 0, -1):         # "Rembrandt van Rijn": "Rijn, Rembrandt van", "van Rijn, Rembrandt"
            last, given = " ".join(parts[i:]), " ".join(parts[:i])
            forms += [f"{last}, {given}", f"{last}, {parts[0]}"]
        forms += [f"{parts[-1]}, {' '.join(parts[:-1])}", f"{parts[-1]}, {parts[0]}"]
        for f in dict.fromkeys(forms):
            if (self._kid_get("es_artworks", creatorsNames=f).get("hydra:totalItems") or 0) > 0:
                return f
        # filed some other way ("Rembrandt Harmensz. van Rijn"): the creator names a plain search returns
        from collections import Counter
        sur, first = fold(parts[-1]), fold(parts[0])
        names = Counter(n for m in (self._kid_get("es_artworks", q=artist).get("hydra:member") or [])
                        for n in m.get("creatorsNames") or [] if sur in fold(n) and first in fold(n))
        return names.most_common(1)[0][0] if names else ""

    @staticmethod
    def _kid_rec(m: dict) -> Record:
        start, end = m.get("productionStartString") or "", m.get("productionEndString") or ""
        date = start if not end or end == start else f"{start}–{end}"
        tech = [t for t in (m.get("techniquePosition") or []) if t] or [t for t in (m.get("materialPosition") or []) if t]
        nat = next((_KID_NATION.get(n.lower(), n) for n in (m.get("creatorsNationalities") or []) if n), "")
        born = (m.get("creatorsBirthDates") or [""])[0][:4]
        died = (m.get("creatorsDeathDates") or [""])[0][:4]
        artist = (m.get("creatorsNamesExact") or [""])[0]
        if not artist and m.get("creatorsNames"):
            last, _, given = m["creatorsNames"][0].partition(", ")
            artist = f"{given} {last}".strip()
        return Record("kunstindeks.dk", f"{KID}/samlinger/{m.get('id', '')}", m.get("title", ""), [], artist,
                      ", ".join(x for x in (nat, "–".join(y for y in (born, died) if y)) if x), date,
                      kid_medium(tech[0]) if tech else "", m.get("museum", "").strip(), "", "", "image",
                      artist_born=born, artist_died=died, artist_nationality=nat, overrule=True,
                      text=fold(" ".join([m.get("title", ""), m.get("description", "") or "", m.get("museum", ""), date])))

    @source("Yale University (LUX)")
    def _yale(self, img, img_vec, queries: list[str], surname: str, titles: list[str] = ()) -> list[Record]:
        """Yale (LUX): the Art Gallery, the Center for British Art and the libraries. Yale-provided metadata is
        CC0 (LUX Open Access page, read 2026-11-05); one lookup per work, never a harvest. Only object records
        (a /data/digital/ stub is an image file, not a work: 2026-11-06)."""
        self.consulted.append("Yale University (LUX)")
        seen: set[str] = set()
        for q in queries[:3]:
            d = self._api(f"{LUX}/api/search/item", "type", q=json.dumps({"text": q}))   # an empty page has no orderedItems
            for stub in (d.get("orderedItems") or [])[:6]:
                u = stub.get("id", "")
                if not u.startswith(f"{LUX}/data/object/") or u in seen:
                    continue
                seen.add(u)
                rec = self._lux_rec(self._api(u, "id"))
                if not rec or surname not in fold(rec.artist):
                    continue
                if not rec.image_url:
                    if any(same_title(t, x) for t in titles for x in rec.all_titles()):
                        self._unchecked.append(f"Yale {u.rsplit('/', 1)[-1]} '{rec.title}' (no image served)")
                    continue
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url, f"Yale {u.rsplit('/', 1)[-1]}")
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
        prod = _one(x.get("produced_by"))
        def name(maker):          # notation is a list of language forms, or a single one
            notes = maker.get("notation") or []
            notes = [notes] if isinstance(notes, dict) else notes
            return next((n.get("@value", "") for n in notes if n.get("@language") == "en"), notes[0].get("@value", "") if notes else "")
        parts = [prod, *(p for p in prod.get("part") or [] if isinstance(p, dict))]
        makers = [name(a) for part in parts for a in part.get("carried_out_by") or []]
        # most records name a maker only by an attribute assignment (a bare person id) and an English statement
        # on the part: "printmaker: Kobayashi Kiyochika (mentioned on object), Japan". Read the name from it,
        # publishers last (every Kiyochika and Zeshin print read as artistless and was skipped, 2026-11-06)
        said = [(re.sub(r"\s*\(.*$|,.*$", "", c.split(":", 1)[-1]).strip(), c.split(":", 1)[0].strip().lower() if ":" in c else "")
                for part in parts if part.get("assigned_by") for r in part.get("referred_to_by") or []
                if en(r) for c in [str(r.get("content", ""))] if c]
        makers += [n for n, role in sorted(said, key=lambda s: s[1] in ("publisher", "printer")) if n]
        makers = list(dict.fromkeys(m for m in makers if m))
        artist = next((m for m in makers if surname and surname in fold(m)), makers[0] if makers else "")
        stamps = _one(prod.get("timespan")).get("identified_by") or []
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

    @source("Rijksmuseum")
    def _rijksmuseum(self, img, img_vec, titles: list[str], artist: str, surname: str) -> list[Record]:
        """Rijksmuseum data services (keyless Linked Art; the old key API is retired): artist AND title,
        then the artist's works with images."""
        if not artist:
            return []
        self.consulted.append("Rijksmuseum")
        seen: set[str] = set()
        searches = [{"creator": artist, "title": t} for t in titles[:4]] + [{"creator": artist, "imageAvailable": "true"}]
        for params in searches:
            d = self._api("https://data.rijksmuseum.nl/search/collection", "type", **params)   # empty: no orderedItems
            for stub in (d.get("orderedItems") or [])[:6]:
                u = stub.get("id", "")
                if u in seen:
                    continue
                seen.add(u)
                rec = self._rijks_rec(self._api(u, "id"), surname)
                if not rec or surname not in fold(rec.artist):
                    continue
                if not rec.image_url:
                    if any(same_title(t, x) for t in titles for x in rec.all_titles()):
                        self._unchecked.append(f"Rijksmuseum {u.rsplit('/', 1)[-1]} '{rec.title}' (no image served)")
                    continue
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url, f"Rijksmuseum {u.rsplit('/', 1)[-1]}")
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
    @source("Wikidata")
    def _wikidata_items(self, img, img_vec, artist: str, ident: dict, titles: list[str] = ()) -> list[str]:
        """Catalogue items of THIS picture: the identification's own item; else the artist's works whose
        Wikidata title matches a lead (one listing query, seconds), then the cached catalogue's nearest by
        embedding. Each is verified on its picture. A run never sweeps a catalogue cold (Sir, 2026-11-05), so
        works not yet embedded were never compared: with no match that is an error, not a no-match (2026-11-06)."""
        if ident.get("wikidata"):
            return [ident["wikidata"]]
        qid = self.wiki.find_artist(artist) if artist else None
        if not qid:
            return []
        listing = self.wiki.oeuvre(qid)
        named = [w for w in listing if any(same_title(t, w["label"]) for t in titles if t)][:4]
        if len(named) < 4 and titles and self.cfg is not None:
            # the English label is one name of many: "Der Blaue Reiter" is labelled "The Blue Rider", "Escargots"
            # "The Snail" (2026-11-12: the item never joined, and Artvee's wrong 1914 printed). The cached listing
            # carries every work's labels and aliases in nine languages; a match is still verified on its picture
            from .indexer import works_listing
            have = {w["qid"] for w in named}
            try:
                every = works_listing(self.cfg.path("state_dir"), self.wiki, qid)
            except Exception as e:      # the English match stands; the listing is a widening, never a gate
                log.warning("works listing for %s unavailable: %s", qid, e)
                every = []
            named += [{"qid": w["qid"], "label": (w["labels"] or [""])[0], "file": w["file"]} for w in every
                      if w["file"] and w["qid"] not in have
                      and any(same_title(t, l) for t in titles if t for l in w["labels"])][:4 - len(named)]
        works, vecs = self.idf._oeuvre_vectors(qid)
        near = [works[int(i)] for i in np.argsort(-(vecs @ img_vec))[:4]] if works else []
        out = []
        for w in list({w["qid"]: w for w in named + near}.values()):
            ok, sim, inl = self._verify(img, img_vec, file_thumb(w["file"], 500), f"Wikidata {w['qid']}")
            if ok:
                out.append(w["qid"])
        if not out and titles:
            # a work with no Commons picture (in copyright) is never in the listing above: its item is taken when the
            # creator and the title pick out exactly one work and that item has no picture that could be compared
            from .indexer import works_listing
            state = self.cfg.path("state_dir") if self.cfg is not None else None
            listing_all = works_listing(state, self.wiki, qid) if state is not None else self.wiki.works_by(qid)
            byname = {w["qid"]: w for w in listing_all
                      if any(same_title(t, l) for t in titles[:4] if t for l in w["labels"])}
            if len(byname) == 1 and not next(iter(byname.values()))["file"]:
                q1 = next(iter(byname))
                self._named_items.add(q1)
                out.append(q1)
        cached = {w["qid"] for w in works} | {w["qid"] for w in named}
        if len({w["qid"] for w in listing} - cached):
            self._unchecked.append(f"{len({w['qid'] for w in listing} - cached)} of {artist}'s {len(listing)} Wikidata "
                                   f"works not embedded yet ({qid}; the background indexer embeds them)")
        return out

    def _wikidata_artist(self, qid: str, surname: str) -> Record | None:
        """The artist's own Wikidata item as an authority (life and citizenships only, never the work)."""
        f = self.wiki.artist_facts(qid)
        if not f.get("artist_born") or (surname and surname not in fold(f.get("artist", ""))):
            return None
        movs = f.get("movements") or []
        # the artist's movement on their Wikipedia article: a second, independent statement of it
        box = self._safe("wikipedia_artist", lambda: [self._infobox(f["enwiki"])] if f.get("enwiki") else [])
        wp_mov = ((box[0] if box else {}) or {}).get("movement", "")
        return Record("wikidata", f"https://www.wikidata.org/wiki/{qid}", "", [], f["artist"], "",
                      movement="|".join(movs), verified_by="the artist's Wikidata item",
                      authority=True, artist_born=f["artist_born"], artist_died=f.get("artist_died", ""),
                      artist_nationality=f.get("artist_nationality", ""),
                      artist_nationalities=f.get("artist_nationalities") or [],
                      text=fold(f"movement {wp_mov}") if wp_mov else "", article_movement=wp_mov,
                      images=[f"https://en.wikipedia.org/wiki/{quote(f['enwiki'].replace(' ', '_'))}"] if f.get("enwiki") else [])

    def _wikidata_record(self, qid: str) -> tuple[Record, dict]:
        d = self.wiki._get(self.wiki.wikidata_api, {"action": "wbgetentities", "ids": qid,
                                                    "props": "labels|aliases|claims|sitelinks", "languages": LANGS})
        ent = (d.get("entities") or {}).get(qid, {})
        labels = ent.get("labels") or {}
        en = (labels.get("en") or {}).get("value", "") or (labels.get("mul") or {}).get("value", "")
        others = [v["value"] for k, v in labels.items() if k not in ("en", "mul")]
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
                     artist_nationality=f.get("artist_nationality", ""),
                     artist_nationalities=f.get("artist_nationalities") or [])
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
        # 843 px is the Art Institute's standard derivative; widths off its allowlist are refused (HTTP 403), and so
        # is any width past the original's: a 526 px drawing is served at 400 or 200 only (6 works, 2026-11-06)
        full = int((x.get("thumbnail") or {}).get("width") or 843)
        w = next((w for w in (843, 400, 200) if w <= full), 200)
        img = f"https://www.artic.edu/iiif/2/{x['image_id']}/full/{w},/0/default.jpg" if x.get("image_id") else ""
        return Record("artic", f"https://www.artic.edu/artworks/{x['id']}", x.get("title", ""), x.get("alt_titles") or [],
                      x.get("artist_title") or "", bio, x.get("date_display") or "", x.get("medium_display") or "",
                      "Art Institute of Chicago", x.get("style_title") or "", img, verified_by)

    def _artic_post(self, artist: str, title: str) -> dict:
        """The Art Institute's fielded search: artist AND title must both match (plain q ranks the
        artist's other works first)."""
        return self._api("https://api.artic.edu/api/v1/artworks/search", "data", "POST", {
            "size": 5, "fields": ARTIC_FIELDS.split(","),
            "query": {"bool": {"must": [{"match": {"artist_title": artist}}, {"match": {"title": title}}]}}})

    @source("Art Institute of Chicago")
    def _artic_search(self, img, img_vec, queries: list[str], surname: str,
                      artist: str = "", titles: tuple[str, ...] | list[str] = ()) -> list[Record]:
        seen: set = set()
        precise = (self._artic_post(artist, t) for t in titles[:4]) if artist else ()
        plain = (self._api("https://api.artic.edu/api/v1/artworks/search", "data", q=q, limit=10, fields=ARTIC_FIELDS)
                 for q in queries)
        for d in chain(precise, plain):          # plain queries run only if the fielded ones find nothing
            for x in d.get("data") or []:
                if x["id"] in seen or surname not in fold(x.get("artist_title") or x.get("artist_display") or ""):
                    continue
                seen.add(x["id"])
                rec = self._artic_rec(x, "image")
                if not rec.image_url:
                    if any(same_title(t, y) for t in titles for y in rec.all_titles()):
                        self._unchecked.append(f"Art Institute {x['id']} '{rec.title}' (no image served)")
                    continue
                # its image host answers a burst with a Cloudflare challenge: that image is unchecked, never a no-match
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url, f"Art Institute {x['id']}")
                if ok:
                    return [rec]
        return []

    def _met(self, oid: str | int, verified_by: str, strict: bool = False) -> Record | None:
        url = f"{MET_API}/v1/objects/{oid}"
        # its search lists withdrawn objects the object API calls "Not a valid object" (404): no record, not a wall
        d = self._api(url, "objectID", gone=(404,)) if strict else self._get(url)
        if not d.get("title"):
            return None
        return Record("met", d.get("objectURL") or f"https://www.metmuseum.org/art/collection/search/{oid}", d["title"], [],
                      d.get("artistDisplayName", ""), d.get("artistDisplayBio", ""), d.get("objectDate", ""), d.get("medium", ""),
                      "The Metropolitan Museum of Art", "", d.get("primaryImageSmall") or "", verified_by)

    def _met_ids(self, q: str, most: int = MET_IDS_MAX, **flags) -> list[int]:
        """Object IDs a Met search returns. The v1 search was retired on 2026-10-01; v1.1 pages (100 IDs by default,
        500 at most), so one bare call saw 100 of Whistler's 569 works and 100 of 205 "Mouse" titles (2026-11-06)."""
        ids: list[int] = []
        while len(ids) < most:
            d = self._api(f"{MET_API}/v1.1/search", "total", q=q, offset=len(ids), limit=min(500, most - len(ids)), **flags)
            page = d.get("objectIDs") or []
            ids += page
            if not page or len(ids) >= d["total"]:
                return ids
        if most >= MET_IDS_MAX:          # a whole read cut short (a plain query asks for its first 5 on purpose)
            log.warning("met search %r: read the first %d of its matches", q, len(ids))
        return ids

    @source("Metropolitan Museum")
    def _met_search(self, img, img_vec, queries: list[str], surname: str, artist: str = "",
                    titles: list[str] = ()) -> list[Record]:
        """The Met serves an image only for public-domain works; a title match without one is unchecked (Signac's
        'Le Pont des Arts', 396898, 2026-11-06), so a no-match there is an error, never an empty list."""
        # the Met honours one field flag per call: artist-only AND title-only, intersected here
        both = []
        by_artist = set(self._met_ids(artist, artistOrCulture="true")) if artist and titles else set()
        for t in titles[:3] if by_artist else []:
            ids = [i for i in self._met_ids(t, title="true") if i in by_artist]
            if ids:
                both.append(ids)
        plain = (self._met_ids(q, 5) for q in queries[:3])
        seen: set = set()
        for ids in chain(both, plain):
            for oid in ids[:5]:
                if oid in seen:
                    continue
                seen.add(oid)
                rec = self._met(oid, "image", strict=True)
                if not rec or surname not in fold(rec.artist):
                    continue
                if not rec.image_url:
                    if any(same_title(t, rec.title) for t in titles):
                        self._unchecked.append(f"Met {oid} '{rec.title}' (no image served: not public domain)")
                    continue
                ok, rec.clip, rec.inliers = self._verify(img, img_vec, rec.image_url, f"Met {oid}")
                if ok:
                    return [rec]
        return []

    @source("Cleveland Museum of Art")
    def _cleveland_search(self, img, img_vec, queries: list[str], surname: str, titles: list[str] = ()) -> list[Record]:
        """Fielded artist AND title first (its q ranks an artist's 8 best-known works, so a lesser one never came up),
        then q, images only (2026-11-06). Every creator is read, not just the first."""
        api = "https://openaccess-api.clevelandart.org/api/artworks/"
        fielded = (self._api(api, "data", artists=surname, title=t, has_image=1, limit=8) for t in titles[:4] if surname)
        plain = (self._api(api, "data", q=q, has_image=1, limit=8) for q in queries)
        seen: set = set()
        for d in chain(fielded, plain):
            for x in d.get("data") or []:
                cr = next((c for c in x.get("creators") or [] if surname in fold(c.get("description") or "")), None)
                if cr is None or x.get("id") in seen:
                    continue
                seen.add(x.get("id"))
                url = ((x.get("images") or {}).get("web") or {}).get("url") or ""
                ok, sim, inl = self._verify(img, img_vec, url, f"Cleveland {x.get('accession_number') or x.get('id')}")
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
        medium = f.get("Medium") or f.get("Media") or f.get("Technique") or f.get("Materials") or ""
        collection = f.get("Collection") or f.get("Current location") or f.get("Location") or ""
        movement = f.get("Movement") or f.get("Style") or f.get("Period") or ""
        rec = Record(institution(url), url, "" if non_title(title) else title, [], f.get("Artist", ""), "",
                     f.get("Date", ""), medium, collection, movement, image_url, verified_by,
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

    def _nationality(self, artist: str, wd: Record, bios: list[Record]) -> tuple[str, list[str], dict]:
        """The nationality a card prints, as demonyms (Sir, 2026-11-05): "American", or "Swiss-French" for an
        artist known by two. The model words it from what the sources state; it is ruled only when every part
        of the wording is stated by a source and two sources state some part of it."""
        stated: dict[str, set[str]] = {}
        countries = wd.artist_nationalities or ([wd.artist_nationality] if wd.artist_nationality else [])
        dems = {_country_demonym(n) for n in countries} - {""}
        if dems:
            stated["wikidata"] = dems
        for r in bios:
            dems = {d.capitalize() for d in DEMONYM if re.search(rf"\b{d}\b", fold(r.artist_bio))}
            if dems:
                stated.setdefault(r.source, set()).update(dems)
        if not stated:
            return "", [], stated
        label = ""
        vlm = getattr(getattr(self, "idf", None), "vlm", None)
        if vlm is not None:
            lines = [f"- Wikidata citizenship: {', '.join(countries)}"] if countries else []
            lines += [f"- {r.source}: {r.artist_bio}" for r in bios]
            try:
                label = str(vlm.ask_json(NATIONALITY_PROMPT.format(artist=artist, statements="\n".join(lines)),
                                         required=("label",), num_predict=256).get("label", "")).strip()
            except Exception as e:
                log.info("nationality wording failed, using the shared demonym: %s", e)
        if not label:                     # no model: the one demonym every stating source shares, if there is one
            common = set.intersection(*stated.values())
            label = next(iter(common)) if len(common) == 1 else ""
        parts = [p.strip() for p in re.split(r"[-–/]", label) if p.strip()]
        said = {a for s in stated.values() for d in s for a in _alias(d)}
        sup = sorted(src for src, s in stated.items() if {a for d in s for a in _alias(d)} & {p.lower() for p in parts})
        if parts and all(p.lower() in said for p in parts) and len(sup) >= 1:      # one reliable witness (Sir, 2026-11-10)
            return "-".join(p.capitalize() for p in parts), sup, stated
        return "", sup, stated

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
        """For a work the ruling could not settle, and only then: Google's results, at most 3 searches, read and
        image-verified like any other page. SerpAPI's monthly plan goes first; when it is low on credits the
        next service with credits answers instead (Serper, Oxylabs, SearchAPI: the Lens pool's text search,
        same ledger and floor; Sir, 2026-11-10)."""
        web = getattr(self.idf, "web", None)
        pool = getattr(self.idf, "lens", None)
        search = pool.pages if pool is not None and hasattr(pool, "pages") else getattr(web, "serpapi_pages", None)
        if search is None:
            return []
        artist = ident.get("artist", "")
        titles = [t for t in leads_ if t][:2]
        queries = list(dict.fromkeys([f"{artist} {t}" for t in titles] + [f"{artist} painting"]))[:3]
        img_vec = self.emb.embed([img])[0]
        have = {institution(r.source) for r in recs}
        self._artist = artist
        return self._safe("google text", lambda: self._web_text(img, img_vec, queries, have, search,
                                                                "Google results (fallback for an unresolved work)"))

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

    def _walled(self, img, img_vec, artist: str, titles: list[str]) -> list[Record]:
        """A walled catalogue raisonné through Google's index (Sir, 2026-11-05): the picture is verified on Google's
        copy, and the record votes on the title the index shows, nothing else. The site itself is never contacted."""
        web = getattr(self.idf, "web", None)
        sites = WalledCatalogue.sites(artist)
        if web is None or not hasattr(web, "index_images") or not sites:
            return []
        self.consulted += [f"{s} (through Google's index)" for s in sites]
        out, done = [], set()
        for r in WalledCatalogue(web).search(artist, titles[:2]):
            site = r.meta.get("catalogue", "")
            if site in done or not r.meta.get("title_from_index"):
                continue
            ok, sim, inl = self._verify(img, img_vec, r.image_url)
            if ok:
                done.add(site)
                out.append(Record(site, r.page_url, r.meta["title_from_index"], image_url=r.image_url,
                                  verified_by="image (Google's indexed copy)", clip=sim, inliers=inl))
        return out

    @source("National Gallery of Art")
    def _nga(self, img, img_vec, titles: list[str], artist: str) -> list[Record]:
        """National Gallery of Art, from its open data (nga.gov itself is walled): the artist's NGA works whose
        titles match a lead first, then all of them ranked by embedding; images from its IIIF server. Works with
        no cached embedding were never ranked, so a no-match among them is an error (the 2026-11-05 benchmark ran
        before Corot's cache existed: 6 arbitrary works of his were compared, San Bartolomeo was not)."""
        from .nga import NGA
        if not artist:
            return []
        path = (self.cfg["sources"].get("nga_db") if self.cfg else "") or ""
        nga = NGA(Path(path).expanduser()) if path else None
        if nga is None or not nga.available:
            raise SourceError(f"National Gallery of Art: no open-data index at {path or '(sources.nga_db unset)'} "
                              "(build it with `neural-canvas nga-index <csv dir>`)")
        self.consulted.append("National Gallery of Art (open data)")
        rows = [r for r in nga.works_by(artist) if r.get("iiif")]
        if not rows:
            return []
        titled = [r for r in rows if any(same_title(t, r["title"]) for t in titles)]
        rest = [r for r in rows if r not in titled]
        ranked = False
        if rest and len(rest) <= 800:          # the artist's other NGA works by embedding (thumbnails cached)
            vecs = self._nga_vectors(artist, rest)
            if vecs is not None:
                ranked = bool(np.any(vecs, axis=1).all())          # a work missing from the cache ranks as zeros
                rest = [rest[int(i)] for i in np.argsort(-(vecs @ img_vec))]
        if len(rest) > 6 and not ranked:
            self._unchecked.append(f"{len(rest) - 6} of {artist}'s {len(rows)} NGA works were never ranked by the picture "
                                   "(no embedding cache yet; the background indexer fills it)")
        for r in (titled[:6] + rest[:6]):
            ok, sim, inl = self._verify(img, img_vec, NGA.image(r), f"NGA {r['objectid']}")
            if ok:
                return [Record("nga", f"https://www.nga.gov/collection/art-object-page.{r['objectid']}.html", r["title"], [],
                               r["artist"], ", ".join(x for x in (r["nationality"], r["artist_dates"]) if x),
                               r["displaydate"] or "", r["medium"] or "", "National Gallery of Art", "", NGA.image(r),
                               "image", sim, inl,
                               text=fold(" ".join(str(r[k] or "") for k in ("title", "attribution", "inscription", "medium",
                                                                           "displaydate", "accessionnum"))),
                               wikidata=r.get("wikidataid") or "")]
        return []

    def _nga_vectors(self, artist: str, rows: list[dict], cold: bool = False):
        """Embeddings of an artist's NGA thumbnails, cached by object id; a run uses the cache only,
        the background indexer (cold=True) fills it."""
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
        todo = [r for r in rows if r["objectid"] not in have] if cold else []
        if not have and not todo:
            return None
        if todo:
            log.info("nga: embedding %d thumbnails for %s", len(todo), artist)
            ok = [(r, im) for r in todo for im in [fetch_image(NGA.image(r, 250), UA)] if im is not None]
            for i in range(0, len(ok), 64):
                chunk = ok[i:i + 64]
                for (r, _), v in zip(chunk, self.emb.embed([im for _, im in chunk])):
                    have[r["objectid"]] = v
            if have:
                save_npz(cache, ids=np.array(list(have)), vecs=np.stack(list(have.values())))
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
        except Exception as e:      # the reason travels with the run: a SourceError names the wall (2026-11-06)
            log.warning("catalogue source %s failed: %s", name, e)
            self.consulted.append(f"{name} (failed: {type(e).__name__}: {str(e)[:240]})")
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

        def from_wikidata(qid: str, via: str = "") -> list[Record]:
            """The work's Wikidata item and every record it links to. ``via`` names the source whose own
            link led here when the item was not found by the picture."""
            nonlocal artist_qid
            recs: list[Record] = []
            rec, ids = self._wikidata_record(qid)
            if via:
                rec.verified_by = f"link from {via}"
            elif qid in self._named_items:
                rec.verified_by = "creator and unique title (the item has no picture)"
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
            return recs

        for qid in self._safe("wikidata", lambda: self._wikidata_items(img, img_vec, artist, ident, titles)):
            recs += from_wikidata(qid)
        over(recs)                      # the Wikidata item and every record it links to
        have = {r.source for r in recs}
        if "artic" not in have:
            recs += over(self._safe("artic_search", lambda: self._artic_search(img, img_vec, queries, surname, artist, titles)))
        if "met" not in have:
            recs += over(self._safe("met_search", lambda: self._met_search(img, img_vec, queries, surname, artist, titles)))
        recs += over(self._safe("cleveland_search", lambda: self._cleveland_search(img, img_vec, queries, surname, titles)))
        recs += over(self._safe("europeana", lambda: self._europeana(img, img_vec, queries, surname, titles)))
        recs += over(self._safe("harvard", lambda: self._harvard(img, img_vec, queries, surname, titles)))
        if "si" not in {institution(r.source) for r in recs}:
            recs += over(self._safe("smithsonian", lambda: self._smithsonian(img, img_vec, queries, surname, titles)))
        if "yale" not in {institution(r.source) for r in recs}:
            recs += over(self._safe("yale", lambda: self._yale(img, img_vec, queries, surname, titles)))
        if "kunstindeks" not in {institution(r.source) for r in recs}:
            recs += over(self._safe("kunstindeks", lambda: self._kid(img, img_vec, artist, titles)))
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
        nga = over(self._safe("nga", lambda: self._nga(img, img_vec, titles, artist)))
        recs += nga
        # a museum record found by the picture names the work's Wikidata item; the oeuvre sweep may have missed
        # it (drawings, works far down the embedding ranking: Redon's Five Butterflies, 2026-11-05)
        if "wikidata" not in {r.source for r in recs}:
            for qid in dict.fromkeys(r.wikidata for r in nga if r.wikidata):
                have = {r.source for r in recs}
                recs += over([r for r in self._safe("wikidata_join", lambda: from_wikidata(qid, "nga"))
                              if r.source not in have])
        recs += over(self._safe("catalogue_sites", lambda: self._registry(img, img_vec, artist, titles,
                                                                  {institution(r.source) for r in recs})))
        recs += over(self._safe("walled_catalogues", lambda: self._walled(img, img_vec, artist, titles)))
        if artist_qid:
            recs += self._safe("wikidata_artist", lambda: [r for r in [self._wikidata_artist(artist_qid, surname)] if r])
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
            if r.source in ("wikipedia.org", "commons.wikimedia.org", "wikimedia", "artvee.com") or "meisterdrucke" in r.source \
                    or any(d in r.source for d in ("wikiart", "artchive", "wga.hu", "britannica", "oxfordartonline", "jstor")):
                return 2
            return 1 if r.source in ("wikidata", "rkd.nl") else 0
        pref = sorted(recs, key=lambda r: (rank(r), r.verified_by != "image"))
        out = {"facts": {}, "support": {}, "unruled": {}, "sources": [{k: v for k, v in asdict(r).items() if k not in ("text", "images")} for r in recs],
               "consulted": self.consulted, "unavailable": UNAVAILABLE, "failures": [],
               "corroborating": [{"source": r.source, "url": r.url, "title": r.title} for r in corroborating],
               "discarded": {}}
        over = {r.source for r in recs if r.overrule}

        def choose(name, values, support_of, agrees=None, floor=2):
            """Sir, 2026-11-05: the overrule tier (Wikidata and its links, museum APIs, catalogues raisonnés)
            votes among itself. When it states a fact, only its values can be ruled; lower-tier sources may
            add support by agreeing, and their disagreeing values are discarded. Most overrule support wins."""
            top = [(r, v) for r, v in values if r.overrule]
            best = None
            for r, v in top or values:
                sup = support_of(r, v)
                key = (len(over & set(sup)), len(sup))
                if len(sup) >= floor and (best is None or key > best[0]):
                    best = (key, v, sup)
            if not best:
                return None
            if top:      # discarded = disagrees with the ruling (a source agreeing in other words is not)
                gone = [f"{r.source}: {v}" for r, v in values if not r.overrule and r.source not in best[2]
                        and not (agrees and agrees(v, best[1], best[2]))]
                if gone:
                    out["discarded"][name] = gone
            return best[1], best[2]

        def pick(name, values, agree, floor=2):
            got = choose(name, values, lambda r, v: sorted({r2.source for r2, v2 in values if agree(v, v2)} |
                                                           {r2.source for r2 in recs if r2.text and r2.states(v)}),
                         lambda v, won, sup: agree(v, won), floor)
            if got:
                out["facts"][name], out["support"][name] = got
                return got[0]
            out["unruled"][name] = [f"{r.source}: {v}" for r, v in values] or ["no curated source states it"]
            return ""

        # title: any of a source's titles (languages, aliases) may agree; the display is the preferred source's
        tvals = [(r, t) for r in pref if (t := display_title(r.title, ident.get("artist", "")))]
        title = ""
        # a near-same title in another medium is another work (Whistler's Riva pastel, the Salute oil)
        got = choose("title", tvals, lambda r, t: sorted({r2.source for r2 in recs if not medium_clash(r.medium, r2.medium)
                                                          and (any(same_title(t, t2) for t2 in r2.all_titles()) or r2.states(t))}),
                     # agreeing with any wording of any supporting source (the card may print Wikidata's English)
                     lambda t, won, sup: any(same_title(t, t2) for r2 in recs if r2.source in sup for t2 in r2.all_titles()))
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
        # "Art Nouveau" and WikiArt's "Art Nouveau (Modern)" are one movement: one name contains the other
        same_mov = lambda a, b: bool(fold(a)) and bool(fold(b)) and (fold(a) in fold(b) or fold(b) in fold(a))  # noqa: E731
        # one reliable witness rules everything but the title and the artist (Sir, 2026-11-10)
        if not pick("movement", [(r, r.movement) for r in pref if r.movement], same_mov, floor=1):
            # the artist's movement (Sir's card lists "movement/period"): the movements on the artist's Wikipedia
            # article, in its order, confirmed by Wikidata's; Wikidata's own prints only when the article names none
            # (it gives Andrew Wyeth "French Realism", Q316325 P135, against the article's "Regionalist"). Names are compared whole, nationality and adjective form aside
            # ("Belgian surrealism" is the article's "Surrealist"), never as a substring: "Impressionism" inside the
            # article's "Post-Impressionism" printed Matisse an Impressionist (2026-11-12)
            who = next((r for r in authorities if r.source == "wikidata" and (r.movement or r.article_movement)), None)
            listed = [x.strip() for x in re.split(r"\s*[,;/]\s*|\s+and\s+", (who.article_movement if who else "") or "")
                      if fold(x)]
            def name(m):
                return " ".join(_ism(w) for w in fold(m).split() if w not in DEMONYM)
            wd = {name(m) for m in (who.movement.split("|") if who else []) if name(m)}
            both = [x for x in listed if name(x) in wd]
            # one reliable witness (Sir, 2026-11-10): the article's movement when it names one (Wyeth's "Regionalist"
            # outvotes Wikidata's "French Realism"); Wikidata's when the article names none (Vallotton: Les Nabis)
            first_wd = next((m for m in (who.movement.split("|") if who else []) if name(m)), "")
            pick_ = both[0] if both else listed[0] if listed else first_wd
            if pick_:
                shown = " ".join(_ism(w) for w in pick_.split())
                out["facts"]["movement"] = shown[:1].upper() + shown[1:]
                out["support"]["movement"] = (["wikipedia (artist)"] if listed else []) + (
                    ["wikidata (artist)"] if both or not listed else [])
                out["unruled"].pop("movement", None)
            elif who:
                out["unruled"]["movement"] = [f"wikidata (artist): {who.movement.replace('|', ', ')}, "
                                              "not named by the artist's Wikipedia article"]
        # the artist's life: Wikidata's years (the work's item, else the artist's own) against a printed biography
        wd = next((r for r in recs + authorities if r.source == "wikidata" and r.artist_born), None)
        bios = [r for r in recs + authorities if r.artist_bio and r.source != "wikidata"]
        if not wd:        # no Wikidata: the first printed biography with the years (an authority's first: RKD, KID)
            first = next((r for r in sorted(bios, key=lambda r: not r.authority) if r.artist_born or
                          re.search(r"\d{4}", r.artist_bio)), None)
            if first:
                yrs = re.findall(r"\d{4}", first.artist_bio)
                first.artist_born = first.artist_born or (yrs[0] if yrs else "")
                first.artist_died = first.artist_died or (yrs[1] if len(yrs) > 1 else "")
                wd = first
        if wd:
            agree = [r.source for r in bios if wd.artist_born in r.artist_bio and (not wd.artist_died or wd.artist_died in r.artist_bio)]
            agree += [r.source for r in recs if r.text and wd.artist_born and wd.artist_born in r.text
                      and (not wd.artist_died or wd.artist_died in r.text) and r.source not in agree]
            # one reliable witness rules the life (Sir, 2026-11-10); the sources that agree are its support
            out["facts"]["artist_born"], out["facts"]["artist_died"] = wd.artist_born, wd.artist_died
            out["support"]["artist_life"] = [wd.source, *[a for a in agree if a != wd.source]]
            label, sup, stated = self._nationality(ident.get("artist", "") or wd.artist, wd, bios)
            if label:
                out["facts"]["artist_nationality"] = label
                out["support"]["artist_nationality"] = sup
            else:
                out["unruled"]["artist_nationality"] = [f"{k}: {', '.join(sorted(v))}" for k, v in stated.items()] or \
                    ["no source states a nationality"]
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


def _ism(word: str) -> str:
    """A movement's adjective as its name: "Regionalist" -> "Regionalism", "surrealists" -> "surrealism"."""
    m = re.match(r"^(.{3,}?)ists?$", word)
    return m.group(1) + "ism" if m and word.lower() not in ("artist", "artists") else word


def labelled(text: str) -> dict:
    """"Label: value" fields as catalogue pages print them (Date: 1880 Collection: Yale ... Medium: ...)."""
    alt = "|".join(re.escape(l) for l in LABELS)
    out = {}
    for m in re.finditer(rf"\b({alt}):\s*(.+?)(?=\s+(?:{alt}):|$)", text[:20000]):
        # a value ends where the page's furniture begins: a run of capitals ("SHOP FOR POSTERS ..."), "Share:"
        # ... or where an unlisted label starts ("Style: Northern Renaissance Series: The Ghent Altarpiece")
        v = re.split(r"\s+(?:[A-Z]{2,}\s+){2,}[A-Z]{2,}|\s+(?:Share|Tags|Buy|Order)\b|\s+[A-Z][a-z]+(?:\s[a-z]+)?:\s",
                     m.group(2))[0].strip(" ;,.")
        if v and fold(v) not in ("unknown", "n a", "na", "none", "?"):
            out.setdefault(m.group(1), v[:200])
    return out


_FILE_WORDS = {"default", "image", "images", "img", "photo", "picture", "thumbnail", "thumb", "original", "large", "main",
               "full", "scaled", "larger", "preview", "download", "file", "media", "artwork", "hero", "banner"}


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
        # a picture's file name is often furniture ("default", "image", "main"): one of the fallback's three Google
        # searches was "Carl Heinrich Bloch default" (2026-11-12)
        if len(t) >= 4 and not re.fullmatch(r"[\d\s]+", t) and fold(t) not in _FILE_WORDS:
            clean.append(t)
    return list(dict.fromkeys(clean))


def record_items(ledger, recs: list[Record]) -> int:
    """Catalogue records join the work's evidence ledger (Sir, 2026-11-07): each is a cited witness for the
    agents, no longer a vote. Artist authority records rule the artist's life only and stay out; auction and
    press pages corroborate as captions."""
    n0 = len(ledger.items)
    items = {r.url.rsplit("/", 1)[-1] for r in recs if r.source == "wikidata" and not r.authority}
    for r in recs:
        if r.authority:
            continue
        from .sources import holder
        claims = {"title": r.title, "artist": r.artist, "date": r.date,
                  "medium": r.medium if _materials(r.medium) else "",
                  # a museum's own record of a picture it matched holds the work (Sir, 2026-11-10)
                  "collection": r.collection or (holder(r.url) if r.verified_by == "image" and not r.support else "")}
        if r.source == "commons.wikimedia.org" and r.wikidata and r.wikidata in items:
            # the fields its Artwork template fills from that same item are an echo, not a second witness; the ones
            # the file page sets itself are its own (2026-11-10)
            claims.update({k: "" for k in r.echo})
        also = [t for t in r.titles if t and t != r.title][:3]
        stated = "; ".join(f"{k} {v!r}" for k, v in claims.items() if v) or "no fields read"
        text = (f"{r.source} {'page' if r.support else 'catalogue record'} ({r.verified_by or 'found by text'}): {stated}"
                + (f"; also titled {', '.join(map(repr, also))}" if also else ""))
        if r.support:
            ledger.add("caption", r.source, text, r.url)
        else:
            ledger.add("record", r.source, text, r.url, claims=claims, overrule=r.overrule,
                       clip=r.clip, inliers=r.inliers, verified=False)
    return len(ledger.items) - n0


AUCTION_HOUSES = ("christies.com", "sothebys.com", "bonhams.com", "phillips.com", "kollerauktionen.ch", "koller.ch",
                  "dorotheum.com", "lempertz.com", "bukowskis.com", "hindmanauctions.com", "ketterer-kunst.com",
                  "ketterer-kunst.de", "vanham.de", "grisebach.com", "artcurial.com", "tajan.com", "doyle.com",
                  "freemansauction.com", "skinnerinc.com", "nagel-auktionen.de", "hampel-auctions.com", "lelandlittle.com",
                  "heritageauctions.com", "ha.com", "stockholmsauktionsverk.se", "bruun-rasmussen.dk", "fischerauktionen.ch")


def sale_page(url: str) -> bool:
    """A page that sells this very object at auction: an auction house's site, or a lot page on a sale aggregator
    (Invaluable's "/auction-lot/", LiveAuctioneers' "/price-result/")."""
    from .evidence import site_of
    u = (url or "").lower()
    # a lot page, never the house's editorial: Christie's "stories" article on the Royal Academy's Riviera show
    # made Signac's "Cassis, Cap Lombard" (Kunstmuseum Den Haag) a "Private collection" (2026-11-12)
    lot = re.search(r"/(auction-lot|price-result|lots?|detail|buy/auction)[/-]", u)
    if site_of(u) in ("kollerauktionen.ch", "koller.ch"):
        lot = lot or re.search(r"/\d{5,}-\d{4}-", u)            # Koller's lot pages: "/de/319324-0001-----5039-..."
    return bool(u) and bool(lot) and (site_of(u) in AUCTION_HOUSES or bool(re.search(r"/(auction-lot|price-result)[/-]", u))
                                      or "/lot" in u)


_EDITION = re.compile(r"lithograph|print|woodcut|woodblock|etching|engraving|screen ?print|serigraph|poster|aquatint|"
                      r"mezzotint|linocut|giclee|photograph|chromolitho", re.I)


_PRINTMAKER = re.compile(r"\b(lithographer|printmaker|poster artist|poster designer|engraver|etcher|woodblock|woodcut|"
                         r"serigraph|screenprint)", re.I)


def in_editions(medium: str) -> bool:
    """A medium made in editions (a lithograph, a woodblock print): no single collection holds the work."""
    return bool(_EDITION.search(medium or ""))


_DASHES = re.compile(r"[‐-―−]")


def life_from_ledger(items, artist: str) -> dict:
    """The artist's life and nationality as pages print them right after the name: "Waldemar Fink (Swiss Artist,
    1883–1948)", "FINK, WALDEMAR (1883 Bern 1948)", "Tomás Sánchez (Cuban b. 1948)", "Soichiro Tomioka, Japanese,
    1922–1994". Each site votes once; the birth year most sites print wins, then its most printed death year (two
    of Fink's pages print 1893, five print 1883)."""
    return decide_life(*life_votes(items, artist))


def life_votes(items, artist: str) -> tuple[dict, dict]:
    """One work's votes on its artist's life and nationality: {origin: [born, died, weight, id]} and
    {origin: [demonym, id]}. A site votes once, with its weightiest item."""
    from .evidence import site_rank
    lives: dict = {}
    nats: dict = {}
    sur = fold(artist).split()[-1] if fold(artist) else ""
    if len(sur) < 3:
        return lives, nats
    for i in items:
        if i.kind not in ("copy", "caption", "engine", "record") or i.site_kind in ("shop", "music"):
            continue
        raw = unicodedata.normalize("NFKD", _DASHES.sub("-", i.text)).encode("ascii", "ignore").decode()
        for m in re.finditer(rf"\b{re.escape(sur)}\b", raw, re.I):
            tail = raw[m.end():m.end() + 80]
            p = re.match(r"[^()]{0,40}?\(([^()]{1,60})\)", tail)
            bare = re.match(r"[,\s]+([A-Za-z]+)[,\s]+(\d{4})\s*-\s*(\d{4})", tail)
            if p:
                inner = p.group(1)
            elif bare and bare.group(1).lower() in DEMONYM:
                inner = bare.group(0)
            else:
                continue
            years = re.findall(r"\b(1[3-9]\d\d|20[0-2]\d)\b", inner)
            if len(years) == 1 and re.search(r"\b(b\.|born|geb\.|nee)\s*(in\s+\w+\s*,?\s*)?\d{4}", inner, re.I):
                life = [years[0], ""]
            elif len(years) == 2 and 15 <= int(years[1]) - int(years[0]) <= 105:
                life = years[:2]
            else:
                continue
            # sites weigh by kind: a catalogue, auction house or art reference outweighs social feeds, which copy
            # each other's errors (Fink's 1893 runs through Instagram, X and Pinterest; Koller and artprice: 1883)
            weight = (3 if i.kind == "record" or site_rank(i.url) <= 2 else 2 if site_rank(i.url) <= 4 else 1) \
                + (1 if i.kind == "copy" and i.verified else 0)
            if weight > (lives.get(i.origin) or ["", "", 0])[2]:
                lives[i.origin] = [*life, weight, i.id]
            # a compound stays whole: "French-American" is not "French" (Maggiori, 2026-11-12)
            pair = re.search(r"\b([A-Za-z]+)-([A-Za-z]+)\b", inner)
            if pair and pair.group(1).lower() in DEMONYM and pair.group(2).lower() in DEMONYM:
                dem = f"{pair.group(1).capitalize()}-{pair.group(2).capitalize()}"
            else:
                dem = next((w.capitalize() for w in fold(inner).split() if w in DEMONYM), "") or _country_demonym(inner)
            if dem:
                nats.setdefault(i.origin, [dem, i.id])
            break
    return lives, nats


def decide_life(lives: dict, nats: dict) -> dict:
    """The winning life and nationality over votes: the most weight wins (an earlier birth year breaks a tie)."""
    out = {"born": "", "died": "", "nationality": "", "life_ids": [], "nationality_ids": []}
    if lives:
        votes: dict = {}
        for born, died, w, _ in lives.values():
            votes[born] = votes.get(born, 0) + w
        born = max(votes, key=lambda b: (votes[b], -int(b)))
        deaths: dict = {}
        for b, d, w, _ in lives.values():
            if b == born and d:
                deaths[d] = deaths.get(d, 0) + w
        died = max(deaths, key=lambda d: deaths[d]) if deaths else ""
        out.update(born=born, died=died, life_ids=[x for b, d, _, x in lives.values() if b == born and d in (died, "")])
    if nats:
        tally: dict = {}
        for dem, _ in nats.values():
            tally[dem] = tally.get(dem, 0) + 1
        dem = max(tally, key=lambda d: (tally[d], "-" in d))      # a tie goes to the compound, the fuller statement
        out.update(nationality=dem, nationality_ids=[x for d, x in nats.values() if d == dem])
    return out


def card_medium(medium: str) -> str:
    """"Oil painting" names the object, not its materials: the card prints "Oil" (Mønsted, 2026-11-12). Wikidata's
    "Oil paint on canvas" prints as museums write it, "Oil on canvas" (Thayer's "Angel")."""
    m = re.sub(r"\b(oil|acrylic|tempera)\s+paint\s+(?=on\b)", r"\1 ", (medium or "").strip(), flags=re.I)
    # WikiArt lists media: "oil, canvas" is "Oil on canvas" (Rockwell's "Girl at the Mirror", 2026-11-12)
    pair = re.fullmatch(r"(oil|acrylic|tempera|gouache|watercolou?r|pastel|ink)\s*,\s*"
                        r"(canvas|panel|paper|board|cardboard|wood|copper|linen|masonite|silk)", m, flags=re.I)
    if pair:
        m = f"{pair.group(1).capitalize()} on {pair.group(2).lower()}"
    bare = re.sub(r"\s+paint(?:ing|ings)?\s*$", "", m, flags=re.I)
    return bare[:1].upper() + bare[1:] if bare != m and _materials(bare) else m


def date_from_ledger(items, artist: str, title: str, born: str = "", died: str = "") -> tuple[str, list[str]]:
    """The work's year as pages print it right after its title, on pages naming the artist: "Richard Savoie,
    Canada (1959-) There Was a Time (2015)" (2026-11-12: the model ruled Savoie's birth year). Each site votes
    once, weighted as for the life. The year must sit in brackets or after a comma right behind the title, and fall
    in the painter's working life: a print's or an exhibition's year printed there is later (Botticelli's
    "Primavera 1910", Bruce's "Composition IV 1941", in the replay over 281 works)."""
    from .agents import states
    from .evidence import site_rank
    t = fold(title)
    if len(t) < 3 or not artist:
        return "", []
    floor = int(born) + 8 if str(born).isdigit() else 1200
    ceiling = int(died) if str(died).isdigit() else time.localtime().tm_year
    pat = re.compile(r"\b" + r"\W+".join(map(re.escape, t.split())) + r"\b\W{0,3}(?:\(|,)\s*(?:c\.|ca\.|circa)?\s*(\d{4})(?:\s*[-/]\s*(\d{4}))?\s*(?:\)|[,.;|]|$)", re.I)
    votes: dict = {}
    for i in items:
        if i.kind not in ("copy", "caption", "engine", "record") or i.site_kind in ("shop", "music") \
                or states(i, "artist", artist) is not True:
            continue
        text = unicodedata.normalize("NFKD", _DASHES.sub("-", i.text)).encode("ascii", "ignore").decode()
        for y in pat.finditer(text):
            if not floor <= int(y.group(1)) <= ceiling:
                continue
            year = y.group(1) + (f"–{y.group(2)}" if y.group(2) and 0 < int(y.group(2)) - int(y.group(1)) <= 15 else "")
            w = (3 if i.kind == "record" or site_rank(i.url) <= 2 else 2 if site_rank(i.url) <= 4 else 1) \
                + (1 if i.kind == "copy" and i.verified else 0)
            if w > (votes.get(i.origin) or ("", 0, ""))[1]:
                votes[i.origin] = (year, w, i.id)
            break
    if not votes:
        return "", []
    tally: dict = {}
    for year, w, _ in votes.values():
        tally[year] = tally.get(year, 0) + w
    best = max(tally, key=tally.get)
    return best, [x for y, _, x in votes.values() if y == best]


class ArtistLives:
    """Every work's votes on its artist's life and nationality, kept across works (var/state/artist_lives.json):
    one painter prints one life and one nationality on every card. Voting work by work printed Mark Maggiori
    "French" on two cards and "American" on a third (2026-11-12). A site's vote from another work is referenced
    as "<run>:<item>"."""

    def __init__(self, state_dir):
        self.path = Path(state_dir) / "artist_lives.json"

    def merge(self, artist: str, run: str, lives: dict, nats: dict) -> dict:
        import fcntl
        key = fold(artist)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                db = json.loads(self.path.read_text()) if self.path.exists() else {}
            except ValueError:
                db = {}
            mine = db.setdefault(key, {"lives": {}, "nats": {}})
            for origin, (b, d, w, x) in lives.items():
                if w >= (mine["lives"].get(origin) or ["", "", 0])[2]:
                    mine["lives"][origin] = [b, d, w, f"{run}:{x}"]
            for origin, (dem, x) in nats.items():
                mine["nats"][origin] = [dem, f"{run}:{x}"]
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(db, ensure_ascii=False, indent=0))
            tmp.replace(self.path)
        return decide_life(mine["lives"], mine["nats"])


def identification_ledger(ev: dict):
    """The identification's evidence ledger. One made before the ledger existed (2026-11-07) kept only its verified
    match: that match, checked on both instruments, becomes the ledger's verified copy, or a rerun of the catalogue
    blocks for want of one (af Klint, 2026-11-10)."""
    from .evidence import Ledger
    ledger = Ledger(ev.get("ledger") or [])
    if not ledger.items and ev.get("confirmed") and ev.get("inliers"):
        ledger.add("copy", str(ev.get("tier") or "identify"), f"identification's verified match: {ev.get('reference', '')}",
                   ev.get("page_url") or ev.get("thumb_url") or "", verified=True, clip=float(ev.get("clip") or 0),
                   inliers=int(ev.get("inliers") or 0))
    return ledger


def catalogue_stage(cat: Catalogue, img: Image.Image, ident: dict, run: str = "work") -> dict:
    """The catalogue stage as the pipeline runs it: gather curated records by the identification's leads, add
    them to the work's ledger, and let the agents rule the card's facts under the same rules as identification,
    plus one: the date, medium and collection need a catalogue record among their witnesses. When a fact stays
    unruled, one more round runs after Google's results through SerpAPI. The artist's life and nationality are
    still ruled by ``rule`` (Wikidata against a printed biography)."""
    from .agents import rule_facts
    idf = cat.idf
    lead_list = leads(ident)
    found = cat.gather(img, ident, lead_list)
    ledger = identification_ledger(ident.get("evidence") or {})
    added = record_items(ledger, found)
    spent = []

    def more(ruling: dict, queries: list[str]) -> int:
        if spent:
            return 0
        spent.append(1)
        extra = cat.fallback(img, ident, found, lead_list)
        found.extend(extra)
        return record_items(ledger, extra)

    prior = (ident.get("evidence") or {}).get("verdict")
    last = ((ident.get("evidence") or {}).get("adjudication") or [{}])[-1]
    # identification's proven artist and title keep their witnesses when the catalogue rules them again
    carry = {k: (prior["facts"][k], prior.get("support", {}).get(k, [])) for k in ("artist", "title")
             if prior and prior.get("facts", {}).get(k)}
    card_rule = dict(need=("artist", "title"), record_facts=("date", "medium", "collection"), carry=carry)
    if not added and prior and not prior.get("failures"):
        # nothing new to weigh (a living painter in no catalogue): identification's last ruling is ruled again
        # under the card's rules, with no model call. Its date, medium and collection had been dropped here
        # outright, though verified copies and pages naming the work stated them (Musil's "Acrylic on Panel,
        # 2025", 2026-11-12)
        verdict = rule_facts(ledger, last["ruling"], last["check"], **card_rule) if last.get("ruling") else {"failures": [1]}
        if verdict["failures"]:
            verdict = {**prior, "facts": {k: v for k, v in prior["facts"].items() if k in ("artist", "title")}}
        verdict.update(rounds=[], note="no catalogue record found: identification's ruling ruled again under the card's rules")
    else:
        try:
            verdict = idf.agents.settle(img, ledger, **card_rule, rounds_max=idf.rounds_max, search=more)
        except Exception as e:      # a failed agent blocks the stage with its reason; it never prints a guess
            log.error("catalogue adjudication failed: %s", e)
            verdict = {"facts": {}, "support": {}, "unruled": {}, "rounds": [],
                       "failures": [f"catalogue adjudication failed: {type(e).__name__}: {str(e)[:200]}"]}
    gaps = [k for k in ("date", "medium", "collection") if not verdict.get("facts", {}).get(k)]
    ruled_title = verdict.get("facts", {}).get("title", "")
    if ruled_title and ruled_title not in lead_list:
        lead_list.insert(0, ruled_title)     # search the title just ruled first, not only identification's leads
    if not verdict.get("failures") and gaps and not spent and more(None, []):
        # the fallback round was meant for any unruled fact, but the loop only searched again when the artist or
        # the title failed: Bloch's "Casting out the Money Changers" ruled both at once, never searched, and lost
        # the WikiArt record (1874, oil on copper) an earlier run had found (2026-11-12). One more ruling over
        # the new evidence; what the first ruled is carried, so the second can only add
        keep = {k: (v, verdict.get("support", {}).get(k, [])) for k, v in verdict["facts"].items()
                if k in ("artist", "title", "date", "medium", "collection")}
        try:
            again = idf.agents.settle(img, ledger, need=("artist", "title"), record_facts=("date", "medium", "collection"),
                                      carry=keep, rounds_max=1, search=None)
        except Exception as e:      # the first ruling stands; the extra round is a widening, never a gate
            log.warning("catalogue fallback round failed: %s", e)
            again = None
        if again and not again["failures"]:
            added = {k: v for k, v in again["facts"].items() if k not in verdict["facts"]}
            verdict["facts"] = {**verdict["facts"], **added}
            verdict["support"] = {**verdict.get("support", {}), **{k: again["support"].get(k, []) for k in added}}
            verdict["unruled"] = {k: v for k, v in {**verdict.get("unruled", {}), **again.get("unruled", {})}.items()
                                  if k not in verdict["facts"]}
            verdict["rounds"] = [*verdict.get("rounds", []), *again.get("rounds", [])]
    facts = dict(verdict["facts"])
    if facts.get("medium"):
        facts["medium"] = card_medium(facts["medium"])
    if facts.get("medium") and not _materials(facts["medium"]):
        # "painting" is an object type, not a medium (Bloch, Christus Consolator, 2026-11-10): it never prints
        verdict.setdefault("unruled", {})["medium"] = f"{facts.pop('medium')!r}: names no material"
    # one name per painter on every card: the artist's own Wikidata label, their common name, when its surname is
    # the ruled one ("Carl Heinrich Bloch" on seven cards and "Carl Bloch" on six; "Hilaire Germain Edgar Degas",
    # 2026-11-12)
    who = next((r for r in found if r.authority and r.source == "wikidata" and r.artist), None)
    if who and facts.get("artist") and same_person(who.artist, facts["artist"]):
        facts["artist"] = who.artist
    if facts.get("title"):
        facts["title"] = display_title(facts["title"], facts.get("artist", ""))
    if fold(facts.get("collection", "")) == "private collection":
        facts["collection"] = "Private collection"
    life = cat.rule(found, {**ident, "artist": facts.get("artist") or ident.get("artist", "")})
    # the life, nationality and movement are ruled by ``rule``: the agents rule only the work's own five facts, and
    # the movement had been left off every card since the ledger redesign (2026-11-10)
    for k in ("artist_born", "artist_died", "artist_nationality", "inscribed_title", "movement"):
        if life["facts"].get(k):
            facts[k] = clean_movement(life["facts"][k]) if k == "movement" else life["facts"][k]
    support = {**verdict["support"], **{k: v for k, v in life["support"].items() if (k.startswith("artist_")
                                        and k not in ("artist",)) or k == "movement"}}
    unruled = {**verdict["unruled"], **{k: v for k, v in life["unruled"].items() if k.startswith("artist_l")
                                        or k in ("artist_nationality", "movement")}}
    # a painter with no Wikidata item and no printed biography: the life and nationality the pages print beside
    # the name, "Waldemar Fink (Swiss, 1883–1948)" (one reliable witness, Sir, 2026-11-10; most sites win a conflict)
    if not facts.get("artist_born") or not facts.get("artist_nationality"):
        who = facts.get("artist") or ident.get("artist", "")
        votes = life_votes(ledger.items, who)
        state = cat.cfg.path("state_dir") if getattr(cat, "cfg", None) is not None else None
        got = ArtistLives(state).merge(who, run, *votes) if state and who \
            else decide_life(*votes)
        if not facts.get("artist_born") and got["born"]:
            facts["artist_born"], facts["artist_died"] = got["born"], got["died"]
            support["artist_life"] = got["life_ids"]
            unruled.pop("artist_life", None)
        if not facts.get("artist_nationality") and got["nationality"]:
            facts["artist_nationality"] = got["nationality"]
            support["artist_nationality"] = got["nationality_ids"]
            unruled.pop("artist_nationality", None)
    # a work dated before its painter was eight is the painter's life misread as the work's date ("Richard Savoie
    # (b. 1959)" ruled Savoie's 2015 painting "1959", 2026-11-12)
    years = [int(y) for y in re.findall(r"\d{4}", facts.get("date") or "")]
    if years and facts.get("artist_born", "").isdigit() and min(years) < int(facts["artist_born"]) + 8:
        unruled["date"] = f"{facts.pop('date')!r}: before the artist (born {facts['artist_born']}) could paint it"
        support.pop("date", None)
    if not facts.get("date") and facts.get("title"):
        year, ids = date_from_ledger(ledger.items, facts.get("artist", ""), facts["title"], facts.get("artist_born", ""),
                                     facts.get("artist_died", ""))
        if year:
            facts["date"], support["date"] = year, ids
            unruled.pop("date", None)
    # the artist's movement describes the artist, but a work painted before the movement began cannot belong to it:
    # Mondrian's c. 1899 farm landscape printed "De Stijl" (founded 1917), Signac's 1884 harbour "Pointillism"
    # (2026-11-12). Only an artist-level movement is held to it; a work's own record speaks for the work
    years = [int(y) for y in re.findall(r"\d{4}", facts.get("date") or "")]
    began = MOVEMENT_START.get(fold(facts.get("movement", "")))
    if years and began and max(years) < began and any("(artist)" in x for x in support.get("movement", [])):
        unruled["movement"] = f"{facts.pop('movement')!r}: the artist's movement began in {began}, after this work"
        support.pop("movement", None)
    # a verified copy on an auction lot page that names this work: it was sold, and no museum's record claims it.
    # Not for a print: an edition has no single holder (Mucha's lithographs)
    from .agents import states
    # with no medium, a painter known for prints may have sold one impression of many: Mucha's "Precious Stones:
    # Emerald" (colour lithographs) printed "Private collection" from two Invaluable lots (2026-11-12)
    printmaker = not facts.get("medium") and facts.get("artist") and any(
        i.site_kind not in ("shop", "music") and _PRINTMAKER.search(i.text) and states(i, "artist", facts["artist"]) is True
        for i in ledger.items)
    if not facts.get("collection") and not in_editions(facts.get("medium", "")) and not printmaker:
        lots = [i.id for i in ledger.items if i.kind == "copy" and i.verified and sale_page(i.url)
                and facts.get("artist") and states(i, "artist", facts["artist"]) is True
                and (states(i, "title", facts.get("title", "")) is True
                     or (facts.get("date") and states(i, "date", facts["date"]) is True))]
        if lots:
            facts["collection"] = "Private collection"
            support["collection"] = lots
            unruled.pop("collection", None)
    return {"facts": facts, "support": support, "unruled": unruled, "failures": verdict["failures"],
            "contradicted": verdict.get("contradicted", {}), "adjudication": verdict.get("rounds", []),
            "sources": life["sources"], "consulted": life["consulted"], "unavailable": life["unavailable"],
            "corroborating": life["corroborating"], "ledger": ledger.to_list()}
