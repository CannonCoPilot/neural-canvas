"""How far a web page can be trusted about an artwork's facts.

Sir's rule (2026-11-05): facts are RULED only from curated sources (museums, national art-history
institutes, catalogues raisonnés, authority files, structured catalogues), two or more agreeing.
Everything else (file names, social media, blogs, shops, print sellers, auction listings) may steer the
SEARCH, never the ruling.

``classify(url)`` -> "curated" | "reference" | "market" | "commercial" | "social" | "blog" | "unknown".
``non_title(text)`` -> why a caption string is not an artwork title (a Pinterest board, a bare place
name), or "" when it may be one. A non-title is still search text.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

RULING = {"curated"}

_CURATED = (
    # catalogues and authorities
    "wikidata.org", "vocab.getty.edu", "getty.edu", "rkd.nl", "europeana.eu", "si.edu", "loc.gov",
    # museums (collection sites)
    "metmuseum.org", "artic.edu", "clevelandart.org", "nga.gov", "rijksmuseum.nl", "tate.org.uk",
    "nationalgallery.org.uk", "louvre.fr", "musee-orsay.fr", "moma.org", "guggenheim.org", "brooklynmuseum.org",
    "mfa.org", "philamuseum.org", "lacma.org", "smk.dk", "nationalmuseum.se", "belvedere.at", "vangoghmuseum.nl",
    "harvardartmuseums.org", "britishmuseum.org", "vam.ac.uk", "nationalgalleries.org", "museodelprado.es",
    "uffizi.it", "khm.at", "smb.museum", "pinakothek.de", "staedelmuseum.de", "kunsthalle-bremen.de",
    "hermitagemuseum.org", "pushkinmuseum.art", "metmuseum.org", "collections.mfa.org", "artmuseum.princeton.edu",
    "artgallery.yale.edu", "britishart.yale.edu", "mia.org", "collections.artsmia.org", "dia.org", "saam.si.edu",
    "npg.si.edu", "hirshhorn.si.edu", "crystalbridges.org", "nrm.org", "terraamericanart.org", "denverartmuseum.org",
    "nelson-atkins.org", "slam.org", "carnegieart.org", "toledomuseum.org", "high.org", "mfah.org", "lacma.org",
    "getty.edu", "nortonsimon.org", "huntington.org", "famsf.org", "sfmoma.org", "seattleartmuseum.org",
    "agsa.sa.gov.au", "ngv.vic.gov.au", "artgallery.nsw.gov.au", "nga.gov.au", "gallery.ca", "ago.ca", "mbam.qc.ca",
    "munchmuseet.no", "nasjonalmuseet.no", "ateneum.fi", "kansallisgalleria.fi", "kokoelmat.fng.fi", "skd.museum",
    "mucha-foundation.org", "muchafoundation.org", "parismuseescollections.paris.fr", "pop.culture.gouv.fr",
    "bildindex.de", "zeno.org", "emuseum", "collections.", "collection.", "artsandculture.google.com",
    "ndl.go.jp", "tnm.jp", "momat.go.jp", "ukiyo-e.org", "honolulumuseum.org", "asia.si.edu", "lacma.org",
    "fitzmuseum.cam.ac.uk", "ashmolean.org", "wallacecollection.org", "nationaltrustcollections.org.uk", "artuk.org",
    "klee.ch", "zpk.org", "fondationbeyeler.ch", "kunstmuseumbasel.ch", "kunsthaus.ch", "albertina.at",
    "lenbachhaus.de", "hilmaafklint.se", "modernamuseet.se", "churchhistorymuseum.org", "lds.org",
    "wyethfoundation.org", "brandywine.org", "farnsworthmuseum.org", "remington-museum.org", "centerofthewest.org",
    "amoncarter.org", "gilcrease.org", "nationalcowboymuseum.org",
    "whistlerpaintings.gla.ac.uk", "saturdayeveningpost.com",   # catalogue raisonné; the Post's own cover archive
)
# a museum's own collection site names the work's holder (Sir, 2026-11-10: "The holding museum's own record counts
# for collection"). SMK's record of Bloch's "Hyrdernes tilbedelse" carried no collection field, and the card lost
# "Statens Museum for Kunst" (2026-11-12)
MUSEUM_NAMES = {
    "smk.dk": "Statens Museum for Kunst", "metmuseum.org": "The Metropolitan Museum of Art",
    "artic.edu": "Art Institute of Chicago", "clevelandart.org": "Cleveland Museum of Art",
    "nga.gov": "National Gallery of Art", "rijksmuseum.nl": "Rijksmuseum", "tate.org.uk": "Tate",
    "nationalgallery.org.uk": "National Gallery, London", "louvre.fr": "Musée du Louvre",
    "musee-orsay.fr": "Musée d'Orsay", "moma.org": "Museum of Modern Art", "guggenheim.org": "Solomon R. Guggenheim Museum",
    "brooklynmuseum.org": "Brooklyn Museum", "mfa.org": "Museum of Fine Arts, Boston",
    "philamuseum.org": "Philadelphia Museum of Art", "lacma.org": "Los Angeles County Museum of Art",
    "nationalmuseum.se": "Nationalmuseum", "belvedere.at": "Belvedere", "vangoghmuseum.nl": "Van Gogh Museum",
    "harvardartmuseums.org": "Harvard Art Museums", "museodelprado.es": "Museo del Prado", "uffizi.it": "Uffizi Gallery",
    "khm.at": "Kunsthistorisches Museum", "albertina.at": "Albertina", "lenbachhaus.de": "Lenbachhaus",
    "kunsthaus.ch": "Kunsthaus Zürich", "kunstmuseumbasel.ch": "Kunstmuseum Basel", "munchmuseet.no": "Munch Museum",
    "nasjonalmuseet.no": "Nasjonalmuseet", "ateneum.fi": "Ateneum", "artmuseum.princeton.edu": "Princeton University Art Museum",
    "artgallery.yale.edu": "Yale University Art Gallery", "britishart.yale.edu": "Yale Center for British Art",
    "dia.org": "Detroit Institute of Arts", "saam.si.edu": "Smithsonian American Art Museum",
    "crystalbridges.org": "Crystal Bridges Museum of American Art", "nrm.org": "Norman Rockwell Museum",
    "terraamericanart.org": "Terra Foundation for American Art", "denverartmuseum.org": "Denver Art Museum",
    "nelson-atkins.org": "Nelson-Atkins Museum of Art", "slam.org": "Saint Louis Art Museum",
    "toledomuseum.org": "Toledo Museum of Art", "high.org": "High Museum of Art", "mfah.org": "Museum of Fine Arts, Houston",
    "nortonsimon.org": "Norton Simon Museum", "huntington.org": "The Huntington", "famsf.org": "Fine Arts Museums of San Francisco",
    "sfmoma.org": "San Francisco Museum of Modern Art", "seattleartmuseum.org": "Seattle Art Museum",
    "gallery.ca": "National Gallery of Canada", "ngv.vic.gov.au": "National Gallery of Victoria",
    "fitzmuseum.cam.ac.uk": "Fitzwilliam Museum", "ashmolean.org": "Ashmolean Museum", "wallacecollection.org": "The Wallace Collection",
    "churchhistorymuseum.org": "Church History Museum", "brandywine.org": "Brandywine Museum of Art",
    "centerofthewest.org": "Buffalo Bill Center of the West", "farnsworthmuseum.org": "Farnsworth Art Museum",
    "amoncarter.org": "Amon Carter Museum of American Art", "gilcrease.org": "Gilcrease Museum",
    "mia.org": "Minneapolis Institute of Art", "artsmia.org": "Minneapolis Institute of Art",
}


def holder(url: str) -> str:
    """The museum whose own collection site this page is on, or ""."""
    host = urlparse(url or "").netloc.lower().removeprefix("www.")
    return next((name for dom, name in MUSEUM_NAMES.items() if host == dom or host.endswith("." + dom)), "")


_REFERENCE = ("wikipedia.org", "wikimedia.org", "wikiart.org", "britannica.com", "artchive.com", "artcyclopedia.com",
              "wga.hu", "the-athenaeum.org", "jstor.org", "oxfordartonline.com")
_MARKET = ("christies.com", "sothebys.com", "bonhams.com", "phillips.com", "ha.com", "fineart.ha.com", "invaluable.com",
           "mutualart.com", "artnet.com", "liveauctioneers.com", "lempertz.com", "dorotheum.com", "artprice.com",
           "the-saleroom.com", "artsy.net", "doyle.com", "swanngalleries.com")
_COMMERCIAL = ("etsy.com", "amazon.", "ebay.", "fineartamerica.com", "pixels.com", "meisterdrucke",
               "redbubble.com", "society6.com", "posterlounge", "juniqe", "allposters", "art.com", "artsdot",
               "1st-art-gallery", "overstockart", "wahooart", "paintingmania", "canvasprints", "displate",
               "onlly.art", "joy-cadeaux", "artprintimages", "mediastorehouse", "alamy.com", "shutterstock.com",
               "gettyimages.", "istockphoto", "dreamstime", "depositphotos", "123rf", "bridgemanimages",
               "magnoliabox", "king-and-mcgaw", "galeriesoftheworld", "reproduction", "prints",
               "artera.ae")       # an aggregator re-hosting Artvee: its /artworks/ path is no museum's
_SOCIAL = ("pinterest.", "pinimg.com", "tumblr.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
           "reddit.com", "flickr.com", "deviantart.com", "yandex.", "vk.com", "threads.net", "tiktok.com",
           "bing.net", "bing.com", "googleusercontent.com")
_BLOG = ("wordpress.com", "blogspot.", "medium.com", "substack.com", "livejournal.com", "weebly.com", "wixsite.com",
         "squarespace.com", "tumblr.", "typepad.com", "/blog")


def classify(url: str) -> str:
    u = (url or "").lower()
    host = urlparse(u).netloc or u
    def one(d):
        if d.startswith("/"):
            return d in u                                   # a path marker
        if "." in d.strip(".") and not d.endswith(".") and not d.startswith("."):
            return host == d or host.endswith("." + d)      # a domain: suffix match ("art.com" != "mutualart.com")
        return d in host                                    # a token ("amazon.", "collections.", "meisterdrucke")

    def hit(lst):
        return any(one(d) for d in lst)
    for name, lst in (("social", _SOCIAL), ("blog", _BLOG), ("commercial", _COMMERCIAL), ("market", _MARKET),
                      ("reference", _REFERENCE), ("curated", _CURATED)):
        if hit(lst):
            return name
    return "unknown"


GENERIC = {"untitled", "landscape", "portrait", "venice", "paris", "london", "rome", "study", "sketch", "painting",
           "still life", "nude", "seascape", "figure", "composition", "abstract", "flowers", "the sea", "trees",
           "self portrait", "self-portrait", "art", "artwork", "image", "picture", "drawing", "print", "untitled work"}


def non_title(text: str) -> str:
    """Why ``text`` is not an artwork's title, or "" when it may be one."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    low = t.lower().strip(" .,:;-|")
    if not low:
        return "empty"
    if re.match(r"^\d+\+?\s.*\bideas?\b", low) or re.search(r"\bideas?\s*(\||$)", low) or low.startswith(("pin on ", "pin by ")):
        return "a Pinterest board name"
    if re.search(r"\b(for sale|buy|shop|prints?|poster|canvas print|framed|reproduction|wall art|art print)\b", low):
        return "shop wording"
    if re.search(r"\b(blog|posts?|what they tell us|tips|how to|reviews?)\b", low):
        return "a blog or article heading"
    if low in GENERIC:
        return "a generic word, not a title"
    return ""


_INSTITUTION = re.compile(r"museum|museet|musee|museo|muzeum|museu|galerie|gallery|galleria|kunst|pinakothek|"
                          r"collection|catalogue|raisonne|database|archiv|foundation|stiftung|fondation|institut", re.I)
_RECORD_PATH = re.compile(r"/(collections?|artworks?|objects?|object-page|kunstwerken|werke|oeuvres?|"
                          r"painting|art-object|fotoweb|catalog|catalogue|detail|item)s?\b", re.I)


# Sir, 2026-11-05: Wikipedia and Wikimedia Commons count, and so does Meisterdrucke (a fine-art print
# publisher whose pages carry catalogue-style captions)
ALSO_RULING = ("wikipedia.org", "commons.wikimedia.org", "meisterdrucke", "artvee.com",   # Artvee: Sir, 2026-11-05
               # promoted from reference to votes, Sir 2026-11-05
               "wikiart.org", "artchive.com", "wga.hu", "britannica.com", "oxfordartonline.com", "jstor.org")


# Sir, 2026-11-05: auction houses and reputable press are "leads and support". A verified page from one
# is listed beside a ruling as corroboration; it never counts toward the two credible sources.
# Auction aggregators (MutualArt, artnet, Invaluable, ...) stay leads only.
AUCTION_HOUSES = ("christies.com", "sothebys.com", "bonhams.com", "phillips.com", "ha.com", "lempertz.com",
                  "dorotheum.com", "doyle.com", "swanngalleries.com")
PRESS = ("smithsonianmag.com", "nytimes.com", "theguardian.com", "washingtonpost.com", "bbc.com", "bbc.co.uk",
         "artnews.com", "theartnewspaper.com", "apollo-magazine.com", "burlington.org.uk", "frieze.com",
         "artforum.com", "newyorker.com", "economist.com", "ft.com", "wsj.com", "npr.org", "pbs.org")


def supporting(url: str) -> bool:
    """An auction house's own catalogue or a reputable publication: corroborates, never rules."""
    host = urlparse(url or "").netloc.lower().removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in AUCTION_HOUSES + PRESS)


def rulable(url: str, artist: str = "", living: bool = False, linked: bool = False) -> bool:
    """May this page's statements count toward a ruling? Curated and institutional pages; Wikipedia,
    Commons and Meisterdrucke (Sir's list); an artist's foundation or estate; and, for a LIVING artist,
    the artist's own site (a host that carries the artist's surname). An unlisted museum-looking host
    counts only when a catalogue links to it (``linked``): found by search, "gallery" is often a shop."""
    if classify(url) == "curated" or (linked and institutional(url)):
        return True
    host = urlparse(url or "").netloc.lower()
    if any(d in host for d in ALSO_RULING):
        return True
    sur = re.sub(r"[^a-z]", "", (artist or "").split()[-1].lower()) if artist else ""
    if sur and len(sur) >= 3 and sur in re.sub(r"[^a-z]", "", host):
        if re.search(r"foundation|stiftung|fondation|fundacion|estate|trust", host + urlparse(url).path, re.I):
            return True
        return living and classify(url) not in ("social", "blog", "market")
    return False


def institutional(url: str) -> bool:
    """A page that speaks for an institution: a curated domain, or an unlisted one (reached from a
    catalogue's own link, never from a search) whose host or path is a museum or collection record."""
    kind = classify(url)
    if kind == "curated":
        return True
    if kind != "unknown":
        return False
    u = urlparse(url)
    return bool(_INSTITUTION.search(u.netloc) or _RECORD_PATH.search(u.path))
