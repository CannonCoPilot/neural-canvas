"""The evidence ledger: one per work. Every search adds what it saw; nothing refuses along the way.

Sir, 2026-11-07: identification was a chain of gates, each re-reading page titles and refusing, so a work
found by one search was thrown away by the next (Suzan Visser: Google's own answer named her, the verified
copies were on Reddit and Instagram, and a caption gate refused them). Now every arm is a witness. It adds
typed items to the ledger, and two local model agents rule on the facts by citing item ids (``agents``).

Kinds of item:

* ``copy``      a copy of the picture on a page, its caption the page's title. ``verified`` when it matched
                the input on both instruments (embedding + geometric inliers): the same artwork, proven.
* ``caption``   a page an engine returned for the picture whose copy was not verified (fetch failed, or
                it fell short of the bar). What the page says, weaker than a copy.
* ``engine``    a search engine's own answer about the uploaded picture: Lens web results and knowledge
                panel, related searches, Bing's best-guess caption, Yandex's tags, Vision's entities.
* ``record``    a catalogue entry: Wikidata, a museum's API, a Commons file's metadata, a curated page.
* ``lead``      a model's guess (candidates, style look-alikes). Aims searches; never evidence of a fact.
* ``signature`` the model's reading of the corners. A lead too.

Independence is counted by ``origin``: the page's site, or the engine itself when an answer has no page. Two
providers returning the same Reddit post are one origin.
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass, field
from urllib.parse import urlsplit

KINDS = ("copy", "caption", "engine", "record", "lead", "signature")
EVIDENCE = ("copy", "caption", "engine", "record")     # what a ruling may count; leads only aim searches
TEXT_MAX = 300
_GOOGLE_HOP = ("google.com/goto", "google.com/url", "lens.google.com/url")
_SECOND = {"co", "com", "org", "net", "ac", "gov", "edu"}     # two-part public suffixes: bbc.co.uk, artuk.org.uk


def site_of(url: str) -> str:
    """The registered site of a URL ('https://www.reddit.com/r/x' -> 'reddit.com', 'a.b.co.uk' -> 'b.co.uk')."""
    host = urlsplit(url or "").netloc.lower().split(":")[0].removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


_RANK = {"curated": 0, "reference": 1, "market": 2, "unknown": 3, "blog": 4, "commercial": 5, "social": 6}
_ART_SITES = ("artvee.com", "wikiart.org", "the-athenaeum.org", "wga.hu", "artuk.org")
_MUSIC = ("spotify.", "music.apple", "soundcloud", "deezer", "bandcamp", "youtube.", "amselcom", "beatport", "tidal.com",
          "shazam", "last.fm", "discogs", "audiomack")


def site_rank(url: str) -> int:
    """How far a page's caption can be trusted to name the work: collections first, shops and feeds last."""
    from .sources import classify
    u = (url or "").lower()
    if any(d in u for d in _ART_SITES):
        return 1
    if any(d in u for d in _MUSIC):
        return 7      # music pages use paintings as covers and name the band (James Ward -> "Latteo")
    return _RANK.get(classify(u), 3)


def site_kind(url: str) -> str:
    """What sort of site a page is on, for the agents to weigh (never a refusal)."""
    return {0: "museum or collection", 1: "art reference", 2: "art market", 3: "other", 4: "blog",
            5: "shop", 6: "social media", 7: "music"}.get(site_rank(url), "other")


@dataclass
class Item:
    id: str
    kind: str
    source: str                  # the arm or provider: "google_lens:serpapi", "bing", "yandex", "wikidata", "model"
    text: str
    url: str = ""
    site: str = ""               # registered site, or the engine's own label when the page is unknown
    site_kind: str = ""
    verified: bool = False       # a copy that passed both instruments
    clip: float = 0.0
    inliers: int = 0
    claims: dict = field(default_factory=dict)   # structured facts a record states: artist, title, date, medium
    overrule: bool = False       # Wikidata and what it links, a museum's own API, a catalogue raisonné

    @property
    def origin(self) -> str:
        if self.kind in ("lead", "signature"):
            return "model"
        if self.site:          # 'reddit.com' and the engine's label 'Reddit · r/oilpaintings' are one origin
            return re.split(r"[.\s]", self.site)[0]
        # no page behind it (Bing's guess, Yandex's tags, a lost redirect): all such items are ONE witness, so a
        # fact always needs a page-backed origin too (code review, 2026-11-07: two page-less answers could
        # otherwise pass as two independent sources)
        return "no-page"

    def line(self) -> str:
        """One line of the agents' prompt."""
        if self.kind == "copy":
            tag = (f"VERIFIED COPY (same artwork: embedding {self.clip:.2f}, {self.inliers} matching points)"
                   if self.verified else f"unverified copy (embedding {self.clip:.2f})")
        else:
            tag = {"caption": "page caption, picture not verified", "engine": "search engine's answer about this picture",
                   "record": "catalogue record" + (" (curated: outranks web pages)" if self.overrule else ""),
                   "lead": "model guess (a lead, NOT evidence)", "signature": "model reading of the signature (a lead)"}[self.kind]
        where = f" | site: {self.site} ({self.site_kind})" if self.site and self.site_kind else (
            f" | site: {self.site}" if self.site else "")
        claims = "".join(f" | {k}: {v}" for k, v in self.claims.items() if v)
        return f"{self.id} [{tag}{where} | via {self.source}] {self.text}{claims}"


class Ledger:
    """Append-only and thread-safe: first look's arms add to it at once."""

    def __init__(self, items: list[dict] | None = None):
        self.items: list[Item] = [Item(**d) for d in items or []]
        self._lock = threading.Lock()
        self._keys = {self._key(i) for i in self.items}

    @staticmethod
    def _key(i: Item) -> tuple:
        kind = "page" if i.kind in ("copy", "caption") else i.kind      # a caption later verified is the same item
        return (kind, i.url or i.source, re.sub(r"\s+", " ", i.text.lower())[:160])

    def add(self, kind: str, source: str, text: str, url: str = "", site: str = "", **kw) -> Item | None:
        assert kind in KINDS, kind
        text = re.sub(r"\s+", " ", str(text or "")).strip()[:TEXT_MAX]
        claims = {k: str(v).strip() for k, v in (kw.pop("claims", None) or {}).items() if v and str(v).strip()}
        if not text and not claims:
            return None
        if any(h in (url or "") for h in _GOOGLE_HOP):
            url = ""                          # an opaque Google redirect names no page: keep the engine's site label
        page = url if url.startswith("http") else ""
        site = (site_of(page) if page else re.sub(r"\s*[·|].*$", "", (site or "").strip().lower())).strip()
        item = Item("", kind, source, text, page, site, site_kind(page) if page else "", claims=claims, **kw)
        with self._lock:
            k = self._key(item)
            if k in self._keys:
                old = next(i for i in self.items if self._key(i) == k)
                if item.verified and not old.verified:   # the same copy, now verified: upgrade in place
                    old.kind, old.verified, old.clip, old.inliers = "copy", True, item.clip, item.inliers
                elif old.kind in ("copy", "caption") and not old.verified and item.clip > old.clip:
                    old.clip, old.inliers = item.clip, item.inliers
                return old
            self._keys.add(k)
            item.id = f"E{len(self.items) + 1}"
            self.items.append(item)
            return item

    def add_copy(self, ref, source: str, clip: float, inliers: int, verified: bool) -> Item | None:
        """A reference that was scored against the input. A verified one is a copy; the rest are captions."""
        url = getattr(ref, "page_url", "") or getattr(ref, "image_url", "")
        text = getattr(ref, "file_title", "") or ""
        fname = _file_words(getattr(ref, "image_url", "") or getattr(ref, "thumb_url", ""))
        if fname and fname.lower() not in text.lower():
            text = f"{text} | file: {fname}" if text else f"file: {fname}"
        site = (getattr(ref, "meta", None) or {}).get("site", "")
        return self.add("copy" if verified else "caption", source, text, url, site, verified=verified,
                        clip=round(float(clip), 4), inliers=int(inliers))

    def get(self, ident: str) -> Item | None:
        return next((i for i in self.items if i.id == ident), None)

    def verified(self) -> list[Item]:
        return [i for i in self.items if i.kind == "copy" and i.verified]

    def origins(self, ids) -> set[str]:
        """Distinct origins among the given items that count as evidence (leads never do)."""
        return {i.origin for x in ids if (i := self.get(x)) and i.kind in EVIDENCE}

    def to_prompt(self, limit: int = 70) -> str:
        """Verified copies, records and engine answers first, then the best captions, up to ``limit`` lines.
        Captions whose picture scored far from the input (another picture on the page) are left out."""
        def rank(i: Item):
            order = {"copy": 0 if i.verified else 3, "record": 1, "engine": 2, "caption": 4, "lead": 5, "signature": 5}
            return (order[i.kind], -i.clip)
        keep = [i for i in self.items if not (i.kind in ("caption", "copy") and not i.verified and 0 < i.clip < 0.6)]
        return "\n".join(i.line() for i in sorted(keep, key=rank)[:limit])

    def to_list(self) -> list[dict]:
        return [asdict(i) for i in self.items]


def _file_words(u: str) -> str:
    """An image's file name as words, when it reads as words ('hummingbird-and-fish-suzan-visser.jpg')."""
    name = urlsplit(u or "").path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    words = re.sub(r"[+_%20-]+", " ", name).strip()
    if len(words) < 6 or re.fullmatch(r"[\w]{16,}|[\d\s]+", words) or not re.search(r"[a-z]{3}", words.lower()):
        return ""
    return words[:80]


# ── engine answers in provider replies ──────────────────────────────────────────────────────────────────────
_TEXT_LISTS = ("organic_results", "web_results", "knowledge_graph", "entities", "related_searches",
               "related_content", "text_results")


def engine_texts(reply: dict) -> list[dict]:
    """What a Google Lens-type reply says about the picture besides its image matches: web results (title +
    snippet), the knowledge panel, named entities, related searches and the AI Overview's text when it is
    inline. Field names differ by provider; every one seen in the saved replies (2026-11-07) is read here.
    Each entry: {"text", "url", "site"}."""
    out: list[dict] = []
    if not isinstance(reply, dict):
        return out
    roots = [reply] + [reply[k] for k in ("data", "results") if isinstance(reply.get(k), dict)]
    for root in roots:
        for key in _TEXT_LISTS:
            val = root.get(key)
            rows = val if isinstance(val, list) else [val] if isinstance(val, dict) else []
            for x in rows:
                if isinstance(x, str):
                    out.append({"text": x, "url": "", "site": ""})
                    continue
                if not isinstance(x, dict):
                    continue
                title = x.get("title") or x.get("name") or x.get("query") or ""
                snippet = x.get("snippet") or x.get("description") or x.get("subtitle") or ""
                text = f"{title} — {snippet}" if title and snippet and snippet not in title else (title or snippet)
                if text:
                    out.append({"text": text, "url": x.get("link") or x.get("url") or "",
                                "site": x.get("source") or x.get("domain") or ""})
        ai = root.get("ai_overview")
        if isinstance(ai, dict):
            blocks = [b.get("snippet") or b.get("text") or "" for b in ai.get("text_blocks") or [] if isinstance(b, dict)]
            if any(blocks):
                out.append({"text": "Google AI Overview: " + " ".join(b for b in blocks if b), "url": "", "site": ""})
    return out
