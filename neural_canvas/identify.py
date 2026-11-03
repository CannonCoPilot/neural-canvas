"""Stage 1 — identify the artwork from its pixels alone.

The model only *proposes*. An identity is CONFIRMED when a reference photograph from an open
collection matches the input on two independent instruments (see ``matching``); the label's
facts are then read from that reference's catalogue record, never from the model's guess.

Search widens round by round: the model's candidates by title → a sweep of each proposed
artist's works (catches right-artist/wrong-title) → a fresh model round told which candidates
were already refuted. No confirmation after the last round ⇒ the artwork stays UNCONFIRMED and
the run blocks; nothing downstream is allowed to print a guessed attribution.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .matching import Embedder, fetch_image, geometric_inliers
from .references import Reference, Wikimedia, file_thumb
from .style import StyleShortlist
from .websearch import WebImageSearch, WebReference
from .vlm import VLM

log = logging.getLogger(__name__)
WEB_UA = "Mozilla/5.0 (compatible; neural-canvas/1.0; +https://github.com/CannonCoPilot/neural-canvas)"

PROPOSE_PROMPT = """You are a museum curator identifying an artwork from a photograph of it.
Nothing is known about it except the image. Study style, subject, palette, technique, and read any
visible signature, monogram, date or inscription exactly as written.
{refuted}
{signature}
Propose up to 5 candidate identifications, most likely first, covering AT LEAST TWO DIFFERENT ARTISTS
(similar styles are easily confused, so include the closest alternatives). Prefer specific known works;
if you recognise the artist but not the work, still give your best title guess for that artist.
Reply ONLY with JSON:
{{"signature_text": "exact text of any signature/inscription or empty string",
  "visual_summary": "one sentence describing the image",
  "subject_keywords": ["1-2 plain words naming the main subject, e.g. 'fishing boats'", "a second subject word"],
  "search_phrases": ["3-5 concrete phrases (4-8 words) a person would type into an image search to find this exact picture: count, colour, subject, setting and medium, e.g. 'three fishing boats on a grey beach at dusk oil painting'"],
  "candidates": [{{"artist": "full name", "title": "title of the work", "year": "year or circa",
                   "confidence": 0.0}}]}}"""

SIGNATURE_PROMPT = """These are enlarged corners of a painting, where artists sign. Read any signature,
monogram, initials or date EXACTLY as painted (letters you cannot read: use ?). If the signature names a
known artist, give that artist's full name. Reply ONLY with JSON:
{"signature_text": "exact text or empty string", "artist": "full name or empty string", "confidence": 0.0}"""

TITLE_PROMPT = """A museum photograph file is catalogued as:
file name: {file_title}
object name: {object_name}
description: {description}
artist: {artist}
date: {date}
Extract the artwork's own title (no artist, museum, accession number or file suffix) and the
artist's full name and creation year. Reply ONLY with JSON:
{{"title": "...", "artist": "...", "date": "..."}}"""


WEB_FACTS_PROMPT = """Each line below is a web image that was verified, pixel by pixel, to be the SAME artwork,
with the image's own file name and the title of the page it appeared on:
{captions}
An image's file name usually names the pictured work; a page title may describe a DIFFERENT work shown on
the same page (shop pages, related-items strips, galleries). Weigh the evidence across all lines.
Give the artwork's artist (full name), its own title (no artist, gallery, size, price or "print" wording),
creation year and medium if any line states them. Use "" for anything no line states. Reply ONLY with JSON:
{{"artist": "...", "title": "...", "date": "...", "medium": "..."}}"""


@dataclass
class Match:
    reference: str
    page_url: str
    thumb_url: str
    clip: float
    inliers: int
    confirmed: bool
    query: str


@dataclass
class Identification:
    status: str                      # "confirmed" | "unconfirmed"
    artist: str = ""
    title: str = ""
    date: str = ""
    medium: str = ""
    collection: str = ""
    artist_born: str = ""
    artist_died: str = ""
    artist_nationality: str = ""
    facts_source: str = ""           # wikidata | commons-metadata
    wikidata: str = ""
    evidence: dict = field(default_factory=dict)
    rounds: list = field(default_factory=list)
    signature_text: str = ""
    visual_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


class Identifier:
    def __init__(self, cfg, vlm: VLM, wiki: Wikimedia, embedder: Embedder, web: WebImageSearch | None = None):
        self.cfg, self.vlm, self.wiki, self.emb, self.web = cfg, vlm, wiki, embedder, web
        self.t = cfg.thresholds
        self.ua = cfg["sources"]["user_agent"]
        self.style = StyleShortlist(wiki, embedder, cfg.path("state_dir"))
        self.cache_dir = cfg.path("state_dir") / "oeuvre"
        self.cache_dir.mkdir(exist_ok=True)

    # ── catalogue sweep: an artist's whole Wikidata oeuvre, embedded once and cached ──
    def _oeuvre_vectors(self, artist_qid: str) -> tuple[list[dict], np.ndarray]:
        works = self.wiki.oeuvre(artist_qid)
        cache = self.cache_dir / f"{artist_qid}.npz"
        have: dict[str, np.ndarray] = {}
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            have = dict(zip(z["qids"].tolist(), z["vecs"]))
        todo = [w for w in works if w["qid"] not in have]
        if todo:
            log.info("embedding %d catalogued works for %s", len(todo), artist_qid)
            ims = [fetch_image(file_thumb(w["file"], 250), self.ua) for w in todo]   # serial, paced
            ok = [(w, im) for w, im in zip(todo, ims) if im is not None]
            for i in range(0, len(ok), 64):
                chunk = ok[i:i + 64]
                for (w, _), v in zip(chunk, self.emb.embed([im for _, im in chunk])):
                    have[w["qid"]] = v
            np.savez(cache, qids=np.array(list(have)), vecs=np.stack(list(have.values())))
        works = [w for w in works if w["qid"] in have]
        return works, (np.stack([have[w["qid"]] for w in works]) if works else np.zeros((0, 512)))

    def _oeuvre_refs(self, artist: str, img_vec: np.ndarray, k: int = 4) -> list[Reference]:
        qid = self.wiki.find_artist(artist)
        if not qid:
            raise LookupError(f"no Wikidata artist record for {artist!r}")
        works, vecs = self._oeuvre_vectors(qid)
        if not works:
            return []
        top = np.argsort(-(vecs @ img_vec))[:k]
        return [Reference(file_title=f"{works[i]['label']} [{works[i]['qid']}]", page_id=10**12 + int(works[i]["qid"][1:]),
                          thumb_url=file_thumb(works[i]["file"], 500),
                          page_url=f"https://www.wikidata.org/wiki/{works[i]['qid']}", width=500, height=500,
                          meta={"wikidata": works[i]["qid"], "artist_wikidata": qid}) for i in top]

    # ── scoring one batch of references against the input ────────────────────
    def _score(self, img: Image.Image, img_vec: np.ndarray, refs: list[Reference], query: str,
               seen: set[int]) -> list[Match]:
        refs = [r for r in refs if r.page_id not in seen]
        seen.update(r.page_id for r in refs)
        def get(r):
            if isinstance(r, WebReference):
                return fetch_image(r.image_url, WEB_UA) or fetch_image(r.thumb_url, WEB_UA)
            return fetch_image(r.thumb_url, self.ua)
        if refs and all(isinstance(r, WebReference) for r in refs):
            with ThreadPoolExecutor(6) as ex:   # many independent sites: no shared rate limit
                pairs = list(zip(refs, ex.map(get, refs)))
        else:                                    # Wikimedia: serial and paced inside fetch_image
            pairs = [(r, get(r)) for r in refs]
        pairs = [(r, im) for r, im in pairs if im is not None]
        if not pairs:
            return []
        sims = self.emb.similarity(img_vec, self.emb.embed([im for _, im in pairs]))
        order = np.argsort(-sims)[:4]          # geometry only for the closest few
        out = []
        for i in order:
            r, im = pairs[i]
            sim = float(sims[i])
            inl = geometric_inliers(img, im) if sim >= self.t["id_clip_min"] - 0.08 else 0
            ok = sim >= self.t["id_clip_min"] and inl >= self.t["id_inliers_min"]
            out.append(Match(r.file_title, r.page_url, r.thumb_url, round(sim, 4), inl, ok, query))
            out[-1].reference_obj = r   # type: ignore[attr-defined]
        return out

    def read_signature(self, img: Image.Image) -> dict:
        """Bottom corners (where most painters sign) at native resolution, plus the top corners."""
        w, h = img.size
        cw, ch = max(1, int(w * 0.32)), max(1, int(h * 0.2))
        boxes = [(0, h - ch, cw, h), (w - cw, h - ch, w, h), (0, 0, cw, ch), (w - cw, 0, w, ch)]
        crops = [img.crop(b) for b in boxes]
        try:
            return self.vlm.ask_json(SIGNATURE_PROMPT, crops, required=("signature_text",), max_side=1024)
        except Exception as e:
            log.warning("signature pass failed: %s", e)
            return {"signature_text": "", "artist": ""}

    def _propose(self, img: Image.Image, refuted: list[dict]) -> dict:
        note = ""
        if refuted:
            lst = "; ".join(f"{c['artist']} — {c['title']}" for c in refuted[:15])
            note = (f"These identifications were already checked against museum photographs and DID NOT "
                    f"match this image: {lst}. Do not repeat them; propose different artists and works.")
        if self._used_phrases:
            note += (" These search phrases were already tried without success: "
                     + "; ".join(sorted(self._used_phrases)[:15]) + ". Give new, differently worded phrases.")
        sig = self._signature
        sig_note = (f"A close reading of the corners found this signature: {sig.get('signature_text')!r}"
                    + (f" (suggesting {sig['artist']})." if sig.get("artist") else ".")) if sig.get("signature_text") else ""
        return self.vlm.ask_json(PROPOSE_PROMPT.format(refuted=note, signature=sig_note), [img],
                                 required=("candidates",), max_side=1280)

    # ── main entry ─────────────────────────────────────────────────────────
    def _tiers(self, cands: list[dict], swept: set[str]):
        """Ordered (tier, query, search-fn) list for one round, cheapest and most specific first:
        catalogued titles, web titles, description-only web, artist+description web, then the
        expensive whole-catalogue sweeps and generic artist queries."""
        out = []
        for c in cands:
            for q in (f"{c.get('title', '')} {c['artist']}", c.get("title", "")):
                if len(q.strip()) >= 4:
                    out.append(("commons", q, lambda q=q: self.wiki.search_files(q, 12)))
        artists = [a for a in dict.fromkeys(c["artist"] for c in cands) if _norm(a) not in swept]
        by_style = []
        if artists:   # painters ordered by how much the picture resembles their manner
            sty = self.emb.embed_text([f"a painting by {a}" for a in artists]) @ self._img_vec
            by_style = [artists[i] for i in np.argsort(-sty)]
        for a in by_style[:2]:   # the two closest styles: one catalogue sweep each, early
            out.append(("oeuvre", f"oeuvre:{a}", lambda a=a: self._oeuvre_refs(a, self._img_vec)))
        for kw in self._keywords:      # artist-agnostic on the catalogued collection: short keyword AND query
            q = f"{kw} painting"
            out.append(("commons", q, lambda q=q: self.wiki.search_files(q, 50)))
        if self.web:
            for c in cands:
                if c.get("title"):
                    q = f"{c['artist']} {c['title']} painting"
                    out.append(("web", q, lambda q=q: self.web.search(q, 15)))
            for ph in self._phrases[:5]:      # artist-agnostic: when the model cannot name the painter
                q = ph if re.search(r"paint|oil|canvas|watercolou?r|acrylic|art", ph, re.I) else f"{ph} painting"
                out.append(("web", q, lambda q=q: self.web.search(q, 25)))
            for a in artists[:3]:
                for ph in self._phrases[:2]:
                    q = f"{a} {ph}"
                    out.append(("web", q, lambda q=q: self.web.search(q, 20)))
        for a in by_style[2:4]:  # the rest of the four closest; farther styles wait for a later round
            out.append(("oeuvre", f"oeuvre:{a}", lambda a=a: self._oeuvre_refs(a, self._img_vec)))
        if self.web:
            for a in by_style[:3]:
                out.append(("web", f"{a} painting", lambda a=a: self.web.search(f"{a} painting", 30)))
        swept.update(_norm(a) for a in by_style[:4])
        return out

    def identify(self, img: Image.Image) -> Identification:
        img_vec = self.emb.embed([img])[0]
        self._img_vec = img_vec
        self._img = img
        self._phrases: list[str] = []
        self._keywords: list[str] = []
        self._used_phrases: set[str] = set()
        self._used_keywords: set[str] = set()
        self._signature = self.read_signature(img)
        log.info("signature pass: %s", self._signature)
        seen: set[int] = set()
        refuted: list[dict] = []
        swept: set[str] = set()
        rounds = []
        sig, summary = "", ""
        for rnd in range(self.t["id_max_rounds"]):
            prop = self._propose(img, refuted)
            sig = sig or self._signature.get("signature_text", "") or prop.get("signature_text", "")
            summary = summary or prop.get("visual_summary", "")
            done = {(_norm(c["artist"]), _norm(c.get("title", ""))) for c in refuted}
            cands = [c for c in prop.get("candidates", []) if c.get("artist")
                     and (_norm(c["artist"]), _norm(c.get("title", ""))) not in done][:5]
            self._phrases = [str(x) for x in (prop.get("search_phrases") or [])
                             if str(x).strip() and str(x).strip().lower() not in self._used_phrases]
            self._used_phrases.update(p.strip().lower() for p in self._phrases)
            self._keywords = [str(x).strip() for x in (prop.get("subject_keywords") or [])
                              if 2 <= len(str(x).strip()) <= 30 and str(x).strip().lower() not in self._used_keywords][:2]
            self._used_keywords.update(k.lower() for k in self._keywords)
            if rnd == 0:   # painters the picture *looks like*, independent of the model's memory
                try:
                    style = self.style.top(img_vec, 4)
                    log.info("style shortlist: %s", style)
                    for name, _ in style:
                        if _norm(name) not in {_norm(c["artist"]) for c in cands}:
                            cands.append({"artist": name, "title": "", "year": "", "confidence": 0, "source": "style"})
                except Exception as e:
                    log.warning("style shortlist unavailable: %s", e)
            sa = (self._signature.get("artist") or "").strip()
            if sa and _norm(sa) not in {_norm(c["artist"]) for c in cands}:
                cands.append({"artist": sa, "title": "", "year": "", "confidence": self._signature.get("confidence", 0)})
            log.info("round %d candidates: %s", rnd + 1, [(c["artist"], c.get("title")) for c in cands])
            matches: list[Match] = []
            for tier, q, fn in self._tiers(cands, swept):
                log.info("  tier %s: %s", tier, q[:80])
                try:
                    batch = self._score(img, img_vec, fn(), q, seen)
                except Exception as e:   # a dead source is recorded in the round, not fatal
                    log.warning("%s search failed for %r: %s", tier, q, e)
                    matches.append(Match(f"ERROR {tier}: {e}", "", "", 0.0, 0, False, q))
                    continue
                matches += batch
                hits = [m for m in batch if m.confirmed]
                if hits:
                    ident = self._finish(tier, hits, matches, rounds, rnd, cands, sig, summary)
                    if ident.status == "confirmed":
                        return ident
                    rounds.pop()          # facts failed their gate; keep searching
            best = sorted(matches, key=lambda m: (-m.clip, -m.inliers))[:5]
            rounds.append({"round": rnd + 1, "candidates": cands, "best_matches": [_mdict(m) for m in best]})
            refuted += cands
        return Identification(status="unconfirmed", rounds=rounds, signature_text=sig, visual_summary=summary,
                              evidence={"reason": "no reference image matched on both instruments"})

    def _corroborate(self, hits: list, cands: list[dict]) -> list:
        """After a first web match, look for the same image on more sites so the facts rest on several
        sources. Searches by what the matched copy itself says (caption, file name) first: prefixing a
        guessed artist would steer away from the copies that carry the real name."""
        img, vec = self._img, self._img_vec
        seen = {h.reference_obj.page_id for h in hits}  # type: ignore[attr-defined]
        r0 = hits[0].reference_obj  # type: ignore[attr-defined]
        fname = re.sub(r"[+_%20-]+", " ", r0.image_url.split("?")[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]).strip()
        caption = re.split(r"\s[|—–-]\s", r0.file_title)[0].strip()
        qs = [q for q in (caption, fname if not re.fullmatch(r"[\w]{20,}|[\d\s]+", fname) else "",
                          f"{caption} artist") if len(q) >= 6]
        for q in list(dict.fromkeys(qs))[:3]:
            try:
                hits += [m for m in self._score(img, vec, self.web.search(q, 20), q, seen) if m.confirmed]
            except Exception as e:
                log.warning("corroboration search failed for %r: %s", q, e)
        return hits

    def _caption_artist(self, hits: list) -> str:
        caps = "\n".join(_web_line(h.reference_obj) for h in hits)  # type: ignore[attr-defined]
        try:
            d = self.vlm.ask_json(WEB_FACTS_PROMPT.format(captions=caps), required=("artist",))
            return str(d.get("artist", "")).strip()
        except Exception:
            return ""

    def _finish(self, tier, hits, matches, rounds, rnd, cands, sig, summary) -> Identification:
        hit = max(hits, key=lambda m: m.inliers)
        ref = hit.reference_obj  # type: ignore[attr-defined]
        rounds.append({"round": rnd + 1, "candidates": cands, "tier": tier, "confirmed_by": [_mdict(h) for h in hits],
                       "best_matches": [_mdict(m) for m in sorted(matches, key=lambda m: -m.clip)[:5]]})
        ident = Identification(status="confirmed", rounds=rounds, signature_text=sig, visual_summary=summary,
                               evidence={"tier": tier, **_mdict(hit)})
        if tier in ("commons", "oeuvre"):
            qid = ref.meta.get("wikidata") or self.wiki.depicted_item(ref.page_id)
            facts = self.wiki.artwork_facts(qid) if qid else {}
            if facts.get("title") and facts.get("artist"):
                ident.facts_source = "wikidata"
                for k, v in facts.items():
                    setattr(ident, k, v)
            else:
                m = ref.meta
                parsed = self.vlm.ask_json(TITLE_PROMPT.format(
                    file_title=ref.file_title, object_name=m.get("ObjectName", ""),
                    description=m.get("ImageDescription", "")[:600], artist=m.get("Artist", "")[:200],
                    date=m.get("DateTimeOriginal", "")), required=("title", "artist"))
                ident.facts_source = "commons-metadata"
                ident.title, ident.artist = parsed["title"].strip(), parsed["artist"].strip()
                ident.date = str(parsed.get("date", "")).strip()
        else:
            hits = self._corroborate(hits, cands)
            caption_artist = self._caption_artist(hits)
            # upgrade: find the same image in the caption artist's catalogued works -> Wikidata facts
            if caption_artist:
                try:
                    cat = [m for m in self._score(self._img, self._img_vec,
                                                  self._oeuvre_refs(caption_artist, self._img_vec), f"oeuvre:{caption_artist}",
                                                  set()) if m.confirmed]
                except Exception as e:
                    log.info("no catalogue upgrade for %r: %s", caption_artist, e)
                    cat = []
                if cat:
                    rounds.pop()
                    up = self._finish("oeuvre", cat, matches + cat, rounds, rnd, cands, sig, summary)
                    up.evidence["found_via"] = {"tier": "web", **_mdict(hit)}
                    return up
            caps = "\n".join(_web_line(h.reference_obj) for h in hits)  # type: ignore[attr-defined]
            parsed = self.vlm.ask_json(WEB_FACTS_PROMPT.format(captions=caps), required=("artist", "title"))
            ident.facts_source = "web-captions"
            ident.title, ident.artist = parsed["title"].strip(), parsed["artist"].strip()
            ident.date, ident.medium = str(parsed.get("date", "")).strip(), str(parsed.get("medium", "")).strip()
            ident.evidence["web_references"] = [_web_line(h.reference_obj) for h in hits]  # type: ignore[attr-defined]
            # independent corroboration 1: the caption's artist is the one the model saw
            seen_names = " ".join([sig] + [c["artist"] for c in cands])
            surname = _norm(ident.artist).split()[-1:] or [""]
            by_model = bool(surname[0]) and surname[0] in _norm(seen_names)
            sites = {h.reference_obj.domain for h in hits  # type: ignore[attr-defined]
                     if surname[0] and surname[0] in _norm(_web_line(h.reference_obj)).split()}  # type: ignore
            ident.evidence["artist_corroborated"] = by_model or len(sites) >= 2
            ident.evidence["artist_corroborated_by"] = ("the model's own reading" if by_model
                                                       else f"{len(sites)} independent sites")
            # 2: the title is actually written in at least one verified reference
            tw = {w for w in _norm(ident.title).split() if len(w) > 2}
            support = [h for h in hits if tw and len(tw & set(_norm(_web_line(h.reference_obj)).split())) / len(tw) >= 0.6]  # type: ignore
            ident.evidence["title_support"] = len(support)
            if not ident.evidence["artist_corroborated"]:
                ident.status = "unconfirmed"
                ident.evidence["reason"] = f"caption artist {ident.artist!r} not corroborated by the image reading"
            elif not support:
                ident.status = "unconfirmed"
                ident.evidence["reason"] = f"title {ident.title!r} is not written in any verified reference"
        if ident.facts_source != "wikidata" and ident.artist:
            ident.__dict__.update(self._artist_facts(ident.artist))
        if not (ident.title and ident.artist):
            ident.status = "unconfirmed"
            ident.evidence["reason"] = "image matched but the reference carries no usable title/artist"
        return ident

    def _artist_facts(self, name: str) -> dict:
        data = self.wiki._get(self.wiki.wikidata_api, {"action": "wbsearchentities", "search": name,
                                                       "language": "en", "type": "item", "limit": 5})
        art = re.compile(r"paint|artist|illustrat|printmak|engrav|sculpt|draughts|woodcut")
        ids = [e["id"] for e in data.get("search", []) if art.search((e.get("description") or "").lower())]
        ents = self.wiki.entities(ids)
        for qid in ids:
            ent = ents.get(qid, {})
            if "Q5" in self.wiki.claim_ids(ent, "P31"):   # a human
                nat = self.wiki.claim_ids(ent, "P27")
                return {"artist_born": self.wiki.claim_year(ent, "P569"),
                        "artist_died": self.wiki.claim_year(ent, "P570"),
                        "artist_nationality": self.wiki.label(self.wiki.entities(nat[:1]).get(nat[0], {}))
                        if nat else ""}
        return {}


def _web_line(r) -> str:
    fname = re.sub(r"[+_%20-]+", " ", (r.image_url.split("?")[0].rsplit("/", 1)[-1]).rsplit(".", 1)[0])
    return f"file: {fname[:80]} | page: {r.file_title[:120]} | site: {r.domain}"


def _mdict(m: Match) -> dict:
    d = {k: v for k, v in m.__dict__.items() if k != "reference_obj"}
    return d
