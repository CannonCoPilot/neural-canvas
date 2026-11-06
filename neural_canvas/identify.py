"""Stage 1 — identify the artwork from its pixels alone.

The model only *proposes*. An identity is CONFIRMED when a reference photograph from an open
collection matches the input on two independent instruments (see ``matching``); the label's
facts are then read from that reference's catalogue record, never from the model's guess.

Search widens round by round: the model's candidates by title → a sweep of each proposed
artist's works (catches right-artist/wrong-title) → a fresh model round told which candidates
were already refuted. A free reverse-image search (Imify) runs alongside from the start. Google
Vision web detection, which is billed, is the fallback: once per image, on a near-miss or when the
free rounds are spent, and its hints buy one more round. No confirmation after the last round ⇒
the artwork stays UNCONFIRMED and the run blocks; nothing downstream is allowed to print a guessed
attribution.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .matching import Embedder, fetch_image, geometric_inliers, save_npz
from .references import Reference, Wikimedia, file_thumb
from .artsites import Arthur, Artvee, WalledCatalogue
from .imify import Imify
from .style import StyleShortlist
from .vision_web import VisionWebDetection
from .websearch import WebImageSearch, WebReference
from .vlm import VLM

log = logging.getLogger(__name__)
WEB_UA = "Mozilla/5.0 (compatible; neural-canvas/1.0; +https://github.com/CannonCoPilot/neural-canvas)"
FETCH_WORKERS = 6     # reference fetches in flight across hosts; Wikimedia and the Art Institute take one at a time

PROPOSE_PROMPT = """You are a museum curator identifying an artwork from a photograph of it.
Nothing is known about it except the image. It IS a work of art — a painting, print or drawing — even
when it looks photographic: propose painters and printmakers, never photographers. Study style, subject, palette, technique, and read any
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
known artist, give that artist's full name. Some artists also write the work's TITLE on it or on its mount
(e.g. Paul Klee: "1920.92 Giftbeeren"): give such a title exactly as written, without the signature, date or
catalogue number, and say whether every letter of it is clearly legible. Reply ONLY with JSON:
{"signature_text": "exact text or empty string", "artist": "full name or empty string", "confidence": 0.0,
 "inscribed_title": "title as written or empty string", "inscribed_title_legible": false}"""

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
Give the artwork's artist (full name only: no dates, nationality or brackets), its own title (no artist, gallery, size, price or "print" wording),
creation year and medium if any line states them. Use "" for anything no line states. Two years in brackets
beside the artist's name, e.g. "(Japan, 1922-1994)", are the artist's life, NOT the work's date. A medium
cut off mid-phrase (e.g. "Oil on |") is not stated: use "". Reply ONLY with JSON:
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
    inscribed_title: str = ""        # a title written on the work, kept only when every letter is legible
    visual_summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


class Identifier:
    def __init__(self, cfg, vlm: VLM, wiki: Wikimedia, embedder: Embedder, web: WebImageSearch | None = None,
                 imify: Imify | None = None, vision: VisionWebDetection | None = None):
        self.cfg, self.vlm, self.wiki, self.emb, self.web = cfg, vlm, wiki, embedder, web
        self.imify, self.vision = imify, vision
        self.t = cfg.thresholds
        self.ua = cfg["sources"]["user_agent"]
        self.style = StyleShortlist(wiki, embedder, cfg.path("state_dir"))
        self.cache_dir = cfg.path("state_dir") / "oeuvre"
        self.cache_dir.mkdir(exist_ok=True)
        self._artist_qids: dict[str, str] = {}
        self._free_sweeps = 0
        self.artvee = Artvee(cfg.path("state_dir"), embedder)
        self.arthur = Arthur(web) if web else None
        self.walled = WalledCatalogue(web) if web and hasattr(web, "index_images") else None

    # ── catalogue sweep: an artist's whole Wikidata oeuvre, embedded once and cached ──
    def _oeuvre_vectors(self, artist_qid: str, cold: bool = False) -> tuple[list[dict], np.ndarray]:
        """The artist's catalogued works and their embeddings. A run uses only what is cached (cold=False):
        embedding a whole catalogue takes minutes at Wikimedia's pace, so only the background indexer
        (indexer.py, after a work is identified) sweeps cold (Sir, 2026-11-05)."""
        works = self.wiki.oeuvre(artist_qid)
        cache = self.cache_dir / f"{artist_qid}.npz"
        have: dict[str, np.ndarray] = {}
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            have = dict(zip(z["qids"].tolist(), z["vecs"]))
        todo = [w for w in works if w["qid"] not in have] if cold else []
        if todo:
            log.info("embedding %d catalogued works for %s", len(todo), artist_qid)
            ims = [fetch_image(file_thumb(w["file"], 250), self.ua) for w in todo]   # one host: serial, paced
            ok = [(w, im) for w, im in zip(todo, ims) if im is not None]
            for i in range(0, len(ok), 64):
                chunk = ok[i:i + 64]
                for (w, _), v in zip(chunk, self.emb.embed([im for _, im in chunk])):
                    have[w["qid"]] = v
            save_npz(cache, qids=np.array(list(have)), vecs=np.stack(list(have.values())))
            # the listing beside the vectors: the global index needs each work's title and image
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"qid": artist_qid, "works": works}))
            tmp.replace(cache.with_suffix(".json"))
        works = [w for w in works if w["qid"] in have]
        return works, (np.stack([have[w["qid"]] for w in works]) if works else np.zeros((0, 512)))

    def _oeuvre_cached(self, artist: str) -> bool:
        qid = self._artist_qids.get(artist)
        if qid is None:
            try:
                qid = self.wiki.find_artist(artist) or ""
            except Exception:
                qid = ""
            self._artist_qids[artist] = qid
        return bool(qid) and (self.cache_dir / f"{qid}.npz").exists()

    def _oeuvre_refs(self, artist: str, img_vec: np.ndarray, k: int = 4, cold: bool = False) -> list[Reference]:
        qid = self.wiki.find_artist(artist)
        if not qid:
            raise LookupError(f"no Wikidata artist record for {artist!r}")
        works, vecs = self._oeuvre_vectors(qid, cold=cold)
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
        with ThreadPoolExecutor(FETCH_WORKERS) as ex:   # rate-limited hosts are paced inside fetch_image
            pairs = list(zip(refs, ex.map(get, refs)))
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
        if self._vision_hints:
            h = self._vision_hints
            note += (" A reverse-image search of the web found this picture on pages titled: "
                     + "; ".join(t for t in h["page_titles"] if t)[:600]
                     + f". The search engine's best guess: {', '.join(h['best_guess'])}; "
                     + f"related names: {', '.join(h['entities'])}. These are leads, not facts.")
        sig = self._signature
        # Sir, 2026-11-05: signatures and inscriptions are leads only. A faint signature is easily misread
        # (an Aivazovsky read as "A. Sisley" steered a whole run), so it never outweighs the painting itself.
        sig_note = (f"A machine reading of the corners suggests a signature like {sig.get('signature_text')!r}"
                    + (f" (perhaps {sig['artist']})" if sig.get("artist") else "")
                    + ". Signatures are often faint and misread: treat this as an unverified lead, and let the "
                      "painting's style, subject and technique decide the candidates.") if sig.get("signature_text") else ""
        if sig.get("inscribed_title"):
            sig_note += f" Something like a title may be written on the work: {sig['inscribed_title']!r} (unverified)."
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
        if artists:   # interleave the model's picks (its order) with the closest styles, then cached first
            sty = self.emb.embed_text([f"a painting by {a}" for a in artists]) @ self._img_vec
            styled = [artists[i] for i in np.argsort(-sty)]
            model = [c["artist"] for c in cands if c.get("source") != "style"]
            mixed = [a for pair in zip(model, styled) for a in pair] + model + styled
            prio = list(dict.fromkeys(a for a in mixed if a in artists))
            cached = [a for a in prio if self._oeuvre_cached(a)]
            # the closest style and the model's first pick sweep first; cached catalogues (free) come next,
            # never ahead of them (Turner and Monet, cached, once took both early slots from Aivazovsky)
            early = list(dict.fromkeys([styled[0]] + [a for a in prio if a != styled[0]][:1]))
            by_style = list(dict.fromkeys(early + cached + prio))
            self._free_sweeps = len(set(early) | set(cached))
        # only catalogues already embedded are searched in a run; a new artist's catalogue is embedded in the
        # background after its work is identified (indexer.py), so no run waits on a cold sweep
        by_style = [a for a in by_style if self._oeuvre_cached(a)]
        for a in by_style[:2]:   # the two closest cached catalogues, early
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
                core = re.sub(r"\b(oil|acrylic|watercolou?r)?\s*(painting|print|on canvas|artwork)\b", "", ph, flags=re.I).strip()
                # contemporary work circulates mostly as print and shop listings: search each medium wording
                for q in dict.fromkeys([ph, f"{core} acrylic painting", f"{core} art print"]):
                    out.append(("web", q, lambda q=q: self.web.search(q, 25)))
            for a in artists[:3]:
                for ph in self._phrases[:2]:
                    q = f"{a} {ph}"
                    out.append(("web", q, lambda q=q: self.web.search(q, 20)))
        # Artvee and Arthur.io (Sir, 2026-11-05): the picture in two large collections. Their captions are
        # leads, never rulings. Titles first, then each likely artist's whole Artvee catalogue by embedding.
        titled = [c for c in cands if c.get("title")]
        insc = (self._signature.get("inscribed_title") or "").strip()
        for q in dict.fromkeys([insc] + [c["title"] for c in titled]):
            if len(q) >= 4:
                out.append(("artvee", f"artvee: {q}", lambda q=q: self.artvee.search(q)))
        for a in artists[:2]:
            words = [c["title"] for c in titled if c["artist"] == a] + ([insc] if insc else [])
            if self.arthur:
                out.append(("arthur", f"arthur: {a}", lambda a=a, w=words: self.arthur.search(a, w)))
            if self.walled and WalledCatalogue.sites(a):    # a walled catalogue raisonné, through Google's index
                out.append(("walled", f"walled catalogue: {a}", lambda a=a, w=words: self.walled.search(a, w)))
            out.append(("artvee", f"artvee catalogue: {a}", lambda a=a: self.artvee.nearest(a, self._img_vec, 6)))
        for a in by_style[2:max(4, self._free_sweeps)]:  # four per round (cached catalogues are free)
            out.append(("oeuvre", f"oeuvre:{a}", lambda a=a: self._oeuvre_refs(a, self._img_vec)))
        if self.web:
            for a in by_style[:3]:
                out.append(("web", f"{a} painting", lambda a=a: self.web.search(f"{a} painting", 30)))
        swept.update(_norm(a) for a in by_style[:max(4, self._free_sweeps)])
        return out

    def identify(self, img: Image.Image) -> Identification:
        calls0 = self.vision.calls if self.vision else 0
        pool = ThreadPoolExecutor(1)
        self._imify_pending = None
        try:
            ident = self._identify(img, pool)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if self._imify_pending is not None:      # confirmed before the reverse search answered
            ident.evidence["imify"] = "not consulted: identification confirmed first"
            self._imify_pending.add_done_callback(lambda f: log.info(
                "late imify result (unused): %s", f.exception() or f"{len(f.result())} candidates"))
        ident.evidence["vision_web_calls"] = (self.vision.calls - calls0) if self.vision else 0
        sig = getattr(self, "_signature", None) or {}
        it = str(sig.get("inscribed_title") or "").strip()
        if it and sig.get("inscribed_title_legible") is True and "?" not in it:
            ident.inscribed_title = it
        ident.evidence["signature"] = sig
        return ident

    def _identify(self, img: Image.Image, pool: ThreadPoolExecutor) -> Identification:
        # the reverse-image arm starts now and is scored the moment it answers
        imify_fut = pool.submit(self.imify.search_image, img) if self.imify else None
        self._imify_pending = imify_fut
        img_vec = self.emb.embed([img])[0]
        self._img_vec = img_vec
        self._img = img
        self._phrases: list[str] = []
        self._keywords: list[str] = []
        self._used_phrases: set[str] = set()
        self._used_keywords: set[str] = set()
        self._vision_hints: dict = {}
        self._hint_round = False
        self._lens_done = False
        vision_due = bool(self.vision and self.vision.enabled)
        max_rounds = self.t["id_max_rounds"]
        self._signature = self.read_signature(img)
        log.info("signature pass: %s", self._signature)
        seen: set[int] = set()
        refuted: list[dict] = []
        swept: set[str] = set()
        rounds = []
        sig, summary = "", ""
        rnd = -1
        while rnd + 1 < max_rounds:
            rnd += 1
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
                cands.append({"artist": sa, "title": "", "year": "", "confidence": 0, "source": "signature"})
            log.info("round %d candidates: %s", rnd + 1, [(c["artist"], c.get("title")) for c in cands])
            matches: list[Match] = []
            pending = self._tiers(cands, swept)
            while True:
                if imify_fut is not None and (imify_fut.done() or not pending):
                    f, imify_fut = imify_fut, None   # waits only when nothing else is left to try
                    self._imify_pending = None
                    pending.insert(0, ("imify", "reverse-image search", f.result))
                if not pending:
                    break
                tier, q, fn = pending.pop(0)
                if vision_due and (tier not in ("commons", "imify")
                                   or (tier == "oeuvre" and not self._oeuvre_cached(q.removeprefix("oeuvre:")))):
                    # one web-detection call answers in seconds; sweeps and web searches take many minutes
                    vision_due = False
                    ident = self._vision_tier(img, img_vec, "before a catalogue sweep", seen, matches, rounds, rnd,
                                              cands, sig, summary) \
                        or self._lens_tier(img, img_vec, "before a catalogue sweep", seen, matches, rounds, rnd, cands,
                                           sig, summary)
                    if ident:
                        return ident
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
            near = [m for m in matches if m.clip >= self.cfg["vision_web"]["near_miss_clip"]]
            if vision_due and (near or rnd + 1 == max_rounds):
                vision_due = False
                ident = self._vision_tier(img, img_vec, "near-miss" if near else "free rounds exhausted", seen, matches,
                                          rounds, rnd, cands, sig, summary)
                if ident:
                    return ident
            if self._vision_hints and rnd + 1 == max_rounds and not self._hint_round:
                self._hint_round = True
                max_rounds += 1   # its hints name new candidates: they get one round of their own
            best = sorted(matches, key=lambda m: (-m.clip, -m.inliers))[:5]
            rounds.append({"round": rnd + 1, "candidates": cands, "best_matches": [_mdict(m) for m in best],
                           **({"vision_hints": self._vision_hints} if self._vision_hints else {})})
            refuted += cands
        # unresolved after every free tier: Google Images through SerpAPI (free plan), two searches at most
        if self.web is not None and hasattr(self.web, "serpapi_images") and refuted:
            for q in list(dict.fromkeys(f"{c['artist']} {c.get('title', '')}".strip() for c in refuted[:3]))[:2]:
                log.info("  tier serpapi (fallback): %s", q)
                try:
                    batch = self._score(img, img_vec, self.web.serpapi_images(q), q, seen)
                except Exception as e:
                    log.warning("serpapi fallback for %r: %s", q, e)
                    break
                hits = [m for m in batch if m.confirmed]
                if hits:
                    ident = self._finish("serpapi", hits, batch, rounds, rnd, refuted[:5], sig, summary)
                    if ident.status == "confirmed":
                        return ident
                    rounds.pop()
        # Google Lens, when no earlier round reached it
        ident = self._lens_tier(img, img_vec, "fallback", seen, [], rounds, rnd, refuted[:5], sig, summary)
        if ident:
            return ident
        return Identification(status="unconfirmed", rounds=rounds, signature_text=sig, visual_summary=summary,
                              evidence={"reason": "no reference image matched on both instruments"})

    def _lens_tier(self, img, img_vec, why, seen, matches, rounds, rnd, cands, sig, summary) -> Identification | None:
        """Google Lens on a public copy of the picture that deletes itself within the hour (Sir, 2026-11-05).
        One SerpAPI search per work: it runs before the first slow catalogue sweep (it found a living painter
        in seconds where sweeps spent half an hour), or last if no round got that far."""
        if self._lens_done or self.web is None or not hasattr(self.web, "serpapi_lens"):
            return None
        self._lens_done = True
        log.info("  tier google_lens (%s)", why)
        try:
            batch = self._score(img, img_vec, self.web.serpapi_lens(self.web.temp_public_url(img)), "google lens", seen)
        except Exception as e:
            log.warning("google lens failed: %s", e)
            return None
        matches += batch
        hits = [m for m in batch if m.confirmed]
        if hits:
            ident = self._finish("google_lens", hits, matches, rounds, rnd, cands, sig, summary)
            if ident.status == "confirmed":
                return ident
            rounds.pop()
        return None

    def _vision_tier(self, img, img_vec, why, seen, matches, rounds, rnd, cands, sig, summary) -> Identification | None:
        """Web detection: one billed call per image, ever (the response is cached on disk). Its pages are
        scored like any other tier; its entity hints steer the next round's candidates."""
        log.info("  tier vision_web (%s)", why)
        try:
            web = self.vision.detect(img)
            self._vision_hints = VisionWebDetection.hints(web)
            batch = self._score(img, img_vec, VisionWebDetection.references(web), f"vision web detection ({why})", seen)
        except Exception as e:
            log.warning("vision web detection failed: %s", e)
            batch = [Match(f"ERROR vision_web: {e}", "", "", 0.0, 0, False, "vision web detection")]
        matches += batch
        hits = [m for m in batch if m.confirmed]
        if hits:
            ident = self._finish("vision_web", hits, matches, rounds, rnd, cands, sig, summary)
            if ident.status == "confirmed":
                return ident
            rounds.pop()
        return None

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
                ident.title, ident.artist = parsed["title"].strip(), clean_artist(parsed["artist"])
                ident.date = str(parsed.get("date", "")).strip()
        else:
            hits = self._corroborate(hits, cands)
            caption_artist = self._caption_artist(hits)
            # upgrade: find the same image in the caption artist's catalogued works -> Wikidata facts.
            # Only from a catalogue already embedded: a first sweep after a confirmed match buys nothing
            # the catalogue stage does not do anyway (its Wikidata item search sweeps the same oeuvre).
            if caption_artist and self._oeuvre_cached(caption_artist):
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
            ident.title, ident.artist = parsed["title"].strip(), clean_artist(parsed["artist"])
            ident.date, ident.medium = str(parsed.get("date", "")).strip(), str(parsed.get("medium", "")).strip()
            ident.evidence["web_references"] = [_web_line(h.reference_obj) for h in hits]  # type: ignore[attr-defined]
            # independent corroboration 1: the caption's artist is the one the model saw
            seen_names = " ".join([sig] + [c["artist"] for c in cands])
            surname = _norm(ident.artist).split()[-1:] or [""]
            by_model = bool(surname[0]) and surname[0] in _norm(seen_names)
            sites = {h.reference_obj.domain for h in hits  # type: ignore[attr-defined]
                     if surname[0] and surname[0] in _norm(_web_line(h.reference_obj)).split()}  # type: ignore
            if not by_model and len(sites) < 2 and ident.artist and ident.title:
                # one more look, now that the facts name an artist and title: the artist's own site,
                # galleries and catalogues usually title the same image the same way
                q = f"{ident.artist} {ident.title}"
                hits = hits + [m for m in self._score(self._img, self._img_vec, self.web.search(q, 20), q,
                                                      {h.reference_obj.page_id for h in hits}) if m.confirmed]  # type: ignore
                sites = {h.reference_obj.domain for h in hits  # type: ignore[attr-defined]
                         if surname[0] and surname[0] in _norm(_web_line(h.reference_obj)).split()}  # type: ignore
                ident.evidence["web_references"] = [_web_line(h.reference_obj) for h in hits]  # type: ignore
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
            rejected = check_facts(ident)
            if rejected:   # a missing fact closes its row; a wrong one would be printed on the card
                ident.evidence["rejected_facts"] = rejected
                log.warning("facts rejected for %s — %s: %s", ident.artist, ident.title, rejected)
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


_DANGLING = re.compile(r"\b(on|and|with|of|in|over|en|sur|auf|su|y|et|und)\s*[|,;:.\u2026-]*\s*$|\.\.\.$|\u2026$", re.I)


def clean_artist(name: str) -> str:
    """A name only: captions write "Tom Lovell (American, 1909-1997)" or "Eyvind Earle, 1916-2000"; the
    bracket or trailing dates are the artist's facts, which come from Wikidata, not part of the name."""
    name = re.sub(r"\s*\([^)]*\)", "", str(name or ""))
    name = re.sub(r"[\s,]+(b\.\s*|c\.\s*)?\d{4}(\s*[-\u2013]\s*\d{4})?\s*$", "", name)
    return re.sub(r"\s+", " ", name).strip(" ,;")


def check_facts(ident) -> dict:
    """Blank the facts read from captions that cannot be right, and say why. A date made of the artist's
    birth and death years is their life, not the work's; a date outside that life is another work's; a
    medium ending mid-phrase ("Oil on") was cut off by a page title."""
    rejected = {}
    born, died = (int(v) if str(v or "").isdigit() else None for v in (ident.artist_born, ident.artist_died))
    years = [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", ident.date or "")]
    if years and born and died and sorted(set(years)) == sorted({born, died}):
        rejected["date"] = f"{ident.date!r} is the artist's life ({born}-{died})"
    elif years and ((born and min(years) < born) or (died and min(years) > died)):
        rejected["date"] = f"{ident.date!r} falls outside the artist's life ({born or '?'}-{died or '?'})"
    if ident.medium and _DANGLING.search(ident.medium.strip()):
        rejected["medium"] = f"{ident.medium!r} is cut off mid-phrase"
    for k in rejected:
        setattr(ident, k, "")
    return rejected


def _web_line(r) -> str:
    fname = re.sub(r"[+_%20-]+", " ", (r.image_url.split("?")[0].rsplit("/", 1)[-1]).rsplit(".", 1)[0])
    return f"file: {fname[:80]} | page: {r.file_title[:120]} | site: {r.domain}"


def _mdict(m: Match) -> dict:
    d = {k: v for k, v in m.__dict__.items() if k != "reference_obj"}
    return d
