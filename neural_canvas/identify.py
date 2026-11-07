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
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .matching import Embedder, fetch_image, geometric_inliers, save_npz
from .references import Reference, Wikimedia, file_thumb
from .artsites import Artvee
from .global_index import GlobalIndex
from .reverse_image import ReverseImage
from .imify import Imify
from .style import StyleShortlist
from .vision_web import VisionWebDetection
from .lens_pool import build_pool
from .websearch import WebImageSearch, WebReference
from .vlm import VLM

log = logging.getLogger(__name__)
WEB_UA = "Mozilla/5.0 (compatible; neural-canvas/1.0; +https://github.com/CannonCoPilot/neural-canvas)"
YANDEX_TOP = 60
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
        # first look's free arms (Sir, 2026-11-06): Bing and Yandex visual search called directly (the Imify
        # service filtered out good Yandex matches and its Bing engine landed on the homepage), and the
        # cached picture index. Arthur.io, walled catalogues and Commons keyword searches are cut.
        self.reverse = ReverseImage(self.ua) if web is not None else None
        self.index = GlobalIndex(cfg.path("state_dir"))
        # the Lens slot (Sir, 2026-11-06): one Google Lens-type search per work from a rotating free-credit pool
        self.lens = build_pool(cfg, web, vision) if web is not None and hasattr(web, "temp_public_url") else None

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
    def identify(self, img: Image.Image) -> Identification:
        calls0 = self.vision.calls if self.vision else 0
        pool = ThreadPoolExecutor(4)      # first look's web arms (Bing, Yandex, the Lens slot) run at once
        self._pending: dict[str, object] = {}
        self._lens_log = {"providers": [], "errors": []}
        self._timing: dict[str, float] = {}
        try:
            ident = self._identify(img, pool)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        for name, fut in self._pending.items():   # confirmed before these answered: their calls are spent anyway
            ident.evidence.setdefault("unused_arms", []).append(name)
            fut.add_done_callback(lambda f, n=name: log.info("late %s result (unused): %s", n, f.exception() or "answered"))
        ident.evidence["vision_web_calls"] = (self.vision.calls - calls0) if self.vision else 0
        ident.evidence["lens_slot"] = self._lens_log
        ident.evidence["seconds_by_arm"] = self._timing
        sig = getattr(self, "_signature", None) or {}
        it = str(sig.get("inscribed_title") or "").strip()
        if it and sig.get("inscribed_title_legible") is True and "?" not in it:
            ident.inscribed_title = it
        ident.evidence["signature"] = sig
        return ident

    def _timed(self, name: str, fn):
        t = time.time()
        try:
            return fn()
        finally:
            self._timing[name] = round(self._timing.get(name, 0.0) + time.time() - t, 2)

    def _try(self, tier: str, q: str, refs, img, img_vec, seen, matches, rounds, cands, sig, summary
             ) -> Identification | None:
        """Score one arm's references; a confirmed identification ends the search."""
        try:
            batch = self._timed(f"verify:{tier}", lambda: self._score(img, img_vec, refs() if callable(refs) else refs, q, seen))
        except Exception as e:      # a dead source is recorded in the run, never read as "no match"
            log.warning("%s search failed for %r: %s", tier, q, e)
            matches.append(Match(f"ERROR {tier}: {str(e)[:200]}", "", "", 0.0, 0, False, q))
            return None
        matches += batch
        hits = [m for m in batch if m.confirmed]
        if not hits:
            return None
        for h in hits:      # an Artvee copy's caption is the page title: the artist is read from the page
            r = h.reference_obj  # type: ignore[attr-defined]
            if "artvee.com/dl/" in (r.page_url or "") and " by " not in (r.file_title or ""):
                try:
                    artist = self.artvee.page_artist(r.page_url)
                    if artist:
                        r.file_title = f"{re.sub(r' [-–] Artvee$', '', r.file_title or '')} by {artist}"
                except Exception as e:
                    log.info("artvee page %s unread: %s", r.page_url, e)
        # verified copies are judged most trusted first: a collection's or reference's caption alone, then every
        # copy together. Album covers, posters and shops reuse pictures under other names (withheld bench,
        # 2026-11-06: Spotify covers of a James Ward colour study named the band; Artvee named the work).
        groups: dict[int, list] = {}
        for h in hits:
            groups.setdefault(_site_rank(h.reference_obj.page_url), []).append(h)  # type: ignore[attr-defined]
        trials = [(f"site rank {r} alone", groups[r]) for r in sorted(groups) if r <= 2 and len(groups[r]) < len(hits)]
        for label, group in trials + [("every verified copy", hits)]:
            ident = self._finish(tier, group, matches, rounds, len(rounds), cands, sig, summary)
            log.info("  %s, %s (%d of %d): %s", tier, label, len(group), len(hits),
                     "confirmed" if ident.status == "confirmed" else ident.evidence.get("reason"))
            if ident.status == "confirmed":
                if group is not hits:
                    ident.evidence["hits_judged"] = f"{label} ({len(group)} of {len(hits)})"
                return ident
            rounds.pop()          # facts failed their gate
        return None

    def _identify(self, img: Image.Image, pool: ThreadPoolExecutor) -> Identification:
        """FIRST LOOK on every work, all arms at once: Bing and Yandex visual search, one Google Lens search
        from the rotating free-credit pool, our cached picture index, and the style shortlist (names only).
        Each arm is scored the moment it answers. RESCUE runs only when first look verified nothing: one model
        round, then free searches by the names it gives, then paid credits (Sir's plan, 2026-11-06). The model's
        repeated guessing rounds are gone: re-guessing alone identified 0 works."""
        img_vec = self.emb.embed([img])[0]
        self._img_vec, self._img = img_vec, img
        self._signature: dict = {}
        self._bing_guess = ""
        seen: set[int] = set()
        matches: list[Match] = []
        rounds: list = []
        sig, summary = "", ""
        arms: dict = {}
        if self.reverse is not None:
            arms[pool.submit(self._timed, "bing", lambda: self._bing(img))] = "bing"
            # Yandex returns up to ~340 candidates for a famous work; its first 60 hold the copies (agent trial)
            arms[pool.submit(self._timed, "yandex", lambda: self.reverse.yandex(img)[:YANDEX_TOP])] = "yandex"
        if self.lens is not None:
            arms[pool.submit(self._timed, "google_lens", lambda: self._lens_refs(self.lens.search(img)))] = "google_lens"
        self._pending = {n: f for f, n in arms.items()}
        try:
            style = self._timed("style", lambda: self.style.top(img_vec, 4))
        except Exception as e:
            log.warning("style shortlist unavailable: %s", e)
            style = []
        cands = [{"artist": n, "title": "", "year": "", "confidence": 0, "source": "style"} for n, _ in style]
        log.info("style shortlist: %s", [n for n, _ in style])
        if self.index is not None:
            ident = self._try("cached_index", "cached picture index",
                              lambda: self._timed("cached_index", lambda: self.index.nearest(img_vec, 10)),
                              img, img_vec, seen, matches, rounds, cands, sig, summary)
            if ident:
                return self._absorb(ident, arms, img, img_vec, seen)
        lens_retried = False
        while arms:
            done, _ = wait(list(arms), return_when=FIRST_COMPLETED)
            for f in done:
                name = arms.pop(f)
                self._pending.pop(name, None)
                ident = self._try(name, name, f.result, img, img_vec, seen, matches, rounds, cands, sig, summary)
                if ident:
                    return self._absorb(ident, arms, img, img_vec, seen)
                if name == "google_lens" and not lens_retried:
                    # hit rates differ by provider (Serper found 7 of 10): one more provider before rescue
                    lens_retried = True
                    tried = set(self._lens_log["providers"]) | {e.split(":", 1)[0] for e in self._lens_log["errors"]}
                    if self.lens is not None and len(tried) < len(self.lens.providers):
                        g = pool.submit(self._timed, "google_lens", lambda: self._lens_refs(self.lens.search(img, tried)))
                        arms[g] = "google_lens"
                        self._pending["google_lens"] = g
        rounds.append({"round": 1, "stage": "first look", "candidates": cands,
                       "best_matches": [_mdict(m) for m in sorted(matches, key=lambda m: (-m.clip, -m.inliers))[:5]]})
        return self._rescue(img, img_vec, seen, matches, rounds, cands)

    def _absorb(self, ident: Identification, arms: dict, img, img_vec, seen, timeout_s: float = 90) -> Identification:
        """The other first-look arms are already running and paid for: their verified copies join the evidence the
        catalogue reads (confirmed_by). Without them the card depended on which arm answered first (withheld
        bench, 2026-11-07: Bing won the race for a Jawlensky, and the Artvee and Wikimedia copies Lens had found,
        which ruled the facts in the passes before, never reached the catalogue)."""
        from concurrent.futures import TimeoutError as Timeout, as_completed
        got = ident.rounds[-1].setdefault("confirmed_by", []) if ident.rounds else []
        lines = list(ident.evidence.get("web_references") or [])
        also = []
        try:
            for f in as_completed(list(arms), timeout=timeout_s):
                name = arms.pop(f)
                self._pending.pop(name, None)
                try:
                    hits = [m for m in self._timed(f"verify:{name}", lambda: self._score(img, img_vec, f.result(), name, seen))
                            if m.confirmed]
                except Exception as e:
                    log.info("  %s (after confirmation) failed: %s", name, e)
                    continue
                if hits:
                    also.append(name)
                    got += [_mdict(h) for h in hits]
                    lines += [_web_line(h.reference_obj) for h in hits if isinstance(h.reference_obj, WebReference)]  # type: ignore
        except Timeout:
            log.info("  first look arms still out after %.0f s: %s", timeout_s, list(arms.values()))
        ident.evidence["web_references"] = list(dict.fromkeys(lines))
        ident.evidence["also_verified_by"] = also
        return ident

    def _bing(self, img) -> list:
        refs, guess = self.reverse.bing(img)
        self._bing_guess = guess
        return refs

    def _rescue(self, img, img_vec, seen, matches, rounds, style_cands) -> Identification:
        """First look verified nothing. One model round names candidates (with the signature and the style
        shortlist); free searches by those names run first, then searches that spend credits."""
        self._signature = self._timed("signature", lambda: self.read_signature(img))
        self._phrases, self._used_phrases, self._vision_hints = [], set(), {}
        try:
            prop = self._timed("propose", lambda: self._propose(img, []))
        except Exception as e:
            log.warning("model round failed: %s", e)
            prop = {}
        sig = self._signature.get("signature_text", "") or prop.get("signature_text", "")
        summary = prop.get("visual_summary", "")
        cands = [c for c in prop.get("candidates", []) if c.get("artist")][:5]
        sa = (self._signature.get("artist") or "").strip()
        for c in ([{"artist": sa, "title": "", "source": "signature"}] if sa else []) + style_cands:
            if _norm(c["artist"]) not in {_norm(x["artist"]) for x in cands}:
                cands.append(c)
        log.info("rescue candidates: %s", [(c["artist"], c.get("title")) for c in cands])
        artists = list(dict.fromkeys(c["artist"] for c in cands))
        titled = [c for c in cands if c.get("title")]
        insc = (self._signature.get("inscribed_title") or "").strip()
        tiers = []
        # free: Artvee, Wikidata titles, Commons titles, the web by guess, cached catalogues
        for q in dict.fromkeys([insc] + [c["title"] for c in titled]):
            if len(q) >= 4:
                tiers.append(("artvee", f"artvee: {q}", lambda q=q: self.artvee.search(q)))
        for a in artists[:3]:
            tiers.append(("artvee", f"artvee catalogue: {a}", lambda a=a: self.artvee.nearest(a, img_vec, 6)))
        for c in titled[:4]:
            tiers.append(("wikidata_title", f"wikidata: {c['artist']} {c['title']}",
                          lambda c=c: self._wikidata_title_refs(c["artist"], c["title"])))
        for c in titled[:4]:
            q = f"{c['title']} {c['artist']}"
            tiers.append(("commons", q, lambda q=q: self.wiki.search_files(q, 12)))
        if self.web:
            for c in titled[:4]:
                q = f"{c['artist']} {c['title']} painting"
                tiers.append(("web", q, lambda q=q: self.web.search(q, 15)))
        for a in artists[:4]:
            if self._oeuvre_cached(a):
                tiers.append(("oeuvre", f"oeuvre:{a}", lambda a=a: self._oeuvre_refs(a, img_vec)))
        # credits, cheapest first; each pool stops at 10% of its free allowance
        if self.lens is not None:
            tiers.append(("yandex_searchapi", "yandex via SearchAPI", lambda: self.lens.extra("searchapi_yandex", img)))
            tiers.append(("lens_exact", "Google Lens exact matches", lambda: self.lens.extra("serpapi_exact", img)))
        if self.web is not None:
            for c in titled[:2]:
                q = f"{c['artist']} {c['title']}"
                tiers.append(("google_images", q, lambda q=q: self.web.serpapi_images(q)))
            if summary:
                tiers.append(("google_images_desc", summary[:120], lambda: self.web.index_images(summary[:120], 30)))
        for tier, q, fn in tiers:
            log.info("  rescue %s: %s", tier, q[:80])
            ident = self._try(tier, q, lambda fn=fn, tier=tier: self._timed(tier, fn), img, img_vec, seen, matches,
                              rounds, cands, sig, summary)
            if ident:
                return ident
        best = sorted(matches, key=lambda m: (-m.clip, -m.inliers))[:5]
        rounds.append({"round": 2, "stage": "rescue", "candidates": cands, "best_matches": [_mdict(m) for m in best]})
        return Identification(status="unconfirmed", rounds=rounds, signature_text=sig, visual_summary=summary,
                              evidence={"reason": "no reference image matched on both instruments",
                                        "errors": [m.reference for m in matches if m.reference.startswith("ERROR")]})

    def _wikidata_title_refs(self, artist: str, title: str) -> list:
        """The artist's catalogued works whose title shares most words with the guessed title."""
        qid = self.wiki.find_artist(artist)
        if not qid:
            raise LookupError(f"no Wikidata artist record for {artist!r}")
        want = {w for w in _norm(title).split() if len(w) > 2}
        out = []
        for w in self.wiki.oeuvre(qid):
            have = set(_norm(w["label"]).split())
            if want and len(want & have) / len(want) >= 0.6:
                out.append(Reference(file_title=f"{w['label']} [{w['qid']}]", page_id=10**12 + int(w["qid"][1:]),
                                     thumb_url=file_thumb(w["file"], 500), page_url=f"https://www.wikidata.org/wiki/{w['qid']}",
                                     width=500, height=500, meta={"wikidata": w["qid"], "artist_wikidata": qid}))
        return out[:8]

    def _lens_refs(self, res) -> list:
        """The Lens slot's answer, with the provider and any failed providers kept for the run record."""
        prov, refs, errs = res
        self._lens_log["providers"].append(prov)
        self._lens_log["errors"] += errs
        return refs

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
        captions = True
        if tier in ("commons", "oeuvre", "wikidata_title") or ref.meta.get("wikidata"):
            qid = ref.meta.get("wikidata") or self.wiki.depicted_item(ref.page_id)
            facts = self.wiki.artwork_facts(qid) if qid else {}
            if facts.get("title") and facts.get("artist"):
                captions = False
                ident.facts_source = "wikidata"
                for k, v in facts.items():
                    setattr(ident, k, v)
            elif not isinstance(ref, WebReference):      # a Commons file: its own metadata names the work
                captions = False
                m = ref.meta
                parsed = self.vlm.ask_json(TITLE_PROMPT.format(
                    file_title=ref.file_title, object_name=m.get("ObjectName", ""),
                    description=m.get("ImageDescription", "")[:600], artist=m.get("Artist", "")[:200],
                    date=m.get("DateTimeOriginal", "")), required=("title", "artist"))
                ident.facts_source = "commons-metadata"
                ident.title, ident.artist = parsed["title"].strip(), clean_artist(parsed["artist"])
                ident.date = str(parsed.get("date", "")).strip()
            # else: a Wikidata row of the cached index whose item lacks a creator or title (Vallotton's
            # "Route from Goubing to Sierre" carries the Tower of Goubing's picture and no artist, withheld
            # bench 2026-11-06): the verified copies' captions are read instead
        if captions:
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
            # a witness is a page that can attribute art. Print shops and blogs usually name the painter (they carry
            # Wyeth's Crows in Winter); music and social pages name whoever reused the picture (two music pages
            # named a cover's band for a James Ward study, withheld bench 2026-11-07)
            sites = {h.reference_obj.domain for h in hits  # type: ignore[attr-defined]
                     if surname[0] and surname[0] in _norm(_web_line(h.reference_obj)).split()  # type: ignore
                     and _site_rank(h.reference_obj.page_url) <= 5}  # type: ignore[attr-defined]
            if not by_model and len(sites) < 2 and ident.artist and ident.title:
                # one more look, now that the facts name an artist and title: the artist's own site,
                # galleries and catalogues usually title the same image the same way
                q = f"{ident.artist} {ident.title}"
                hits = hits + [m for m in self._score(self._img, self._img_vec, self.web.search(q, 20), q,
                                                      {h.reference_obj.page_id for h in hits}) if m.confirmed]  # type: ignore
                sites = {h.reference_obj.domain for h in hits  # type: ignore[attr-defined]
                         if surname[0] and surname[0] in _norm(_web_line(h.reference_obj)).split()  # type: ignore
                         and _site_rank(h.reference_obj.page_url) <= 5}  # type: ignore[attr-defined]
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
        if ident.status == "confirmed":
            self._confirm(ident)
            # artist leads for the catalogue, best first: a wrong first name no longer hides the work from
            # the museums (the benchmark asked only the model's top 2, right for 13 of 24 works)
            ident.evidence["artist_leads"] = list(dict.fromkeys(      # the model's names; style look-alikes are not leads
                a for a in [ident.artist, *[c["artist"] for c in cands if c.get("source") != "style"]] if a))[:5]
        return ident

    def _confirm(self, ident: Identification):
        """CONFIRM (Sir, 2026-11-06: all three on every found work): the found artist and title searched on the
        web, in Google Images through SerpAPI, and in Google's image index through Serper. Copies that pass
        both instruments become more references for the catalogue; credits stop at each pool's 10% floor."""
        if self.web is None or not (ident.artist and ident.title):
            return
        q = f"{ident.artist} {ident.title}"
        seen: set[int] = set()
        out, refs = {}, list(ident.evidence.get("web_references", []) or [])
        for name, fn in (("web_images", lambda: self.web.search(f"{q} painting", 20)),
                         ("google_images", lambda: self.web.serpapi_images(q)),
                         ("google_images_desc", lambda: self.web.index_images(q, 30))):
            try:
                ms = self._timed(f"confirm:{name}", lambda fn=fn: self._score(self._img, self._img_vec, fn(), q, set(seen)))
                hits = [m for m in ms if m.confirmed]
                out[name] = {"verified": len(hits), "best_clip": max((m.clip for m in ms), default=0.0)}
                refs += [_web_line(h.reference_obj) for h in hits if isinstance(h.reference_obj, WebReference)]  # type: ignore
            except Exception as e:     # recorded, never read as "no match"
                out[name] = {"error": str(e)[:200]}
        ident.evidence["confirmation"] = out
        ident.evidence["web_references"] = list(dict.fromkeys(refs))

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


_RANK = {"curated": 0, "reference": 1, "market": 2, "unknown": 3, "blog": 4, "commercial": 5, "social": 6}
_ART_SITES = ("artvee.com", "wikiart.org", "the-athenaeum.org", "wga.hu", "artuk.org")


def _site_rank(url: str) -> int:
    """How far a page's caption can be trusted to name the work: collections first, shops and feeds last."""
    from .sources import classify
    u = (url or "").lower()
    if any(d in u for d in _ART_SITES):
        return 1
    if any(d in u for d in ("spotify.", "music.apple", "soundcloud", "deezer", "bandcamp", "youtube.", "amselcom",
                            "beatport", "tidal.com", "shazam", "last.fm", "discogs", "audiomack")):
        return 7      # music pages use paintings as covers and name the band (James Ward -> "Latteo")
    return _RANK.get(classify(u), 3)


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
