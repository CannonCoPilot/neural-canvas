"""Stage 1 — identify the artwork from its pixels alone, by collecting evidence and then ruling on it.

FIRST LOOK runs every arm at once: Bing and Yandex visual search, one Google Lens search from the rotating
free-credit pool, the cached picture index and the style shortlist. RESCUE (one model round, then searches by
the names it gives) runs only when no copy of the picture was verified. Every arm adds typed items to one
evidence ledger per work (``evidence``): copies verified on two instruments (``matching``), page captions,
the engines' own answers, catalogue records and model leads. Nothing refuses along the way.

Then two local model agents rule (``agents``): the adjudicator states artist and title citing ledger items,
the checker verifies every citation, and mechanical rules decide: a picture verified on both instruments,
and each fact stated by two independent items, one of them tied to the picture. Between rounds the aimed
searches run (confirm by the ruled names, the artist's cached catalogue, the agents' own requests). A work
the evidence cannot settle stays UNCONFIRMED, blocks, and alerts (Sir's redesign, 2026-11-07).
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
from .lens_pool import Work as LensWork, build_pool
from .agents import Agents, _fact, states
from .evidence import Ledger
from .evidence import site_rank
from .websearch import WebImageSearch, WebReference
from .vlm import VLM

log = logging.getLogger(__name__)
WEB_UA = "Mozilla/5.0 (compatible; neural-canvas/1.0; +https://github.com/CannonCoPilot/neural-canvas)"
YANDEX_TOP = 60
ARM_TIMEOUT_S = 240   # a first-look arm with no answer by then is recorded as an error
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
        m = cfg["models"]
        self.agents = Agents(vlm, VLM(m["ollama_url"], m.get("checker", "qwen3:8b"), m.get("vlm_timeout_s", 900)))
        self.rounds_max = int(m.get("rounds_max", 3))
        self.alert = None          # set by the service: a Pulse alert when a work cannot be settled

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
               seen: set[int], tier: str = "") -> list[Match]:
        """Score references on both instruments. Every reference lands in the ledger: a verified one as a copy,
        the rest as captions with their embedding score (what a page says is evidence even when its picture
        could not be checked)."""
        refs = [r for r in refs if r.page_id not in seen]
        seen.update(r.page_id for r in refs)
        def get(r):
            if isinstance(r, WebReference):
                return fetch_image(r.image_url, WEB_UA) or fetch_image(r.thumb_url, WEB_UA)
            return fetch_image(r.thumb_url, self.ua)
        with ThreadPoolExecutor(FETCH_WORKERS) as ex:   # rate-limited hosts are paced inside fetch_image
            got = list(zip(refs, ex.map(get, refs)))
        pairs = [(r, im) for r, im in got if im is not None]
        led = getattr(self, "ledger", None)
        src = lambda r: (getattr(r, "meta", None) or {}).get("source") or tier or query[:40]   # noqa: E731
        if led is not None:
            for r, im in got:
                if im is None:
                    led.add_copy(r, src(r), 0.0, 0, False)
        if not pairs:
            return []
        sims = self.emb.similarity(img_vec, self.emb.embed([im for _, im in pairs]))
        order = np.argsort(-sims)
        out = []
        for rank, i in enumerate(order):
            r, im = pairs[i]
            sim = float(sims[i])
            # geometry for the closest few, and for any other copy close enough to be the same picture
            near = sim >= self.t["id_clip_min"] - (0.08 if rank < 4 else 0.0)
            inl = geometric_inliers(img, im) if near and rank < 12 else 0
            ok = sim >= self.t["id_clip_min"] and inl >= self.t["id_inliers_min"]
            if led is not None:
                led.add_copy(r, src(r), sim, inl, ok)
            if rank < 4 or ok:
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

    def _propose(self, img: Image.Image) -> dict:
        note = ""
        texts = [i.text for i in self.ledger.items if i.kind in ("engine", "caption", "copy")][:25]
        if texts:
            note = ("Reverse-image searches of the web returned these captions and answers for this picture "
                    "(none of their pictures matched it exactly): " + "; ".join(texts)[:1500]
                    + ". These are leads, not facts.")
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
        self.ledger = Ledger()
        self._lens_log = {"providers": [], "errors": []}
        self._timing: dict[str, float] = {}
        self._signature: dict = {}
        self._confirmed_pairs: set[tuple[str, str]] = set()
        self._anchor_queries: set[str] = set()
        self._lens_more = False
        self._confirmation: dict = {}
        self._record_qids: set[str] = set()
        self._verified: list[Match] = []      # every copy verified on both instruments, from any search
        try:
            ident = self._identify(img, pool)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        ident.evidence["vision_web_calls"] = (self.vision.calls - calls0) if self.vision else 0
        ident.evidence["lens_slot"] = self._lens_log
        ident.evidence["seconds_by_arm"] = self._timing
        ident.evidence["confirmation"] = self._confirmation
        ident.evidence["ledger"] = self.ledger.to_list()
        sig = self._signature or {}
        it = str(sig.get("inscribed_title") or "").strip()
        if it and sig.get("inscribed_title_legible") is True and "?" not in it:
            ident.inscribed_title = it
        ident.evidence["signature"] = sig
        if ident.status != "confirmed" and self.alert is not None:
            # a work the evidence could not settle stays OPEN and blocks; the approach needs another look
            try:
                self.alert(f"identification unsettled: {ident.evidence.get('reason', '')}"[:300],
                           {"verified_copies": len(self.ledger.verified()), "items": len(self.ledger.items)})
            except Exception as e:
                log.warning("identify alert failed: %s", e)
        return ident

    def _timed(self, name: str, fn):
        t = time.time()
        try:
            return fn()
        finally:
            self._timing[name] = round(self._timing.get(name, 0.0) + time.time() - t, 2)

    def _arm(self, tier: str, refs, img, img_vec, seen, matches) -> list[Match]:
        """Score one arm's references into the ledger; a dead source is recorded, never read as "no match"."""
        try:
            batch = self._timed(f"verify:{tier}", lambda: self._score(img, img_vec, refs() if callable(refs) else refs,
                                                                     tier, seen, tier))
        except Exception as e:
            log.warning("%s search failed: %s", tier, e)
            matches.append(Match(f"ERROR {tier}: {str(e)[:200]}", "", "", 0.0, 0, False, tier))
            return []
        matches += batch
        hits = [m for m in batch if m.confirmed]
        for h in hits:      # an Artvee copy's caption is the page title: the artist is read from the page
            r = h.reference_obj  # type: ignore[attr-defined]
            if "artvee.com/dl/" in (r.page_url or "") and " by " not in (r.file_title or ""):
                try:
                    artist = self.artvee.page_artist(r.page_url)
                    if artist:
                        self.ledger.add("record", "artvee", f"Artvee page names the artist: {artist}", r.page_url,
                                        claims={"artist": artist})
                except Exception as e:
                    log.info("artvee page %s unread: %s", r.page_url, e)
        if hits:
            self._verified += hits
            self._first_tier = getattr(self, "_first_tier", "") or tier
            log.info("  %s: %d verified cop%s", tier, len(hits), "y" if len(hits) == 1 else "ies")
        return hits

    def _identify(self, img: Image.Image, pool: ThreadPoolExecutor) -> Identification:
        """FIRST LOOK on every work, all arms at once: Bing and Yandex visual search, one Google Lens search from
        the rotating free-credit pool, our cached picture index, and the style shortlist. Every arm adds to the
        ledger; none refuses. RESCUE runs only when no copy was verified: one model round, then searches by the
        names it gives. Then the agents rule on the facts (Sir's redesign, 2026-11-07)."""
        img_vec = self.emb.embed([img])[0]
        self._img_vec, self._img = img_vec, img
        self._first_tier = ""
        seen: set[int] = set()
        matches: list[Match] = []
        hits: list[Match] = []
        arms: dict = {}
        work = LensWork(img, self.lens.url_fn) if self.lens is not None else None
        if self.reverse is not None:
            arms[pool.submit(self._timed, "bing", lambda: self._bing(img))] = "bing"
            # Yandex returns up to ~340 candidates for a famous work; its first 60 hold the copies (agent trial)
            arms[pool.submit(self._timed, "yandex", lambda: self._yandex(img))] = "yandex"
        if self.lens is not None:
            arms[pool.submit(self._timed, "google_lens", lambda: self._lens_refs(self.lens.search(img, work=work), work))] = "google_lens"
        try:
            style = self._timed("style", lambda: self.style.top(img_vec, 4))
        except Exception as e:
            log.warning("style shortlist unavailable: %s", e)
            style = []
        cands = [{"artist": n, "title": "", "year": "", "confidence": 0, "source": "style"} for n, _ in style]
        for n, s in style:
            self.ledger.add("lead", "style", f"style look-alike: {n} (similarity {s:.2f})")
        log.info("style shortlist: %s", [n for n, _ in style])
        if self.index is not None:
            hits += self._arm("cached_index", lambda: self._timed("cached_index", lambda: self.index.nearest(img_vec, 10)),
                              img, img_vec, seen, matches)
        lens_retried = False
        while arms:
            done, _ = wait(list(arms), timeout=ARM_TIMEOUT_S, return_when=FIRST_COMPLETED)
            if not done:
                log.warning("first look arms still out after %d s: %s", ARM_TIMEOUT_S, list(arms.values()))
                matches += [Match(f"ERROR {n}: no answer in {ARM_TIMEOUT_S} s", "", "", 0.0, 0, False, n) for n in arms.values()]
                break
            for f in done:
                name = arms.pop(f)
                got = self._arm(name, f.result, img, img_vec, seen, matches)
                hits += got
                if name == "google_lens" and not got and not lens_retried:
                    # hit rates differ by provider: one more provider when this one verified nothing
                    lens_retried = True
                    tried = set(self._lens_log["providers"]) | {e.split(":", 1)[0] for e in self._lens_log["errors"]}
                    if self.lens is not None and len(tried) < len(self.lens.providers):
                        w2 = LensWork(img, self.lens.url_fn)
                        g = pool.submit(self._timed, "google_lens",
                                        lambda: self._lens_refs(self.lens.search(img, tried, work=w2), w2))
                        arms[g] = "google_lens"
        rounds: list = [{"round": 1, "stage": "first look", "candidates": cands,
                         "best_matches": [_mdict(m) for m in sorted(matches, key=lambda m: (-m.clip, -m.inliers))[:5]]}]
        if not hits:
            hits = self._rescue(img, img_vec, seen, matches, rounds, cands)
        rounds[-1]["confirmed_by"] = [_mdict(h) for h in hits]
        sig = (self._signature or {}).get("signature_text", "")
        summary = getattr(self, "_summary", "")
        if not hits:
            best = sorted(matches, key=lambda m: (-m.clip, -m.inliers))[:5]
            rounds[-1]["best_matches"] = [_mdict(m) for m in best]
            return Identification(status="unconfirmed", rounds=rounds, signature_text=sig, visual_summary=summary,
                                  evidence={"reason": "no reference image matched on both instruments",
                                            "errors": [m.reference for m in matches if m.reference.startswith("ERROR")]})
        self._records(hits)
        return self._adjudicate(img, hits, matches, rounds, cands, sig, summary)

    def _adjudicate(self, img, hits, matches, rounds, cands, sig, summary) -> Identification:
        """The agents rule on artist and title from the ledger (agents.settle). Between rounds the aimed searches
        run: confirm by the ruled names, the ruled artist's cached catalogue, and any search the agents ask for."""
        best = max(hits, key=lambda m: (m.inliers, m.clip))
        ident = Identification(status="unconfirmed", rounds=rounds, signature_text=sig, visual_summary=summary,
                               evidence={"tier": self._first_tier, **_mdict(best)})
        try:
            verdict = self._timed("adjudicate", lambda: self.agents.settle(
                img, self.ledger, need=("artist", "title"), rounds_max=self.rounds_max,
                search=lambda ruling, qs: self._aimed(ruling, qs, rounds)))
        except Exception as e:          # an agent that fails leaves the work open; it never guesses
            log.error("adjudication failed: %s", e)
            ident.evidence["reason"] = f"adjudication failed: {type(e).__name__}: {str(e)[:200]}"
            return ident
        ident.evidence["adjudication"] = verdict.pop("rounds", [])
        ident.evidence["verdict"] = verdict
        facts = verdict["facts"]
        ident.artist, ident.title = clean_artist(facts.get("artist", "")), facts.get("title", "")
        ident.date, ident.medium = facts.get("date", ""), facts.get("medium", "")
        ident.collection = facts.get("collection", "")
        ident.facts_source = "evidence"
        if verdict["failures"]:
            ident.evidence["reason"] = "; ".join(verdict["failures"])[:400]
            return ident
        ident.status = "confirmed"
        if (_norm(ident.artist), _norm(ident.title)) not in self._confirmed_pairs:  # Sir, 2026-11-06: confirm every found work
            # confirm only: the anchors' own words are searched between failing rounds, not after a ruling stands
            # (bench pass 8: 139 s of searches on a work already confirmed)
            self._aimed({"artist": {"value": ident.artist}, "title": {"value": ident.title}}, [], rounds, anchors=False)
        rounds[-1]["confirmed_by"] = [_mdict(m) for m in self._verified]
        qid = next((r.claims.get("wikidata") for r in self.ledger.items if r.kind == "record" and r.claims.get("wikidata")
                    and _norm(r.claims.get("artist", "")).split()[-1:] == _norm(ident.artist).split()[-1:]), "")
        ident.wikidata = qid or ""
        ident.__dict__.update(self._artist_facts(ident.artist))
        rejected = check_facts(ident)
        if rejected:   # a missing fact closes its row; a wrong one would be printed on the card
            ident.evidence["rejected_facts"] = rejected
            log.warning("facts rejected for %s — %s: %s", ident.artist, ident.title, rejected)
        ident.evidence["web_references"] = list(dict.fromkeys(
            _web_line(m.reference_obj) for m in self._verified  # type: ignore[attr-defined]
            if isinstance(getattr(m, "reference_obj", None), WebReference)))
        # artist leads for the catalogue, best first: the ruled artist, then the model's names
        ident.evidence["artist_leads"] = list(dict.fromkeys(
            a for a in [ident.artist, *[c["artist"] for c in cands if c.get("source") != "style"]] if a))[:5]
        return ident

    def _aimed(self, ruling: dict, queries: list[str], rounds: list, anchors: bool = True) -> int:
        """Searches aimed by a ruling: the three confirm searches (once per work), the ruled artist's cached
        catalogue, and the agents' own requests. Returns how many items the ledger gained."""
        n0 = len(self.ledger.items)
        artist, title = _fact(ruling, "artist")[0], _fact(ruling, "title")[0]
        found: list[Match] = []
        if anchors and self.lens is not None and not self._lens_more:
            # an unsettled work asks one more Lens provider: Cloud Vision returns no web results, and Visser was
            # settled only on runs where a text-rich provider answered (bench passes 4-8). One credit, 10% floor.
            self._lens_more = True
            tried = set(self._lens_log["providers"]) | {e.split(":", 1)[0] for e in self._lens_log["errors"]}
            if len(tried) < len(self.lens.providers):
                w = LensWork(self._img, self.lens.url_fn)
                try:
                    refs = self._timed("google_lens_more", lambda: self._lens_refs(self.lens.search(self._img, tried, work=w), w))
                    self._records(self._arm("google_lens_more", refs, self._img, self._img_vec, set(), []))  # _arm keeps them
                except Exception as e:      # recorded; an unavailable pool already alerted
                    log.warning("second Lens provider failed: %s", e)
        if artist and title and (_norm(artist), _norm(title)) not in self._confirmed_pairs:
            # once per ruled pair: a feedback round that turns the ruling gets confirm searches of its own
            self._confirmed_pairs.add((_norm(artist), _norm(title)))
            found += self._confirm(artist, title)
        if artist and self._oeuvre_cached(artist):
            try:
                found += [m for m in self._score(self._img, self._img_vec, self._oeuvre_refs(artist, self._img_vec),
                                                 f"oeuvre:{artist}", set(), "oeuvre") if m.confirmed]
            except Exception as e:
                log.info("no catalogue look-up for %r: %s", artist, e)
        if anchors and self.web is not None and (artist or title):
            # the anchors' own words: a verified copy that names the ruled artist or title is searched by its caption,
            # which finds the other sites carrying that wording (James Ward, bench pass 7: Artvee's caption led to the
            # second and third sites that the ruled names alone did not)
            for i in sorted(self.ledger.verified(), key=lambda i: -i.inliers)[:6]:
                if not ((artist and states(i, "artist", artist)) or (title and states(i, "title", title))):
                    continue
                q = re.split(r"\s+[|]\s+|\s+[-–—]\s+(?=[^-–—]*$)", i.text)[0].strip(" .…")
                if len(q) >= 12 and _norm(q) not in self._anchor_queries and len(self._anchor_queries) < 4:
                    self._anchor_queries.add(_norm(q))
                    queries = [q, *queries]
        if self.web is not None:
            for q in queries:
                try:
                    found += [m for m in self._timed("agent_search", lambda q=q: self._score(
                        self._img, self._img_vec, self.web.search(q, 20), q, set(), "agent_search")) if m.confirmed]
                except Exception as e:
                    log.warning("agent search %r failed: %s", q, e)
        if found:
            self._verified += found
            self._records(found)
            rounds[-1]["confirmed_by"] = [_mdict(m) for m in self._verified]
        log.info("aimed searches (%s — %s; %d asked): %d new items", artist, title, len(queries), len(self.ledger.items) - n0)
        return len(self.ledger.items) - n0

    def _records(self, hits: list[Match]):
        """Catalogue records for verified copies that are catalogue entries: a Wikidata item (the cached index, an
        artist's catalogue), another indexed collection's row, or a Commons file. Each becomes a record in the
        ledger, never a ruling: a Wikidata item can carry the wrong picture (Vallotton's "Route from Goubing to
        Sierre", withheld bench 2026-11-06)."""
        keep = ("title", "artist", "date", "medium", "collection")
        for h in sorted(hits, key=lambda m: -m.inliers)[:6]:
            ref = h.reference_obj  # type: ignore[attr-defined]
            meta = getattr(ref, "meta", None) or {}
            qid = meta.get("wikidata", "")
            if not qid and not isinstance(ref, WebReference) and ref.page_id < 10**12:
                try:
                    qid = self.wiki.depicted_item(ref.page_id) or ""
                except Exception as e:
                    log.info("no Wikidata item for %s: %s", ref.file_title, e)
            if qid:
                if qid in self._record_qids:
                    continue
                self._record_qids.add(qid)
                try:
                    facts = self.wiki.artwork_facts(qid) or {}
                except Exception as e:
                    log.info("Wikidata %s unread: %s", qid, e)
                    facts = {}
                stated = "; ".join(f"{k}: {facts[k]}" for k in keep if facts.get(k)) or "no title or creator recorded"
                self.ledger.add("record", "wikidata", f"Wikidata item {qid}, whose image is this picture: {stated}",
                                f"https://www.wikidata.org/wiki/{qid}", overrule=True,
                                claims={**{k: facts[k] for k in keep if facts.get(k)}, "wikidata": qid})
            elif meta.get("artist") or meta.get("title"):      # a row of another indexed collection (NGA, Artvee)
                self.ledger.add("record", meta.get("source", "index"),
                                f"indexed collection lists this picture as {meta.get('title', '')!r} by {meta.get('artist', '')}",
                                ref.page_url, claims={"title": meta.get("title", ""), "artist": meta.get("artist", "")})
            elif not isinstance(ref, WebReference) and meta:     # a Commons file: its own metadata
                text = "; ".join(f"{k}: {str(meta.get(k, ''))[:160]}" for k in ("ObjectName", "Artist", "ImageDescription",
                                                                                "DateTimeOriginal") if meta.get(k))
                if text:
                    self.ledger.add("record", "commons", f"Commons file {ref.file_title}: {text}", ref.page_url)

    def _bing(self, img) -> list:
        refs, guess = self.reverse.bing(img)
        if guess:
            self.ledger.add("engine", "bing", f"Bing's best guess for this picture: {guess}")
        return refs

    def _yandex(self, img) -> list:
        tags: list[str] = []
        refs = self.reverse.yandex(img, tags)[:YANDEX_TOP]
        for t in tags:
            self.ledger.add("engine", "yandex", f"Yandex tag for this picture: {t}")
        return refs

    def _rescue(self, img, img_vec, seen, matches, rounds, style_cands) -> list[Match]:
        """First look verified nothing. One model round names candidates (with the signature, the style shortlist
        and what the engines said); free searches by those names run first, then searches that spend credits.
        Stops at the first tier that verifies a copy."""
        self._signature = self._timed("signature", lambda: self.read_signature(img))
        try:
            prop = self._timed("propose", lambda: self._propose(img))
        except Exception as e:
            log.warning("model round failed: %s", e)
            prop = {}
        self._summary = prop.get("visual_summary", "")
        summary = self._summary
        cands = [c for c in prop.get("candidates", []) if c.get("artist")][:5]
        for c in cands:
            self.ledger.add("lead", "model", f"model guess: {c.get('artist')} — {c.get('title', '')} ({c.get('year', '')})")
        sa = (self._signature.get("artist") or "").strip()
        if self._signature.get("signature_text"):
            self.ledger.add("signature", "model", f"signature read as {self._signature['signature_text']!r}"
                            + (f", perhaps {sa}" if sa else ""))
        for c in ([{"artist": sa, "title": "", "source": "signature"}] if sa else []) + style_cands:
            if _norm(c["artist"]) not in {_norm(x["artist"]) for x in cands}:
                cands.append(c)
        log.info("rescue candidates: %s", [(c["artist"], c.get("title")) for c in cands])
        rounds.append({"round": 2, "stage": "rescue", "candidates": cands})
        artists = list(dict.fromkeys(c["artist"] for c in cands))
        titled = [c for c in cands if c.get("title")]
        insc = (self._signature.get("inscribed_title") or "").strip()
        tiers = []
        # free: Artvee, Wikidata titles, Commons titles, the web by guess, cached catalogues
        for q in dict.fromkeys([insc] + [c["title"] for c in titled]):
            if len(q) >= 4:
                tiers.append(("artvee", lambda q=q: self.artvee.search(q)))
        for a in artists[:3]:
            tiers.append(("artvee", lambda a=a: self.artvee.nearest(a, img_vec, 6)))
        for c in titled[:4]:
            tiers.append(("wikidata_title", lambda c=c: self._wikidata_title_refs(c["artist"], c["title"])))
        for c in titled[:4]:
            tiers.append(("commons", lambda q=f"{c['title']} {c['artist']}": self.wiki.search_files(q, 12)))
        if self.web:
            for c in titled[:4]:
                tiers.append(("web", lambda q=f"{c['artist']} {c['title']} painting": self.web.search(q, 15)))
        for a in artists[:4]:
            if self._oeuvre_cached(a):
                tiers.append(("oeuvre", lambda a=a: self._oeuvre_refs(a, img_vec)))
        # credits, cheapest first; each pool stops at 10% of its free allowance
        if self.lens is not None:
            for name, tier in (("searchapi_yandex", "yandex_searchapi"), ("serpapi_exact", "lens_exact")):
                def run(name=name, tier=tier):
                    w = LensWork(img, self.lens.url_fn)
                    try:
                        return self.lens.extra(name, img, w)
                    finally:
                        for t in w.texts:
                            self.ledger.add("engine", t.get("source", tier), t["text"], t.get("url", ""), t.get("site", ""))
                tiers.append((tier, run))
        if self.web is not None:
            for c in titled[:2]:
                tiers.append(("google_images", lambda q=f"{c['artist']} {c['title']}": self.web.serpapi_images(q)))
            if summary:
                tiers.append(("google_images_desc", lambda: self.web.index_images(summary[:120], 30)))
        for tier, fn in tiers:
            log.info("  rescue %s", tier)
            hits = self._arm(tier, lambda fn=fn, tier=tier: self._timed(tier, fn), img, img_vec, seen, matches)
            if hits:
                return hits
        return []

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

    def _lens_refs(self, res, work) -> list:
        """The Lens slot's answer: its references to score, its text answers into the ledger, and the provider
        and any failed providers kept for the run record."""
        prov, refs, errs = res
        self._lens_log["providers"].append(prov)
        self._lens_log["errors"] += errs
        for t in work.texts:
            self.ledger.add("engine", t.get("source", f"google_lens:{prov}"), t["text"], t.get("url", ""), t.get("site", ""))
        return refs

    def _confirm(self, artist: str, title: str) -> list[Match]:
        """CONFIRM (Sir, 2026-11-06: all three on every found work): the ruled artist and title searched on the
        web, in Google Images through SerpAPI, and in Google's image index through Serper. Everything they
        return joins the ledger; credits stop at each pool's 10% floor."""
        if self.web is None:
            return []
        q = f"{artist} {title}"
        found: list[Match] = []
        for name, fn in (("web_images", lambda: self.web.search(f"{q} painting", 20)),
                         ("google_images", lambda: self.web.serpapi_images(q)),
                         ("google_images_desc", lambda: self.web.index_images(q, 30))):
            try:
                ms = self._timed(f"confirm:{name}", lambda fn=fn, name=name: self._score(self._img, self._img_vec, fn(), q,
                                                                                        set(), f"confirm:{name}"))
                hits = [m for m in ms if m.confirmed]
                self._confirmation[name] = {"verified": len(hits), "best_clip": max((m.clip for m in ms), default=0.0)}
                found += hits
            except Exception as e:     # recorded, never read as "no match"
                self._confirmation[name] = {"error": str(e)[:200]}
        return found


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


_site_rank = site_rank      # kept under its old name for callers


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
