"""The docent pipeline: ingest → identify → interpret → compose → upscale → label → deliver.

Every stage has a gate. A failed gate stops the run with status ``blocked`` and the stage's reasons;
there is no fallback that lets a below-threshold artwork through. Stage results are kept in the
run record, so ``resume`` re-runs only what did not pass (e.g. delivery after OAuth consent).

The pipeline never sees the original file name: ingest copies the image to ``<work>/<id>/source.<ext>``
(``id`` = content hash) and only that path reaches the stages.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .config import Config
from .borders import trim_borders
from .crop import Composer, CropPlan
from .dedup import ArtworkRegistry, phash
from .deliver import AuthRequired, GooglePhotos
from .identify import Identifier
from .larger import LargerSource, needed_scale
from .interpret import Interpretation, Subject, interpret
from .catalogue import Catalogue, catalogue_stage
from .colour import rebalance
from .identify import WEB_UA
from .matching import Embedder, fetch_image
from .placard import card_margin, card_size, compose, viewer_grid
from .references import Wikimedia
from .upscale import Upscaler
from .vision_web import VisionWebDetection
from .lens_pool import Ledger
from .vlm import VLM
from .websearch import WebImageSearch

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".tif", ".tiff"}
STAGES = ["identify", "catalogue", "interpret", "colour", "compose", "upscale", "label", "deliver"]


def content_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ingest(cfg: Config, src: Path) -> tuple[str, Path]:
    rid = content_id(src)
    d = cfg.path("work_dir") / rid
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"source{src.suffix.lower()}"
    if not dst.exists():
        shutil.copy2(src, dst)
    return rid, dst


def load_image(path: Path) -> Image.Image:
    """Pixels only: EXIF orientation is applied, every descriptive tag is dropped."""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:70]


@dataclass
class RunResult:
    run_id: str
    status: str                  # complete | blocked | error
    stage: str = ""              # stage that blocked
    reasons: list = field(default_factory=list)
    record_path: str = ""
    frame_path: str = ""


# The service runs works in parallel threads (one Docent each). Delivery is the one step that must be
# serial: the duplicate check reads the album's state, and two photographs of one painting finishing
# together would otherwise both pass it and both be delivered.
_DELIVERY = threading.Lock()


class RunStopped(Exception):
    """Raised by an on_stage callback to end a run between stages (its task was closed on the board)."""


class Docent:
    """Holds the loaded models; one instance serves many runs."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        m, s = cfg["models"], cfg["sources"]
        self.vlm = VLM(m["ollama_url"], m["vlm"], m["vlm_timeout_s"])
        self.wiki = Wikimedia(s["commons_api"], s["wikidata_api"], s["user_agent"])
        self.embedder = Embedder(m["clip"])
        vw = cfg["vision_web"]
        self.identifier = Identifier(cfg, self.vlm, self.wiki, self.embedder,
                                     WebImageSearch(searxng_url=s.get("searxng_url", ""), text_search=s.get("text_search")),
                                     None,       # Imify retired (Sir, 2026-11-06): Bing and Yandex are called directly
                                     VisionWebDetection(vw, cfg.path("state_dir") / "vision_web",
                                                        Ledger(cfg.path("state_dir") / "lens_pool.json"))
                                     if vw.get("enabled") else None)
        W, H = cfg["display"]["width"], cfg["display"]["height"]
        dsp = cfg["display"]
        self.grid = viewer_grid(dsp["height"], dsp.get("viewer_height", 0))
        cw, ch = card_size(dsp["placard_scale"], self.grid)
        m = card_margin(dsp["placard_scale"], self.grid)
        self.composer = Composer(W, H, cfg.thresholds["essential_retained_min"],
                                 card=((W - m - cw) / W, (H - m - ch) / H, (W - m) / W, (H - m) / H),
                                 detail_min=cfg.thresholds["detail_min_retained"])
        self.larger = LargerSource(self.wiki, self.identifier.web, self.embedder, cfg.thresholds)
        from .indexer import ArtistIndexer
        self.first_sight = ArtistIndexer(cfg.path("state_dir"), lambda: None)   # queue only; the service indexes
        self._upscaler = None
        self.photos = GooglePhotos(cfg["google_photos"], cfg.path("state_dir"))
        self.registry = ArtworkRegistry(cfg.path("state_dir"))

    def _looks_like(self, img: Image.Image, other_rid: str) -> bool:
        """Names alone never make a duplicate: the delivered twin's own source must also match this
        picture on the embedding (at the identification bar), so two works sharing a title stay apart."""
        srcs = sorted((self.cfg.path("work_dir") / other_rid).glob("source.*"))
        if not srcs:
            return False
        vecs = self.embedder.embed([img, load_image(srcs[0])])
        sim = float(self.embedder.similarity(vecs[0], vecs[1:])[0])
        log.info("duplicate check vs %s: CLIP %.3f", other_rid[:8], sim)
        return sim >= self.cfg.thresholds["id_clip_min"]

    def _delivered_twin(self, img: Image.Image, facts: dict, run_id: str, rec: dict) -> tuple[str, dict] | None:
        """A delivered run of the same artwork (by name, then confirmed on the picture), or None."""
        self.registry.reload()                     # another worker may have delivered since this run began
        twin = self.registry.delivered_twin(facts["artist"], facts["title"], exclude=run_id)
        if twin and not self._looks_like(img, twin[0]):
            rec["duplicate_check"] = {"twin": twin[0], "verdict": "same name, different picture: delivered"}
            twin = None
        return twin

    def _duplicate(self, twin, run_id: str, rec: dict, save, rec_path) -> RunResult:
        reason = (f"same artwork as run {twin[0][:8]} ({twin[1].get('artist')} — {twin[1].get('title')}), "
                  "already delivered to the album")
        rec["status"], rec["duplicate_of"], rec["reasons"] = "duplicate", twin[0], [reason]
        save()
        return RunResult(run_id, "duplicate", "identify", [reason], str(rec_path), "")

    @property
    def upscaler(self) -> Upscaler:
        if self._upscaler is None:
            self._upscaler = Upscaler(self.cfg["models"]["upscaler_weights"])
        return self._upscaler

    # ── one run ──────────────────────────────────────────────────────────
    def _versions(self, st: dict) -> list:
        out = []
        for u in _version_urls(st):
            im = fetch_image(u, WEB_UA)
            if im is not None:
                out.append((u, im))
        return out

    def run(self, run_id: str, source: Path, resume: bool = True, deliver: bool = True,
            on_stage=None, redeliver: bool = False, redo: tuple[str, ...] = ()) -> RunResult:
        work = source.parent
        rec_path = work / "record.json"
        rec = json.loads(rec_path.read_text()) if (resume and rec_path.exists()) else {}
        rec.setdefault("run_id", run_id)
        rec.setdefault("stages", {})
        for s in redo:                    # run these again; delivery keeps its record, so a redelivery replaces
            if s == "deliver":
                raise ValueError("redo cannot drop the delivery record; use redeliver")
            rec["stages"].pop(s, None)
            if s == "source":
                (source.parent / "source-large.jpg").unlink(missing_ok=True)
        rec["source"] = source.name
        rec["started"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.vlm.calls = []
        st = rec["stages"]

        def save():
            rec["vlm_calls"] = rec.get("vlm_calls", []) + self.vlm.calls
            self.vlm.calls = []
            rec_path.write_text(json.dumps(rec, indent=2, default=str))

        def block(stage, reasons):
            rec["status"], rec["blocked_stage"], rec["reasons"] = "blocked", stage, reasons
            save()
            return RunResult(run_id, "blocked", stage, reasons, str(rec_path), rec.get("frame", ""))

        def done(stage):
            return resume and st.get(stage, {}).get("passed")

        raw = load_image(source)
        rec["source_size"] = raw.size
        img, rec["trim"] = trim_borders(raw)      # everything below sees the artwork, not its scan border
        if img is not raw:
            log.info("trimmed non-art borders %s -> %s", raw.size, img.size)
        try:
            stage = "identify"
            if on_stage:
                on_stage(stage)
            if not done(stage):
                t0 = time.time()
                ident = self.identifier.identify(img)
                st[stage] = {"passed": ident.status == "confirmed", "seconds": round(time.time() - t0, 1),
                             **ident.to_dict()}
                save()
                if ident.status != "confirmed":
                    return block(stage, [ident.evidence.get("reason", "unconfirmed")])
            # first sight (Sir, 2026-11-05): the artist is queued for the background index as soon as the work is
            # identified, so a run that blocks later, or one started from the command line, still fills it
            try:
                self.first_sight.enqueue(st["identify"].get("artist", ""))
            except Exception as e:
                log.warning("first-sight queue failed: %s", e)
            # identification's facts are LEADS only; the card prints what curated sources rule (Sir, 2026-11-05)
            stage = "catalogue"
            if on_stage:
                on_stage(stage)
            if not done(stage):
                t0 = time.time()
                cat = Catalogue(self.identifier, self.cfg.thresholds)
                # curated records join the work's evidence ledger; the agents rule the card (Sir, 2026-11-07)
                ruling = catalogue_stage(cat, img, st["identify"], run=run_id)
                st[stage] = {"passed": not ruling["failures"], "seconds": round(time.time() - t0, 1), **ruling}
                save()
                if ruling["failures"]:
                    return block(stage, ruling["failures"])
            ruled = st["catalogue"]["facts"]
            facts = {k: ruled.get(k, "") for k in (
                "artist", "title", "date", "medium", "collection", "artist_nationality",
                "artist_born", "artist_died", "inscribed_title")}
            # a different photograph of a painting already in the album: done, not delivered twice
            twin = self._delivered_twin(img, facts, run_id, rec)
            if twin:
                return self._duplicate(twin, run_id, rec, save, rec_path)
            self.registry.record(run_id, phash=phash(raw), artist=facts["artist"], title=facts["title"],
                                 status="identified")

            # a source too small for the display is replaced by a larger verified copy, when one is found
            # (Sir, 2026-11-10); the reading below is then redone on it, since its trimmed box changes
            stage = "source"
            large = work / "source-large.jpg"
            disp = (self.cfg["display"]["width"], self.cfg["display"]["height"])
            max_up = self.cfg.thresholds["max_upscale"]
            if not large.exists() and needed_scale(img.size, disp) > max_up:
                if on_stage:
                    on_stage(stage)
                t0 = time.time()
                found, search = self.larger.find(img, st, facts["artist"], facts["title"], disp, max_up)
                st[stage] = {"passed": True, "seconds": round(time.time() - t0, 1), "replaced": found is not None,
                             **search.to_dict()}
                if found is not None:
                    found.save(large, "JPEG", quality=95)
                save()
            if large.exists():
                img = load_image(large)
                rec["trim"] = {"cut": {}, "box": [0, 0, *img.size], "source_size": list(img.size),
                               "larger_copy": str(large)}

            stage = "interpret"
            if on_stage:
                on_stage(stage)
            # an interpretation made before faces were located is redone: the crop needs them
            prev = st.get(stage, {})
            # redone when made before faces or subject kinds were located, or on a differently trimmed image
            if not done(stage) or "faces" not in prev or any("kind" not in x for x in prev.get("subjects", [])) \
                    or tuple(prev.get("image_box", ())) != tuple(rec["trim"]["box"]):
                t0 = time.time()
                it = interpret(self.vlm, img, facts)
                # an abstract with nothing located is whole as a field of colour and form: there is no subject to mark
                field_only = it.genre == "abstract" and not it.subjects
                fails = [] if field_only or any(s.essential for s in it.subjects) else ["no essential subject marked"]
                fails += [] if it.label_text and len(it.label_text.split()) >= 25 else ["label paragraph too short"]
                st[stage] = {"passed": not fails, "failures": fails, "seconds": round(time.time() - t0, 1),
                             "image_box": list(rec["trim"]["box"]), **it.to_dict()}
                save()
                if fails:
                    return block(stage, fails)
            it = _interp_from(st["interpret"])

            stage = "colour"
            if on_stage:
                on_stage(stage)
            img, st[stage] = rebalance(img, self._versions(st))
            st[stage]["passed"] = True     # a work with too few copies keeps its own colour; it is recorded
            save()

            stage = "compose"
            if on_stage:
                on_stage(stage)
            plan = self.composer.plan(img, it)
            st[stage] = {"passed": plan.passed, **plan.to_dict()}
            facts["detail"] = plan.detail          # the best crop cuts something essential: say so
            save()
            if not plan.passed:
                return block(stage, plan.failures)

            stage = "upscale"
            if on_stage:
                on_stage(stage)
            crop = img.crop(plan.window_px)
            art, up = self.upscaler.to_size(crop, plan.art_size, self.cfg.thresholds["max_upscale"])
            st[stage] = {"passed": up.passed, **up.to_dict()}
            save()
            if not up.passed:
                return block(stage, up.failures)

            stage = "label"
            if on_stage:
                on_stage(stage)
            from .catalogue import clean_movement
            facts["movement"] = clean_movement(ruled.get("movement", ""))   # one movement, as the card prints it
            win = plan.window

            def in_window(b):
                bx = ((b[0] - win[0]) / (win[2] - win[0]), (b[1] - win[1]) / (win[3] - win[1]),
                      (b[2] - win[0]) / (win[2] - win[0]), (b[3] - win[1]) / (win[3] - win[1]))
                return tuple(min(1.0, max(0.0, v)) for v in bx)
            ess = [in_window(s.box) for s in it.subjects if s.essential]
            faces = [in_window(f.box) for f in it.faces]
            disp = (self.cfg["display"]["width"], self.cfg["display"]["height"])
            frame, pl = compose(art, plan, disp, facts, ess, self.cfg.thresholds["placard_min_contrast"],
                                self.cfg["display"]["placard_scale"], self.cfg["display"]["card_texture"],
                                faces=faces, grain=self.cfg["display"]["card_texture_grain"], grid=self.grid)
            name = f"{_slug(facts['artist'])}--{_slug(facts['title'])}--{run_id[:8]}.jpg"
            out = self.cfg.path("output_dir") / name
            frame.save(out, "JPEG", quality=92, optimize=True)
            fails = list(pl.failures)
            if frame.size != disp:
                fails.append(f"frame {frame.size} != display {disp}")
            st[stage] = {"passed": not fails, **pl.to_dict(), "failures": fails, "frame": str(out)}
            rec["frame"] = str(out)
            save()
            if fails:
                return block(stage, fails)

            stage = "deliver"
            if on_stage:
                on_stage(stage)
            if deliver and (redeliver or not done(stage)):
                old = st.get(stage, {}).get("media_item_id", "") if redeliver else ""
                desc = f"{facts['title']}" + (f" '{facts['inscribed_title']}'" if facts.get("inscribed_title") else "") + \
                       (f", {facts['date']}" if facts["date"] else "") + \
                       (" (detail)" if facts.get("detail") else "") + \
                       f" — {facts['artist']}. {it.label_text}"
                with _DELIVERY:
                    twin = None if redeliver else self._delivered_twin(img, facts, run_id, rec)
                    if twin:                      # a parallel run of the same painting got there first
                        return self._duplicate(twin, run_id, rec, save, rec_path)
                    try:
                        dr = self.photos.deliver(out, desc)
                    except AuthRequired as e:
                        st[stage] = {"passed": False, "failures": [str(e)], "auth_required": True}
                        save()
                        return block(stage, [str(e)])
                    st[stage] = {"passed": dr.passed, **dr.to_dict()}
                    save()
                    if not dr.passed:
                        return block(stage, dr.failures)
                    if old and old != dr.media_item_id:   # the new frame replaces the old one in the album
                        try:
                            self.photos.remove_from_album(old)
                            st[stage]["replaced"] = old
                        except Exception as e:     # never a silent duplicate: kept, and retried at the next start
                            st[stage]["pending_remove"] = old
                            log.error("old frame %s still in the album beside its replacement: %s", old[:12], e)
                        save()
                    self.registry.record(run_id, status="delivered", frame=str(out))
            elif not deliver:
                if not st.get(stage, {}).get("passed"):   # a render-only rerun never forgets a real delivery
                    st[stage] = {"passed": False, "skipped": "delivery disabled for this run"}
                rec["status"] = "rendered"
                save()
                return RunResult(run_id, "rendered", "deliver", ["delivery disabled for this run"],
                                 str(rec_path), str(out))

            rec["status"] = "complete"
            rec.pop("blocked_stage", None)
            rec.pop("reasons", None)
            save()
            return RunResult(run_id, "complete", record_path=str(rec_path), frame_path=str(out))
        except RunStopped as e:
            rec["status"], rec["stopped"] = "stopped", f"before {stage}: {e}"
            save()
            return RunResult(run_id, "stopped", stage, [str(e)], str(rec_path))
        except Exception as e:
            rec["status"], rec["error"] = "error", traceback.format_exc()[-3000:]
            save()
            return RunResult(run_id, "error", stage, [f"{type(e).__name__}: {e}"], str(rec_path))


def _version_urls(st: dict) -> list[str]:
    """Every verified copy of the work: the identification's confirmed matches and the catalogue's images."""
    urls = [m.get("thumb_url") for r in st.get("identify", {}).get("rounds", []) for m in r.get("confirmed_by", [])]
    urls += [c.get("image_url") for c in st.get("catalogue", {}).get("sources", []) if c.get("image_url")]
    return list(dict.fromkeys(u for u in urls if u))


def _interp_from(d: dict) -> Interpretation:
    subs = [Subject(s["label"], tuple(s["box"]), s["importance"], s["essential"], s.get("kind", "object"))
            for s in d["subjects"]]
    faces = [Subject(f["label"], tuple(f["box"]), f["importance"], f["essential"], "head") for f in d.get("faces", [])]
    return Interpretation(d["genre"], d["scene"], d["artistic_intent"], tuple(d["focal_point"]),
                          d.get("horizon_y"), subs, d.get("label_text", ""), faces)
