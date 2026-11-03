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
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

from .config import Config
from .crop import Composer, CropPlan
from .deliver import AuthRequired, GooglePhotos
from .identify import Identifier
from .interpret import Interpretation, Subject, interpret
from .matching import Embedder
from .placard import compose
from .references import Wikimedia
from .upscale import Upscaler
from .vlm import VLM
from .websearch import WebImageSearch

log = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
STAGES = ["identify", "interpret", "compose", "upscale", "label", "deliver"]


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


class Docent:
    """Holds the loaded models; one instance serves many runs."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        m, s = cfg["models"], cfg["sources"]
        self.vlm = VLM(m["ollama_url"], m["vlm"], m["vlm_timeout_s"])
        self.wiki = Wikimedia(s["commons_api"], s["wikidata_api"], s["user_agent"])
        self.embedder = Embedder(m["clip"])
        self.identifier = Identifier(cfg, self.vlm, self.wiki, self.embedder, WebImageSearch())
        self.composer = Composer(cfg["display"]["width"], cfg["display"]["height"],
                                 cfg.thresholds["essential_retained_min"])
        self._upscaler = None
        self.photos = GooglePhotos(cfg["google_photos"], cfg.path("state_dir"))

    @property
    def upscaler(self) -> Upscaler:
        if self._upscaler is None:
            self._upscaler = Upscaler(self.cfg["models"]["upscaler_weights"])
        return self._upscaler

    # ── one run ──────────────────────────────────────────────────────────
    def run(self, run_id: str, source: Path, resume: bool = True, deliver: bool = True,
            on_stage=None) -> RunResult:
        work = source.parent
        rec_path = work / "record.json"
        rec = json.loads(rec_path.read_text()) if (resume and rec_path.exists()) else {}
        rec.setdefault("run_id", run_id)
        rec.setdefault("stages", {})
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

        img = load_image(source)
        rec["source_size"] = img.size
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
            facts = {k: st["identify"].get(k, "") for k in (
                "artist", "title", "date", "medium", "collection", "artist_nationality",
                "artist_born", "artist_died")}

            stage = "interpret"
            if on_stage:
                on_stage(stage)
            if not done(stage):
                t0 = time.time()
                it = interpret(self.vlm, img, facts)
                fails = [] if any(s.essential for s in it.subjects) else ["no essential subject marked"]
                fails += [] if it.label_text and len(it.label_text.split()) >= 25 else ["label paragraph too short"]
                st[stage] = {"passed": not fails, "failures": fails, "seconds": round(time.time() - t0, 1),
                             **it.to_dict()}
                save()
                if fails:
                    return block(stage, fails)
            it = _interp_from(st["interpret"])

            stage = "compose"
            if on_stage:
                on_stage(stage)
            plan = self.composer.plan(img, it)
            st[stage] = {"passed": plan.passed, **plan.to_dict()}
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
            win = plan.window
            ess = []
            for s in it.subjects:
                if s.essential:
                    bx = ((s.box[0] - win[0]) / (win[2] - win[0]), (s.box[1] - win[1]) / (win[3] - win[1]),
                          (s.box[2] - win[0]) / (win[2] - win[0]), (s.box[3] - win[1]) / (win[3] - win[1]))
                    ess.append(tuple(min(1.0, max(0.0, v)) for v in bx))
            disp = (self.cfg["display"]["width"], self.cfg["display"]["height"])
            frame, pl = compose(art, plan, disp, facts, it.label_text, ess,
                                self.cfg.thresholds["placard_min_contrast"])
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
            if deliver and not done(stage):
                desc = f"{facts['title']}" + (f", {facts['date']}" if facts["date"] else "") + \
                       f" — {facts['artist']}. {it.label_text}"
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
            elif not deliver:
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
        except Exception as e:
            rec["status"], rec["error"] = "error", traceback.format_exc()[-3000:]
            save()
            return RunResult(run_id, "error", stage, [f"{type(e).__name__}: {e}"], str(rec_path))


def _interp_from(d: dict) -> Interpretation:
    subs = [Subject(s["label"], tuple(s["box"]), s["importance"], s["essential"]) for s in d["subjects"]]
    return Interpretation(d["genre"], d["scene"], d["artistic_intent"], tuple(d["focal_point"]),
                          d.get("horizon_y"), subs, d.get("label_text", ""))
