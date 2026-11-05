"""Stage 3 — fill the 16:9 display, losing as little of the artwork's framing as the screen allows.

Every work is shown full-screen: the largest 16:9 window of the artwork (the whole of its short
axis), slid along its long axis, then upscaled to the display. Never stretched, never matted. What
remains to decide is only WHERE the window sits, and that follows the artwork, in this order:

1. **faces** — every face of a principal figure (an essential subject of top importance) stays
   wholly inside. This is the gate: a work whose principal faces cannot all share one 16:9 window
   blocks, and its run says so. Beyond that, as many other faces as possible.
2. **subjects** — importance-weighted retention, with a penalty for slicing a subject through
   (keeping 15-85% of it) rather than keeping or dropping it whole.
3. **composition** — the primary subject keeps its centrality or offset; the focal point sits near
   a third or the centre; visual saliency; the genre's rule (landscapes give up height from the
   bottom first).
4. **the label corner** — faces and essential subjects are steered out of the bottom-right corner
   where the placard is printed.

Any essential subject the window trims below ``essential_retained_min`` is listed in ``warnings``
with how much of it survives: truncation is reported, never hidden.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from .interpret import Interpretation

GENRE_WEIGHTS = {   # (subject importance, saliency, thirds composition)
    "landscape": (0.45, 0.35, 0.20), "seascape": (0.45, 0.35, 0.20), "cityscape": (0.5, 0.3, 0.2),
    "portrait": (0.75, 0.15, 0.10), "figure-group": (0.70, 0.20, 0.10),
    "religious-historical": (0.70, 0.20, 0.10), "genre-scene": (0.65, 0.25, 0.10),
    "still-life": (0.60, 0.30, 0.10), "animal": (0.70, 0.20, 0.10), "interior": (0.6, 0.3, 0.1),
    "abstract": (0.30, 0.60, 0.10),
}
FACE_MARGIN = 0.01        # a kept face keeps this much air (fraction of the image) inside the window
W_FACES, W_SLICE, W_OFFSET, W_TOP, W_CARD = 0.6, 0.25, 0.15, 0.05, 0.3


def saliency_map(img: Image.Image, side: int = 128) -> np.ndarray:
    """Spectral-residual saliency (Hou & Zhang 2007) on a small grey copy, normalised to sum 1."""
    g = np.asarray(img.convert("L").resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.0
    f = np.fft.fft2(g)
    log_amp = np.log(np.abs(f) + 1e-8)
    resid = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(resid + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 3)
    return sal / sal.sum()


def _retained(box, win) -> float:
    ix = max(0.0, min(box[2], win[2]) - max(box[0], win[0]))
    iy = max(0.0, min(box[3], win[3]) - max(box[1], win[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return (ix * iy) / area if area > 0 else 0.0


def _inside(box, win, margin: float = FACE_MARGIN) -> bool:
    return (box[0] >= win[0] + margin or win[0] <= 0) and (box[1] >= win[1] + margin or win[1] <= 0) \
        and (box[2] <= win[2] - margin or win[2] >= 1) and (box[3] <= win[3] - margin or win[3] >= 1)


def principal_faces(interp: Interpretation) -> list:
    """The faces the crop may never cut: those of the principal figures, i.e. the essential subjects of
    top importance. A crowd or choir marked essential has faces everywhere; they are scored, not gated.
    A face whose subject cannot be matched by label stays gated (the strict side)."""
    ess = [s for s in interp.subjects if s.essential]
    top = max((s.importance for s in ess), default=0)
    imp = {s.label.strip().lower(): s.importance for s in ess}
    return [f for f in interp.faces if f.essential and imp.get(f.label.strip().lower(), top) >= top]


@dataclass
class CropPlan:
    mode: str                                  # always "full-bleed" (kept for the record format)
    window: tuple[float, float, float, float]  # fractions of source
    window_px: tuple[int, int, int, int]
    art_size: tuple[int, int]                  # displayed artwork size on the canvas
    art_offset: tuple[int, int]
    retention: dict = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    faces_kept: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Composer:
    def __init__(self, width: int, height: int, essential_min: float,
                 card: tuple[float, float, float, float] = (0.867, 0.894, 0.99, 0.981)):
        """``card`` is the placard's box as fractions of the display (bottom-right)."""
        self.W, self.H, self.ess_min, self.card = width, height, essential_min, card
        self.A = width / height

    def _windows(self, src_aspect: float, axis: str):
        """The largest 16:9 windows (fractions). axis 'x' = slide horizontally (wide source), 'y' = vertically."""
        length = (self.A / src_aspect) if axis == "x" else (src_aspect / self.A)
        length = min(1.0, length)
        steps = max(1, int((1.0 - length) / 0.004))
        for i in range(steps + 1):
            o = (1.0 - length) * i / steps
            yield (o, 0.0, o + length, 1.0) if axis == "x" else (0.0, o, 1.0, o + length)

    def _card_in_source(self, win):
        x0, y0, x1, y1 = self.card
        return (win[0] + x0 * (win[2] - win[0]), win[1] + y0 * (win[3] - win[1]),
                win[0] + x1 * (win[2] - win[0]), win[1] + y1 * (win[3] - win[1]))

    def _score(self, win, interp: Interpretation, sal: np.ndarray, weights, axis: str) -> float:
        subs = interp.subjects
        tot = sum(s.importance for s in subs) or 1
        rets = [_retained(s.box, win) for s in subs]
        s_imp = sum(s.importance * r for s, r in zip(subs, rets)) / tot
        s_slice = sum(s.importance for s, r in zip(subs, rets) if 0.15 < r < 0.85) / tot
        n = sal.shape[0]
        x0, y0, x1, y1 = (int(round(v * n)) for v in win)
        s_sal = float(sal[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].sum())
        fx, fy = interp.focal_point
        wx, wy = (fx - win[0]) / (win[2] - win[0]), (fy - win[1]) / (win[3] - win[1])
        thirds = 0.0
        if 0 <= wx <= 1 and 0 <= wy <= 1:
            d = min(abs(wx - t) for t in (1 / 3, 1 / 2, 2 / 3)) + min(abs(wy - t) for t in (1 / 3, 1 / 2, 2 / 3))
            thirds = max(0.0, 1 - 3 * d)
        faces = interp.faces
        s_face = sum(_inside(f.box, win) for f in faces) / len(faces) if faces else 1.0
        # the primary subject keeps the centrality or offset it has in the whole work
        s_off = 1.0
        prim = max((s for s in subs if s.essential), key=lambda s: s.importance, default=None)
        if prim is not None:
            k = 0 if axis == "x" else 1
            c = (prim.box[k] + prim.box[k + 2]) / 2
            in_win = (c - win[k]) / (win[k + 2] - win[k])
            s_off = max(0.0, 1 - 2 * abs(in_win - c))
        # landscapes give up height from the bottom first
        s_top = (1 - win[1] / max(1e-6, 1 - (win[3] - win[1]))) if axis == "y" and \
            interp.genre in ("landscape", "seascape", "cityscape") and win[3] - win[1] < 1 else 0.0
        card = self._card_in_source(win)
        s_card = max([1.0 for f in faces if _retained(f.box, card) > 0]          # a face under the label
                     + [_retained(s.box, card) for s in subs if s.essential], default=0.0)
        a, b, c3 = weights
        return (a * s_imp + b * s_sal + c3 * thirds + W_FACES * s_face - W_SLICE * s_slice
                + W_OFFSET * s_off + W_TOP * s_top - W_CARD * s_card)

    def plan(self, img: Image.Image, interp: Interpretation) -> CropPlan:
        w, h = img.size
        a = w / h
        axis = "x" if a >= self.A else "y"
        sal = saliency_map(img)
        weights = GENRE_WEIGHTS.get(interp.genre, (0.6, 0.3, 0.1))
        ess_faces = principal_faces(interp)
        wins = list(self._windows(a, axis))
        keep = [win for win in wins if all(_inside(f.box, win) for f in ess_faces)]
        failures = []
        if not keep:
            best_n = max(sum(_inside(f.box, win) for f in ess_faces) for win in wins)
            failures.append(f"no 16:9 window keeps every essential face ({best_n} of {len(ess_faces)} at best): "
                            "the faces span more of the long axis than the screen allows")
            keep = wins
        scored = [(self._score(win, interp, sal, weights, axis), win) for win in keep]
        sc, win = max(scored)
        px = (round(win[0] * w), round(win[1] * h), round(win[2] * w), round(win[3] * h))
        ret = {f"{i}:{s.label}": round(_retained(s.box, win), 3) for i, s in enumerate(interp.subjects)}
        warnings = [f"essential subject {s.label!r} truncated: {_retained(s.box, win):.0%} kept"
                    for s in interp.subjects if s.essential and _retained(s.box, win) < self.ess_min]
        kept = sum(_inside(f.box, win) for f in interp.faces)
        return CropPlan("full-bleed", tuple(round(v, 4) for v in win), px, (self.W, self.H), (0, 0), ret,
                        round(sc, 4), passed=not failures, failures=failures, warnings=warnings,
                        faces_kept=f"{kept}/{len(interp.faces)}")
