"""Stage 3 — subject-preserving composition onto the 16:9 display.

One algorithm for every genre (the old code had nine copies, none of which ran). The genre only
changes small composition weights. Gate: every *essential* subject keeps ≥ ``essential_retained_min``
of its area. Search order:

1. **full-bleed** — the largest 16:9 window of the artwork, slid along the free axis; best-scoring
   window that passes the gate wins.
2. **matted** — when no 16:9 window can hold every essential subject (tall or very wide works), trim
   the free axis only as far as the essential subjects allow and show the rest of the screen as a
   museum-wall mat (pillarbox or letterbox). The artwork is never squeezed and never cut through a
   protagonist to fill the screen.
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
MARGIN = 0.03   # breathing room kept around essential subjects in matted mode


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


@dataclass
class CropPlan:
    mode: str                                  # full-bleed | pillarbox | letterbox
    window: tuple[float, float, float, float]  # fractions of source
    window_px: tuple[int, int, int, int]
    art_size: tuple[int, int]                  # displayed artwork size on the canvas
    art_offset: tuple[int, int]
    retention: dict = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False
    failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Composer:
    def __init__(self, width: int, height: int, essential_min: float):
        self.W, self.H, self.ess_min = width, height, essential_min
        self.A = width / height

    def _score(self, win, interp: Interpretation, sal: np.ndarray, weights) -> float:
        subs = interp.subjects
        tot = sum(s.importance for s in subs) or 1
        s_imp = sum(s.importance * _retained(s.box, win) for s in subs) / tot
        n = sal.shape[0]
        x0, y0, x1, y1 = (int(round(v * n)) for v in win)
        s_sal = float(sal[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].sum())
        fx, fy = interp.focal_point
        wx, wy = (fx - win[0]) / (win[2] - win[0]), (fy - win[1]) / (win[3] - win[1])
        thirds = 0.0
        if 0 <= wx <= 1 and 0 <= wy <= 1:
            d = min(abs(wx - t) for t in (1 / 3, 1 / 2, 2 / 3)) + min(abs(wy - t) for t in (1 / 3, 1 / 2, 2 / 3))
            thirds = max(0.0, 1 - 3 * d)
        a, b, c = weights
        return a * s_imp + b * s_sal + c * thirds

    def _windows(self, src_aspect: float, span: tuple[float, float] | None, axis: str):
        """Yield windows (fractions). axis 'x' = slide horizontally (wide source), 'y' = vertically."""
        full = (self.A / src_aspect) if axis == "x" else (src_aspect / self.A)   # 16:9 length on free axis
        length = full if span is None else min(1.0, max(full, span[1] - span[0] + 2 * MARGIN))
        lo, hi = 0.0, 1.0 - length                     # feasible offsets along the free axis
        if span is not None:                           # window must contain the essential span + margin
            lo = max(lo, min(1.0, span[1] + MARGIN) - length)
            hi = min(hi, max(0.0, span[0] - MARGIN))
        if lo > hi + 1e-9:
            return
        steps = max(1, int((hi - lo) / 0.005))
        for i in range(steps + 1):
            o = lo + (hi - lo) * i / steps
            yield ((o, 0.0, o + length, 1.0) if axis == "x" else (0.0, o, 1.0, o + length)), length

    def plan(self, img: Image.Image, interp: Interpretation) -> CropPlan:
        w, h = img.size
        a = w / h
        axis = "x" if a >= self.A else "y"
        sal = saliency_map(img)
        weights = GENRE_WEIGHTS.get(interp.genre, (0.6, 0.3, 0.1))
        ess = [s for s in interp.subjects if s.essential]

        def ok(win):
            return all(_retained(s.box, win) >= self.ess_min for s in ess)

        best = None
        for win, _ in self._windows(a, None, axis):
            if ok(win):
                sc = self._score(win, interp, sal, weights)
                if best is None or sc > best[1]:
                    best = (win, sc)
        mode = "full-bleed"
        if best is None and ess:
            lo = min(s.box[0 if axis == "x" else 1] for s in ess)
            hi = max(s.box[2 if axis == "x" else 3] for s in ess)
            for win, _ in self._windows(a, (lo, hi), axis):
                if ok(win):
                    sc = self._score(win, interp, sal, weights)
                    if best is None or sc > best[1]:
                        best = (win, sc)
            mode = "pillarbox" if axis == "y" else "letterbox"
        failures = []
        if best is None:
            # unreachable when essentials are within the image; kept as an explicit failure
            failures.append("no window retains every essential subject")
            best = ((0.0, 0.0, 1.0, 1.0), 0.0)
            mode = "pillarbox" if axis == "y" else "letterbox"
        win, sc = best
        px = (round(win[0] * w), round(win[1] * h), round(win[2] * w), round(win[3] * h))
        cw, ch = px[2] - px[0], px[3] - px[1]
        if mode == "full-bleed":
            art = (self.W, self.H)
        elif mode == "pillarbox":
            art = (round(self.H * cw / ch), self.H)
        else:
            art = (self.W, round(self.W * ch / cw))
        art = (min(art[0], self.W), min(art[1], self.H))
        off = ((self.W - art[0]) // 2, (self.H - art[1]) // 2)
        ret = {f"{i}:{s.label}": round(_retained(s.box, win), 3) for i, s in enumerate(interp.subjects)}
        for i, s in enumerate(interp.subjects):
            if s.essential and _retained(s.box, win) < self.ess_min:
                failures.append(f"essential subject {s.label!r} retained {_retained(s.box, win):.2f}")
        return CropPlan(mode, tuple(round(v, 4) for v in win), px, art, off, ret, round(sc, 4),
                        passed=not failures, failures=failures)
