"""Stage 3 — fill the 16:9 display: the largest window, centred on what matters, nudged to cut less.

Every work is shown full-screen and nothing is ever added (no mats, no fill). The method is the same
for every genre (Sir's rule, 2026-11-04):

1. **window** — the artwork's shorter side is kept whole; the other side is cut to 16:9. This is the
   largest window the screen allows, and its only freedom is where it sits along the long side.
2. **centroid** — the centre of every located box (heads, figures, animals, objects; regions such as
   sky or meadow count lightly) is weighted by its importance, and the window starts centred on the
   weighted centroid. A work with no such boxes (an abstract) uses its centre of visual weight.
3. **adjust** — the window then slides to the position of least cost: distance from the centroid,
   plus what it cuts off the subjects at the edges. One rule is a ban, the rest are costs ("try not
   to"). The ban: the work has ONE main subject, and its head and face are never cropped (Sir's rule,
   2026-11-10, after Bloch's *Christus Consolator* lost Christ's head to a crowd of lesser faces). Of
   the windows that keep that head whole, the cheapest wins: try not to cut a secondary face (each
   face costs on its own, by its importance, so a crowd of them can never outvote the main head; and
   a face cut through the middle costs most), then figures, animals and objects, and regions hardly
   at all.
4. **genre suggestions** — small costs that guide the final position: figures keep headroom; a
   landscape gives up foreground before sky and keeps its horizon off the centre line; faces and
   essential subjects keep out of the bottom-right corner, where the placard is printed.

When the best window still cuts the MAIN subject (it keeps less than ``detail_min`` of itself, or its
head is too large for any window to hold whole), the frame is a DETAIL of the work: ``detail`` is set
with its reasons, and the label says "(detail)". It is delivered as the best crop available, never
blocked and never padded. Cuts to other essential subjects and to secondary faces are listed in
``warnings``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from .interpret import Interpretation

KIND_WEIGHT = {"head": 1.0, "figure": 1.0, "animal": 1.0, "object": 1.0, "region": 0.15}
# genres whose regions ARE the subject (the sky, the sea, a colour field): regions count fully there
REGION_GENRES = {"landscape", "seascape", "cityscape", "abstract"}
FIGURE_GENRES = {"portrait", "figure-group", "religious-historical", "genre-scene", "animal", "interior"}
ESSENTIAL_BOOST = 1.5
# costs, all "try not to"
W_CENTROID = 0.5      # per unit of the window's sliding range away from the centroid
W_CUT = 1.0           # weighted share of subject area cut off
W_FACE = 1.0          # per secondary face, its share cut off, scaled by its importance (not shared out)
W_FACE_SPLIT = 0.5    # per secondary face cut through (part in, part out): the ugliest cut there is
W_HEADROOM = 0.15     # figures: the main head pressed against the top edge
W_LANDSCAPE = 0.2     # landscapes: foreground goes before sky; horizon off the centre line
W_CARD = 0.3          # faces and essential subjects under the placard corner
HEADROOM = 0.04       # fraction of the window kept above a principal head
STEP = 0.002          # window positions tried, as a fraction of the long side


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


def _centre(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _area(box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _distance(pt, box) -> float:
    """From a point to a box (0 inside it)."""
    dx, dy = max(box[0] - pt[0], 0.0, pt[0] - box[2]), max(box[1] - pt[1], 0.0, pt[1] - box[3])
    return (dx * dx + dy * dy) ** 0.5


def main_subject(interp: Interpretation):
    """The ONE main subject: essential first, then the most important, then the one with a face of its
    own, then the one the focal point falls on (its face, else its box; the VLM often ties two figures
    at the top importance, as Christ and the angel in Bloch's Gethsemane), then the largest. None when
    nothing is located."""
    faces = {}
    for f in interp.faces:
        faces.setdefault(f.label.strip().lower(), f)

    def key(s):
        face = faces.get(s.label.strip().lower())
        return (s.essential, s.importance, face is not None,
                -_distance(interp.focal_point, face.box if face else s.box), _area(s.box))
    return max(interp.subjects, default=None, key=key)


def main_heads(interp: Interpretation, main=None) -> list:
    """The main subject's head: the face that bears its label, else the largest face centred inside its
    box (essential and important ones first). Empty when the main subject shows no face."""
    main = main or main_subject(interp)
    if main is None:
        return []
    named = [f for f in interp.faces if f.label.strip().lower() == main.label.strip().lower()]
    if named:
        return named
    inside = [f for f in interp.faces
              if main.box[0] <= _centre(f.box)[0] <= main.box[2] and main.box[1] <= _centre(f.box)[1] <= main.box[3]]
    return [max(inside, key=lambda f: (f.essential, f.importance, _area(f.box)))] if inside else []


def secondary_faces(interp: Interpretation, main=None, heads=None) -> list[tuple[object, float]]:
    """(face, weight) for every face but the main head; weight = its importance relative to the main
    subject's, never above 1. A face whose figure was not named takes its own importance, held below
    the main subject's."""
    main = main or main_subject(interp)
    heads = main_heads(interp, main) if heads is None else heads
    top = max(1, main.importance if main else max((f.importance for f in interp.faces), default=1))
    imp = {s.label.strip().lower(): s.importance for s in interp.subjects}
    out = []
    for f in interp.faces:
        if any(f is h for h in heads):
            continue
        named = imp.get(f.label.strip().lower())
        i = named if named is not None else min(f.importance, top - 1)
        out.append((f, min(1.0, max(1, i) / top)))
    return out


@dataclass
class CropPlan:
    mode: str                                  # always "full-bleed" (kept for the record format)
    window: tuple[float, float, float, float]  # fractions of source
    window_px: tuple[int, int, int, int]
    art_size: tuple[int, int]                  # displayed artwork size on the canvas
    art_offset: tuple[int, int]
    retention: dict = field(default_factory=dict)
    score: float = 0.0                         # the cost of the chosen window (lower is better)
    passed: bool = False
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    faces_kept: str = ""
    detail: bool = False                       # the frame shows a detail: the main subject is cut
    detail_reasons: list = field(default_factory=list)
    centroid: tuple[float, float] = (0.5, 0.5)
    centroid_window: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)   # step 2's window
    costs: dict = field(default_factory=dict)  # the chosen window's cost, term by term

    def to_dict(self) -> dict:
        return asdict(self)


class Composer:
    def __init__(self, width: int, height: int, essential_min: float,
                 card: tuple[float, float, float, float] = (0.867, 0.894, 0.99, 0.981), detail_min: float = 0.75):
        """``card`` is the placard's box as fractions of the display (bottom-right)."""
        self.W, self.H, self.ess_min, self.card, self.detail_min = width, height, essential_min, card, detail_min
        self.A = width / height

    def _card_in_source(self, win):
        x0, y0, x1, y1 = self.card
        return (win[0] + x0 * (win[2] - win[0]), win[1] + y0 * (win[3] - win[1]),
                win[0] + x1 * (win[2] - win[0]), win[1] + y1 * (win[3] - win[1]))

    @staticmethod
    def _boxes(interp: Interpretation) -> list[tuple[tuple, float, str, object]]:
        """(box, weight, kind, subject) for every located thing: subjects plus heads."""
        region_w = 1.0 if interp.genre in REGION_GENRES else KIND_WEIGHT["region"]
        out = []
        for s in interp.subjects:
            kind = getattr(s, "kind", "object") or "object"
            w = s.importance * (region_w if kind == "region" else KIND_WEIGHT.get(kind, 1.0))
            out.append((s.box, w * (ESSENTIAL_BOOST if s.essential else 1.0), kind, s))
        imp = {s.label.strip().lower(): s.importance for s in interp.subjects}
        for f in interp.faces:
            w = imp.get(f.label.strip().lower(), f.importance)
            out.append((f.box, w * (ESSENTIAL_BOOST if f.essential else 1.0), "head", f))
        return out

    def centroid(self, img: Image.Image, interp: Interpretation) -> tuple[float, float]:
        """Importance-weighted centroid of the box centres; the centre of visual weight when the work
        has nothing but regions (or nothing located at all)."""
        things = [(b, w) for b, w, k, _ in self._boxes(interp) if k != "region" or interp.genre in REGION_GENRES]
        if things:
            tot = sum(w for _, w in things)
            return (sum(_centre(b)[0] * w for b, w in things) / tot, sum(_centre(b)[1] * w for b, w in things) / tot)
        sal = saliency_map(img)
        n = sal.shape[0]
        ys, xs = np.mgrid[0:n, 0:n]
        return float((xs * sal).sum() / (n - 1)), float((ys * sal).sum() / (n - 1))

    def _costs(self, win, axis: str, cen, interp: Interpretation, boxes, heads, faces) -> dict:
        k = 0 if axis == "x" else 1
        length = win[k + 2] - win[k]
        span = max(1e-6, 1.0 - length)
        c = {"centroid": W_CENTROID * abs((win[k] + win[k + 2]) / 2 - cen[k]) / span}
        tot = sum(w for _, w, _, _ in boxes) or 1.0
        c["cut"] = W_CUT * sum(w * (1 - _retained(b, win)) for b, w, _, _ in boxes) / tot
        kept = [(_retained(f.box, win), w) for f, w in faces]
        c["face_cut"] = W_FACE * sum(w * (1 - r) for r, w in kept)
        c["face_split"] = W_FACE_SPLIT * sum(w for r, w in kept if 0.0 < r < 0.999)
        c["headroom"] = 0.0
        if interp.genre in FIGURE_GENRES and heads and axis == "y":
            top = min(h.box[1] for h in heads)
            room = (top - win[1]) / length
            if 0 <= room < HEADROOM:
                c["headroom"] = W_HEADROOM * (HEADROOM - room) / HEADROOM
        c["landscape"] = 0.0
        if interp.genre in ("landscape", "seascape", "cityscape") and axis == "y" and span > 1e-6:
            c["landscape"] = W_LANDSCAPE * win[1] / span                           # sky goes last
            if interp.horizon_y is not None and win[1] <= interp.horizon_y <= win[3]:
                hz = (interp.horizon_y - win[1]) / length
                c["landscape"] += W_LANDSCAPE * max(0.0, 1 - abs(hz - 0.5) / (1 / 6)) # not on the centre line
        card = self._card_in_source(win)
        c["card"] = W_CARD * max([1.0 for f in interp.faces if _retained(f.box, card) > 0]
                                 + [_retained(s.box, card) for s in interp.subjects if s.essential], default=0.0)
        c["total"] = sum(c.values())
        return c

    def plan(self, img: Image.Image, interp: Interpretation) -> CropPlan:
        w, h = img.size
        a = w / h
        axis = "x" if a >= self.A else "y"
        length = min(1.0, (self.A / a) if axis == "x" else (a / self.A))
        k = 0 if axis == "x" else 1

        def window(o):
            return (o, 0.0, o + length, 1.0) if axis == "x" else (0.0, o, 1.0, o + length)

        cen = self.centroid(img, interp)
        main = main_subject(interp)
        heads = main_heads(interp, main)
        boxes, faces = self._boxes(interp), secondary_faces(interp, main, heads)
        o_c = min(max(cen[k] - length / 2, 0.0), 1.0 - length)
        steps = max(1, int(round((1.0 - length) / STEP)))
        # the grid, plus the positions that put an edge of the window exactly on an edge of the main head
        flush = [v for hd in heads for v in (hd.box[k], hd.box[k + 2] - length)]
        offsets = sorted({round(o_c, 6), *((1.0 - length) * i / steps for i in range(steps + 1)),
                          *(round(min(max(v, 0.0), 1.0 - length), 6) for v in flush)})
        # the ban: only windows that keep the main head whole (or, when no window can, keep the most of it)
        head_kept = {o: min((_retained(hd.box, window(o)) for hd in heads), default=1.0) for o in offsets}
        best_kept = max(head_kept.values())
        allowed = [o for o in offsets if head_kept[o] >= best_kept - 1e-9]
        costed = [(self._costs(window(o), axis, cen, interp, boxes, heads, faces), o) for o in allowed]
        best, o = min(costed, key=lambda t: (round(t[0]["total"], 9), abs(t[1] - o_c)))
        win = window(o)
        px = (round(win[0] * w), round(win[1] * h), round(win[2] * w), round(win[3] * h))
        ret = {f"{i}:{s.label}": round(_retained(s.box, win), 3) for i, s in enumerate(interp.subjects)}
        warnings = [f"essential subject {s.label!r} truncated: {_retained(s.box, win):.0%} kept"
                    for s in interp.subjects if s.essential and _retained(s.box, win) < self.ess_min]
        head_cuts = [f"main head of {f.label!r} cannot fit the window: {_retained(f.box, win):.0%} kept"
                     for f in heads if _retained(f.box, win) < 0.999]
        warnings += head_cuts + [f"secondary face of {f.label!r} cut: {_retained(f.box, win):.0%} kept"
                                 for f, _ in faces if _retained(f.box, win) < 0.999]
        reasons = head_cuts + ([f"main subject {main.label!r}: {_retained(main.box, win):.0%} kept"]
                               if main is not None and _retained(main.box, win) < self.detail_min else [])
        kept = sum(_retained(f.box, win) >= 0.999 for f in interp.faces)
        return CropPlan("full-bleed", tuple(round(v, 4) for v in win), px, (self.W, self.H), (0, 0), ret,
                        round(best["total"], 4), passed=True, failures=[], warnings=warnings,
                        faces_kept=f"{kept}/{len(interp.faces)}", detail=bool(reasons), detail_reasons=reasons,
                        centroid=(round(cen[0], 4), round(cen[1], 4)),
                        centroid_window=tuple(round(v, 4) for v in window(o_c)),
                        costs={key: round(v, 4) for key, v in best.items()})
