"""Stage 5 — compose the display frame: matte, artwork, and a museum wall label.

The card carries facts only: artist, nationality and life, title and date, medium, movement, and
collection. Facts come from the verified identification; the movement line is the catalogue's or, where
the catalogue is silent, the interpreter's (see ``movement``). The interpreter's paragraph is not
printed: it travels as the photo's description instead.

The card is drawn at ``scale`` (1.0 = the original 940 px card) and is only as wide as its longest
line. It sits in the bottom-right corner: over the artwork when that corner is clear of every
essential subject, otherwise on the mat beside a slightly smaller artwork.
Gates: zero overlap with essential subjects, text contrast ≥ threshold, all text fits the card.
"""
from __future__ import annotations

import textwrap
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .crop import CropPlan, saliency_map

FONT_DIR = "/System/Library/Fonts/Supplemental/"
SERIF = FONT_DIR + "Georgia.ttf"
SERIF_BOLD = FONT_DIR + "Georgia Bold.ttf"
SERIF_ITALIC = FONT_DIR + "Georgia Italic.ttf"

CARD_W, PAD, GAP = 940, 44, 22     # at scale 1.0; CARD_W is now the widest a card may grow
INK = (29, 28, 26)
PAPER = (244, 241, 234)
CARD_ALPHA = 236


@dataclass
class PlacardResult:
    position: str
    box: tuple[int, int, int, int]
    contrast: float
    essential_overlap: float
    lines: list
    passed: bool
    failures: list

    def to_dict(self) -> dict:
        return asdict(self)


def _lum(c) -> float:
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = c
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(a, b) -> float:
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def matte_colour(art: Image.Image) -> tuple[int, int, int]:
    """A deep, slightly warm wall tone derived from the artwork's own average colour."""
    m = np.asarray(art.convert("RGB").resize((32, 32))).reshape(-1, 3).mean(axis=0)
    base = np.array([24, 23, 21], dtype=float)
    return tuple(int(v) for v in (0.85 * base + 0.15 * m * 0.25))


def _lines(ident: dict, scale: float) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    f_artist = ImageFont.truetype(SERIF_BOLD, round(46 * scale))
    f_meta = ImageFont.truetype(SERIF, round(30 * scale))
    f_title = ImageFont.truetype(SERIF_ITALIC, round(44 * scale))
    out = [(ident["artist"], f_artist)]
    life = "–".join(v for v in (ident.get("artist_born"), ident.get("artist_died")) if v)
    meta = ", ".join(v for v in (ident.get("artist_nationality"), life) if v)
    if meta:
        out.append((meta, f_meta))
    out.append(("", None))
    title = ident["title"] + (f", {ident['date']}" if ident.get("date") else "")
    for ln in textwrap.wrap(title, 34):
        out.append((ln, f_title))
    for v in (ident.get("medium"), ident.get("movement"), ident.get("collection")):
        if v:
            out.append((v, f_meta))
    return out


def _card_size(lines, pad: int, gap: int) -> tuple[int, int]:
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    w = max(int(probe.textlength(s, font=f)) for s, f in lines if f is not None)
    h = 2 * pad + sum(gap if f is None else int(f.size * 1.32) for _, f in lines)
    return w + 2 * pad, h


def compose(art: Image.Image, plan: CropPlan, canvas_size: tuple[int, int], ident: dict,
            essential_boxes: list[tuple[float, float, float, float]], min_contrast: float, scale: float = 0.5):
    """Return (frame, PlacardResult). ``essential_boxes`` are fractions of the displayed artwork."""
    W, H = canvas_size
    frame = Image.new("RGB", (W, H), matte_colour(art))
    pad, gap = round(PAD * scale), round(GAP * scale)
    lines = _lines(ident, scale)
    cw, ch = _card_size(lines, pad, gap)
    aw, ah = art.size
    margin = max(40, round(64 * scale))

    def mat_layout(aw, ah):
        """Art pushed left, card on the mat to its right."""
        ox = margin if (W - aw) - margin >= cw + 2 * margin else (W - aw) // 2
        oy = (H - ah) // 2
        return ox, oy, [("mat-bottom-right", (W - margin - cw, H - margin - ch, W - margin, H - margin))]

    def overlay_layout(aw, ah, ox, oy):
        if plan.mode == "letterbox" and (H - ah) // 2 >= ch + margin:
            return [("band-bottom-right", (W - margin - cw, H - margin // 2 - ch, W - margin, H - margin // 2))]
        return [("bottom-right", (ox + aw - margin - cw, oy + ah - margin - ch, ox + aw - margin, oy + ah - margin))]

    sal = saliency_map(art)
    n = sal.shape[0]
    geo = {}

    def art_frac(box):
        x0, y0, x1, y1 = box
        ox, oy, aw, ah = geo["ox"], geo["oy"], geo["aw"], geo["ah"]
        return ((x0 - ox) / aw, (y0 - oy) / ah, (x1 - ox) / aw, (y1 - oy) / ah)

    def overlap(box):
        fx0, fy0, fx1, fy1 = art_frac(box)
        worst = 0.0
        for b in essential_boxes:
            ix = max(0.0, min(fx1, b[2]) - max(fx0, b[0]))
            iy = max(0.0, min(fy1, b[3]) - max(fy0, b[1]))
            area = (b[2] - b[0]) * (b[3] - b[1])
            worst = max(worst, ix * iy / area if area else 0.0)
        return worst

    def sal_cost(box):
        fx0, fy0, fx1, fy1 = (min(1.0, max(0.0, v)) for v in art_frac(box))
        return float(sal[int(fy0 * n):max(int(fy1 * n), int(fy0 * n) + 1),
                         int(fx0 * n):max(int(fx1 * n), int(fx0 * n) + 1)].sum())

    aw, ah = art.size
    if plan.mode == "pillarbox" and W - aw >= cw + 3 * margin:
        ox, oy, candidates = mat_layout(aw, ah)
    else:
        ox, oy = plan.art_offset
        candidates = overlay_layout(aw, ah, ox, oy)
        geo.update(ox=ox, oy=oy, aw=aw, ah=ah)
        if min(overlap(c[1]) for c in candidates) > 0:
            # the bottom-right corner covers a protagonist: hang the work a little smaller, label on the mat
            k = min((W - cw - 3 * margin) / aw, (H - 2 * margin) / ah)
            aw, ah = round(aw * k), round(ah * k)
            art = art.resize((aw, ah), Image.LANCZOS)
            ox, oy, candidates = mat_layout(aw, ah)
    geo.update(ox=ox, oy=oy, aw=aw, ah=ah)
    frame.paste(art, (ox, oy))

    scored = sorted(candidates, key=lambda c: (overlap(c[1]) > 0, overlap(c[1]), sal_cost(c[1])))
    pos, box = scored[0]
    ov = overlap(box)

    beneath = np.asarray(frame.crop(box).convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    a = CARD_ALPHA / 255.0
    card_bg = tuple(a * np.array(PAPER) + (1 - a) * beneath)
    card = Image.new("RGBA", (cw, ch), PAPER + (CARD_ALPHA,))
    d = ImageDraw.Draw(card)
    y = pad
    overflow = cw > round(CARD_W * scale) + 2 * pad   # a line too long for any sensible card
    for s, f in lines:
        if f is None:
            y += gap
            continue
        d.text((pad, y), s, font=f, fill=INK)
        y += int(f.size * 1.32)
    d.rectangle((0, 0, max(3, round(6 * scale)), ch), fill=(150, 120, 70, 255))   # brass edge, museum style
    frame.paste(card, box[:2], card)

    contrast = contrast_ratio(INK, card_bg)
    failures = []
    if ov > 0:
        failures.append(f"label overlaps an essential subject ({ov:.2f})")
    if contrast < min_contrast:
        failures.append(f"text contrast {contrast:.1f} < {min_contrast}")
    if overflow:
        failures.append("label text overflows the card")
    if not (0 <= box[0] and 0 <= box[1] and box[2] <= W and box[3] <= H):
        failures.append("label runs off the frame")
    return frame, PlacardResult(pos, box, round(contrast, 2), round(ov, 3), [s for s, _ in lines],
                                not failures, failures)
