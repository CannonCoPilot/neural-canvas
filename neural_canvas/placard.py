"""Stage 5 — compose the display frame: matte, artwork, and a museum wall label.

Label facts come only from the verified identification; the paragraph is the interpreter's.
Placement prefers the mat (outside the artwork entirely); on a full-bleed frame the card goes to
the corner with the least visual saliency and must not touch any essential subject.
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

CARD_W, PAD, GAP = 940, 44, 22
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


def _lines(ident: dict, text: str) -> list[tuple[str, ImageFont.FreeTypeFont]]:
    f_artist = ImageFont.truetype(SERIF_BOLD, 46)
    f_meta = ImageFont.truetype(SERIF, 30)
    f_title = ImageFont.truetype(SERIF_ITALIC, 44)
    f_body = ImageFont.truetype(SERIF, 31)
    out = [(ident["artist"], f_artist)]
    life = "–".join(v for v in (ident.get("artist_born"), ident.get("artist_died")) if v)
    meta = ", ".join(v for v in (ident.get("artist_nationality"), life) if v)
    if meta:
        out.append((meta, f_meta))
    out.append(("", None))
    title = ident["title"] + (f", {ident['date']}" if ident.get("date") else "")
    for ln in textwrap.wrap(title, 34):
        out.append((ln, f_title))
    for v in (ident.get("medium"), ident.get("collection")):
        if v:
            out.append((v, f_meta))
    if text:
        out.append(("", None))
        for ln in textwrap.wrap(text, 52):
            out.append((ln, f_body))
    return out


def _card_height(lines) -> int:
    h = 2 * PAD
    for s, f in lines:
        h += GAP if f is None else int(f.size * 1.32)
    return h


def compose(art: Image.Image, plan: CropPlan, canvas_size: tuple[int, int], ident: dict, text: str,
            essential_boxes: list[tuple[float, float, float, float]], min_contrast: float):
    """Return (frame, PlacardResult). ``essential_boxes`` are fractions of the displayed artwork."""
    W, H = canvas_size
    frame = Image.new("RGB", (W, H), matte_colour(art))
    lines = _lines(ident, text)
    ch = _card_height(lines)
    aw, ah = art.size
    margin = 64

    # artwork placement: when the mat is wide enough, push the art aside to give the card the mat
    ox, oy = plan.art_offset
    side_room = W - aw
    in_mat = plan.mode == "pillarbox" and side_room >= CARD_W + 3 * margin
    if in_mat:
        ox = margin if side_room - margin >= CARD_W + 2 * margin else (W - aw) // 2
        frame.paste(art, (ox, oy))
        cx = ox + aw + (W - ox - aw - CARD_W) // 2
        cy = H - margin - ch
        candidates = [("mat-right", (cx, cy, cx + CARD_W, cy + ch))]
    else:
        frame.paste(art, (ox, oy))
        bands = H - ah
        if plan.mode == "letterbox" and bands // 2 >= ch + margin:
            candidates = [("band-bottom-right", (W - margin - CARD_W, H - margin // 2 - ch, W - margin, H - margin // 2))]
        else:
            candidates = [(n, (x, y, x + CARD_W, y + ch)) for n, x, y in (
                ("bottom-right", ox + aw - margin - CARD_W, oy + ah - margin - ch),
                ("bottom-left", ox + margin, oy + ah - margin - ch),
                ("top-right", ox + aw - margin - CARD_W, oy + margin),
                ("top-left", ox + margin, oy + margin))]

    sal = saliency_map(art)
    n = sal.shape[0]

    def art_frac(box):
        x0, y0, x1, y1 = box
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

    scored = sorted(candidates, key=lambda c: (overlap(c[1]) > 0, overlap(c[1]), sal_cost(c[1])))
    pos, box = scored[0]
    ov = overlap(box)

    beneath = np.asarray(frame.crop(box).convert("RGB"), dtype=float).reshape(-1, 3).mean(axis=0)
    a = CARD_ALPHA / 255.0
    card_bg = tuple(a * np.array(PAPER) + (1 - a) * beneath)
    card = Image.new("RGBA", (CARD_W, ch), PAPER + (CARD_ALPHA,))
    d = ImageDraw.Draw(card)
    y = PAD
    overflow = False
    for s, f in lines:
        if f is None:
            y += GAP
            continue
        if d.textlength(s, font=f) > CARD_W - 2 * PAD:
            overflow = True
        d.text((PAD, y), s, font=f, fill=INK)
        y += int(f.size * 1.32)
    d.rectangle((0, 0, 6, ch), fill=(150, 120, 70, 255))   # brass edge, museum style
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
