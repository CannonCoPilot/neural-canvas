"""Stage 5 — compose the display frame: matte, artwork, and a museum wall label.

The card carries facts only: artist, nationality and life, title and date, medium, movement, and
collection. Facts come from the verified identification; the movement line is the catalogue's or, where
the catalogue is silent, the interpreter's (see ``movement``). The interpreter's paragraph is not
printed: it travels as the photo's description instead.

Every card is identical in size, layout, type and stock; only the facts change. At ``scale`` 0.5 it
is 470 x 190 px on a 3840 x 2160 frame: artist, nationality and life, a gap, then "title, date" on ONE
line, medium, movement, collection. Rows with nothing to say close up; the card does not shrink. A
line too long for the card is set smaller (to 60% for the title, 70% otherwise); a title still too
long takes two lines at 60%, in the same height. Past that the label FAILS rather than spill.

The card is cut from a photographed watercolour card stock (``card_texture``), its grain scaled
fine and its relief softened so text contrast holds; the texture file is required, never silently
replaced with a flat colour. The card sits in the bottom-right corner: over the artwork when that
corner is clear of every essential subject, otherwise on the mat beside a slightly smaller artwork.
Gates: zero overlap with essential subjects, text contrast ≥ threshold, all text fits the card.
"""
from __future__ import annotations

import textwrap
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .crop import CropPlan

FONT_DIR = "/System/Library/Fonts/Supplemental/"
SERIF = FONT_DIR + "Georgia.ttf"
SERIF_BOLD = FONT_DIR + "Georgia Bold.ttf"
SERIF_ITALIC = FONT_DIR + "Georgia Italic.ttf"

CARD_W, PAD, GAP = 940, 44, 22     # at scale 1.0; every card is CARD_W wide
SIZES = {"artist": 46, "meta": 30, "title": 44}   # px at scale 1.0
INK = (29, 28, 26)
PAPER = (244, 241, 234)
TEXTURE_GRAIN = 0.35     # the stock photo is downscaled this much: its pebble grain ~20 px -> ~7 px
TEXTURE_RELIEF = 0.55    # and its light/shadow relief kept at this fraction


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


_FACE = {"artist": SERIF_BOLD, "meta": SERIF, "title": SERIF_ITALIC}
_ROWS = ("artist", "meta", "gap", "title", "meta", "meta", "meta")   # the fixed template


def _font(kind: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FACE[kind], max(6, px))


def card_size(scale: float) -> tuple[int, int]:
    """The one card size: CARD_W wide, tall enough for the full template."""
    h = 2 * round(PAD * scale) + sum(round(GAP * scale) if k == "gap" else int(round(SIZES[k] * scale) * 1.32)
                                     for k in _ROWS)
    return round(CARD_W * scale), h


def _layout(ident: dict, scale: float, inner_w: int) -> tuple[list[tuple[str, ImageFont.FreeTypeFont, int]], list[str]]:
    """Rows of (text, font, advance). Facts that are missing close up; a row that will not fit is set
    smaller, then (title only) on two lines; past that it is reported, never cut."""
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    life = "–".join(v for v in (ident.get("artist_born"), ident.get("artist_died")) if v)
    rows = [("artist", ident["artist"]),
            ("meta", ", ".join(v for v in (ident.get("artist_nationality"), life) if v)),
            ("gap", ""),
            ("title", ident["title"] + (f", {ident['date']}" if ident.get("date") else "")),
            ("meta", ident.get("medium", "")), ("meta", ident.get("movement", "")), ("meta", ident.get("collection", ""))]
    out, failures = [], []
    for kind, text in rows:
        if kind == "gap":
            out.append(("", None, round(GAP * scale)))
            continue
        if not text:
            continue
        full = round(SIZES[kind] * scale)
        slot = int(full * 1.32)
        floor = 0.6 if kind == "title" else 0.7
        px = full
        while px > full * floor and probe.textlength(text, font=_font(kind, px)) > inner_w:
            px -= 1
        f = _font(kind, px)
        if probe.textlength(text, font=f) <= inner_w:
            out.append((text, f, slot))
            continue
        if kind == "title":
            lines = textwrap.wrap(text, max(8, int(len(text) * inner_w / probe.textlength(text, font=f))))
            if len(lines) == 2 and all(probe.textlength(ln, font=f) <= inner_w for ln in lines):
                out += [(lines[0], f, slot // 2), (lines[1], f, slot - slot // 2)]
                continue
        failures.append(f"{kind} line too long for the card: {text[:60]!r}")
        out.append((text, f, slot))
    return out, failures


@lru_cache(maxsize=4)
def card_stock(path: str, w: int, h: int) -> Image.Image:
    """The card background: the centre of the stock photo, grain scaled fine, relief softened, lifted
    to the paper tone while keeping the stock's own warm hue. Identical for every card."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"placard card texture not found: {p} (display.card_texture)")
    src = Image.open(p).convert("RGB")
    cw, chh = round(w / TEXTURE_GRAIN), round(h / TEXTURE_GRAIN)
    x0, y0 = (src.width - cw) // 2, (src.height - chh) // 2
    a = np.asarray(src.crop((x0, y0, x0 + cw, y0 + chh)).resize((w, h), Image.LANCZOS), dtype=float)
    mean = a.reshape(-1, 3).mean(axis=0)
    tone = mean * (PAPER[0] / mean.max())          # the stock's hue at the paper's brightness
    a = tone + (a - mean) * TEXTURE_RELIEF
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def compose(art: Image.Image, plan: CropPlan, canvas_size: tuple[int, int], ident: dict,
            essential_boxes: list[tuple[float, float, float, float]], min_contrast: float, scale: float = 0.5,
            texture: str = ""):
    """Return (frame, PlacardResult). ``essential_boxes`` are fractions of the displayed artwork."""
    W, H = canvas_size
    frame = Image.new("RGB", (W, H), matte_colour(art))
    pad = round(PAD * scale)
    cw, ch = card_size(scale)
    lines, fit_failures = _layout(ident, scale, cw - 2 * pad)
    aw, ah = art.size
    margin = max(40, round(64 * scale))

    # the card has ONE place: the frame's bottom-right corner. Only the artwork ever moves.
    box = (W - margin - cw, H - margin - ch, W - margin, H - margin)
    geo = {}

    def art_frac(b):
        x0, y0, x1, y1 = b
        ox, oy, aw, ah = geo["ox"], geo["oy"], geo["aw"], geo["ah"]
        return ((x0 - ox) / aw, (y0 - oy) / ah, (x1 - ox) / aw, (y1 - oy) / ah)

    def overlap(b):
        fx0, fy0, fx1, fy1 = art_frac(b)
        worst = 0.0
        for e in essential_boxes:
            ix = max(0.0, min(fx1, e[2]) - max(fx0, e[0]))
            iy = max(0.0, min(fy1, e[3]) - max(fy0, e[1]))
            area = (e[2] - e[0]) * (e[3] - e[1])
            worst = max(worst, ix * iy / area if area else 0.0)
        return worst

    right_room, bottom_room = box[0] - margin, box[1] - margin   # art edges that keep the card on the mat

    def on_mat(ox, oy, aw, ah):
        return ox + aw <= right_room or oy + ah <= bottom_room

    aw, ah = art.size
    ox, oy = plan.art_offset
    geo.update(ox=ox, oy=oy, aw=aw, ah=ah)
    if on_mat(ox, oy, aw, ah):
        pos = "mat-bottom-right"                         # the border already holds the card
    elif overlap(box) == 0:
        pos = "bottom-right"                             # over the artwork, clear of every essential subject
    else:
        # the corner covers a protagonist: move the artwork as little as possible to free it
        pos = "mat-bottom-right"
        moves = []
        if aw <= right_room:
            moves.append((ox - min(ox, right_room - aw), min(ox, right_room - aw), oy))
        if ah <= bottom_room:
            moves.append((oy - min(oy, bottom_room - ah), ox, min(oy, bottom_room - ah)))
        if moves:
            _, ox, oy = min(moves)                       # the smaller shift
        else:                                            # the smaller shrink: clear it sideways or from below
            k_side, k_below = right_room / aw, bottom_room / ah
            k = max(k_side, k_below)
            aw, ah = round(aw * k), round(ah * k)
            art = art.resize((aw, ah), Image.LANCZOS)
            if k == k_side:
                ox, oy = (right_room - aw) // 2, (H - ah) // 2
            else:
                ox, oy = (W - aw) // 2, (bottom_room - ah) // 2
        geo.update(ox=ox, oy=oy, aw=aw, ah=ah)
    frame.paste(art, (ox, oy))
    ov = overlap(box) if not on_mat(ox, oy, aw, ah) else 0.0

    card = card_stock(texture, cw, ch).copy() if texture else Image.new("RGB", (cw, ch), PAPER)
    # contrast is judged on the stock's darkest 5% (its shadowed grain), not its average
    lum = np.asarray(card.convert("L"), dtype=float)
    dark = np.asarray(card, dtype=float).reshape(-1, 3)[lum.reshape(-1) <= np.percentile(lum, 5)].mean(axis=0)
    d = ImageDraw.Draw(card)
    y = pad
    for s, f, adv in lines:
        if f is not None:
            d.text((pad, y), s, font=f, fill=INK)
        y += adv
    d.rectangle((0, 0, max(3, round(6 * scale)) - 1, ch), fill=(150, 120, 70))   # brass edge, museum style
    frame.paste(card, box[:2])

    contrast = contrast_ratio(INK, tuple(dark))
    failures = list(fit_failures)
    overflow = y > ch - pad // 2
    if ov > 0:
        failures.append(f"label overlaps an essential subject ({ov:.2f})")
    if contrast < min_contrast:
        failures.append(f"text contrast {contrast:.1f} < {min_contrast}")
    if overflow:
        failures.append("label text overflows the card")
    if not (0 <= box[0] and 0 <= box[1] and box[2] <= W and box[3] <= H):
        failures.append("label runs off the frame")
    return frame, PlacardResult(pos, box, round(contrast, 2), round(ov, 3), [s for s, _, _ in lines],
                                not failures, failures)
