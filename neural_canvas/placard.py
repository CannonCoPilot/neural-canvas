"""Stage 5 — compose the display frame: matte, artwork, and a museum wall label.

The card carries facts only: artist, nationality and life, title and date, medium, movement, and
collection. Facts come from the verified identification; the movement line is the catalogue's or, where
the catalogue is silent, the interpreter's (see ``movement``). The interpreter's paragraph is not
printed: it travels as the photo's description instead.

Every card is identical in size, layout, type and stock; only the facts change. At ``scale`` 0.6 it
is 494 x 224 px on a 3840 x 2160 frame: artist, nationality and life, a gap, then "title, date" on ONE
line, medium, movement, collection. Rows with nothing to say close up; the card does not shrink. A
line too long for the card is set smaller (to 60% for the title, 70% otherwise); a title still too
long takes two lines at 60%, in the same height. Past that the label FAILS rather than spill.

The card is cut from a photographed paper (``card_texture``) at a fixed grain (``card_texture_grain``:
card pixels per photo pixel), its relief softened so text contrast holds; the texture file is
required, never silently replaced with a flat colour, and must be large enough for the card.

The card is drawn on the VIEWER's pixel grid. The TV shows the album at ~1080p (measured 2026-11-04:
2-px stripes resolve, 1-px do not), so a card drawn at 4K is shrunk 2:1 by the viewer and its small
type blurs. Instead it is rendered at the viewer's resolution, with the font hinted to that grid, and
each pixel is written as a 2x2 block on a block-aligned position: the viewer's 2:1 reduction then
returns the card as drawn. The card sits in the bottom-right corner, over the full-screen
artwork; the crop stage steers faces and essential subjects out of that corner, and the label FAILS
rather than cover a face.
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

CARD_W, PAD, GAP = 822, 44, 22     # at scale 1.0; every card is CARD_W wide (940 trimmed 12.5%, Sir 2026-11-04)
SIZES = {"artist": 46, "meta": 30, "title": 44}   # px at scale 1.0
INK = (29, 28, 26)
PAPER = (244, 241, 234)
TEXTURE_RELIEF = 0.55    # the paper's light/shadow relief kept at this fraction


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


def card_margin(scale: float, grid: int = 1) -> int:
    """Gap between the card and the frame's bottom and right edges, a whole number of viewer pixels."""
    return -(-max(40, round(64 * scale)) // grid) * grid


def card_size(scale: float, grid: int = 1) -> tuple[int, int]:
    """The one card size in frame pixels: CARD_W wide, tall enough for the full template. It is laid out
    on the viewer's grid (``grid`` frame pixels per viewer pixel), so both sides divide by ``grid``."""
    s = scale / grid
    h = 2 * round(PAD * s) + sum(round(GAP * s) if k == "gap" else int(round(SIZES[k] * s) * 1.32) for k in _ROWS)
    return round(CARD_W * s) * grid, h * grid


def viewer_grid(frame_h: int, viewer_h: int) -> int:
    """Frame pixels per viewer pixel: 2 for a 2160-line frame shown on a 1080-line viewer."""
    return max(1, frame_h // viewer_h) if viewer_h else 1


def _layout(ident: dict, scale: float, inner_w: int) -> tuple[list[tuple[str, ImageFont.FreeTypeFont, int]], list[str]]:
    """Rows of (text, font, advance). Facts that are missing close up; a row that will not fit is set
    smaller, then (title only) on two lines; past that it is reported, never cut."""
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    life = "–".join(v for v in (ident.get("artist_born"), ident.get("artist_died")) if v)
    rows = [("artist", ident["artist"]),
            ("meta", ", ".join(v for v in (ident.get("artist_nationality"), life) if v)),
            ("gap", ""),
            ("title", ident["title"] + (f" '{ident['inscribed_title']}'" if ident.get("inscribed_title") else "")
             + (f", {ident['date']}" if ident.get("date") else "") + (" (detail)" if ident.get("detail") else "")),
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
def card_stock(path: str, frame_w: int, frame_h: int, grain: float, w: int, h: int) -> Image.Image:
    """The card background: the centre of the paper photo, a card of ``frame_w`` x ``frame_h`` frame
    pixels covering ``1/grain`` photo pixels each, drawn at ``w`` x ``h``; relief softened, lifted to
    the paper tone while keeping the paper's own hue. Identical for every card."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"placard card texture not found: {p} (display.card_texture)")
    src = Image.open(p).convert("RGB")
    cw, chh = round(frame_w / grain), round(frame_h / grain)
    if cw > src.width or chh > src.height:      # PIL would pad the card with black
        raise ValueError(f"card texture {p.name} {src.size} is smaller than the card needs ({cw}x{chh} "
                         f"at grain {grain}): use a larger photo or a higher display.card_texture_grain")
    x0, y0 = (src.width - cw) // 2, (src.height - chh) // 2
    a = np.asarray(src.crop((x0, y0, x0 + cw, y0 + chh)).resize((w, h), Image.LANCZOS), dtype=float)
    mean = a.reshape(-1, 3).mean(axis=0)
    tone = mean * (PAPER[0] / mean.max())          # the stock's hue at the paper's brightness
    a = tone + (a - mean) * TEXTURE_RELIEF
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def compose(art: Image.Image, plan: CropPlan, canvas_size: tuple[int, int], ident: dict,
            essential_boxes: list[tuple[float, float, float, float]], min_contrast: float, scale: float = 0.5,
            texture: str = "", faces: list | None = None, grain: float = 0.47, grid: int = 1):
    """Return (frame, PlacardResult). ``essential_boxes`` and ``faces`` are fractions of the displayed
    artwork, which fills the frame. Gate: the card never covers a face."""
    W, H = canvas_size
    frame = Image.new("RGB", (W, H), matte_colour(art))
    s = scale / grid                      # the card is drawn on the viewer's grid, then blown up by grid
    pad = round(PAD * s)
    cw, ch = card_size(scale, grid)
    sw, sh = cw // grid, ch // grid
    lines, fit_failures = _layout(ident, s, sw - 2 * pad)
    margin = card_margin(scale, grid)

    # the art fills the frame (the crop stage made it 16:9); the card has one place, bottom-right, over it
    box = (W - margin - cw, H - margin - ch, W - margin, H - margin)
    if art.size != (W, H):
        raise ValueError(f"artwork {art.size} does not fill the {W}x{H} frame")
    frame.paste(art, (0, 0))
    pos = "bottom-right"
    card_f = (box[0] / W, box[1] / H, box[2] / W, box[3] / H)

    def covered(e):
        ix = max(0.0, min(card_f[2], e[2]) - max(card_f[0], e[0]))
        iy = max(0.0, min(card_f[3], e[3]) - max(card_f[1], e[1]))
        area = (e[2] - e[0]) * (e[3] - e[1])
        return ix * iy / area if area else 0.0

    ov = max((covered(e) for e in essential_boxes), default=0.0)     # recorded; the crop minimises it
    faces_under = sum(1 for f in (faces or []) if covered(f) > 0)

    card = card_stock(texture, cw, ch, grain, sw, sh).copy() if texture else Image.new("RGB", (sw, sh), PAPER)
    # contrast is judged on the stock's darkest 5% (its shadowed grain), not its average
    lum = np.asarray(card.convert("L"), dtype=float)
    dark = np.asarray(card, dtype=float).reshape(-1, 3)[lum.reshape(-1) <= np.percentile(lum, 5)].mean(axis=0)
    d = ImageDraw.Draw(card)
    y = pad
    for s, f, adv in lines:
        if f is not None:
            d.text((pad, y), s, font=f, fill=INK)
        y += adv
    d.rectangle((0, 0, max(1, round(max(3, round(6 * scale)) / grid)) - 1, sh), fill=(150, 120, 70))  # brass edge
    frame.paste(card.resize((cw, ch), Image.NEAREST), box[:2])     # each viewer pixel -> a grid x grid block

    contrast = contrast_ratio(INK, tuple(dark))
    failures = list(fit_failures)
    overflow = y > sh - pad // 2
    if faces_under:
        failures.append(f"label covers {faces_under} face(s)")
    if contrast < min_contrast:
        failures.append(f"text contrast {contrast:.1f} < {min_contrast}")
    if overflow:
        failures.append("label text overflows the card")
    if not (0 <= box[0] and 0 <= box[1] and box[2] <= W and box[3] <= H):
        failures.append("label runs off the frame")
    return frame, PlacardResult(pos, box, round(contrast, 2), round(ov, 3), [s for s, _, _ in lines],
                                not failures, failures)
