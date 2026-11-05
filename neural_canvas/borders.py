"""Stage 0 — trim what is not the artwork: flat bands of black, white or grey along an edge, including
any caption printed in them ("Summer Morning (1891)." on a black band below a scan).

A band is recognised from the outside in. Its colour is the edge's own (the median of the outermost
rows) and must be neutral: black, white or grey, not a hue. In every column, the band's depth is how
far that colour runs in from the edge. A border made by a scan, a mount or a caption strip ends on a
STRAIGHT line, so most columns agree on its depth (lettering inside the band only shortens a few).
A painted ground (a vignette on white paper, a black field between shapes, a pale sky) ends wherever
the drawing begins, so its columns disagree, and it is left whole. The picture beyond the line must
also plainly differ from the band colour, and no side gives up more than MAX_SIDE of its length.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

NEUTRAL = 24          # max channel spread (0-255) for a band colour: black, white or grey, not a hue
TOL = 18              # a pixel within this of the band colour belongs to the band
STRAIGHT = 0.5        # share of columns whose depth agrees with the band's: a straight edge
AGREE = 0.004         # "agrees": within this fraction of the side's length (3 px at least)
STEP_MIN = 40         # the picture beyond the band differs from the band colour by at least this (mean)
MAX_SIDE = 0.3        # never trim more than this fraction from one side
MIN_BAND = 3          # thinner than this is the scan's edge noise, not a band


def _band(a: np.ndarray) -> int:
    """Rows of ``a`` (rows x cols x 3, outermost row first) that are a non-art band; 0 when none."""
    n = a.shape[0]
    limit = int(n * MAX_SIDE)
    edge = np.median(a[:2].reshape(-1, 3), axis=0)
    if edge.max() - edge.min() > NEUTRAL:
        return 0
    off = np.abs(a[:limit + 1] - edge).max(axis=2) > TOL           # pixels that are not band colour
    depth = np.where(off.any(axis=0), off.argmax(axis=0), limit + 1)  # band depth, column by column
    band = int(np.bincount(depth).argmax())
    if band < MIN_BAND or band > limit:
        return 0
    if (np.abs(depth - band) <= max(3, AGREE * n)).mean() < STRAIGHT:
        return 0                                                      # a ragged edge: a painted ground
    beyond = a[band:band + 12].reshape(-1, 3).mean(axis=0)
    return band if np.abs(beyond - edge).mean() >= STEP_MIN else 0


def trim_borders(img: Image.Image) -> tuple[Image.Image, dict]:
    """Return (artwork, record): the image without its non-art bands, and what was cut from each side."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    cut = {"top": _band(a), "bottom": _band(a[::-1]),
           "left": _band(a.transpose(1, 0, 2)), "right": _band(a.transpose(1, 0, 2)[::-1])}
    h, w = a.shape[:2]
    box = (cut["left"], cut["top"], w - cut["right"], h - cut["bottom"])
    rec = {"cut": cut, "box": box, "source_size": (w, h)}
    return (img.crop(box) if any(cut.values()) else img), rec
