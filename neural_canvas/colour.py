"""Colour: one artwork, many photographs of it, each with its own white balance and exposure.

``rebalance`` — Sir's rule (2026-11-05): when several copies of the same work are found on different
source pages, take a colour middle ground and move the input toward it. Each verified copy is
registered onto the input (ORB + RANSAC homography) so only the shared area is measured; the CIELAB
mean and spread of every copy (the input included) are pooled by MEDIAN, which a single badly
balanced copy cannot drag; the input is then mapped to that median (per-channel mean/spread transfer).
Fewer than MIN_OTHERS other copies: no middle ground exists and the input is left as it is (recorded).
A greyscale copy (a black-and-white reproduction of a colour work) does not vote on colour.

``match_low_frequency`` — after super-resolution, the network's colour and tone drift is removed by
moving the large-scale colour of the result back onto its source, leaving the new fine detail alone.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from .matching import homography

log = logging.getLogger(__name__)

WORK = 384            # long side of the working copies
MIN_OTHERS = 2        # other copies needed before a middle ground means anything
MIN_INLIERS = 25      # a copy is used only when it registers as the same picture
MIN_OVERLAP = 0.5     # ... and covers at least this share of the input
GREY_CHROMA = 3.0     # a copy whose a*/b* spread is below this is a greyscale reproduction


def _lab(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _small(im: Image.Image) -> Image.Image:
    s = WORK / max(im.size)
    return im.convert("RGB").resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)


def _stats(lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = lab[mask]
    return px.mean(axis=0), px.std(axis=0) + 1e-3


def rebalance(img: Image.Image, versions: list[tuple[str, Image.Image]]) -> tuple[Image.Image, dict]:
    """Return (image moved to the copies' colour middle ground, record). ``versions`` are (source, image)."""
    src = _small(img)
    lab_s = _lab(src)
    full = np.ones(lab_s.shape[:2], bool)
    means, stds, used, skipped = [], [], [], []
    for name, v in versions:
        H, inl, scale = homography(src, v)
        if H is None or inl < MIN_INLIERS:
            skipped.append({"source": name, "why": f"does not register ({inl} inliers)"}); continue
        vs = v.convert("RGB")
        vs.thumbnail((900, 900), Image.LANCZOS)               # the working copy homography() used
        S = np.diag([1 / scale, 1 / scale, 1.0])
        warped = cv2.warpPerspective(np.asarray(vs), S @ H, (src.width, src.height), flags=cv2.INTER_AREA)
        valid = cv2.warpPerspective(np.full(vs.size[::-1], 255, np.uint8), S @ H, (src.width, src.height)) > 0
        valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if valid.mean() < MIN_OVERLAP:
            skipped.append({"source": name, "why": f"covers {valid.mean():.0%} of the work"}); continue
        lab_v = cv2.cvtColor(warped.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        m, sd = _stats(lab_v, valid)
        if max(sd[1], sd[2]) < GREY_CHROMA and max(_stats(lab_s, full)[1][1:]) >= GREY_CHROMA:
            skipped.append({"source": name, "why": "greyscale reproduction"}); continue
        # the input measured over the same area, so crop and border differences do not count as colour
        ms, ss = _stats(lab_s, valid)
        means.append(m - ms); stds.append(sd / ss); used.append(name)
    rec = {"copies_used": used, "copies_skipped": skipped, "applied": False}
    if len(used) < MIN_OTHERS:
        rec["why_not"] = f"{len(used)} other verified cop{'y' if len(used) == 1 else 'ies'}; {MIN_OTHERS} needed for a middle ground"
        return img, rec
    # the input itself votes too: zero offset, unit spread
    d_mean = np.median(np.vstack([np.zeros(3)] + means), axis=0)
    d_std = np.median(np.vstack([np.ones(3)] + stds), axis=0)
    lab = _lab(img)
    m_all, s_all = lab.reshape(-1, 3).mean(axis=0), lab.reshape(-1, 3).std(axis=0) + 1e-3
    out = (lab - m_all) * d_std + m_all + d_mean
    out[..., 0] = np.clip(out[..., 0], 0, 100)
    rgb = np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)
    res = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))
    shift = float(np.sqrt(((_lab(_small(res)) - lab_s) ** 2).sum(axis=2)).mean())
    rec.update(applied=True, mean_shift_lab=[round(float(x), 2) for x in d_mean],
               spread_ratio=[round(float(x), 3) for x in d_std], delta_e=round(shift, 2))
    return res, rec


def match_low_frequency(out: Image.Image, ref: Image.Image, side: int = 64) -> Image.Image:
    """Remove colour/tone drift from ``out`` by moving its large-scale CIELAB content onto ``ref``'s
    (same size); detail finer than ~1/``side`` of the frame is untouched."""
    lo = lambda im: cv2.resize(_lab(im), (side, round(side * im.height / im.width)), interpolation=cv2.INTER_AREA)
    diff = lo(ref) - lo(out)
    diff = cv2.GaussianBlur(diff, (0, 0), 1.0)
    field = cv2.resize(diff, out.size, interpolation=cv2.INTER_CUBIC)
    lab = _lab(out) + field
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    rgb = np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)
    return Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))
