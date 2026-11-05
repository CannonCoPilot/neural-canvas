"""Stage 4 — local super-resolution (Real-ESRGAN x4plus on Apple MPS) to the exact display size.

The network's colour and tone drift is removed by matching the result's large-scale colour back to the
source (``colour.match_low_frequency``); the drift it had before that is recorded (``delta_e_raw``).
Gates: the result is exactly the planned size, the needed scale is within what one x4 pass is
trusted for, and the super-resolved image keeps the source's colour (mean CIELAB ΔE against a plain
Lanczos resize, measured at low resolution where the two should agree).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

TILE, PAD = 384, 24
MAX_DELTA_E = 4.0


@dataclass
class UpscaleResult:
    method: str
    scale: float
    size: tuple[int, int]
    delta_e: float
    seconds: float
    passed: bool
    failures: list
    delta_e_raw: float = 0.0          # the network's drift before its colour was matched back

    def to_dict(self) -> dict:
        return asdict(self)


class Upscaler:
    def __init__(self, weights: str):
        import spandrel
        self.dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        desc = spandrel.ModelLoader().load_from_file(weights)
        self.scale = desc.scale
        self.net = desc.model.eval().to(self.dev)
        if self.dev.type == "mps":
            self.net = self.net.half()

    @torch.no_grad()
    def _sr(self, img: Image.Image) -> Image.Image:
        x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = x.shape
        s = self.scale
        out = np.zeros((h * s, w * s, 3), dtype=np.float32)
        dtype = torch.float16 if self.dev.type == "mps" else torch.float32
        for ty in range(0, h, TILE):
            for tx in range(0, w, TILE):
                y0, x0 = max(0, ty - PAD), max(0, tx - PAD)
                y1, x1 = min(h, ty + TILE + PAD), min(w, tx + TILE + PAD)
                t = torch.from_numpy(np.ascontiguousarray(x[y0:y1, x0:x1].transpose(2, 0, 1)))[None]
                r = self.net(t.to(self.dev, dtype)).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                cy, cx = (ty - y0) * s, (tx - x0) * s
                th, tw = (min(ty + TILE, h) - ty) * s, (min(tx + TILE, w) - tx) * s
                out[ty * s:ty * s + th, tx * s:tx * s + tw] = r[cy:cy + th, cx:cx + tw]
        if self.dev.type == "mps":
            torch.mps.empty_cache()
        return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8))

    def to_size(self, img: Image.Image, size: tuple[int, int], max_scale: float) -> tuple[Image.Image, UpscaleResult]:
        import time
        t0 = time.time()
        scale = max(size[0] / img.width, size[1] / img.height)
        failures = []
        if scale > max_scale:
            failures.append(f"needs {scale:.2f}x, above the {max_scale:.1f}x single-pass limit")
        if scale > 1.15:
            method = "realesrgan-x4plus+lanczos"
            out = self._sr(img).resize(size, Image.LANCZOS)
        else:
            method = "lanczos"
            out = img.convert("RGB").resize(size, Image.LANCZOS)
        ref = img.convert("RGB").resize(size, Image.LANCZOS)
        de_raw = round(_mean_delta_e(out, ref), 3)
        if method != "lanczos":
            from .colour import match_low_frequency
            out = match_low_frequency(out, ref)
        de = _mean_delta_e(out, ref)
        if out.size != tuple(size):
            failures.append(f"output {out.size} != planned {size}")
        if de > MAX_DELTA_E:
            failures.append(f"colour drift ΔE {de:.2f} > {MAX_DELTA_E}")
        return out, UpscaleResult(method, round(scale, 3), out.size, round(de, 3), round(time.time() - t0, 1),
                                  not failures, failures, de_raw)


def _mean_delta_e(a: Image.Image, b: Image.Image, side: int = 256) -> float:
    def lab(im):
        im = im.resize((side, round(side * im.height / im.width)), Image.BILINEAR)
        return cv2.cvtColor(np.asarray(im, dtype=np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return float(np.sqrt(((lab(a) - lab(b)) ** 2).sum(axis=2)).mean())
