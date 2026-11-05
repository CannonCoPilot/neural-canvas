"""Runtime configuration.

One YAML file (``config.yaml`` at the repo root, gitignored) overrides the defaults below.
Every threshold the pipeline gates on lives here, so a gate can be audited in one place.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "paths": {
        "watch_dir": str(REPO_ROOT / "var" / "inbox"),
        "work_dir": str(REPO_ROOT / "var" / "work"),
        "output_dir": str(REPO_ROOT / "var" / "output"),
        "state_dir": str(REPO_ROOT / "var" / "state"),
    },
    "models": {
        "ollama_url": "http://localhost:11434",
        # Non-thinking instruct VLM: the 8b tag in the estate is a thinking model that burns
        # its whole budget reasoning and returns empty content.
        "vlm": "qwen3-vl:32b-instruct",
        "vlm_timeout_s": 900,
        "clip": "clip-ViT-B-32",
        "upscaler_weights": str(Path.home() / "Claude" / "Staging" / "verified" / "models"
                                / "realesrgan" / "RealESRGAN_x4plus.pth"),
    },
    # placard_scale: the wall label's size; 1.0 is the original 940 px card
    "display": {"width": 3840, "height": 2160, "placard_scale": 0.5,
                # photographed watercolour card stock (Art_AI input/card_stock.jpg); local asset, not in git
                "card_texture": str(Path.home() / "Claude" / "Staging" / "verified" / "neural-canvas"
                                    / "assets" / "card_stock.jpg")},
    "thresholds": {
        # identification: a candidate is CONFIRMED only when a reference image of it from an
        # open collection matches the input on both instruments (embedding + geometry).
        "id_clip_min": 0.86,
        "id_inliers_min": 25,
        "id_max_rounds": 3,
        # crop: every essential subject must be (almost) wholly inside the visible artwork
        "essential_retained_min": 0.97,
        # upscale: largest single-pass scale Real-ESRGAN x4 is trusted for
        "max_upscale": 4.0,
        # placard: minimum text contrast (WCAG ratio); overlap with essential subjects must be zero
        "placard_min_contrast": 7.0,
    },
    "sources": {
        "user_agent": "neural-canvas/1.0 (https://github.com/CannonCoPilot/neural-canvas)",
        "commons_api": "https://commons.wikimedia.org/w/api.php",
        "wikidata_api": "https://www.wikidata.org/w/api.php",
        # Imify's bundled SearXNG: a second free text-to-image engine beside DuckDuckGo ("" = off)
        "searxng_url": "http://127.0.0.1:8888",
    },
    # free reverse-image arm: the local Imify service (~/Claude/Staging/verified/imify/run-imify.sh)
    "imify": {"enabled": True, "url": "http://127.0.0.1:8890", "timeout_s": 90, "num_results": 40},
    # billed reverse-image fallback: Google Cloud Vision web detection, at most one call per image.
    # The key is read from the credentials file at call time; it never appears in config.
    "vision_web": {
        "enabled": True,
        "credentials_path": str(Path.home() / "Claude" / "Project_Aion" / ".claude" / "secrets" / "credentials.yaml"),
        "key_path": "image.google_vision",
        # a round that ends unconfirmed with any reference this close is a near-miss: the work is
        # out there, but no free copy matched on both instruments
        "near_miss_clip": 0.80,
        "max_side": 1600,
        "max_results": 30,
    },
    "pulse": {
        "url": "http://localhost:8800/api/v1",
        "runner_label": "runner:neural-canvas",
        "project_label": "project:neural-canvas",
        "poll_s": 10,
    },
    "google_photos": {
        "enabled": True,
        "oauth_client_path": str(Path.home() / "Claude" / "Project_Aion" / ".claude" / "secrets"
                                 / "neural-canvas-gphotos-client.json"),
        "token_path": str(Path.home() / "Claude" / "Project_Aion" / ".claude" / "secrets"
                          / "neural-canvas-gphotos-token.json"),
        "album_name": "Neural Canvas",
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class Config:
    data: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS))

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def thresholds(self) -> dict:
        return self.data["thresholds"]

    def path(self, key: str) -> Path:
        p = Path(os.path.expanduser(self.data["paths"][key]))
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else Path(os.environ.get("NEURAL_CANVAS_CONFIG", REPO_ROOT / "config.yaml"))
    over = yaml.safe_load(path.read_text()) if path.exists() else {}
    return Config(_merge(DEFAULTS, over or {}))
