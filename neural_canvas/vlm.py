"""Local vision-language model client (Ollama).

Every model answer is parsed against an explicit key list; a malformed answer is retried
with the parse error fed back, and raises after the retry budget instead of being patched up.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time

import requests
from PIL import Image

log = logging.getLogger(__name__)


class VLMError(RuntimeError):
    pass


def image_b64(img: Image.Image, max_side: int = 1024, quality: int = 90) -> str:
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in answer")
    return json.loads(text[start:end + 1])


class VLM:
    def __init__(self, url: str, model: str, timeout_s: int = 900):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.calls: list[dict] = []   # per-call telemetry, copied into the run record

    def ask_json(self, prompt: str, images: list[Image.Image] | None = None, required: tuple[str, ...] = (),
                 max_side: int = 1024, retries: int = 2, num_predict: int = 2048) -> dict:
        imgs = [image_b64(i, max_side) for i in (images or [])]
        messages = [{"role": "user", "content": prompt, **({"images": imgs} if imgs else {})}]
        last_err = None
        for attempt in range(retries + 1):
            t0 = time.time()
            r = requests.post(f"{self.url}/api/chat", timeout=self.timeout_s, json={
                "model": self.model, "stream": False, "think": False, "format": "json",
                "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 16384},
                "messages": messages,
            })
            r.raise_for_status()
            body = r.json()
            content = body.get("message", {}).get("content", "")
            self.calls.append({"model": self.model, "seconds": round(time.time() - t0, 1),
                               "eval_count": body.get("eval_count"), "attempt": attempt})
            try:
                data = _extract_json(content)
                missing = [k for k in required if k not in data]
                if missing:
                    raise ValueError(f"missing keys {missing}")
                return data
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("VLM answer unusable (%s); retrying", e)
                messages = messages + [
                    {"role": "assistant", "content": content[:4000]},
                    {"role": "user", "content": f"That answer was not usable ({e}). Reply again with ONLY "
                                                f"the JSON object, including keys: {', '.join(required)}."},
                ]
        raise VLMError(f"no usable answer after {retries + 1} attempts: {last_err}")
