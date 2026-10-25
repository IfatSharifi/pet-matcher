"""Minimal Claude Messages API client for image -> JSON extraction.

Standard library only (no SDK needed), with:
  - structured outputs (the reply must match a JSON schema)
  - retries on rate limits and overload
  - a disk cache, so re-running an eval never pays twice for the same image + prompt
  - token and cost accounting

Set your key first:  $env:ANTHROPIC_API_KEY = "sk-ant-..."   (PowerShell)
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
CACHE_DIR = Path(".cache/claude")

# USD per million tokens (input, output), prompts under 100K tokens. https://claude.com/pricing
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    by_call: list = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, inp: int, out: int, secs: float, cached: bool) -> None:
        with self._lock:  # evals call Claude from several threads
            self.calls += 1
            if cached:
                self.cached_calls += 1
                return
            price_in, price_out = PRICES.get(model, (0.0, 0.0))
            self.input_tokens += inp
            self.output_tokens += out
            self.cost_usd += inp / 1e6 * price_in + out / 1e6 * price_out
            self.seconds += secs
            self.by_call.append(secs)


class ClaudeError(RuntimeError):
    pass


def image_block(path: Path, max_edge: int) -> dict:
    """Resize (never upscale) and JPEG-encode: fewer visual tokens, smaller requests."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.b64encode(buf.getvalue()).decode()}}


class ClaudeClient:
    def __init__(self, model: str = "claude-haiku-5-5", api_key: str | None = None,
                 max_retries: int = 5, use_cache: bool = True) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.usage = Usage()

    def extract(self, images: list[Path], prompt: str, schema: dict, system: str = "",
                max_edge: int = 768, max_tokens: int = 1024) -> dict:
        """Send images + prompt, return the JSON object Claude filled in."""
        content = []
        for i, p in enumerate(images, 1):
            if len(images) > 1:
                content.append({"type": "text", "text": f"Image {i}:"})
            content.append(image_block(Path(p), max_edge))
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if system:
            body["system"] = system

        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        if self.use_cache and cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            self.usage.add(self.model, 0, 0, 0.0, cached=True)
            return cached["result"]

        t0 = time.perf_counter()
        response = self._post(body)
        secs = time.perf_counter() - t0
        u = response.get("usage", {})
        self.usage.add(self.model, u.get("input_tokens", 0), u.get("output_tokens", 0), secs, cached=False)

        stop = response.get("stop_reason")
        if stop in ("refusal", "max_tokens"):
            raise ClaudeError(f"stop_reason={stop}: output may not match the schema")
        text = next((b["text"] for b in response.get("content", []) if b.get("type") == "text"), None)
        if text is None:
            raise ClaudeError("no text block in response")
        result = json.loads(text)

        if self.use_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"result": result, "usage": u}, ensure_ascii=False),
                                  encoding="utf-8")
        return result

    def _post(self, body: dict) -> dict:
        if not self.api_key:
            raise ClaudeError("ANTHROPIC_API_KEY is not set")
        data = json.dumps(body).encode()
        headers = {"x-api-key": self.api_key, "anthropic-version": API_VERSION,
                   "content-type": "application/json"}
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                retryable = e.code in (408, 429, 500, 502, 503, 504, 529)
                if not retryable or attempt == self.max_retries:
                    raise ClaudeError(f"HTTP {e.code}: {detail}") from None
                wait = float(e.headers.get("retry-after") or 2 ** attempt)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise ClaudeError(f"network error: {e}") from None
                wait = 2 ** attempt
            time.sleep(min(wait, 60))
        raise ClaudeError("unreachable")
