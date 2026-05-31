"""Shared Anthropic client + cost tracking + JSON-extraction helper."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field

from anthropic import Anthropic

_client: Anthropic | None = None
_client_lock = threading.Lock()


def client() -> Anthropic:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError(
                        "ANTHROPIC_API_KEY missing. Set it in reddit-crawler/.env "
                        "(see .env.example)."
                    )
                _client = Anthropic(api_key=key)
    return _client


# Default worker count for parallel LLM fan-out. The Anthropic SDK is safe to
# call from multiple threads; this caps in-flight requests to stay polite.
MAX_WORKERS = 8


# Pricing per million tokens (USD). Approximate; for in-script estimates only.
PRICING = {
    "claude-haiku-4-5-20251001": {"in": 1.00, "in_cached": 0.10, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "in_cached": 0.30, "out": 15.00},
}


@dataclass
class CostTracker:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    per_model_calls: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, usage) -> None:
        prices = PRICING.get(model, {"in": 1.0, "in_cached": 0.1, "out": 5.0})
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        # Lock only the shared-state mutation; runs from many worker threads.
        with self._lock:
            self.calls += 1
            self.per_model_calls[model] = self.per_model_calls.get(model, 0) + 1
            self.input_tokens += in_tok
            self.cached_input_tokens += cached
            self.output_tokens += out_tok
            self.cost_usd += (
                in_tok / 1_000_000 * prices["in"]
                + cached / 1_000_000 * prices["in_cached"]
                + out_tok / 1_000_000 * prices["out"]
            )

    def summary(self) -> str:
        return (
            f"calls={self.calls} "
            f"in={self.input_tokens} cached={self.cached_input_tokens} "
            f"out={self.output_tokens} cost=${self.cost_usd:.3f}"
        )


COST = CostTracker()


def call_with_cached_system(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
) -> str:
    """Single Claude call with the system block marked for ephemeral caching."""
    resp = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    COST.add(model, resp.usage)
    parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    return "".join(parts).strip()


_json_block_re = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def extract_json(text: str):
    """Pull the first JSON value out of an LLM response. Returns None on failure."""
    s = text.strip()
    if not s:
        return None
    m = _json_block_re.search(s)
    if m:
        s = m.group(1).strip()
    # Try whole-string first
    for candidate in (s, _largest_brace_span(s)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _largest_brace_span(s: str) -> str | None:
    starts = [i for i, c in enumerate(s) if c in "{["]
    ends = [i for i, c in enumerate(s) if c in "}]"]
    if not starts or not ends:
        return None
    return s[starts[0] : ends[-1] + 1]
