"""USD per 1M tokens (input, output). Override with II_PRICE_TABLE='{"model": [in, out]}'.

Unknown models are logged with cost 0 AND flagged `unpriced` in model_runs so a new
model cannot silently disappear from the budget (the lesson from the main backend's
TTS-at-$0 bug, CHANGELOG 2026-08-13).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

DEFAULT_PRICES = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "simulated": (0.0, 0.0),
}


def _table():
    t = dict(DEFAULT_PRICES)
    raw = os.environ.get("II_PRICE_TABLE", "").strip()
    if raw:
        try:
            for k, v in json.loads(raw).items():
                t[k] = (float(v[0]), float(v[1]))
        except (ValueError, TypeError, IndexError):
            pass
    return t


def price(model: str) -> Optional[Tuple[float, float]]:
    t = _table()
    if model in t:
        return t[model]
    for k, v in t.items():  # dated variants, e.g. gpt-4o-mini-2024-07-18
        if model.startswith(k + "-"):
            return v
    return None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> Tuple[float, bool]:
    p = price(model)
    if p is None:
        return 0.0, False
    return round(input_tokens / 1e6 * p[0] + output_tokens / 1e6 * p[1], 6), True


# Realtime speech models bill audio and text separately, per 1M tokens. Audio runs at about
# 600 tokens per minute in and 1,200 per minute out. Override with II_REALTIME_PRICES='{...}'.
REALTIME_PRICES = {"audio_in": 32.00, "audio_in_cached": 0.40, "text_in": 4.00, "text_in_cached": 0.40,
                   "audio_out": 64.00, "text_out": 16.00}


def realtime_cost_usd(usage: dict) -> Tuple[float, dict]:
    """Cost of one realtime `response.done` usage block -> (usd, token breakdown)."""
    prices = dict(REALTIME_PRICES)
    raw = os.environ.get("II_REALTIME_PRICES", "").strip()
    if raw:
        try:
            prices.update({k: float(v) for k, v in json.loads(raw).items() if k in prices})
        except (ValueError, TypeError, AttributeError):
            pass

    def _n(d, *path) -> int:
        cur = d
        for k in path:
            if not isinstance(cur, dict):
                return 0
            cur = cur.get(k)
        try:
            return max(0, min(int(cur or 0), 2_000_000))
        except (TypeError, ValueError):
            return 0

    def _d(x) -> dict:
        return x if isinstance(x, dict) else {}

    usage = _d(usage)
    ind = _d(usage.get("input_token_details"))
    outd = _d(usage.get("output_token_details"))
    cached = _d(ind.get("cached_tokens_details"))
    a_in, t_in = _n(ind, "audio_tokens"), _n(ind, "text_tokens")
    a_cached, t_cached = _n(cached, "audio_tokens"), _n(cached, "text_tokens")
    if not a_cached and not t_cached and _n(ind, "cached_tokens"):
        t_cached = _n(ind, "cached_tokens")
    a_out, t_out = _n(outd, "audio_tokens"), _n(outd, "text_tokens")
    if not (a_in or t_in or a_out or t_out):  # older shape: totals only
        t_in, t_out = _n(usage, "input_tokens"), _n(usage, "output_tokens")
    cost = ((max(0, a_in - a_cached) * prices["audio_in"] + a_cached * prices["audio_in_cached"]
             + max(0, t_in - t_cached) * prices["text_in"] + t_cached * prices["text_in_cached"]
             + a_out * prices["audio_out"] + t_out * prices["text_out"]) / 1e6)
    return round(cost, 6), {"audio_in": a_in, "audio_in_cached": a_cached, "text_in": t_in,
                            "text_in_cached": t_cached, "audio_out": a_out, "text_out": t_out}
