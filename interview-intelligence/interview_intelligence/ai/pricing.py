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
