"""In-process sliding-window rate limiter (per user + route class).

Abuse protection only — business limits (2 active sessions, sessions/day, budgets) are
enforced in the database where they survive restarts and multiple instances.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from ..errors import TooMany

LIMITS = {
    "turn": (40, 60.0),
    "upload": (12, 600.0),
    "session_create": (12, 3600.0),
    "voice": (60, 60.0),
    "admin": (120, 60.0),
    "read": (240, 60.0),
}

_buckets: Dict[str, Deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def check(user_key: str, klass: str) -> None:
    limit, window = LIMITS.get(klass, (120, 60.0))
    now = time.monotonic()
    key = f"{klass}:{user_key}"
    with _lock:
        q = _buckets[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise TooMany("You're going a little fast — wait a moment and try again.")
        q.append(now)


def reset_for_tests() -> None:
    with _lock:
        _buckets.clear()
