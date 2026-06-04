"""
Tiny in-memory rate limiter (no external dependency).

Good enough as a cost guard on a single Render instance: keyed per user,
fixed window. FAIL-OPEN — any internal error returns without blocking, so a
limiter bug can never break a legitimate submission. Resets on restart and is
not shared across instances; for multi-instance scale, swap for Redis later.
"""

import time
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException

_HITS: Dict[str, List[float]] = defaultdict(list)


def check_rate_limit(key: str, max_calls: int, window_seconds: int) -> None:
    """Raise HTTPException(429) if `key` exceeded `max_calls` within the window."""
    try:
        now = time.time()
        cutoff = now - window_seconds
        hits = [t for t in _HITS[key] if t >= cutoff]
        if len(hits) >= max_calls:
            _HITS[key] = hits
            raise HTTPException(
                status_code=429,
                detail="Too many requests — please wait a minute and try again.",
            )
        hits.append(now)
        _HITS[key] = hits
        # opportunistic cleanup to bound memory
        if len(_HITS) > 5000:
            for k in list(_HITS.keys()):
                if not _HITS[k] or _HITS[k][-1] < cutoff:
                    _HITS.pop(k, None)
    except HTTPException:
        raise
    except Exception:
        return  # fail open
