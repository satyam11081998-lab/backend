"""
Per-key mutual exclusion inside this process.

Most route handlers used to be `async def` doing blocking I/O, which (by
accident) made the event loop run them one at a time. Now that they run on
worker threads -- so one slow request no longer freezes every other user --
the few read-then-write sequences that silently relied on that one-at-a-time
behaviour take an explicit lock for just their key (one attempt, one user),
leaving everything else concurrent.

Fail-open like the rest of the product: if a lock cannot be taken within
`timeout` the caller proceeds without it (and it is logged), so a stuck request
can never wedge an interview.
"""

import threading
from contextlib import contextmanager
from typing import Dict, List

_guard = threading.Lock()
_locks: Dict[str, List] = {}  # key -> [Lock, users]


@contextmanager
def keyed_lock(key: str, timeout: float = 30.0):
    with _guard:
        ent = _locks.get(key)
        if ent is None:
            ent = _locks[key] = [threading.Lock(), 0]
        ent[1] += 1
    got = ent[0].acquire(timeout=timeout)
    if not got:
        print(f"[keyed_lock] {key}: not acquired within {timeout}s; proceeding without it")
    try:
        yield got
    finally:
        if got:
            ent[0].release()
        with _guard:
            ent[1] -= 1
            if ent[1] == 0 and _locks.get(key) is ent:
                del _locks[key]
