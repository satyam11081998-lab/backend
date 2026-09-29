"""
Turn idempotency: one completed candidate turn -> one interviewer decision.

Keyed by (attempt_id, turn_id). A second request with the same key replays the
stored outcome instead of deciding again (no second response, no second state
fold, no second row). A concurrent duplicate waits for the first to finish.

In-process (the backend runs one uvicorn worker). Persisted idempotency for
realtime transcript rows is handled separately (attempt_messages.client_turn_id,
migration 0071), so a process restart cannot duplicate rows either.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class LedgerEntry:
    status: str = "pending"          # pending | done | failed
    result: Dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.monotonic)
    event: threading.Event = field(default_factory=threading.Event)


class TurnLedger:
    def __init__(self, ttl_s: float = 900.0, max_entries: int = 5000):
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._d: "OrderedDict[Tuple[str, str], LedgerEntry]" = OrderedDict()

    def _prune(self, now: float) -> None:
        while self._d:
            k, e = next(iter(self._d.items()))
            if now - e.created > self.ttl_s or len(self._d) > self.max_entries:
                self._d.popitem(last=False)
            else:
                break

    def begin(self, attempt_id: str, turn_id: Optional[str]) -> Tuple[bool, Optional[LedgerEntry]]:
        """(True, entry) for a new turn; (False, entry) for a duplicate. No turn_id -> (True, None)."""
        if not turn_id:
            return True, None
        key = (attempt_id, turn_id)
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            e = self._d.get(key)
            if e is not None and e.status != "failed":
                return False, e
            e = LedgerEntry()
            self._d[key] = e
            self._d.move_to_end(key)
            return True, e

    def wait(self, entry: LedgerEntry, timeout_s: float = 30.0) -> LedgerEntry:
        entry.event.wait(timeout_s)
        return entry

    def finish(self, entry: Optional[LedgerEntry], **result: Any) -> None:
        if entry is None:
            return
        entry.result.update(result)
        entry.status = "done"
        entry.event.set()

    def fail(self, entry: Optional[LedgerEntry], **result: Any) -> None:
        if entry is None:
            return
        entry.result.update(result)
        entry.status = "failed"
        entry.event.set()

    def size(self) -> int:
        with self._lock:
            return len(self._d)


class RowIdempotency:
    """(attempt_id, client_turn_id) -> message_id, for /realtime-turn saves."""

    def __init__(self, max_entries: int = 20000):
        self._lock = threading.Lock()
        self._d: "OrderedDict[Tuple[str, str], Optional[str]]" = OrderedDict()
        self._inflight: Dict[Tuple[str, str], threading.Event] = {}
        self.max_entries = max_entries

    def claim(self, attempt_id: str, key: Optional[str]) -> Tuple[bool, Optional[str]]:
        """(True, None): caller should write. (False, message_id): already written (or being written)."""
        if not key:
            return True, None
        k = (attempt_id, key)
        with self._lock:
            if k in self._d:
                return False, self._d[k]
            ev = self._inflight.get(k)
            if ev is None:
                self._inflight[k] = threading.Event()
                return True, None
        ev.wait(10.0)
        with self._lock:
            return False, self._d.get(k)

    def done(self, attempt_id: str, key: Optional[str], message_id: Optional[str]) -> None:
        if not key:
            return
        k = (attempt_id, key)
        with self._lock:
            self._d[k] = message_id
            self._d.move_to_end(k)
            while len(self._d) > self.max_entries:
                self._d.popitem(last=False)
            ev = self._inflight.pop(k, None)
        if ev:
            ev.set()

    def release(self, attempt_id: str, key: Optional[str]) -> None:
        """The write failed: let a retry try again."""
        if not key:
            return
        with self._lock:
            ev = self._inflight.pop((attempt_id, key), None)
        if ev:
            ev.set()


LEDGER = TurnLedger()
ROWS = RowIdempotency()
