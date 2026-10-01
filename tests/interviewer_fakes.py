"""
Fakes for the unified-interviewer-brain test suites: a scripted provider (no
network, no keys) and an in-memory Supabase stand-in with the two behaviours the
brain relies on (a unique (attempt_id, client_turn_id) index, and a switch that
simulates a database where migration 0071 has not been run).
"""
from __future__ import annotations

import os
import re
import sys
import threading
import types
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k, _v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
               "SUPABASE_SERVICE_ROLE_KEY": "test", "GEMINI_API_KEY": "test"}.items():
    os.environ.setdefault(_k, _v)
os.environ.setdefault("INTERVIEWER_TELEMETRY", "off")

from services.interviewer.providers import CallMeta  # noqa: E402
from services.interviewer.types import ProviderError, ProviderTimeout  # noqa: E402

# One clean line per move - what a well-behaved model returns.
GOOD = {
    "MICRO_HINT": "Think about how many households could realistically own a car first.",
    "TARGETED_HINT": "You need the share of those households that buy a new car in a given year.",
    "STRUCTURAL_HINT": "Break it into three pieces: total households, the share that can afford a car, and how often they replace it.",
    "DEMONSTRATION": "Take 30 crore households; if about 10 percent can afford a car, that is 3 crore. You take the replacement cycle next.",
    "DELIVER_SOLUTION": "Start from 30 crore households, keep the 10 percent that can afford a car, divide by a seven-year replacement cycle, and add first-time buyers. That lands near 40 lakh cars a year.",
    "REPAIR": "Fair enough, let's simplify. We only need yearly new-car sales, so start from households that can afford one.",
    "DATA_REVEAL": "Volumes are down 12 percent year on year while prices held flat.",
    "ANSWER_DIRECT": "Use annual figures for India only.",
    "TRANSITION": "Let's move to the numbers: the client sold 2 lakh units last year at an average price of Rs 6 lakh.",
    "DIRECT_CORRECTION": "Check the units on that step: you mixed monthly and yearly figures.",
    "TARGETED_PROBE": "Where does the replacement demand from existing owners come in?",
    "RETHINK_CUE": "Does that figure hold up against roughly 3 crore car-owning households?",
}

REFUSE = "That's the exercise. What's your next step?"
PRAISE = "Great question! "
MULTI_Q = " What else? And why?"


CONTEXTUAL = {"ACKNOWLEDGE_AND_CONTINUE", "REFLECT_PROGRESS", "ACKNOWLEDGE_AND_ORIENT"}


def move_of(messages: List[Dict[str, str]]) -> str:
    sysmsg = messages[0]["content"] if messages else ""
    m = re.search(r'"response_function":\s*"([A-Z_]+)"', sysmsg) or re.search(r"MOVE:\s*([A-Z_]+)", sysmsg)
    return m.group(1) if m else "ANSWER_DIRECT"


def contextual_line(messages: List[Dict[str, str]], move: str) -> str:
    """What a well-behaved model says for a contextual beat: about the candidate's own words."""
    last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    words = re.findall(r"[A-Za-z0-9.]+", last)[:7]
    gist = " ".join(words).lower() or "that step"
    if move == "REFLECT_PROGRESS":
        return f"So your structure runs: {gist}; take it from there."
    if move == "ACKNOWLEDGE_AND_ORIENT":
        return f"That closes out {gist}; the next part of your plan is yours to take on."
    return f"Noted - {gist}; carry on from there."


class FakeLLM:
    """Provider stand-in with the InterviewerLLM interface (complete / stream).

    mode: good | praise | refuse | multi_q | empty | leak | error | timeout | refuse_then_good | callable
    """

    def __init__(self, mode: Union[str, Callable] = "good"):
        self.mode = mode
        self.calls: List[Dict] = []
        self.lock = threading.Lock()

    def _text(self, messages) -> str:
        mv = move_of(messages)
        base = contextual_line(messages, mv) if mv in CONTEXTUAL else GOOD.get(mv, "Use annual figures for India only.")
        mode = self.mode
        with self.lock:
            n = len(self.calls)
        if callable(mode):
            return mode(messages, mv)
        if mode == "praise":
            return PRAISE + base
        if mode == "refuse":
            return REFUSE
        if mode == "refuse_then_good":
            return REFUSE if n <= 1 else base
        if mode == "multi_q":
            return base + MULTI_Q
        if mode == "empty":
            return ""
        if mode == "leak":
            return f"MOVE: {mv}. TASK: follow rules. " + base
        return base

    def _record(self, messages, kind):
        with self.lock:
            self.calls.append({"kind": kind, "move": move_of(messages), "messages": messages})

    def complete(self, messages, *, max_tokens: int) -> Tuple[str, CallMeta]:
        self._record(messages, "complete")
        if self.mode == "error":
            raise ProviderError("fake provider down")
        if self.mode == "timeout":
            raise ProviderTimeout("fake timeout")
        return self._text(messages), CallMeta(provider="fake", model="fake-model", tokens_in=100, tokens_out=20,
                                              latency_ms=1)

    def stream(self, messages, *, max_tokens: int, meta: CallMeta) -> Iterator[str]:
        self._record(messages, "stream")
        if self.mode == "error":
            raise ProviderError("fake provider down")
        if self.mode == "timeout":
            raise ProviderTimeout("fake timeout")
        meta.provider, meta.model, meta.first_token_ms = "fake", "fake-model", 1
        text = self._text(messages)
        for i in range(0, len(text), 5):
            yield text[i:i + 5]
        meta.tokens_in, meta.tokens_out, meta.latency_ms = 100, 20, 2


# ---------------------------------------------------------------------------
# In-memory Supabase
# ---------------------------------------------------------------------------
class _Res:
    def __init__(self, data=None, count=None):
        self.data, self.count = data, count


class APIError(Exception):
    pass


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.filters, self.single = db, table, "select", None, [], False
        self._limit = None

    def select(self, *a, **k): return self
    def order(self, *a, **k): return self

    def limit(self, n, *a, **k):
        self._limit = n
        return self

    def insert(self, p):
        self.op, self.payload = "insert", p
        return self

    def update(self, p):
        self.op, self.payload = "update", p
        return self

    def upsert(self, p, **k):
        self.op, self.payload = "insert", p
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def gte(self, k, v): return self
    def in_(self, k, v): return self

    def maybe_single(self):
        self.single = True
        return self

    def execute(self):
        with self.db.lock:
            rows = self.db.tables.setdefault(self.table, [])
            if self.op == "insert":
                p = dict(self.payload)
                if self.table == "attempt_messages" and "client_turn_id" in p:
                    if not self.db.has_client_turn_id:
                        raise APIError("PGRST204: Could not find the 'client_turn_id' column of 'attempt_messages' in the schema cache")
                    if p["client_turn_id"] and any(r.get("attempt_id") == p.get("attempt_id")
                                                   and r.get("client_turn_id") == p["client_turn_id"] for r in rows):
                        raise APIError('23505: duplicate key value violates unique constraint "attempt_messages_client_turn_uidx"')
                row = dict(p, id=f"{self.table}-{len(rows) + 1}", created_at=f"t{len(rows) + 1:05d}")
                rows.append(row)
                self.db.writes.append(("insert", self.table, row))
                return _Res([row])
            hit = [r for r in rows if all(r.get(k) == v for k, v in self.filters)]
            if self.op == "update":
                for r in hit:
                    r.update(self.payload)
                self.db.writes.append(("update", self.table, dict(self.payload)))
                return _Res(hit)
            if self._limit is not None:
                hit = hit[: self._limit]
            return _Res(dict(hit[0]) if hit else None) if self.single else _Res([dict(h) for h in hit], count=len(hit))


class FakeDB:
    def __init__(self, history=None, case_type="guesstimate", status="active", used=0, quota=20,
                 content="Estimate the number of cars sold in India per year.", has_client_turn_id=True,
                 session_state=None, teaching_policy="coached"):
        self.lock = threading.RLock()
        self.writes: List[Tuple] = []
        self.has_client_turn_id = has_client_turn_id
        self.tables = {
            "attempts": [{"id": "a1", "user_id": "u1", "status": status, "case_id": "c1",
                          "clarification_quota": quota, "clarification_used": used,
                          "tier_at_start": "pro", "session_state": dict(session_state or {})}],
            "cases": [{"id": "c1", "type": case_type, "title": "Cars", "difficulty": "easy",
                       "content": content, "is_active": True, "teaching_policy": teaching_policy,
                       "solution": "HIDDEN-SOLUTION-TEXT-42"}],
            "attempt_messages": [dict(m, attempt_id="a1", id=f"m{i}", created_at=f"t{i:05d}")
                                 for i, m in enumerate(history or [])],
            "ai_usage_log": [],
        }

    def table(self, name):
        return _Q(self, name)

    def rows(self, role=None):
        msgs = self.tables["attempt_messages"]
        return [m for m in msgs if role is None or m.get("role") == role]

    def inserted(self, table="attempt_messages", role=None):
        return [w[2] for w in self.writes if w[0] == "insert" and w[1] == table and (role is None or w[2].get("role") == role)]

    def session_state(self):
        return self.tables["attempts"][0].get("session_state") or {}


def install_route_fakes(att_module, db_getter: Callable[[], FakeDB], *, email: Optional[str] = None):
    """Patch routes.attempts the same way tests/test_v11_voice_integration.py does."""
    from fastapi import HTTPException

    def _user(sb, auth):
        if not auth:
            raise HTTPException(status_code=401, detail="Missing authentication token")
        return "u1", {"id": "u1", "email": email}

    att_module.get_supabase_client = lambda: db_getter()
    att_module.get_verified_user = _user
    att_module.get_verified_user_id = lambda sb, auth: "u1"
    att_module.is_guest_user = lambda u: False
    att_module.check_rate_limit = lambda *a, **k: None
    att_module.assert_daily_budget = lambda *a, **k: None
    att_module.llm_case_content = lambda case: case["content"]


def parse_sse(text: str) -> List[Tuple[str, str]]:
    out = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev = next((l[7:] for l in block.split("\n") if l.startswith("event: ")), "message")
        data = next((l[6:] for l in block.split("\n") if l.startswith("data: ")), "")
        out.append((ev, data))
    return out
