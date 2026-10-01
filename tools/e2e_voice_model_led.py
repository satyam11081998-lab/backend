"""
Backend for the model-led voice browser E2E (frontend qa/e2e-voice/run-model-led.cjs).

The REAL routes (/realtime/session, /attempts/{id}/realtime-turn, /voice-coach,
/voice-tool) on an in-memory database, with OpenAI's client_secrets pointed at the
mock realtime peer and a scripted hint model. No network, no keys, nothing written
anywhere but memory.

    python -m tools.e2e_voice_model_led --port 8765 --mock http://127.0.0.1:8766
"""
from __future__ import annotations

import argparse
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k, _v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
               "SUPABASE_SERVICE_ROLE_KEY": "test"}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"
os.environ["VOICE_INTERVIEWER"] = "model_led"

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

import routes.attempts as att  # noqa: E402
import routes.realtime as rt  # noqa: E402
import routes.voice_coach as rvc  # noqa: E402
import services.ai_providers as ap  # noqa: E402
import services.ai_usage as au  # noqa: E402

CASE = {"id": "c1", "type": "guesstimate", "title": "EV scooters", "difficulty": "medium", "is_active": True,
        "content": "Estimate the number of electric scooters currently operating in Chennai.",
        "hint": "Start from households and two-wheeler ownership before electric penetration.",
        "solution": ("Chennai has about 27 lakh households; 60% own a two-wheeler, half are scooters, 6% electric, "
                     "so the estimate is about 48,000 electric scooters.")}


class _Res:
    def __init__(self, data=None, count=None):
        self.data, self.count = data, count


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.filters, self.single = db, table, "select", None, [], False

    def select(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self

    def insert(self, p):
        self.op, self.payload = "insert", p
        return self

    def update(self, p):
        self.op, self.payload = "update", p
        return self

    def eq(self, k, v):
        self.filters.append((k, v))
        return self

    def maybe_single(self):
        self.single = True
        return self

    def execute(self):
        rows = self.db.tables.setdefault(self.table, [])
        if self.op == "insert":
            row = dict(self.payload, id=f"{self.table}-{len(rows) + 1}", created_at=f"t{len(rows) + 1:04d}")
            rows.append(row)
            return _Res([row])
        hit = [r for r in rows if all(r.get(k) == v for k, v in self.filters)]
        if self.op == "update":
            for r in hit:
                r.update(self.payload)
            return _Res(hit)
        return _Res(hit[0] if hit else None) if self.single else _Res(hit, count=len(hit))


class DB:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tables = {
            "attempts": [{"id": "a1", "user_id": "u1", "status": "active", "case_id": "c1",
                          "clarification_quota": 20, "clarification_used": 0, "tier_at_start": "pro",
                          "session_state": {}}],
            "cases": [dict(CASE)],
            "attempt_messages": [],
        }

    def table(self, name):
        return _Q(self, name)


DBI = DB()


class _Choice:
    def __init__(self, c):
        self.message = types.SimpleNamespace(content=c)


class HintModel:
    """Scripted hint model: a hint, or (for the answer framework) a funnel."""
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        system = kw["messages"][0]["content"]
        text = ("Think of it as a funnel from households to electric scooters." if "asked for the answer" in system
                else "Look at households before scooters.")
        return types.SimpleNamespace(choices=[_Choice(text)], usage=None, id="fake")


def build_app(mock_url: str) -> FastAPI:
    user = types.SimpleNamespace(id="u1", email="owner@example.com", is_anonymous=False)
    for mod in (att, rt, rvc):
        mod.get_supabase_client = lambda: DBI
    for mod in (att, rt, rvc):
        mod.get_verified_user = lambda sb, a: ("u1", user)
        mod.check_rate_limit = lambda *a, **k: None
    att.get_verified_user_id = lambda sb, a: "u1"
    att.is_guest_user = lambda u: False
    att.assert_daily_budget = lambda *a, **k: None
    att.log_realtime_usage = lambda **k: None
    att.deduct_realtime_credit = lambda *a, **k: None
    att._case_cache.clear()
    rvc.assert_daily_budget = lambda *a, **k: None
    rt.assert_daily_budget = lambda *a, **k: None
    rt.get_ai_input_quota = lambda sb, uid: {"tier": "pro"}
    rt.has_credit = lambda *a, **k: True
    rt.get_balance = lambda *a, **k: {"total_remaining": 30}
    rt.log_ai_usage = lambda **k: None
    rt.OPENAI_API_KEY = "sk-test"
    rt.CLIENT_SECRETS_URL = f"{mock_url}/v1/realtime/client_secrets"
    ap.openai_client = lambda: HintModel()
    au.log_ai_usage = lambda **k: None

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(rt.router, prefix="/realtime")
    app.include_router(att.router)
    app.include_router(rvc.router)

    @app.get("/__e2e/db")
    def dump():
        a = DBI.tables["attempts"][0]
        return {"messages": DBI.tables["attempt_messages"], "session_state": a.get("session_state")}

    @app.post("/__e2e/reset")
    def reset():
        DBI.reset()
        att._case_cache.clear()
        return {"ok": True}

    return app


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--mock", default="http://127.0.0.1:8766")
    a = p.parse_args()
    uvicorn.run(build_app(a.mock), host="127.0.0.1", port=a.port, log_level="warning")
