"""
Gemini Live voice: who may start a session (2026-10-03).

Free accounts get the one-time voice trial (14 min, 7 min per session) on Gemini
Live, exactly as on OpenAI realtime. Before this, Gemini refused every non-Pro
account with "Voice interview is a Pro feature", so the trial was never usable
once the admin switched voice_mode to gemini.

Fakes only (no network, no keys, no Supabase):
  guest                    -> 403 "Create an account..."
  free, trial left         -> session; max_session_seconds 420; short token
  free, 3 min left         -> session capped at 180 s
  free, < 30 s left        -> 402 free message
  free, trial used         -> 402 free message (mentions Upgrade)
  free, network over cap   -> 429
  Pro, in credit           -> session, no cap, 30-minute token
  Pro, no credit           -> 402 Pro message (minute pack)

Run:  python -m tests.test_gemini_free_trial
"""
from __future__ import annotations

import datetime
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k, _v in {
    "OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test", "GEMINI_API_KEY": "test",
}.items():
    os.environ.setdefault(_k, _v)
os.environ.pop("REALTIME_FREE_SESSION_SECONDS", None)
os.environ.pop("REALTIME_FREE_IP_PER_DAY", None)
os.environ["VOICE_INTERVIEWER"] = "renderer"   # keep the case prompt builder out of it

# Missing SDKs (openai / supabase / dotenv) are stubbed only when absent; see tests/_sdk_stubs.py.
from tests._sdk_stubs import install as _install_sdk_stubs  # noqa: E402
_install_sdk_stubs()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.realtime_gemini as rg  # noqa: E402

passed = 0


def check(name, cond, extra=""):
    global passed
    if not cond:
        print("  FAIL", name, extra)
        raise SystemExit(1)
    passed += 1
    print("  ok", name)


# ---------------------------------------------------------------- fakes
STATE = {"guest": False, "tier": "free", "remaining": 14.0, "ip_over": False}
MINTED: list = []


class _Q:
    def __init__(self, rows):
        self._rows = rows

    def __getattr__(self, _name):
        return lambda *a, **k: self

    def execute(self):
        return types.SimpleNamespace(data=self._rows)


class _SB:
    def table(self, name):
        if name == "cases":
            return _Q([{"id": "c1", "type": "profitability", "difficulty": "medium", "market": "IN",
                        "content": "x", "title": "t"}])
        return _Q([])


def _rate_limit(key, max_calls, window_seconds):
    if key.startswith("rt_ip_day:") and STATE["ip_over"]:
        raise HTTPException(status_code=429, detail="Too many requests")


rg.get_supabase_client = lambda: _SB()
rg.get_verified_user = lambda sb, auth: ("u1", types.SimpleNamespace(email="a@b.c"))
rg.is_guest_user = lambda u: STATE["guest"]
rg.check_rate_limit = _rate_limit
rg.assert_daily_budget = lambda: None
rg.get_ai_input_quota = lambda sb, uid: {"tier": STATE["tier"]}
rg.get_balance = lambda sb, uid, tier: {"total_remaining": STATE["remaining"], "tier": tier}
rg.log_ai_usage = lambda **kw: None
rg.voice_interviewer_mode = lambda uid, email: "renderer"
rg._resolve_live_model = lambda: "gemini-test-live"


class _Tokens:
    def create(self, config):
        MINTED.append(config)
        return types.SimpleNamespace(name="auth_tokens/test")


_genai = types.ModuleType("google.genai")
_genai.Client = lambda api_key: types.SimpleNamespace(auth_tokens=_Tokens())
_google = sys.modules.get("google") or types.ModuleType("google")
_google.genai = _genai
sys.modules["google"] = _google
sys.modules["google.genai"] = _genai

app = FastAPI()
app.include_router(rg.router, prefix="/realtime-gemini")
client = TestClient(app)


def start(**state):
    STATE.update({"guest": False, "tier": "free", "remaining": 14.0, "ip_over": False})
    STATE.update(state)
    MINTED.clear()
    return client.post("/realtime-gemini/session", json={"case_id": "c1"},
                       headers={"Authorization": "Bearer t", "X-Forwarded-For": "1.2.3.4"})


def token_minutes():
    cfg = MINTED[-1]
    exp = cfg["expire_time"] - datetime.datetime.now(tz=datetime.timezone.utc)
    return exp.total_seconds() / 60


print("pure cap")
check("Pro -> no cap", rg.free_session_cap("pro", 50) is None)
check("free, full trial -> 420 s", rg.free_session_cap("free", 14) == 420)
check("free, 3 min left -> 180 s", rg.free_session_cap("free", 3) == 180)
check("lite counts as non-Pro", rg.free_session_cap("lite", 14) == 420)
check("free, nothing left -> 0", rg.free_session_cap("free", 0) == 0)

print("session gate")
r = start(guest=True)
check("guest -> 403 sign in", r.status_code == 403 and "Create an account" in r.json()["detail"], r.text)

r = start(tier="free", remaining=14.0)
check("free with trial -> 200", r.status_code == 200, r.text)
check("free: max_session_seconds 420", r.json().get("max_session_seconds") == 420, r.text)
check("free: credits echoed", r.json().get("credits", {}).get("total_remaining") == 14.0)
check("free: token expires with the session (~10 min, not 30)", 9.5 < token_minutes() < 10.5, str(token_minutes()))

r = start(tier="free", remaining=3.0)
check("free, 3 min left -> 180 s", r.status_code == 200 and r.json()["max_session_seconds"] == 180, r.text)

r = start(tier="free", remaining=0.3)
check("free, 18 s left -> 402", r.status_code == 402 and "Upgrade" in r.json()["detail"], r.text)

r = start(tier="free", remaining=0.0)
check("free, trial used -> 402 free message", r.status_code == 402 and "free voice interview minutes" in r.json()["detail"], r.text)
check("free, trial used: says Upgrade", "Upgrade" in r.json()["detail"])

r = start(tier="free", ip_over=True)
check("free, network over daily cap -> 429", r.status_code == 429 and "free voice limit" in r.json()["detail"], r.text)

r = start(tier="pro", remaining=40.0, ip_over=True)
check("Pro ignores the network cap and has no session cap", r.status_code == 200 and r.json()["max_session_seconds"] is None, r.text)
check("Pro: 30-minute token as before", 29.5 < token_minutes() < 30.5, str(token_minutes()))

r = start(tier="pro", remaining=0.0)
check("Pro, no credit -> 402 minute-pack message", r.status_code == 402 and "minute pack" in r.json()["detail"], r.text)

r = start(tier="free")
check("no response says 'Pro feature' any more", "Pro feature" not in r.text)

print(f"\n{passed} checks passed.")
