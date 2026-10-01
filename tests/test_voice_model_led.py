"""
Model-led realtime voice interviewer (VOICE_INTERVIEWER): the speech model
converses by itself; the server coaches alongside.

Proves, with fakes only (no network, no keys):
  1. SWITCH    -- off by default; on for everyone, or only for an allowlist.
  2. SESSION   -- in model-led mode the realtime session carries the playbook +
                  case (never the solution), the hint/answer tools, and lets the
                  model answer by itself (create_response on, barge-in kept). The
                  default renderer session is unchanged.
  3. COACH     -- the learner read turns into notes: asked for help, stuck,
                  frustrated, asked for the answer, interviewer over-questioning,
                  clarifications used up, progressing, answer already given.
  4. HINTS     -- the ladder climbs cue -> hint -> framework -> partial step,
                  frustration skips the gentle cue, and no hint ever carries the
                  solution's final number (even if the hint model writes it).
  5. ANSWER    -- first ask: a framework + "results page", no solution; asking
                  again: the worked answer with the honest results caveat.
  6. ROUTES    -- /voice-coach and /voice-tool: switch-gated, owner-checked,
                  state persisted in attempts.session_state.

Run:  python -m tests.test_voice_model_led
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k, _v in {
    "OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test", "GEMINI_API_KEY": "test",
}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"
os.environ.pop("VOICE_INTERVIEWER", None)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.realtime as rt  # noqa: E402
import routes.voice_coach as rvc  # noqa: E402
import routes.attempts as att  # noqa: E402
from services import voice_coach as vc  # noqa: E402
from prompts.voice_renderer import VOICE_RENDERER_INSTRUCTIONS  # noqa: E402
from prompts.voice_interviewer_playbook import build_voice_interviewer_instructions  # noqa: E402

_fail: list = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond or not detail else f"   [{detail}]"))
    if not cond:
        _fail.append(name)


CASE_CONTENT = ("Estimate the number of electric scooters currently operating in Chennai. Make assumptions about "
                "population, two-wheeler use, EV penetration and scooter lifecycle.")
SOLUTION = ("Chennai has about 1.1 crore people in roughly 27 lakh households. About 60% of households own a "
            "two-wheeler, so 16 lakh two-wheelers. Scooters are about half, 8 lakh. With 6% electric penetration, "
            "the estimate is about 48,000 electric scooters operating in Chennai.")
HINT = "Start from households and two-wheeler ownership before electric penetration."


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
            self.db.writes.append(("insert", self.table, row))
            return _Res([row])
        hit = [r for r in rows if all(r.get(k) == v for k, v in self.filters)]
        if self.op == "update":
            for r in hit:
                r.update(self.payload)
            self.db.writes.append(("update", self.table, dict(self.payload)))
            return _Res(hit)
        return _Res(hit[0] if hit else None) if self.single else _Res(hit, count=len(hit))


class FakeDB:
    def __init__(self, history, session_state=None, owner="u1", status="active", used=0, quota=20):
        self.writes = []
        self.tables = {
            "attempts": [{"id": "a1", "user_id": owner, "status": status, "case_id": "c1",
                          "clarification_quota": quota, "clarification_used": used,
                          "tier_at_start": "pro", "session_state": dict(session_state or {})}],
            "cases": [{"id": "c1", "type": "guesstimate", "title": "EV scooters", "difficulty": "medium",
                       "content": CASE_CONTENT, "hint": HINT, "solution": SOLUTION, "is_active": True}],
            "attempt_messages": [dict({"kind": "voice"}, **m, attempt_id="a1", id=f"m{i}", created_at=f"t{i:04d}")
                                 for i, m in enumerate(history)],
        }

    def table(self, name):
        return _Q(self, name)

    def state(self):
        return self.tables["attempts"][0]["session_state"]


print("=" * 72)
print("1. SWITCH")
print("=" * 72)
check("default: renderer (today's flow)", vc.voice_interviewer_mode("u1", "a@b.com") == "renderer")
os.environ["VOICE_INTERVIEWER"] = "model_led"
check("VOICE_INTERVIEWER=model_led: everyone", vc.voice_interviewer_mode("u9", None) == "model_led")
os.environ["VOICE_INTERVIEWER"] = "allowlist"
os.environ["VOICE_INTERVIEWER_ALLOWLIST"] = "Owner@Example.com, u7"
check("allowlist: listed email (case-insensitive)", vc.voice_interviewer_mode("u1", "owner@example.com") == "model_led")
check("allowlist: listed user id", vc.voice_interviewer_mode("u7", None) == "model_led")
check("allowlist: everyone else keeps today's flow", vc.voice_interviewer_mode("u8", "x@y.com") == "renderer")
os.environ.pop("VOICE_INTERVIEWER")

print()
print("=" * 72)
print("2. SESSION")
print("=" * 72)


class _FakeResp:
    status_code = 200
    text = "{}"

    def json(self):
        return {"value": "ek_test", "expires_at": 1}


class _FakeAsyncClient:
    last = None

    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def post(self, url, headers=None, json=None):
        _FakeAsyncClient.last = json
        return _FakeResp()


rt.httpx = types.SimpleNamespace(AsyncClient=_FakeAsyncClient)
rt.OPENAI_API_KEY = "sk-test"
rt.check_rate_limit = lambda *a, **k: None
rt.assert_daily_budget = lambda *a, **k: None
rt.get_ai_input_quota = lambda sb, uid: {"tier": "pro"}
rt.has_credit = lambda *a, **k: True
rt.get_balance = lambda *a, **k: {"total_remaining": 10}
rt.log_ai_usage = lambda **k: None
rt.get_verified_user = lambda sb, a: ("u1", types.SimpleNamespace(id="u1", email="owner@example.com"))


def mint(env=None, state=None):
    os.environ.pop("VOICE_INTERVIEWER", None)
    if env:
        os.environ["VOICE_INTERVIEWER"] = env
    db = FakeDB([], session_state=state)
    rt.get_supabase_client = lambda: db
    out = asyncio.run(rt.create_realtime_session(rt.RealtimeSessionRequest(case_id="c1", attempt_id="a1"),
                                                 authorization="Bearer t"))
    os.environ.pop("VOICE_INTERVIEWER", None)
    return out, (_FakeAsyncClient.last or {}).get("session", {})


out, sess = mint()
td = sess["audio"]["input"]["turn_detection"]
check("default session unchanged: renderer instructions, no tools, create_response off",
      sess["instructions"] == VOICE_RENDERER_INSTRUCTIONS and "tools" not in sess and td["create_response"] is False
      and out.get("interviewer") == "renderer", (out.get("interviewer"), td))
out, sess = mint("model_led", state={"voice": {"coach_notes": ["They asked for help: call get_hint."]}})
td = sess["audio"]["input"]["turn_detection"]
ins = sess["instructions"]
check("model-led: the model answers by itself, barge-in kept",
      td["create_response"] is True and td["interrupt_response"] is True, td)
check("model-led: playbook + case in the instructions", "Do NOT question every step" in ins and CASE_CONTENT in ins)
check("model-led: the solution is NOT in the instructions",
      "48,000" not in ins and SOLUTION[:40] not in ins and "16 lakh" not in ins)
check("model-led: hint and answer tools offered",
      [t["name"] for t in sess.get("tools", [])] == ["get_hint", "answer_request"] and sess.get("tool_choice") == "auto")
check("model-led: notes already on the attempt carried into a new session", "They asked for help: call get_hint." in ins)
check("model-led: response tells the client which interviewer it got", out.get("interviewer") == "model_led")

base = build_voice_interviewer_instructions(CASE_CONTENT, "guesstimate")
for phrase in ("results page", "answer_request", "get_hint", "Never refuse help", "At most ONE question",
               "question may come without a question mark"):
    check(f"playbook states: {phrase!r}", phrase in base)

print()
print("=" * 72)
print("3. COACH NOTES")
print("=" * 72)
A = {"clarification_quota": 20, "clarification_used": 0}
H0 = [{"role": "assistant", "content": "Let's begin. How would you approach this?"}]


def notes_for(last, history=H0, state=None, attempt=A):
    n, _ = vc.coach_notes(history + [{"role": "user", "content": last}], attempt, state or {})
    return n


check("asked for help -> get_hint note", vc.NOTE_ASKED_HELP in notes_for("I'm stuck, can you help me?"))
check("asked for the answer -> answer_request note", vc.NOTE_ASKED_ANSWER in notes_for("Just tell me the answer"))
check("frustrated -> no questions, something concrete",
      vc.NOTE_FRUSTRATED in notes_for("This is irritating, you keep asking the same question"))
qs = [{"role": "assistant", "content": "What is the population?"}, {"role": "user", "content": "1 crore"},
      {"role": "assistant", "content": "How many households?"}, {"role": "user", "content": "25 lakh"},
      {"role": "assistant", "content": "And how many own two-wheelers?"}]
check("interviewer asked questions 2-3 turns running -> 'no question next turn'",
      vc.NOTE_TOO_MANY_QUESTIONS in notes_for("Maybe 60 percent of them", history=qs))
check("clarifications used up -> assume and carry on",
      vc.NOTE_NO_CLARIFICATIONS in notes_for("What is the population of Chennai?", attempt={"clarification_quota": 7, "clarification_used": 7}))
check("real progress -> short specific acknowledgement, no quiz",
      notes_for("Chennai has about 1.1 crore people, roughly 27 lakh households, and about 60 percent own a two "
                "wheeler which gives about 16 lakh two wheelers") == [vc.NOTE_PROGRESSING])
check("answer already given -> help them wrap up",
      vc.NOTE_ANSWER_GIVEN in notes_for("ok", state={"voice": {"answer_revealed": True}}))
check("at most three notes", len(notes_for("This is irritating, just tell me the answer", history=qs,
                                           attempt={"clarification_quota": 7, "clarification_used": 7})) <= 3)

print()
print("=" * 72)
print("4. HINT LADDER")
print("=" * 72)


class _Choice:
    def __init__(self, c):
        self.message = types.SimpleNamespace(content=c)


class FakeLLM:
    def __init__(self, replies=()):
        self.replies, self.calls = list(replies), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        text = self.replies.pop(0) if self.replies else "Think about how many households own a two-wheeler first."
        return types.SimpleNamespace(choices=[_Choice(text)], usage=None, id="f")


vc_log = []
import services.ai_usage as _au  # noqa: E402
_au.log_ai_usage = lambda **k: vc_log.append(k)
T = H0 + [{"role": "user", "content": "I don't know where to start"}]
state = {}
levels = []
for _ in range(5):
    out, v = vc.hint({"type": "guesstimate", "hint": HINT, "solution": SOLUTION}, CASE_CONTENT, T, state,
                     reason="asked_for_help", llm=FakeLLM())
    levels.append(v["hint_level"])
    state = vc.with_voice_state(state, v)
check("ladder climbs cue -> hint -> framework -> partial step, then stays", levels == [1, 2, 3, 4, 4], levels)
check("top-level hint_level kept in step for the learner model", state.get("hint_level") == 4)
out, v = vc.hint({"solution": SOLUTION}, CASE_CONTENT, T, {}, reason="frustrated", llm=FakeLLM())
check("frustrated skips the gentle cue (framework or more)", v["hint_level"] >= 3, v)
llm = FakeLLM(["So you end at about 48,000 electric scooters. Start from households."])
out, v = vc.hint({"solution": SOLUTION}, CASE_CONTENT, T, {}, llm=llm)
check("a hint that leaks the final number has that sentence removed", "48,000" not in out and "Start from households" in out, out)
llm = FakeLLM(["The answer is 48,000."])
out, v = vc.hint({"solution": SOLUTION, "hint": HINT}, CASE_CONTENT, T, {}, llm=llm)
check("... and if nothing is left, a safe fallback hint", "48,000" not in out and "Hint to give" in out, out)
check("hint prompt grounds on the solution but forbids revealing its result",
      "never reveal its final result" in llm.calls[0]["messages"][1]["content"])
import services.ai_providers as _ap  # noqa: E402
_ap_real = _ap.openai_client
_ap.openai_client = lambda: None
out, v = vc.hint({"solution": SOLUTION, "hint": HINT}, CASE_CONTENT, T, {})
check("no model available -> generic ladder line, never an error", "Hint to give (a cue)" in out, out)
_ap.openai_client = _ap_real

print()
print("=" * 72)
print("5. ANSWER RULE")
print("=" * 72)
case = {"type": "guesstimate", "hint": HINT, "solution": SOLUTION}
out1, v1, given1 = vc.answer_request(case, CASE_CONTENT, T, {}, llm=FakeLLM(["Think of it as a funnel from households to scooters."]))
check("first ask: no answer, a way of thinking, results page",
      not given1 and "48,000" not in out1 and "results page" in out1 and "funnel" in out1, out1)
s1 = vc.with_voice_state({}, v1)
out2, v2, given2 = vc.answer_request(case, CASE_CONTENT, T, s1, llm=FakeLLM())
check("asking again: the worked answer, with the honest results caveat",
      given2 and "48,000" in out2 and "results will show" in out2 and v2["answer_revealed"] is True, out2)
os.environ["VOICE_ANSWER_AFTER_ASKS"] = "3"
_, va, ga = vc.answer_request(case, CASE_CONTENT, T, {}, llm=FakeLLM())
_, vb, gb = vc.answer_request(case, CASE_CONTENT, T, vc.with_voice_state({}, va), llm=FakeLLM())
_, vcx, gc = vc.answer_request(case, CASE_CONTENT, T, vc.with_voice_state({}, vb), llm=FakeLLM())
check("VOICE_ANSWER_AFTER_ASKS=3 -> answer on the third ask", (ga, gb, gc) == (False, False, True))
os.environ.pop("VOICE_ANSWER_AFTER_ASKS")
out3, _, g3 = vc.answer_request({"type": "guesstimate"}, CASE_CONTENT, T, s1, llm=FakeLLM())
check("no stored solution -> the model walks a sensible approach (still with the caveat)",
      g3 and "no stored model answer" in out3 and "results will show" in out3)

print()
print("=" * 72)
print("6. ROUTES")
print("=" * 72)
DB = FakeDB(T)
for mod in (rvc, att):
    mod.get_supabase_client = lambda: DB
rvc.get_verified_user = lambda sb, a: ("u1", types.SimpleNamespace(id="u1", email="owner@example.com"))
rvc.check_rate_limit = lambda *a, **k: None
rvc.assert_daily_budget = lambda *a, **k: None
att._case_cache.clear()
app = FastAPI()
app.include_router(rvc.router)
client = TestClient(app)
H = {"Authorization": "Bearer t"}

r = client.post("/attempts/a1/voice-coach", headers=H)
check("switch off -> 409 (routes are inert)", r.status_code == 409, r.status_code)
os.environ["VOICE_INTERVIEWER"] = "allowlist"
os.environ["VOICE_INTERVIEWER_ALLOWLIST"] = "owner@example.com"

DB = FakeDB(H0 + [{"role": "user", "content": "I'm stuck, can you help me?"}])
r = client.post("/attempts/a1/voice-coach", headers=H)
j = r.json()
check("/voice-coach: notes + refreshed instructions when they change",
      r.status_code == 200 and j["changed"] and vc.NOTE_ASKED_HELP in j["notes"]
      and vc.NOTE_ASKED_HELP in (j["instructions"] or "") and CASE_CONTENT in j["instructions"], j)
check("/voice-coach: refreshed instructions never carry the solution", "48,000" not in (j["instructions"] or ""))
check("/voice-coach: notes saved on the attempt", DB.state().get("voice", {}).get("coach_notes") == j["notes"])
r = client.post("/attempts/a1/voice-coach", headers=H)
check("/voice-coach: same notes again -> changed false, no instructions resent",
      r.json()["changed"] is False and r.json()["instructions"] is None)

vc_llm = FakeLLM(["Look at households before scooters."] * 4)
_ap.openai_client = lambda: vc_llm
r = client.post("/attempts/a1/voice-tool", json={"name": "get_hint", "arguments": '{"reason": "asked_for_help"}'}, headers=H)
check("/voice-tool get_hint: returns a hint, level saved",
      r.status_code == 200 and "Look at households" in r.json()["output"] and DB.state()["voice"]["hint_level"] == 1, r.text[:200])
r = client.post("/attempts/a1/voice-tool", json={"name": "answer_request", "arguments": "{}"}, headers=H)
check("/voice-tool answer_request #1: no answer", r.json()["answer_given"] is False and "48,000" not in r.json()["output"])
r = client.post("/attempts/a1/voice-tool", json={"name": "answer_request"}, headers=H)
check("/voice-tool answer_request #2: answer given and remembered",
      r.json()["answer_given"] is True and "48,000" in r.json()["output"] and DB.state()["voice"]["answer_revealed"] is True)
r = client.post("/attempts/a1/voice-tool", json={"name": "rm_rf", "arguments": "{}"}, headers=H)
check("/voice-tool: unknown tool -> 400", r.status_code == 400)
DB = FakeDB(T, owner="someone-else")
r = client.post("/attempts/a1/voice-tool", json={"name": "get_hint"}, headers=H)
check("/voice-tool: someone else's attempt -> 403", r.status_code == 403, r.status_code)
DB = FakeDB(T, status="submitted")
r = client.post("/attempts/a1/voice-coach", headers=H)
check("/voice-coach: submitted attempt -> 400", r.status_code == 400, r.status_code)
_ap.openai_client = _ap_real
os.environ.pop("VOICE_INTERVIEWER")

print()
if _fail:
    print(f"{len(_fail)} FAILED: {_fail}")
    sys.exit(1)
print("ALL PASS")
