"""
Model-led realtime voice interviewer (VOICE_INTERVIEWER): the speech model
converses by itself; the server coaches alongside.

Proves, with fakes only (no network, no keys):
  1. SWITCH    -- model-led by default; renderer (V11/V12 per turn) or an allowlist on request.
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
check("default: model-led (the realtime model converses)", vc.voice_interviewer_mode("u1", "a@b.com") == "model_led")
os.environ["VOICE_INTERVIEWER"] = "renderer"
check("VOICE_INTERVIEWER=renderer: back to V11/V12 deciding each turn", vc.voice_interviewer_mode("u1", "a@b.com") == "renderer")
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


out, sess = mint("renderer")
td = sess["audio"]["input"]["turn_detection"]
check("renderer session unchanged: renderer instructions, no tools, create_response off, whisper-1",
      sess["instructions"] == VOICE_RENDERER_INSTRUCTIONS and "tools" not in sess and td["create_response"] is False
      and sess["audio"]["input"]["transcription"]["model"] == "whisper-1"
      and out.get("interviewer") == "renderer" and out.get("open_first") is False, (out.get("interviewer"), td))

HIST = [{"role": "user", "content": "Hi, I'd like to start with Chennai's households."},
        {"role": "assistant", "content": "SAY: Sure, go ahead with households."}]


def mint_with_history(env=None, state=None, history=HIST):
    os.environ.pop("VOICE_INTERVIEWER", None)
    if env:
        os.environ["VOICE_INTERVIEWER"] = env
    db = FakeDB(history, session_state=state)
    rt.get_supabase_client = lambda: db
    out = asyncio.run(rt.create_realtime_session(rt.RealtimeSessionRequest(case_id="c1", attempt_id="a1"),
                                                 authorization="Bearer t"))
    os.environ.pop("VOICE_INTERVIEWER", None)
    return out, (_FakeAsyncClient.last or {}).get("session", {})


out, sess = mint_with_history(state={"voice": {"coach_notes": ["They asked for help: call get_hint."]}})
td = sess["audio"]["input"]["turn_detection"]
ins = sess["instructions"]
check("default session is model-led: the model answers by itself, barge-in kept",
      out.get("interviewer") == "model_led" and td["create_response"] is True and td["interrupt_response"] is True, td)
check("model-led: replies come fast (semantic_vad eagerness high by default)", td.get("eagerness") == "high", td)
check("model-led: NO tools by default (a tool call is a round trip = lag)", "tools" not in sess, sess.get("tools"))
check("model-led: streaming transcription model, in English (no Devanagari for Indian English)",
      sess["audio"]["input"]["transcription"] == {"model": "gpt-4o-mini-transcribe", "language": "en"},
      sess["audio"]["input"]["transcription"])
check("model-led: the interviewer opens the call; per-turn coach off",
      out.get("open_first") is True and out.get("coach") is False, out)
check("prompt: the case sits on top", ins.startswith("=== THE CASE") and CASE_CONTENT in ins.split("===")[2])
check("prompt: private notes carry the stored hint and model solution", HINT in ins and SOLUTION in ins)
check("prompt: notes are marked never to be read out, answer only under the rule",
      "never read these out" in ins and "ONLY under the ANSWER RULE" in ins)
check("prompt: the conversation so far (chat -> voice) is carried over, without the SAY label",
      "CANDIDATE: Hi, I'd like to start with Chennai's households." in ins
      and "INTERVIEWER: Sure, go ahead with households." in ins and "SAY:" not in ins)
check("prompt: coach notes are NOT added when the coach is off", "They asked for help: call get_hint." not in ins)
check("prompt: playbook sits below the case", ins.index("=== THE CASE") < ins.index("=== HOW YOU RUN THIS"))
for phrase in ("HOW A STRUCTURED THINKER WORKS A CASE", "MECE", "Guesstimate / market sizing", "Profitability",
               "Market entry", "Sanity-check", "Pick up THEIR words", "question can come without a question mark",
               "Do NOT question every step", "at most ONE question", "CUE", "FRAMEWORK", "ANALOGY",
               "results page", "Never say \"that's for you to figure out\"", "Natural Indian English",
               # what a human interviewer does (2026-10-02 feedback)
               "SAY ONLY WHAT THIS MOMENT NEEDS", "Never apologise or say sorry", "\"Are you sure about that?\"",
               "food for thought", "Not a question back at them", "never the same question again",
               "on the right track", "Most of your turns are under fifteen words", "never say you \"can't\"",
               "Never read it out, restate it or summarise it", "it is not a turn"):
    check(f"playbook states: {phrase!r}", phrase in ins)
check("playbook: no refusal script and no apology wording to copy",
      "I can't give you" not in ins and "I apologize" not in ins)
os.environ["VOICE_COACH"] = "on"
out, sess = mint_with_history(state={"voice": {"coach_notes": ["They asked for help: call get_hint."]}})
check("VOICE_COACH=on: coach notes join the prompt and the client is told to coach",
      "They asked for help: call get_hint." in sess["instructions"] and out.get("coach") is True)
os.environ.pop("VOICE_COACH")
os.environ["VOICE_TOOLS"] = "on"
out, sess = mint_with_history()
check("VOICE_TOOLS=on: hint/answer tools offered (opt-in)", [t["name"] for t in sess.get("tools", [])] == ["get_hint", "answer_request"])
os.environ.pop("VOICE_TOOLS")
us = build_voice_interviewer_instructions(CASE_CONTENT, "guesstimate", market="US")
check("US cases: American English, dollars", "Natural American English" in us and "lakh, crore" not in us)


class _RejectThenOk:
    calls = []

    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def post(self, url, headers=None, json=None):
        import copy
        _RejectThenOk.calls.append(copy.deepcopy(json))
        model = json["session"]["audio"]["input"]["transcription"]["model"]
        r = _FakeResp()
        if model != "whisper-1":
            r = types.SimpleNamespace(status_code=400, text='{"error":"unknown model"}', json=lambda: {})
        return r


rt.httpx = types.SimpleNamespace(AsyncClient=_RejectThenOk)
os.environ["REALTIME_TRANSCRIBE_MODEL"] = "gpt-live-transcribe"
out, _ = mint_with_history()
models = [c["session"]["audio"]["input"]["transcription"]["model"] for c in _RejectThenOk.calls]
check("a rejected transcription model never takes voice down (retried with whisper-1)",
      models == ["gpt-live-transcribe", "whisper-1"] and out.get("client_secret") == "ek_test", models)
os.environ.pop("REALTIME_TRANSCRIBE_MODEL")
rt.httpx = types.SimpleNamespace(AsyncClient=_FakeAsyncClient)

print()
print("=" * 72)
print("2a. OPENING, RESUME AND DIFFICULTY")
print("=" * 72)
from prompts.voice_interviewer_playbook import normalize_level  # noqa: E402
fresh = build_voice_interviewer_instructions(CASE_CONTENT, "guesstimate")
check("fresh call: the case is on screen - do not explain it, ask them to read it and give their approach",
      "already on the candidate's screen" in fresh and "Do NOT explain, read out or summarise it" in fresh
      and "RESUMING" not in fresh)
out, sess = mint_with_history()
check("reopening voice with saved turns RESUMES (no restart, no re-explaining)",
      out.get("resume") is True and "You are RESUMING this interview" in sess["instructions"]
      and "Greet them in a few words" not in sess["instructions"])
out, sess = mint_with_history(history=[])
check("first voice call on an attempt is fresh", out.get("resume") is False and "RESUMING" not in sess["instructions"])
check("level defaults to the case's own difficulty", out.get("level") == "medium" and "DIFFICULTY: MEDIUM" in sess["instructions"])
check("normalize_level: picked level wins, else case difficulty, else medium",
      (normalize_level("HARD", "easy"), normalize_level(None, "easy"), normalize_level("x", None)) == ("hard", "easy", "medium"))
for lvl, marker in (("easy", "DIFFICULTY: EASY"), ("hard", "DIFFICULTY: HARD")):
    os.environ.pop("VOICE_INTERVIEWER", None)
    db = FakeDB(HIST)
    rt.get_supabase_client = lambda: db
    o = asyncio.run(rt.create_realtime_session(rt.RealtimeSessionRequest(case_id="c1", attempt_id="a1", level=lvl),
                                               authorization="Bearer t"))
    ins_l = (_FakeAsyncClient.last or {}).get("session", {}).get("instructions", "")
    check(f"level={lvl}: the prompt carries the {lvl} block, response echoes it",
          o.get("level") == lvl and marker in ins_l and ins_l.index(marker) > ins_l.index("HOW YOU RUN THIS"))
hard = build_voice_interviewer_instructions(CASE_CONTENT, "guesstimate", level="hard")
easy = build_voice_interviewer_instructions(CASE_CONTENT, "guesstimate", level="easy")
check("hard: pressure-tests, hints only on request", "Pressure-test" in hard and "only when they explicitly ask" in hard)
check("easy: offers cues unprompted, sooner hints", "offer a cue without waiting to be asked" in easy)
brief = ("Hello! Our company is a well-established regional player in Gujarat's traditional namkeen market, with "
         "Rs 100 crores in annual revenue. How would you like to structure your approach?")
hist_ro = [{"role": "user", "content": "I'd focus on general trade."},
           {"role": "assistant", "content": "Hello! Our company is a well-established"},
           {"role": "assistant", "content": "Yes, general trade first makes sense for a mass brand."}]
ro = build_voice_interviewer_instructions(brief, "growth", transcript=hist_ro)
check("history: an interviewer line that read the brief aloud is replaced (never copied again)",
      "INTERVIEWER: Hello! Our company" not in ro and "INTERVIEWER: (read the case brief aloud - never do this again)" in ro
      and "INTERVIEWER: Yes, general trade first makes sense" in ro and "CANDIDATE: I'd focus on general trade." in ro)
check("prompt: the brief may be in the client's voice - those are not lines to read",
      "written in the client's voice" in ro and "not lines for you to read" in ro)

print()
print("=" * 72)
print("2b. GEMINI LIVE SESSION (the admin's current voice mode)")
print("=" * 72)
import routes.realtime_gemini as rtg  # noqa: E402
_gcap = {}


class _FakeTokens:
    def create(self, config):
        _gcap["config"] = config
        return types.SimpleNamespace(name="auth_tokens/test")


import types as _types  # noqa: E402
_fake_genai = _types.ModuleType("google.genai")
_fake_genai.Client = lambda api_key=None: types.SimpleNamespace(auth_tokens=_FakeTokens())
import google  # noqa: E402
sys.modules["google.genai"] = _fake_genai
google.genai = _fake_genai
rtg.GEMINI_API_KEY = "test"
rtg.get_verified_user = lambda sb, a: ("u1", types.SimpleNamespace(id="u1", email="owner@example.com", is_anonymous=False))
rtg.is_guest_user = lambda u: False
rtg.check_rate_limit = lambda *a, **k: None
rtg.assert_daily_budget = lambda *a, **k: None
rtg.get_ai_input_quota = lambda sb, uid: {"tier": "pro"}
rtg.has_credit = lambda *a, **k: True
rtg.get_balance = lambda *a, **k: {"total_remaining": 10}
rtg.log_ai_usage = lambda **k: None
_REAL_RESOLVE = rtg._resolve_live_model
rtg._resolve_live_model = lambda: "fake-live"


def gmint(env=None, history=HIST):
    os.environ.pop("VOICE_INTERVIEWER", None)
    if env:
        os.environ["VOICE_INTERVIEWER"] = env
    db = FakeDB(history)
    rtg.get_supabase_client = lambda: db
    out = rtg.create_gemini_session(rtg.GeminiSessionRequest(case_id="c1", attempt_id="a1"), authorization="Bearer t")
    os.environ.pop("VOICE_INTERVIEWER", None)
    cfg = ((_gcap.get("config") or {}).get("live_connect_constraints") or {}).get("config") or {}
    return out, cfg


gout, gcfg = gmint()
gins = gcfg.get("system_instruction") or ""
check("Gemini default is LIVE: interviewer=model_led, opens the call", gout.get("interviewer") == "model_led" and gout.get("open_first") is True, gout)
check("Gemini live: the session prompt is the playbook with the case on top",
      gins.startswith("=== THE CASE") and CASE_CONTENT in gins and "HOW A STRUCTURED THINKER" in gins)
check("Gemini live: private notes + conversation so far (no SAY label)",
      SOLUTION in gins and "CANDIDATE: Hi, I'd like to start" in gins and "SAY:" not in gins)
check("Gemini live: still speech-to-speech with both transcripts (saved, never in front of a reply)",
      gcfg.get("response_modalities") == ["AUDIO"] and "input_audio_transcription" in gcfg and "output_audio_transcription" in gcfg)
check("Gemini: candidate transcript pinned to English (en-IN for Indian cases)",
      gcfg.get("input_audio_transcription") == {"language_codes": ["en-IN"]}, gcfg.get("input_audio_transcription"))
check("Gemini: the prompt (with private notes) is never sent back to the browser", "instructions" not in gout)
aad = gcfg.get("realtime_input_config", {}).get("automatic_activity_detection", {})
check("Gemini live: fastest config first (quick end-of-turn, echo/noise-resistant start)",
      aad == {"start_of_speech_sensitivity": "START_SENSITIVITY_LOW", "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
              "prefix_padding_ms": 200, "silence_duration_ms": 500} and gout.get("tier") == 0 and gout.get("tiers") == 3, aad)
check("Gemini 3.x live: no thinking setting (those models reject it)", "thinking_config" not in gcfg)
check("Gemini live: resume + level reported", gout.get("resume") is True and gout.get("level") == "medium")

# 2.5 native audio thinks by default: the fast config turns that off.
rtg._resolve_live_model = lambda: "gemini-2.5-flash-native-audio-preview-12-2025"
_, gcfg25 = gmint()
check("Gemini 2.5 native audio: no thinking pass before speaking",
      (gcfg25.get("thinking_config") or {}).get("thinking_budget") == 0)
rtg._resolve_live_model = lambda: "fake-live"

# The browser asks for the next config when Google refuses one at setup.
db_t = FakeDB(HIST)
rtg.get_supabase_client = lambda: db_t
for want_tier, has_lang, has_voice in ((1, False, True), (2, False, False), (9, False, False)):
    o = rtg.create_gemini_session(rtg.GeminiSessionRequest(case_id="c1", attempt_id="a1", tier=want_tier),
                                  authorization="Bearer t")
    c = _gcap["config"]["live_connect_constraints"]["config"]
    check(f"Gemini tier={want_tier}: a simpler config (step-down after a refused setup)",
          o["tier"] == min(want_tier, 2) and ("language_codes" in c["input_audio_transcription"]) == has_lang
          and ("speech_config" in c) == has_voice and c["system_instruction"].startswith("=== THE CASE"), (o["tier"], sorted(c)))

# Model choice: the newest general live model on the key (3.8 Live is Google's
# low-latency default), never translate/transcribe/extended-thinking variants.
_saved_list, _saved_resolve = rtg._list_live_models, rtg._resolve_live_model
rtg._resolve_live_model = _REAL_RESOLVE
os.environ.pop("GEMINI_LIVE_MODEL", None)
for listed, want in (
        (["gemini-2.5-flash-native-audio-preview-12-2025", "gemini-3.1-flash-live-preview", "gemini-3.8-live",
          "gemini-3.8-live-extended-thinking", "gemini-3.5-live-translate-preview", "gemini-3.5-transcribe-live"],
         "gemini-3.8-live"),
        (["gemini-2.5-flash-native-audio-preview-09-2025", "gemini-3.1-flash-live-preview"], "gemini-3.1-flash-live-preview"),
        (["gemini-live-2.5-flash-preview", "gemini-2.5-flash-native-audio-preview-09-2025",
          "gemini-2.5-flash-native-audio-preview-12-2025"], "gemini-2.5-flash-native-audio-preview-12-2025")):
    rtg._MODEL_CACHE["model"] = None
    rtg._list_live_models = lambda listed=listed: listed
    check(f"model choice: {want} from {len(listed)} listed", rtg._resolve_live_model() == want, rtg._resolve_live_model())
rtg._MODEL_CACHE["model"] = None
os.environ["GEMINI_LIVE_MODEL"] = "gemini-2.5-flash-native-audio-preview-12-2025"
rtg._list_live_models = lambda: ["gemini-3.8-live", "gemini-2.5-flash-native-audio-preview-12-2025"]
check("model choice: GEMINI_LIVE_MODEL still wins when it is available",
      rtg._resolve_live_model() == "gemini-2.5-flash-native-audio-preview-12-2025")
os.environ.pop("GEMINI_LIVE_MODEL")
rtg._MODEL_CACHE["model"] = None
rtg._list_live_models, rtg._resolve_live_model = _saved_list, _saved_resolve


class _PickyTokens:
    seen = []

    def create(self, config):
        cfg = config["live_connect_constraints"]["config"]
        _PickyTokens.seen.append(bool(cfg["input_audio_transcription"]))
        if cfg["input_audio_transcription"]:
            raise ValueError("unknown field language_codes")
        _gcap["config"] = config
        return types.SimpleNamespace(name="auth_tokens/test")


_fake_genai.Client = lambda api_key=None: types.SimpleNamespace(auth_tokens=_PickyTokens())
gout2, gcfg2 = gmint()
check("Gemini: if a fast setting is rejected at mint, the next config is used (voice never breaks)",
      gout2.get("token") == "auth_tokens/test" and gout2.get("tier") == 1
      and gcfg2.get("realtime_input_config") == {"automatic_activity_detection": {"silence_duration_ms": 500}}
      and _PickyTokens.seen[:2] == [True, False], _PickyTokens.seen)
_fake_genai.Client = lambda api_key=None: types.SimpleNamespace(auth_tokens=_FakeTokens())
gout, gcfg = gmint("renderer")
check("Gemini VOICE_INTERVIEWER=renderer: old flow (voice-renderer instructions)",
      gcfg.get("system_instruction") == VOICE_RENDERER_INSTRUCTIONS and gout.get("interviewer") == "renderer")

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

os.environ["VOICE_INTERVIEWER"] = "renderer"
r = client.post("/attempts/a1/voice-coach", headers=H)
check("renderer mode -> 409 (routes are inert)", r.status_code == 409, r.status_code)
os.environ["VOICE_INTERVIEWER"] = "allowlist"
os.environ["VOICE_INTERVIEWER_ALLOWLIST"] = "owner@example.com"

DB = FakeDB(H0 + [{"role": "user", "content": "I'm stuck, can you help me?"}])
r = client.post("/attempts/a1/voice-coach", headers=H)
j = r.json()
check("/voice-coach: notes + refreshed instructions when they change",
      r.status_code == 200 and j["changed"] and vc.NOTE_ASKED_HELP in j["notes"]
      and vc.NOTE_ASKED_HELP in (j["instructions"] or "") and CASE_CONTENT in j["instructions"], j)
check("/voice-coach: refreshed instructions are the full prompt (case on top, notes, history)",
      (j["instructions"] or "").startswith("=== THE CASE") and "CANDIDATE: I'm stuck" in (j["instructions"] or ""))
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
