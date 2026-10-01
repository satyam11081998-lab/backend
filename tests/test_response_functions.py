"""
V12 interviewer: function first, language second.

Proves, with a fake model and a fake database (no network, no keys):
  1. POLICY   -- the deterministic layer picks a conversational RESPONSE FUNCTION.
                 A substantive candidate turn never lands on the fixed-phrase lane;
                 a minimal turn still gets a minimal beat; every V10.2 business
                 rule (help, solution, correction, repair, meta, clarification,
                 silence) keeps its route.
  2. PACKET   -- the JSON INTERVIEWER CONTROL PACKET carries every field of the
                 spec and no sentence to copy.
  3. ENGINE   -- contextual beats are worded by the model from the packet,
                 validated (no question, no unverified "that's correct", no leaked
                 metadata, must pick up the candidate's content), regenerated once,
                 and fall back to a plain hand-back instead of an error.
  4. ROUTES   -- /voice-decision returns the beat as PRESENCE with its function;
                 /messages persists it; the function is folded into session_state
                 and drives the next turn's cool-down and rotation.
  5. REPLAYS  -- real transcripts where a long candidate explanation used to get a
                 one-word "Right." / "Got it." now get a contextual line.

Run:  python -m tests.test_response_functions
"""
from __future__ import annotations

import json
import os
import re
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _k, _v in {
    "OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test", "GEMINI_API_KEY": "test",
}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.attempts as att  # noqa: E402
import services.interview_engine as ie  # noqa: E402
import services.interviewer_decision as idec  # noqa: E402
from services.session_signals import compute_signals  # noqa: E402
from services.interviewer_decision import (  # noqa: E402
    decide_response, build_interviewer_control_packet, validate_contextual_line,
    evaluate_intervention_gate, CONTEXTUAL_PRESENCE, FAST_FUNCTIONS, _FAST_PHRASES,
    _SUBSTANTIVE_FUNCTION,
)
from services.clarification_counter import count_clarifications  # noqa: E402

_fail: list = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond or not detail else f"   [{detail}]"))
    if not cond:
        _fail.append(name)


FAST_TEXTS = {o["text"] for opts in _FAST_PHRASES.values() for o in opts}
STOCK = {"got it.", "right.", "okay.", "alright.", "sure.", "understood.", "makes sense.",
         "go ahead.", "that works.", "that makes sense."}

# ------------------------------------------------------------------ real turns
# REAL: typed by the product owner on the live site (2026-10-01, profitability case
# 6bdd8d3a-...), pasted verbatim from the screenshot of that session.
REAL_PLAN = (
    "First, I'll compute the baseline: with an 8% profit margin, total costs are 92% of revenue, "
    "and the cost split is raw materials 50%, labor 20%, overhead 15%, and logistics 15%. Then I'll "
    "apply the changes only to the affected lines: a 10% reduction on raw materials, and a 5% "
    "reduction on labor and overhead via operational efficiency, keeping logistics unchanged. From "
    "the updated total cost, I'll calculate the new profit margin. After that, I'll propose creative "
    "levers, for example supplier renegotiation or hedging for inputs, process automation and quality "
    "improvements to reduce rework, mix and pricing by region or product, and logistics optimization "
    "like better container utilization or consolidation.")
PROFIT_CASE = [{"role": "assistant", "kind": "text", "content":
                "A snacks maker runs at an 8% profit margin. Costs: raw materials 50%, labour 20%, "
                "overhead 15%, logistics 15% of revenue. Raw materials fall 10%, labour and overhead "
                "5%. What happens to the margin, and what else would you do?"}]

# REAL: the realtime voice transcript already used by tests/test_v11_voice_integration.py.
LIVE = [
    {"role": "user", "kind": "voice", "content": "So the number of smartphone sellers in Bangalore can you tell me the population of Bangalore"},
    {"role": "assistant", "kind": "voice", "content": "Take the population of Bangalore as 1.3 crore."},
    {"role": "user", "kind": "voice", "content": "I think there will be approximately 1.3 crore smartphones also."},
    {"role": "assistant", "kind": "voice", "content": "How did you get there?"},
]

# Representative turns from the design brief (illustrative, not logged sessions).
DELHI_STRUCTURE = ("We start with the population of Delhi, convert that into households, estimate "
                   "penetration, then split usage between households and restaurants")
STRATEGIC = "Rather than only reducing cost, I'd also look at increasing realization through premium products."
STEP_DONE = ("So 30 crore households times 10 percent gives 3 crore, and with a 7 year cycle that "
             "comes to about 43 lakh cars a year")
SUBSTANTIVE_TURNS = [REAL_PLAN, DELHI_STRUCTURE, STRATEGIC, STEP_DONE,
                     "I'd split the market into urban and rural households, and within each by "
                     "income band, then estimate car ownership for each segment",
                     "Raw materials go from 50 to 45, labour from 20 to 19, overhead 15 to 14.25, "
                     "logistics stays 15, so total cost is 93.25 percent of the old revenue"]


def decide(text, history=PROFIT_CASE, channel="text", prior_state=None):
    sig = compute_signals(history, text, "coached", prior_state=prior_state or {}, channel=channel)
    return sig, decide_response(sig)


print("=" * 72)
print("1. POLICY: state -> response function")
print("=" * 72)
for ch in ("text", "voice"):
    for t in SUBSTANTIVE_TURNS:
        sig, d = decide(t, channel=ch)
        check(f"[{ch}] substantive turn -> model-worded {d['function']}: {t[:40]!r}",
              d["render"] == "model" and d["function"] in CONTEXTUAL_PRESENCE, d)

sig, d = decide(REAL_PLAN)
check("real plan (110 words) is read as substantive reasoning", sig["is_substantive_reasoning"] and sig["turn_type"] == "plan", sig["turn_type"])
check("real plan -> ACKNOWLEDGE_AND_CONTINUE (not the ACKNOWLEDGE phrase bank)", d["function"] == "ACKNOWLEDGE_AND_CONTINUE", d)
sig, d = decide(STRATEGIC)
check("strategic observation -> REFLECT_PROGRESS", d["function"] == "REFLECT_PROGRESS", d)
sig, d = decide(DELHI_STRUCTURE)
check("structure walk-through -> REFLECT_PROGRESS", d["function"] == "REFLECT_PROGRESS", d)
sig, d = decide(STEP_DONE)
check("finished step with a result -> ACKNOWLEDGE_AND_ORIENT", d["function"] == "ACKNOWLEDGE_AND_ORIENT", d)

sig, d = decide("First, I'll compute the baseline cost split, then apply the 10% raw material and 5% "
                "labour and overhead cuts, then recompute the margin. Shall I proceed?")
check("plan + 'Shall I proceed?' -> acknowledged and floor returned (not a clarification answer)",
      d["lane"] == "PRESENCE" and d["function"] == "ACKNOWLEDGE_AND_CONTINUE" and d["reason"] == "floor_yield_after_work", d)
sig, d = decide("Shall I proceed?")
check("bare 'Shall I proceed?' keeps its V10.2 route (ANSWER_DIRECT)", d["function"] == "ANSWER_DIRECT", d)

for text, fn in [("Okay.", "HAND_BACK"), ("Yeah", "HAND_BACK"), ("100.", "SHORT_ACK"),
                 ("I'll split households into urban and rural first.", "HAND_BACK")]:
    sig, d = decide(text, history=PROFIT_CASE + [{"role": "user", "content": "Let me start."},
                                                 {"role": "assistant", "content": "Go on."}])
    check(f"minimal turn {text!r} -> {fn} on the fast lane (no model call)", d["function"] == fn and d["render"] == "fast", d)

sig, d = decide("Delhi population = 20 million. Assume 5 people per family...")
check("[text] mid-calculation ('...') -> HAND_BACK", d["function"] == "HAND_BACK", d)
sig, d = decide("Delhi population = 20 million. Assume 5 people per family...", channel="voice")
check("[voice] mid-calculation -> NO_OUTPUT (never talk over them)", d["function"] == "NO_OUTPUT" and d["lane"] == "SILENCE", d)
sig, d = decide("okay", history=PROFIT_CASE + [{"role": "user", "content": "I'll take 25 crore households."},
                                              {"role": "assistant", "content": "Go ahead."}])
check("'okay' mid-case no longer re-opens the case", d["function"] != "OPEN", d)

for text, fn in [("Can you help me here?", "MICRO_HINT"), ("Show me the correct approach.", "DELIVER_SOLUTION"),
                 ("Tell me the answer", "DELIVER_SOLUTION"),
                 ("So 1 litre = 100 ml, then I multiply by households", "CORRECT_AND_CONTINUE"),
                 ("This is irritating, you keep asking the same question", "REPAIR_AND_RESET"),
                 ("Are you an AI?", "DEFLECT_META"), ("What is the population of Delhi?", "ANSWER_DIRECT"),
                 ("I'm done, let's wrap up. My recommendation is to cut raw material costs.", "CLOSE")]:
    sig, d = decide(text)
    check(f"business rule kept: {text[:36]!r} -> {fn}", d["function"] == fn and d["lane"] == "SUBSTANTIVE", d)
sig, d = decide("Can you help me here?", channel="voice", history=LIVE)
check("[voice] help still honoured -> MICRO_HINT", d["function"] == "MICRO_HINT", d)
sig = compute_signals(LIVE, "Can you help", "coached", channel="voice", is_voice_partial=True)
check("[voice] partial transcript -> NO_OUTPUT", decide_response(sig)["function"] == "NO_OUTPUT")

# Cool-down and rotation come from the persisted function, not from fixed text.
cand_turns = sum(1 for t in PROFIT_CASE if t.get("role") == "user")
prior = {"last_function": "ACKNOWLEDGE_AND_CONTINUE", "recent_functions": ["ACKNOWLEDGE_AND_CONTINUE"],
         "function_turn": cand_turns + 1}
hist = PROFIT_CASE + [{"role": "user", "content": REAL_PLAN},
                      {"role": "assistant", "content": "Baseline first, then the cuts line by line - carry on with the baseline."}]
sig, d = decide("Raw materials go from 50 to 45, labour from 20 to 19, overhead 15 to 14.25, logistics stays 15, "
                "so total cost is 93.25 percent of the old revenue", history=hist, channel="voice", prior_state=prior)
check("[voice] contextual beat on the previous turn -> cool-down: NO_OUTPUT", d["function"] == "NO_OUTPUT", d)
sig, d = decide("Now I take the margin: 100 minus 93.25 is 6.75, but revenue also needs restating so let me "
                "recompute on the new base before I compare the two margins", history=hist, prior_state=prior)
check("[text] same function never twice in a row (rotated)", d["function"] in CONTEXTUAL_PRESENCE
      and d["function"] != "ACKNOWLEDGE_AND_CONTINUE", d)

os.environ["INTERVIEWER_CONTEXTUAL_PRESENCE"] = "off"
sig, d = decide(REAL_PLAN)
check("kill switch INTERVIEWER_CONTEXTUAL_PRESENCE=off -> pre-V12 fixed short line (0 tokens)",
      d["function"] == "SHORT_ACK" and d["render"] == "fast", d)
os.environ.pop("INTERVIEWER_CONTEXTUAL_PRESENCE")
sig, d = decide(REAL_PLAN)
check("... and back on by default", d["function"] == "ACKNOWLEDGE_AND_CONTINUE", d)

# REAL (2026-10-01 22:34, owner's phone): "Hi." -> interviewer proposed a structure and asked
# "Shall we proceed with that structure?" -> "Yes." -> "Do ahead." (ASR for "Go ahead") got the
# stock "The data confirms that." because "do" was read as a question word -> DATA_REVEAL.
PROPOSED = [{"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "In a typical case, we'd usually start by exploring the context and "
                                             "defining the objective. Shall we proceed with that structure?"}]
for text in ("Yes.", "Do ahead.", "Go ahead", "Yes, go ahead.", "Sure, carry on", "okay"):
    for ch in ("text", "voice"):
        sig, d = decide(text, history=PROPOSED, channel=ch)
        check(f"[{ch}] {text!r} after the interviewer proposed something -> CONTINUE_AS_AGREED (model-worded)",
              d["function"] == "CONTINUE_AS_AGREED" and d["render"] == "model", d)
sig, d = decide("Do ahead.", history=PROPOSED)
check("'Do ahead.' is never a question or a data request", not sig["is_scope_question"] and d["function"] != "DATA_REVEAL", d)
for text, fn in [("Do you know the population?", "ANSWER_DIRECT"), ("Is it okay", "ANSWER_DIRECT"),
                 ("Yes but what is the population?", "ANSWER_DIRECT"), ("Can you help me here?", "MICRO_HINT")]:
    sig, d = decide(text, history=PROPOSED)
    check(f"real questions keep their route: {text!r} -> {fn}", d["function"] == fn, d)
sig, d = decide("Yes.", history=[{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Let's begin."}])
check("'Yes.' after a statement (nothing proposed) -> plain hand-back", d["function"] == "HAND_BACK", d)
from services.interviewer_decision import _MODE_FALLBACK  # noqa: E402
check("the nonsense fallback 'The data confirms that.' is gone", "confirms that" not in _MODE_FALLBACK["DATA_REVEAL"])

# Property over the 50 behaviour scenarios: the V10.2 gate decision is preserved.
try:
    from tools.eval_interviewer_behavior import SCENARIOS
except Exception:  # noqa: BLE001
    SCENARIOS = []
changed, fast_substantive = [], []
for sc in SCENARIOS:
    hist = sc.get("history") or sc.get("transcript") or []
    msg = sc.get("message") or sc.get("candidate") or sc.get("new_message") or ""
    if not msg:
        continue
    sig = compute_signals(hist, msg, sc.get("policy", "coached"))
    lane, mode, _ = evaluate_intervention_gate(sig)
    d = decide_response(sig)
    if lane == "SUBSTANTIVE" and d["reason"] != "floor_yield_after_work" and \
            d["function"] != _SUBSTANTIVE_FUNCTION.get(mode, mode):
        changed.append(sc.get("id"))
    if sig.get("is_substantive_reasoning") and d["render"] == "fast":
        fast_substantive.append(sc.get("id"))
check(f"{len(SCENARIOS)} behaviour scenarios: every substantive gate decision keeps its route",
      SCENARIOS and not changed, changed)
check("... and no substantive-reasoning turn is sent to the fixed-phrase lane", not fast_substantive, fast_substantive)

print()
print("=" * 72)
print("2. PACKET: the decision brief")
print("=" * 72)
sig, d = decide(REAL_PLAN)
pkt = build_interviewer_control_packet(sig, d, candidate_text=REAL_PLAN, recent_lines=["Go on."], allow_questions=False)
flat = json.dumps(pkt)
check("packet is JSON-serialisable", bool(json.loads(flat)))
ic = pkt["interviewer_control"]
REQUIRED = ["candidate_turn_type", "candidate_turn_summary", "reasoning_stage", "progress_state",
            "reasoning_quality", "candidate_confidence", "candidate_is_working", "candidate_needs_space",
            "explicit_help_request", "explicit_solution_request", "material_error", "frustration_state",
            "intervention_level", "response_function", "question_allowed", "hint_allowed",
            "correction_allowed", "solution_allowed", "silence_allowed", "target_length",
            "target_sentence_count", "content_specificity", "conversational_objective",
            "recent_response_functions", "repetition_avoidance"]
missing = [k for k in REQUIRED if f'"{k}"' not in flat]
check("packet carries every field of the spec", not missing, missing)
check("packet: function, no question, must reference the candidate",
      ic["response_function"] == "ACKNOWLEDGE_AND_CONTINUE" and ic["interviewer_decision"]["question_allowed"] is False
      and ic["response_generation"]["must_reference_candidate_content"] is True)
check("packet never allows certifying the work", ic["interviewer_decision"]["may_confirm_correctness"] is False)
check("packet holds no sentence to copy (no phrase-bank line, no 'fallback')",
      not any(t in flat for t in FAST_TEXTS) and "fallback" not in flat)
check("packet summary is taken from the candidate's own words", ic["candidate_state"]["candidate_turn_summary"].startswith("First, I'll compute the baseline"))
check("packet lists the numbers they stated", "8%" in ic["candidate_state"]["numbers_stated"], ic["candidate_state"]["numbers_stated"])
sig, d = decide(STEP_DONE, history=PROFIT_CASE + [{"role": "user", "content": DELHI_STRUCTURE},
                                                  {"role": "assistant", "content": "Go ahead."}])
pkt2 = build_interviewer_control_packet(sig, d, candidate_text=STEP_DONE)
check("ACKNOWLEDGE_AND_ORIENT carries the candidate's OWN earlier plan to orient by",
      (pkt2["interviewer_control"]["conversation_memory"]["candidate_plan"] or "").startswith("We start with the population"),
      pkt2["interviewer_control"]["conversation_memory"]["candidate_plan"])

print()
print("=" * 72)
print("3. LAYER C: validating a model-worded beat")
print("=" * 72)
line, probs = validate_contextual_line("Baseline first, then the cuts line by line. Carry on with the baseline.", REAL_PLAN)
check("specific line passes untouched", line.startswith("Baseline first") and probs == [], (line, probs))
line, probs = validate_contextual_line("Got it.", REAL_PLAN)
check("stock 'Got it.' is flagged as not picking up the candidate", probs == ["unreferenced"], probs)
line, probs = validate_contextual_line("That's correct. Carry the 92% cost base into the cuts.", REAL_PLAN)
check("unverified 'That's correct.' is removed, the rest kept", line == "Carry the 92% cost base into the cuts." and not probs, (line, probs))
line, probs = validate_contextual_line("You've set the baseline. What will you do with logistics?", REAL_PLAN)
check("question removed from a no-question function", "?" not in line and line.startswith("You've set the baseline"), line)
line, probs = validate_contextual_line('{"response_function": "ACKNOWLEDGE_AND_CONTINUE"}', REAL_PLAN)
check("leaked control metadata is rejected outright", probs == ["leak"] and line == "", (line, probs))
line, probs = validate_contextual_line("You've laid out the baseline. Now apply the cuts. Then the levers.", REAL_PLAN, channel="voice")
check("voice keeps one sentence", line == "You've laid out the baseline.", line)

from services.interviewer_decision import scrub_control_leak, StreamLeakGuard  # noqa: E402
check("deep lane: a JSON reply is reduced to its spoken line",
      scrub_control_leak('{"reply": "Start from the cost lines that change."}') == "Start from the cost lines that change.")
check("deep lane: a sentence naming a control field is dropped",
      scrub_control_leak("Look at the cost lines. My response_function is MICRO_HINT.") == "Look at the cost lines.")
g = StreamLeakGuard()
streamed = "".join(x for p in ['{"te', 'xt": "Which cost line moves most?"}'] for x in g.feed(p))
check("streaming: a reply that opens as JSON is held and released scrubbed",
      streamed == "" and g.flush("Okay, continue.") == "Which cost line moves most?")
g = StreamLeakGuard()
streamed = "".join(x for p in ["Which cost ", "line moves most?"] for x in g.feed(p))
check("streaming: a normal reply still streams token by token", streamed == "Which cost line moves most?" and g.flush() == "")

print()
print("=" * 72)
print("4. ENGINE + ROUTES with a fake model")
print("=" * 72)


class _Choice:
    def __init__(self, content):
        self.message = types.SimpleNamespace(content=content)
        self.delta = types.SimpleNamespace(content=content)


class FakeLLM:
    def __init__(self):
        self.replies, self.calls, self.raise_exc = [], [], None
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.raise_exc:
            raise self.raise_exc
        text = self.replies.pop(0) if self.replies else "Split buyers by age first."
        if kw.get("stream"):
            return iter([types.SimpleNamespace(choices=[_Choice(text)], usage=None, id="f")])
        return types.SimpleNamespace(choices=[_Choice(text)], usage=None, id="f")


LLM = FakeLLM()
ie.openai_client = lambda: LLM
ie.resolve_llm = lambda feature: (LLM, "fake-model", "openai")
ie.log_ai_usage = lambda **kw: None
idec.openai_client = lambda: None  # assessor off


def system_text(call):
    return "\n".join(m["content"] for m in call["messages"] if m["role"] == "system")


def run(text, history=PROFIT_CASE, replies=(), channel="text", prior=None, raise_exc=None):
    LLM.replies, LLM.calls, LLM.raise_exc = list(replies), [], raise_exc
    ctl = {}
    out = "".join(ie.stream_interviewer_reply("Snacks maker profitability case.", "profitability", history, text,
                                              control_out=ctl, channel=channel, prior_state=prior or {}))
    return out, ctl, list(LLM.calls)


GOOD = "You've set the 92% cost base and the line-by-line cuts - start with the baseline."
out, ctl, calls = run(REAL_PLAN, replies=[GOOD])
check("REPLAY real plan: one model call, line worded from the candidate's content", out == GOOD and len(calls) == 1, (out, len(calls)))
check("REPLAY real plan: not a phrase-bank line", out not in FAST_TEXTS and out.lower() not in STOCK)
check("prompt carries the JSON control packet and the function",
      "INTERVIEWER CONTROL PACKET" in system_text(calls[0]) and "RESPONSE FUNCTION: ACKNOWLEDGE_AND_CONTINUE" in system_text(calls[0]))
pkt_json = system_text(calls[0]).split("INTERVIEWER CONTROL PACKET (decision brief for this turn):\n", 1)[1].split("\n", 1)[0]
check("packet in the prompt parses as JSON", json.loads(pkt_json)["interviewer_control"]["response_function"] == "ACKNOWLEDGE_AND_CONTINUE")
check("control output: lane PRESENCE, function, rendered by the model",
      (ctl.get("lane"), ctl.get("function"), ctl.get("render")) == ("PRESENCE", "ACKNOWLEDGE_AND_CONTINUE", "model"), ctl)
check("contextual beat is short (small token budget)", calls[0]["max_tokens"] <= 100, calls[0]["max_tokens"])

out, ctl, calls = run(REAL_PLAN, replies=["Got it.", GOOD])
check("stock reply is regenerated once with the reason", len(calls) == 2 and out == GOOD
      and "cannot be used (unreferenced)" in calls[1]["messages"][-1]["content"], (out, len(calls)))
out, ctl, calls = run(REAL_PLAN, replies=['<<mode=interviewer; intervention=continue>>\n\nBaseline first, then the cuts - go ahead with the baseline.'])
check("control tag stripped and kept for the learner-state fold", out.startswith("Baseline first") and ctl["tag"].get("intervention") == "continue", (out, ctl.get("tag")))
out, ctl, calls = run(REAL_PLAN, replies=["What is your baseline margin?", "Is that it?"])
check("two unusable replies -> plain hand-back, never an error, never a question",
      out in FAST_TEXTS and "?" not in out and ctl.get("fallback") is True, (out, ctl))
out, ctl, calls = run(REAL_PLAN, replies=["Got it.", "Right."])
check("model insists on a stock ack twice -> labelled fallback, the stock line is not passed off as contextual",
      ctl.get("fallback") is True and len(calls) == 2, (out, ctl))
out, ctl, calls = run(REAL_PLAN, raise_exc=RuntimeError("provider down"))
check("provider failure on a presence beat -> hand-back line, not an error", out in FAST_TEXTS and ctl.get("fallback") is True, out)
out, ctl, calls = run("Okay.", history=PROFIT_CASE + [{"role": "user", "content": "Let me start."},
                                                     {"role": "assistant", "content": "Go on."}])
check("minimal turn: fast lane, zero model calls", calls == [] and out in FAST_TEXTS, (out, len(calls)))
out, ctl, calls = run("Can you help me here?", replies=["<<intervention=micro_hint; hint=2>>\n\nStart from the cost lines that change."])
check("help turn: substantive deep lane, function MICRO_HINT, V10.2 HINT instruction kept",
      ctl.get("function") == "MICRO_HINT" and ctl.get("mode") == "HINT" and "YOUR MOVE THIS TURN: HINT" in system_text(calls[0]), ctl)

# ----- routes
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
    def __init__(self, history, session_state=None):
        self.writes = []
        self.tables = {
            "attempts": [{"id": "a1", "user_id": "u1", "status": "active", "case_id": "c1",
                          "clarification_quota": 20, "clarification_used": 0,
                          "tier_at_start": "pro", "session_state": dict(session_state or {})}],
            "cases": [{"id": "c1", "type": "profitability", "title": "Snacks", "difficulty": "easy",
                       "content": "Snacks maker profitability case.", "is_active": True}],
            "attempt_messages": [dict(m, attempt_id="a1", id=f"m{i}", created_at=f"t{i:04d}")
                                 for i, m in enumerate(history)],
        }

    def table(self, name):
        return _Q(self, name)

    def state_writes(self):
        return [w[2]["session_state"] for w in self.writes if w[0] == "update" and "session_state" in w[2]]


DB = FakeDB(PROFIT_CASE)
att.get_supabase_client = lambda: DB
att.get_verified_user = lambda sb, auth: ("u1", {"id": "u1"})
att.get_verified_user_id = lambda sb, auth: "u1"
att.is_guest_user = lambda u: False
att.check_rate_limit = lambda *a, **k: None
att.assert_daily_budget = lambda *a, **k: None
att.llm_case_content = lambda case: case["content"]
app = FastAPI()
app.include_router(att.router)
client = TestClient(app)
H = {"Authorization": "Bearer t"}


def voice(text, history, replies=(), state=None):
    global DB
    DB = FakeDB(history, state)
    LLM.replies, LLM.calls, LLM.raise_exc = list(replies), [], None
    r = client.post("/attempts/a1/voice-decision", json={"content": text}, headers=H)
    att._await_after_turn("a1")
    return r, DB


r, db = voice(REAL_PLAN, PROFIT_CASE, replies=["You've set the cost base and the cuts. Start with the baseline."])
d = r.json()
check("/voice-decision: substantive turn -> PRESENCE with the model's contextual line",
      r.status_code == 200 and d.get("lane") == "PRESENCE" and d.get("function") == "ACKNOWLEDGE_AND_CONTINUE"
      and d.get("say") == "You've set the cost base and the cuts.", d)
check("/voice-decision: event has the same line (client contract unchanged)", (d.get("event") or {}).get("data", {}).get("text") == d.get("say"))
check("/voice-decision: control metadata only as the existing fields (no packet in the response)", "interviewer_control" not in r.text)
st = db.state_writes()
check("function folded into session_state for the next turn",
      len(st) == 1 and st[0].get("last_function") == "ACKNOWLEDGE_AND_CONTINUE" and st[0].get("function_turn") == 1, st)

hist2 = PROFIT_CASE + [{"role": "user", "kind": "voice", "content": REAL_PLAN},
                       {"role": "assistant", "kind": "voice", "content": d.get("say") or ""}]
r, db = voice("Raw materials go from 50 to 45, labour from 20 to 19, overhead 15 to 14.25, logistics stays 15, "
              "so total cost is 93.25 percent of the old revenue", hist2, state=st[0])
check("next voice turn honours the cool-down from the folded function (SILENCE, 0 model calls)",
      r.json().get("lane") == "SILENCE" and LLM.calls == [], r.json())

DB = FakeDB(PROFIT_CASE)
LLM.replies, LLM.calls = ["You've set the cost base and the cuts. Start with the baseline."], []
r = client.post("/attempts/a1/messages", json={"content": REAL_PLAN, "kind": "text"}, headers=H)
att._await_after_turn("a1")
rows = [w[2] for w in DB.writes if w[0] == "insert" and w[2].get("role") == "assistant"]
check("/messages: contextual beat streamed and persisted as the assistant row",
      r.status_code == 200 and "event: done" in r.text and rows and rows[0]["content"].startswith("You've set the cost base"), r.text[-200:])
check("C9 counting untouched (count_clarifications on the real plan)", count_clarifications(REAL_PLAN, "text") == 0)

print()
print("=" * 72)
print("5. VOICE PROTOCOL LABEL")
print("=" * 72)
from prompts.voice_renderer import strip_say_label  # noqa: E402
check("strip 'SAY:' label", strip_say_label("SAY: That's an interesting perspective.") == "That's an interesting perspective.")
check("strip repeated / lowercase label", strip_say_label("say: SAY:  Go ahead.") == "Go ahead.")
check("ordinary 'Say,' sentence untouched", strip_say_label("Say, what about costs?") == "Say, what about costs?")
DB = FakeDB(PROFIT_CASE)
r = client.post("/attempts/a1/realtime-turn", json={"role": "assistant", "content": "SAY: The data confirms that."}, headers=H)
rows = [w[2] for w in DB.writes if w[0] == "insert" and w[2].get("role") == "assistant"]
check("/realtime-turn never stores the label", r.status_code == 200 and rows and rows[-1]["content"] == "The data confirms that.", rows)
r = client.post("/attempts/a1/realtime-turn", json={"role": "user", "content": "SAY: hello"}, headers=H)
rows = [w[2] for w in DB.writes if w[0] == "insert" and w[2].get("role") == "user"]
check("... candidate text is stored exactly as spoken", rows and rows[-1]["content"] == "SAY: hello")
LLM.replies, LLM.calls = ["Start from the cost lines that change."], []
out, ctl, calls = run("Can you help me here?", history=PROFIT_CASE + [
    {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "SAY: Let's look at the margin."}])
sent = [m["content"] for m in calls[0]["messages"] if m["role"] == "assistant"]
check("the interviewer model never sees an old 'SAY:' line (it would copy the pattern)",
      sent and not any(c.startswith("SAY:") for c in sent), sent)

print()
if _fail:
    print(f"{len(_fail)} FAILED: {_fail}")
    sys.exit(1)
print("ALL PASS")
