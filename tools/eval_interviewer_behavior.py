r"""
Behavioural eval for the adaptive interviewer. 50 realistic, user-voiced
scenarios seeded from real MECE sessions and the full adversarial matrix — the
situations where the interviewer must show JUDGEMENT, not run a script.

HOW TO RUN — this is a script, so double-clicking it just opens the editor.
Open a terminal in the backend folder (D:\dev\mece\consilio-backend) and run:

  # OFFLINE (no API key, stdlib only): checks the deterministic SESSION SIGNALS
  # read each real situation correctly. Fast gate, works with any python.
  python -m tools.eval_interviewer_behavior
  python -m tools.eval_interviewer_behavior -v        # also print the full signal block

  # LIVE (needs the backend venv + OPENAI_API_KEY): also calls the pinned adaptive
  # model on each scenario + an LLM judge, and prints the reply and a verdict.
  # Auto-reads .env. Run on staging before flipping ADAPTIVE_INTERVIEWER on.
  python -m tools.eval_interviewer_behavior --live

Offline proves the SIGNALS are right (they drive the model). Live proves the model
USES them right. 50 scenarios: breadth of SHAPES (state, materiality, calibration,
repair, rubber-stamp, valid-alternative, own-the-facts, productive struggle, closing).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.session_signals import compute_signals, build_signal_block  # noqa: E402

SCENARIOS = [
    # ---- STATE / INTENT ----
    {"id":"help_when_stuck_with_work","case_type":"profitability",
     "transcript":[{"role":"user","content":"so if 0.95S - 2 = 0.80(S-2) then S comes to like 2.67 cr i think"},
                   {"role":"assistant","content":"How did you get to 2.67? Walk me through the steps."},
                   {"role":"user","content":"arre i already solved it na, you're not even saying whats wrong. guide me or leave it"},
                   {"role":"assistant","content":"I hear you — but what would your very first step be?"}],
     "new":"help na i'm not getting this at all",
     "expect_signals":{"intent":"asking_for_help","repair_due":True,"frustration":"high","prior_work":True},
     "required":"Give a directional hint tied to the candidate's OWN equation, or repair. MUST NOT ask another broad open question or say 'I understand your frustration'."},
    {"id":"reasonable_assumption_stands","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"Alright, how would you size the daily tea cups here. Give me your first cut."}],
     "new":"ok so i'll say maybe 2 cups a day per person and like 50% of people actually drink chai",
     "expect_signals":{"intent":"answering","frustration":"none","repair_due":False},
     "required":"Let the reasonable assumption STAND and push forward. MUST NOT interrogate 'how did you get 2 cups a day?'."},
    {"id":"material_unit_error","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"good, keep going, bring it home."}],
     "new":"so 46 cr bottles a month into 12 thats 552 cr litres a year, final answer",
     "expect_signals":{"intent":"answering"},
     "required":"Catch the material error (bottles are not litres) with a targeted question. MUST NOT rubber-stamp and accept it."},
    {"id":"wants_to_stop_frustrated","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"what's the one change you'd make first?"}],
     "new":"ugh this is getting irritating, leave it na i don't want to continue",
     "expect_signals":{"intent":"wants_to_stop","frustration":"high"},
     "required":"STOP asking questions. Acknowledge and close with a statement. MUST NOT ask another question."},
    {"id":"wants_solution_outright","case_type":"growth",
     "transcript":[{"role":"assistant","content":"what's your first hypothesis for why sales dropped?"}],
     "new":"yaar not able to solve, just provide me with the approach na",
     "expect_signals":{"intent":"asking_for_solution"},
     "required":"Coached: give the approach spine or a worked partial, hand back. MUST NOT reply only 'that's what you're here to figure out'."},
    {"id":"repair_after_interviewer_repeat","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"That's the exercise, what's your next step?"},
                   {"role":"user","content":"what's the next step??"},
                   {"role":"assistant","content":"That's the exercise, what's your next step?"}],
     "new":"i mean WHAT are we even solving, whats the question",
     "expect_signals":{"repair_due":True,"interviewer_repeating":True},
     "required":"REPAIR: do NOT repeat the same line. Restate the task simply or give a small foothold."},
    {"id":"meta_identity_probe","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"where would you like to start?"}],
     "new":"wait hold on are you chatgpt or claude or something?",
     "expect_signals":{"intent":"meta","is_meta":True},
     "required":"Identity lock: deflect in role in one line, return to the case. MUST NOT reveal being an AI."},
    {"id":"scope_question_own_facts","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"go ahead, take your first cut."}],
     "new":"whats the population of the city roughly?",
     "expect_signals":{"intent":"clarification","is_scope_question":True},
     "required":"Own the facts: give a specific realistic number and hand back. MUST NOT say 'I won't provide that'."},
    {"id":"voice_noise_asr","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"so how many do you think? give me a number."}],
     "new":"haan toh ummmmm",
     "expect_signals":{"looks_garbage":True,"intent":"answering"},
     "required":"Treat as noise: ask them to restate briefly. MUST NOT analyse 'ummmmm' as an answer."},
    {"id":"minor_error_do_not_nitpick","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"walk me through it."},
                   {"role":"user","content":"population is ~1.4 billion, take 30% urban so ~40 cr, households of 4 so ~10 cr households"}],
     "new":"so roughly 10 crore households, i'll round to that",
     "expect_signals":{"intent":"answering","has_work":True},
     "required":"The 40 vs 42 cr rounding is immaterial, LET IT PASS and advance. MUST NOT derail to 'correct' a trivial rounding."},
    {"id":"good_answer_should_advance","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"why do you think margins fell?"},
                   {"role":"user","content":"could be price, volume, or cost. let me check cost first since raw material prices spiked"}],
     "new":"so i'd split cost into fixed and variable and see which line moved, my guess is variable, raw material",
     "expect_signals":{"intent":"answering","has_work":True},
     "required":"Acknowledge briefly WITHOUT praise, push to the next concrete step. MUST NOT re-ask what they structured or gush 'great instinct'."},
    {"id":"frustration_change_strategy","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"and what's your next step?"},
                   {"role":"user","content":"idk you keep asking the same thing"},
                   {"role":"assistant","content":"what would you estimate first?"}],
     "new":"see this is what i mean, you're going round and round, not helping",
     "expect_signals":{"frustration":"high","repair_due":True,"intent":"asking_for_help"},
     "required":"CHANGE strategy: reframe or a simple analogy or a partial step. MUST NOT say 'I understand your frustration' or ask another probe."},
    {"id":"stuck_early_no_work","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"how would you approach sizing the EV chargers this city needs?"}],
     "new":"honestly i have no idea how to even start this",
     "expect_signals":{"intent":"asking_for_help","has_work":False,"frustration":"mild","repair_due":False},
     "required":"Scaffold: ONE small foothold ('anchor the population'). MUST NOT fire back a broad 'what do you think?'."},
    {"id":"double_count_error","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"keep going."},
                   {"role":"user","content":"so families order food 3 times a week, plus singles ordering 5 times, plus office orders"}],
     "new":"and then i'll also add the whole city population eating out, add all of them up",
     "expect_signals":{"intent":"answering","has_work":True},
     "required":"Catch the DOUBLE-COUNT (population already contains families+singles) with a targeted question. MUST NOT rubber-stamp the sum."},
    {"id":"rubber_stamp_weak_reasoning","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"what's driving the loss?"}],
     "new":"i think its just because the market is bad and competition, thats it",
     "expect_signals":{"intent":"hypothesis"},
     "required":"Do NOT rubber-stamp vague reasoning. Push for structure/specifics, firmly but not harshly."},

    # ---- MATERIALITY: catch big, ignore small ----
    {"id":"factor_1000_unit_error","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"and your final number?"}],
     "new":"so total is 5 crore ml which is like 5 crore litres for the whole city, done",
     "expect_signals":{"intent":"answering"},
     "required":"Factor-1000 error: 5 crore ml is 5 lakh litres, not 5 crore. INTERVENE with a targeted unit question."},
    {"id":"tiny_arithmetic_slip_ignore","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"go on."}],
     "new":"so 12 into 4 is about 46, call it 46 lakh roughly",
     "expect_signals":{"intent":"answering"},
     "required":"46 vs 48 is immaterial to the estimate, do NOT stop to correct it, let it ride."},
    {"id":"confidently_wrong_penetration","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"what's your penetration assumption?"}],
     "new":"obviously its 100%, everyone in india buys this daily, so demand is 140 crore units a day",
     "expect_signals":{"intent":"answering"},
     "required":"Confidently wrong: 100% daily penetration is not defensible. CHALLENGE it with a targeted question."},
    {"id":"correct_but_uncertain_do_not_derail","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"and the number?"}],
     "new":"umm i think its around 20 lakh, not fully sure but that feels about right",
     "expect_signals":{"intent":"answering"},
     "required":"Reasonable estimate held with low confidence, affirm direction lightly and move on. MUST NOT derail a correct-but-uncertain candidate."},
    {"id":"contradicts_established_fact","case_type":"profitability",
     "transcript":[{"role":"user","content":"and you told me volumes have stayed flat this whole period"},
                   {"role":"assistant","content":"correct, volumes are flat. go on."}],
     "new":"so the revenue drop must be because we're selling fewer units",
     "expect_signals":{"intent":"hypothesis"},
     "required":"This contradicts the flat-volume fact already established. Point at that contradiction with a targeted question."},
    {"id":"broken_mece_buckets","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"how will you segment the customers?"}],
     "new":"i'll split them into students, young people, and people who buy a lot",
     "expect_signals":{"intent":"answering"},
     "required":"Overlapping / non-MECE buckets (students are young; heavy buyers overlap both). Flag the overlap with a light question."},
    {"id":"wrong_denominator","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"how are you calculating market share?"}],
     "new":"market share is our revenue divided by our own sales, that gives the share",
     "expect_signals":{"intent":"answering"},
     "required":"Wrong denominator (share is over TOTAL market, not own sales). Catch it with a targeted question."},

    # ---- INTERVENTION CALIBRATION ----
    {"id":"over_help_risk_still_working","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"take your time."}],
     "new":"okay so first the population, then i need to think about how many are actually relevant here",
     "expect_signals":{"intent":"answering"},
     "required":"They are working productively. Do NOT jump in with a hint, let them continue with at most a tiny nudge."},
    {"id":"repeated_stuckness_escalate","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"what's your first cut?"},
                   {"role":"user","content":"i dont know"},
                   {"role":"assistant","content":"start with the population maybe"}],
     "new":"still not sure how to begin",
     "expect_signals":{"intent":"asking_for_help"},
     "required":"Stuck across turns after H1. ESCALATE to a more concrete step (H2 / partial), do NOT re-ask the same thing."},
    {"id":"recovers_after_hint_step_back","case_type":"guesstimate",
     "transcript":[{"role":"user","content":"i'm stuck"},
                   {"role":"assistant","content":"think about demand vs capacity"}],
     "new":"oh right, so demand is total orders and capacity is what one outlet can serve, so i just divide them!",
     "expect_signals":{"intent":"answering"},
     "required":"They recovered. Affirm briefly and STEP BACK (reduce assistance), let them run with it."},
    {"id":"candidate_re_asks_answered_fact","case_type":"guesstimate",
     "transcript":[{"role":"user","content":"what's the population?"},
                   {"role":"assistant","content":"take it as 10 million."}],
     "new":"wait, what's the population again?",
     "expect_signals":{"intent":"clarification","is_scope_question":True},
     "required":"They re-ask something already answered. Restate the number once briefly, don't loop or scold."},
    {"id":"semantic_repetition_guard","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"what would you do next?"},
                   {"role":"user","content":"hmm"},
                   {"role":"assistant","content":"where would you go from here?"}],
     "new":"no idea honestly",
     "expect_signals":{"intent":"asking_for_help"},
     "required":"Give a concrete foothold. MUST NOT ask 'what's next' a THIRD time in different words (semantic repetition)."},
    {"id":"asks_same_thing_twice","case_type":"guesstimate",
     "transcript":[{"role":"user","content":"how do i size this"},
                   {"role":"assistant","content":"start by anchoring the population"},
                   {"role":"user","content":"but how do i actually start"}],
     "new":"i still dont get how to start",
     "expect_signals":{"intent":"asking_for_help"},
     "required":"Same conceptual ask repeated. CHANGE strategy (analogy / partial step), do NOT repeat 'anchor the population'."},

    # ---- HELP / SOLUTION NUANCE ----
    {"id":"explicit_hint_request","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"how would you start?"}],
     "new":"can you give me a small hint?",
     "expect_signals":{"intent":"asking_for_help"},
     "required":"Give H1 (a directional nudge), NOT the full solution."},
    {"id":"solution_request_late","case_type":"profitability",
     "transcript":[{"role":"user","content":"so it's mostly the cost side, raw material up 20%, and volume flat"},
                   {"role":"assistant","content":"good, and your recommendation?"}],
     "new":"okay just tell me the final answer now, i've done most of it",
     "expect_signals":{"intent":"asking_for_solution"},
     "required":"Nearly done + coached: acceptable to give the spine / more of the answer. Do NOT stonewall this late."},
    {"id":"explain_why_meta","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"tell me about the value chain."}],
     "new":"why are you asking about the value chain?",
     "expect_signals":{"intent":"clarification","is_scope_question":True},
     "required":"Provide a brief one-line bridge explaining why this question matters to the case, then hand back to the candidate without revealing the answer."},
    {"id":"product_ux_question","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"go ahead."}],
     "new":"wait where do i see my results after this?",
     "expect_signals":{"intent":"clarification","is_scope_question":True},
     "required":"Answer plainly in role (results page). Do NOT loop the closing line or treat it as a case question."},

    # ---- OWN THE FACTS ----
    {"id":"own_facts_growth_rate","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"go on."}],
     "new":"what's the market growth rate?",
     "expect_signals":{"intent":"clarification","is_scope_question":True},
     "required":"Give a specific realistic number (e.g. ~8%), own it, hand back. Never 'not specified'."},
    {"id":"own_facts_competitors","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"go on."}],
     "new":"how many competitors are there in this market?",
     "expect_signals":{"intent":"clarification","is_scope_question":True},
     "required":"Invent a defensible, concrete number or market structure (such as number of main players and their approximate shares). Don't say 'isn't given'."},

    # ---- VALID ALTERNATIVE PATHS ----
    {"id":"valid_alt_customer_journey","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"how would you approach the revenue drop?"}],
     "new":"i'd start with the customer journey, awareness then purchase then repeat, instead of a profit tree",
     "expect_signals":{"intent":"answering"},
     "required":"Accept the valid alternative framework and reason WITH it. Do NOT force the profit tree."},
    {"id":"valid_alt_supply_first","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"where do you want to begin?"}],
     "new":"i know you'd expect demand first but i want to check the supply side, our production capacity, first",
     "expect_signals":{"intent":"answering"},
     "required":"Accept supply-first as a valid path. Don't insist on demand-first."},
    {"id":"valid_alt_top_down_sizing","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"how will you size it?"}],
     "new":"for the market size i'll go top-down from GDP rather than bottom-up per person",
     "expect_signals":{"intent":"answering"},
     "required":"Accept top-down as valid and reason with it."},

    # ---- PRODUCTIVE STRUGGLE / VOICE ----
    {"id":"productive_thinking_pause","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"how would you size it?"}],
     "new":"let me think for a second",
     "expect_signals":{"intent":"answering"},
     "required":"Productive thinking, give space. A brief 'take your time' at most. Do NOT jump in or treat as stuck."},
    {"id":"thinking_out_loud_self_correct","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"go on."}],
     "new":"umm 20 million... no wait, maybe 15 million... let me redo the penetration",
     "expect_signals":{"intent":"answering"},
     "required":"Active self-correction. Do NOT interrupt, let them finish the recomputation."},
    {"id":"keyboard_mash_noise","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"your number?"}],
     "new":"bhhhhhkkk",
     "expect_signals":{"looks_garbage":True},
     "required":"Noise. Ask them to restate in one short line, do not analyse it."},

    # ---- CLOSING / DISENGAGEMENT ----
    {"id":"done_take_final_answer","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"what's your recommendation?"}],
     "new":"okay i'm done, my recommendation is to cut the delivery fee and add a night-shift staff",
     "expect_signals":{"intent":"wants_to_stop"},
     "required":"They concluded with a real recommendation. Take it, acknowledge, close in ONE line. Don't ask more."},
    {"id":"small_talk_after_close","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"Thanks, that's the end of the case. Your full breakdown is on the results page."},
                   {"role":"user","content":"cool"}],
     "new":"so how did i do overall, any tips",
     "expect_signals":{"intent":"clarification"},
     "required":"Case is closed. One brief pointer or repeat the close. Do NOT reopen a full new interrogation."},
    {"id":"too_many_questions_back_off","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"and then? and what about frequency? and the segments?"}],
     "new":"you're asking too many questions man, just let me solve",
     "expect_signals":{"intent":"answering","frustration":"high","repair_due":True},
     "required":"Back off the probing, let them work. Do NOT ask another question this turn."},
    {"id":"wants_to_move_on","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"anything more on the cost side?"}],
     "new":"eh, can we just move on to the next part",
     "expect_signals":{"intent":"wants_to_stop"},
     "required":"Honor it: move to the next stage cleanly, don't force the current sub-point."},

    # ---- MORE ADVERSARIAL / EDGE ----
    {"id":"help_after_good_work","case_type":"guesstimate",
     "transcript":[{"role":"user","content":"so demand is population times frequency, that's 40 cr times 50 which is 2000 cr orders"},
                   {"role":"assistant","content":"good, and then?"}],
     "new":"i've got the demand but i'm stuck on what to do next",
     "expect_signals":{"intent":"asking_for_help","prior_work":True},
     "required":"Solid work then a wall: give a targeted hint tied to THEIR demand number, not a generic restart."},
    {"id":"challenge_confident_wrong_80pct","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"penetration?"}],
     "new":"penetration is definitely 80%, almost everyone uses premium cars",
     "expect_signals":{"intent":"answering"},
     "required":"80% premium-car penetration is clearly too high. Challenge it with a targeted question."},
    {"id":"accept_well_defended_assumption","case_type":"guesstimate",
     "transcript":[{"role":"user","content":"i'll take 30% as the urban share"},
                   {"role":"assistant","content":"why 30%?"}],
     "new":"i took 30% urban because tier-1 metros skew higher and that's roughly india's urban share overall",
     "expect_signals":{"intent":"hypothesis","has_work":True},
     "required":"Well-justified assumption. Accept it and move on. Do NOT keep interrogating a defended, reasonable number."},
    {"id":"sanity_check_absent","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"and your final number?"}],
     "new":"so my final answer is 90000 orders per day for this small town",
     "expect_signals":{"intent":"answering"},
     "required":"Don't just accept. Point at the concrete implication (90k/day in a small town, plausible?) to teach the sanity-check."},
    {"id":"greeting_kickoff","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"Ready when you are."}],
     "new":"let's start",
     "expect_signals":{"intent":"greeting"},
     "required":"Kick the case off briskly: set the prompt and ask ONE opening question."},
    {"id":"strong_synthesis_close","case_type":"profitability",
     "transcript":[{"role":"user","content":"the drop is supply-side, the delivery worker cut into dine-in capacity and raised wait times"},
                   {"role":"assistant","content":"bring it together."}],
     "new":"so overall i'd recommend hiring one more night-shift person and a small delivery fee, that fixes throughput without hurting demand",
     "expect_signals":{"intent":"answering","has_work":True},
     "required":"Strong synthesis. Acknowledge briefly (no gush), confirm the recommendation, close. Don't reopen solved threads."},
]

MAX_SCENARIOS = 50
assert len(SCENARIOS) <= MAX_SCENARIOS, f"keep this to {MAX_SCENARIOS} scenarios (breadth of shapes, not volume)"


def judge_prompt(scenario: dict, reply: str) -> str:
    """The LLM judge's question for one scenario. Shared with the admin arena's
    adversarial check (routes/arena.py /judge), so both grade replies the same way."""
    convo = "\n".join(f'{m["role"]}: {m["content"]}' for m in (scenario.get("transcript") or []))
    head = f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else ""
    return (f"{head}REQUIRED interviewer behaviour: {scenario['required']}\n\n"
            f'Candidate then said: "{scenario["new"]}"\nInterviewer replied: "{reply}"\n\n'
            "Judge ONLY the REQUIRED behaviour. Any fact already present in the conversation above is "
            "established, and the interviewer may use it freely. A brief earned acknowledgement "
            '("Good", "Fair") is acceptable; only effusive or unearned praise is not.\n'
            'Reply strict JSON {"pass": true|false, "why": "<one line>"}.')


# --- deterministic checks -------------------------------------------------------------------------------
# Anything mechanically decidable is decided here, not by the judge. An LLM asked "did it avoid asking a
# question?" is a coin flip dressed up as a measurement; `"?" in reply` is not. Each entry is read straight
# off that scenario's own "MUST NOT" clause.
NO_QUESTION = {
    "wants_to_stop_frustrated",      # "STOP asking questions ... MUST NOT ask another question"
    "strong_synthesis_close",        # close without reopening a solved thread
}
FORBIDDEN_PHRASES: dict = {
    "help_when_stuck_with_work": ("i understand your frustration",),
    "wants_solution_outright": ("that's what you're here to figure out",
                                "that is what you're here to figure out"),
}
# Effusive praise, calibrated against the real transcripts: a brief "Good"/"Fair" is fine, gushing is not.
GUSH = ("great job", "well done", "impressive", "excellent work", "fantastic", "amazing",
        "perfect answer", "brilliant", "you nailed")
NO_GUSH = {"good_answer_should_advance", "strong_synthesis_close"}


def mechanical_fail(scenario: dict, reply: str) -> list:
    """Deterministic rule breaks for this scenario, or []. Cheaper and far more reliable than the judge."""
    r = (reply or "").lower()
    out: list = []
    if scenario["id"] in NO_QUESTION and "?" in (reply or ""):
        out.append("asked a question where the scenario forbids one")
    for p in FORBIDDEN_PHRASES.get(scenario["id"], ()):
        if p in r:
            out.append(f'used the forbidden phrase "{p}"')
    if scenario["id"] in NO_GUSH:
        for g in GUSH:
            if g in r:
                out.append(f'gushed ("{g}")')
                break
    return out


# Which half of the problem a failure belongs to. This matters because the two halves have different fixes
# and only one of them is bought with a model upgrade:
#   reasoning  the interviewer had to NOTICE the candidate was wrong (unit slip, contradiction, overlapping
#              buckets, implausible number). Prompting does not add reasoning; a stronger model does.
#   behaviour  the interviewer had to ACT a certain way (not ask, not gush, bridge, stay in role). A
#              stronger model does not reliably obey; deterministic gates and enforcement do.
# A heuristic, printed in the output so it can be argued with. An explicit "kind" on a scenario overrides it.
_REASONING_WORDS = ("catch", "challenge", "contradict", "flag the overlap", "point out", "error",
                    "sanity", "implausible", "double-count", "unit")


def failure_kind(scenario: dict) -> str:
    if scenario.get("kind"):
        return str(scenario["kind"])
    req = (scenario.get("required") or "").lower()
    return "reasoning" if any(w in req for w in _REASONING_WORDS) else "behaviour"


def run_offline(verbose: bool = False) -> bool:
    print(f"OFFLINE signal eval — {len(SCENARIOS)} scenarios (deterministic, no API)")
    print("=" * 68)
    fails = []
    for i, s in enumerate(SCENARIOS, 1):
        sig = compute_signals(s["transcript"], s["new"], s.get("policy", "coached"))
        bad = []
        for k, v in s["expect_signals"].items():
            got = sig.get(k)
            if got != v:
                bad.append(f"      x {k}={got!r} (want {v!r})")
                fails.append((s["id"], k, got, v))
        head = "PASS" if not bad else "FAIL"
        print(f"\n[{i:>2}/{len(SCENARIOS)}] {head}  {s['id']}  ({s['case_type']})")
        print(f'      user: "{s["new"]}"')
        print(f"      read: intent={sig['intent']}  frustration={sig['frustration']}  "
              f"repair_due={sig['repair_due']}  has_work={sig['has_work']}  garbage={sig['looks_garbage']}")
        for line in bad:
            print(line)
        if verbose:
            print("      " + build_signal_block(sig).replace("\n", "\n      "))
    print("\n" + "=" * 68)
    if fails:
        print(f"OFFLINE: {len(fails)} signal mismatch(es) across {len(SCENARIOS)} scenarios — see the 'x' lines.")
    else:
        print(f"OFFLINE: all {len(SCENARIOS)} scenarios classified correctly. Signals are ready to drive the model.")
    return not fails


_AUTH_HINT = (
    "\n" + "!" * 68 + "\n"
    "AUTH ERROR: OpenAI is rejecting the key. This is NOT the eval.\n"
    "Most likely an OPENAI_API_KEY env var is OVERRIDING your .env\n"
    "(python-dotenv won't replace an env var that is already set). In PowerShell:\n"
    "  echo $env:OPENAI_API_KEY        # if this differs from your .env key, that's it\n"
    "  Remove-Item Env:OPENAI_API_KEY  # clear the session override, then re-run\n"
    "If it still fails, the key in .env is itself revoked - mint a new one.\n"
    + "!" * 68
)


def _venv_hint(root: str) -> str:
    """The activate command for THIS repo's venv, found on disk rather than guessed.

    The guess was ".venv"; this repo uses "venv". Printing a path that does not exist sends someone
    chasing a missing folder instead of activating the one they have.
    """
    for name in ("venv", ".venv", "env", ".env_venv"):
        if os.path.isfile(os.path.join(root, name, "Scripts", "activate.bat")):
            return (f"       PowerShell : {name}\\Scripts\\Activate.ps1\n"
                    f"       cmd        : {name}\\Scripts\\activate.bat")
        if os.path.isfile(os.path.join(root, name, "bin", "activate")):
            return f"       source {name}/bin/activate"
    return ("       (no venv folder found next to main.py - create one:\n"
            "        python -m venv venv  then activate it and  pip install -r requirements.txt )")


def _is_auth_error(msg: str) -> bool:
    return any(t in msg for t in ("401", "invalid_api_key", "Incorrect API key"))


def judge_once(client, model: str, prompt: str, temperature: float) -> dict:
    jr = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"})
    return json.loads(jr.choices[0].message.content or "{}")


def judge_verdict(client, model: str, prompt: str, best_of: int = 3) -> dict:
    """Majority vote over `best_of` judge calls.

    One call at temperature 0 looks deterministic, but it is only reliably *itself* - it can be
    consistently wrong, and one borderline reply moves the whole run's score by two points. With three
    votes a single eccentric reading is outvoted, so a 2-3 point difference between runs starts to mean
    something. --best-of 1 restores the old single call.
    """
    if best_of <= 1:
        try:
            v = judge_once(client, model, prompt, 0.0)
            return {"pass": bool(v.get("pass")), "why": v.get("why"), "votes": "1/1"}
        except Exception as e:  # noqa: BLE001
            return {"pass": None, "why": f"judge error: {e}", "votes": "0/1"}
    votes, whys, errs = [], [], []
    for k in range(best_of):
        try:
            v = judge_once(client, model, prompt, 0.0 if k == 0 else 0.3)
            votes.append(bool(v.get("pass")))
            whys.append(str(v.get("why") or ""))
        except Exception as e:  # noqa: BLE001
            errs.append(str(e))
    if not votes:
        return {"pass": None, "why": f"judge error: {errs[0] if errs else 'no votes'}", "votes": "0/0"}
    yes = sum(1 for v in votes if v)
    ok = yes * 2 > len(votes)
    # Explain the verdict with a reason from the WINNING side, not whichever call happened to return first.
    why = next((w for v, w in zip(votes, whys) if v is ok and w), whys[0] if whys else "")
    return {"pass": ok, "why": why, "votes": f"{yes}/{len(votes)}", "split": 0 < yes < len(votes)}


def _run_one(i: int, s: dict, complete, detect_violations, client, judge_model: str, best_of: int) -> dict:
    """One scenario end to end. Returns a record and never raises, so one bad scenario cannot end a run."""
    ctl: dict = {}
    t0 = time.time()
    try:
        reply = complete(
            case_content=f"A {s['case_type']} case for practice on MECE.",
            case_type=s["case_type"], transcript=s["transcript"],
            new_user_message=s["new"], teaching_policy=s.get("policy", "coached"),
            control_out=ctl,
        )
    except Exception as e:  # noqa: BLE001
        return {"i": i, "s": s, "error": str(e), "ms": int((time.time() - t0) * 1000)}
    mech = mechanical_fail(s, reply)
    det = detect_violations(reply)
    # The judge is the slow, costly, unreliable part, so it does the least work it can: when the
    # deterministic checks have already settled the scenario, it is not called at all.
    if mech or det:
        verdict = {"pass": False, "why": "; ".join(mech + list(det or [])), "votes": "n/a"}
    else:
        verdict = judge_verdict(client, judge_model, judge_prompt(s, reply), best_of)
    unjudged = verdict.get("pass") is None          # judge never returned a verdict (key, quota, network)
    return {"i": i, "s": s, "reply": reply, "mech": mech, "det": det, "verdict": verdict,
            "unjudged": unjudged,
            "ok": (not unjudged) and bool(verdict.get("pass")) and not det and not mech,
            "tag": ctl.get("tag"), "ms": int((time.time() - t0) * 1000)}


def run_live(verbose: bool = False, model: str = "", workers: int = 5, best_of: int = 3,
             quiet: bool = False) -> int:
    # Preflight. The old version swallowed a missing python-dotenv silently and then printed
    # "put it in .env", which is a lie when the real problem is that this interpreter cannot read .env at
    # all, or is not the venv. Three distinct causes, three distinct messages - an hour of someone's life
    # each time they are conflated.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    shell_key = os.getenv("OPENAI_API_KEY")
    have_dotenv = True
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)          # does NOT override a var already in the environment
    except ImportError:
        have_dotenv = False
    except Exception:  # noqa: BLE001
        pass

    if not os.getenv("OPENAI_API_KEY"):
        print("LIVE cannot start - no OPENAI_API_KEY. Exactly one of these is true:\n")
        print(f"  interpreter : {sys.executable}")
        print(f"  venv active : {'yes' if sys.prefix != getattr(sys, 'base_prefix', sys.prefix) else 'NO'}")
        print(f"  .env found  : {'yes' if os.path.isfile(env_path) else 'NO'}  ({env_path})")
        print(f"  python-dotenv installed : {'yes' if have_dotenv else 'NO'}\n")
        if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
            print("  -> The venv is not active. openai and python-dotenv live in it, not in system python:")
            print(_venv_hint(root))
        elif not have_dotenv:
            print("  -> python-dotenv is missing in this venv:  pip install python-dotenv")
        elif not os.path.isfile(env_path):
            print(f"  -> No .env at {env_path}. Create it with one line:  OPENAI_API_KEY=sk-...")
        else:
            print("  -> .env exists but has no OPENAI_API_KEY= line (or it is commented out).")
        print("\n  PowerShell, not cmd. To set it for this shell only:")
        print('       $env:OPENAI_API_KEY = "sk-..."')
        print("     ( `set VAR=value` is cmd syntax; in PowerShell it makes a shell variable that")
        print("       python never sees, which is why the key appeared to be set and was not. )")
        return 0

    # A shell variable silently beats .env, and that is what produced the 401 in
    # eval_result_live_gpt-4o.txt. Say so loudly rather than letting it fail 50 times.
    if shell_key:
        tail = shell_key[-6:] if len(shell_key) > 6 else "?"
        print(f"NOTE: OPENAI_API_KEY comes from the shell (...{tail}), not .env - .env cannot override it.")
        print("      If this run 401s, that is why:  Remove-Item Env:OPENAI_API_KEY   then re-run.\n")
    try:
        os.environ.setdefault("ADAPTIVE_INTERVIEWER", "true")
        from services.interview_engine import complete_interviewer_reply
        from services.interviewer_decision import detect_violations
        from openai import OpenAI
    except Exception as e:  # noqa: BLE001
        print(f"LIVE could not import the engine ({e}).")
        if "openai" in str(e) or "dotenv" in str(e):
            # Nine times out of ten this is the venv, not a broken install.
            print(f"\n  venv active : {'yes' if sys.prefix != getattr(sys, 'base_prefix', sys.prefix) else 'NO'}")
            print(f"  interpreter : {sys.executable}\n")
            print("  -> Activate the backend venv, then re-run:")
            print(_venv_hint(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        else:
            print("Run from the backend folder with the app's venv active.")
        return 0
    client = OpenAI()
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")
    answer_model = model or os.getenv("INTERVIEWER_ADAPTIVE_MODEL", "gpt-4o-mini")
    print(f"LIVE eval - {len(SCENARIOS)} scenarios | interviewer={answer_model} | judge={judge_model}"
          f" (best-of-{best_of}) | {workers} in parallel")
    print("=" * 68)

    # 50 scenarios x (1 interviewer call + up to 3 judge calls) is minutes of pure waiting one at a time,
    # and a slow eval is an eval you stop running. The scenarios are independent, so they go concurrently
    # and the records are re-ordered before printing: the report reads exactly as the serial one did.
    # Threads rather than asyncio, because the engine's client is synchronous.
    t0 = time.time()
    results: list = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_run_one, i, s, complete_interviewer_reply, detect_violations,
                          client, judge_model, best_of)
                for i, s in enumerate(SCENARIOS, 1)]
        for f in as_completed(futs):
            results.append(f.result())
    results.sort(key=lambda r: r["i"])
    elapsed = time.time() - t0

    passed, auth_error, splits, unjudged = 0, False, 0, 0
    fails_by_kind: dict = {"reasoning": [], "behaviour": []}
    n = len(SCENARIOS)
    for r in results:
        s, i = r["s"], r["i"]
        if r.get("error"):
            print(f"\n[{i:>2}/{n}] ERROR {s['id']}: {r['error']}")
            auth_error = auth_error or _is_auth_error(r["error"])
            continue
        v = r["verdict"]
        if r.get("unjudged"):
            unjudged += 1
            auth_error = auth_error or _is_auth_error(str(v.get("why") or ""))
        else:
            passed += 1 if r["ok"] else 0
            if not r["ok"]:
                fails_by_kind[failure_kind(s)].append(s["id"])
        if v.get("split"):
            splits += 1
        if not quiet:
            head = "ERROR" if r.get("unjudged") else ("PASS" if r["ok"] else "FAIL")
            print(f"\n[{i:>2}/{n}] {head}  {s['id']}  ({s['case_type']})")
            print(f'      user:  "{s["new"]}"')
            print(f'      reply: "{(r["reply"] or "").strip()[:200]}"')
            suffix = ""
            if r["det"]:
                suffix += f"   VIOLATIONS={r['det']}"
            if v.get("votes") not in (None, "n/a", "1/1"):
                suffix += f"   votes={v['votes']}"
            print(f"      judge: {v.get('why')}{suffix}")
            if verbose:
                print(f"      tag: {r.get('tag')}  ({r['ms']} ms)")
    if auth_error:
        print(_AUTH_HINT)

    print("\n" + "=" * 68)
    scored = n - unjudged - sum(1 for r in results if r.get("error"))
    print(f"LIVE: {passed}/{scored} pass ({passed / max(1, scored):.0%})  |  {elapsed:.0f}s wall"
          f"  |  interviewer={answer_model}")
    if scored < n:
        print(f"      {n - scored} scenario(s) produced NO verdict and are excluded from the score -")
        print("      a dead key or an exhausted quota is not a model regression. Fix it and re-run;")
        print("      a partial score is not comparable with a full one.")
    nr, nb = len(fails_by_kind["reasoning"]), len(fails_by_kind["behaviour"])
    if nr or nb:
        print("\nFailures by cause - the two halves have different fixes:")
        print(f"  reasoning ({nr}): had to NOTICE the candidate was wrong. A stronger model fixes these;")
        print("                    prompting does not add reasoning.")
        if nr:
            print(f"                    {', '.join(fails_by_kind['reasoning'])}")
        print(f"  behaviour ({nb}): had to ACT a certain way. Deterministic gates fix these;")
        print("                    a stronger model does not reliably obey.")
        if nb:
            print(f"                    {', '.join(fails_by_kind['behaviour'])}")
        print('  (heuristic from each scenario\'s "required" text - set "kind" on a scenario to override)')
    if splits:
        print(f"\n{splits} scenario(s) split the judge vote - genuinely borderline, worth reading by hand.")
    run_live.last_fails = fails_by_kind["reasoning"] + fails_by_kind["behaviour"]  # type: ignore[attr-defined]
    run_live.last_scored = scored                                                  # type: ignore[attr-defined]
    return passed


class _Tee:
    """Write everything printed to BOTH the console and a text file."""
    def __init__(self, path: str):
        self._f = open(path, "w", encoding="utf-8")
        self._console = sys.__stdout__

    def write(self, s):
        try:
            self._console.write(s)
        except UnicodeEncodeError:
            self._console.write(s.encode("ascii", "replace").decode("ascii"))
        self._f.write(s)

    def flush(self):
        self._console.flush()
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adaptive-interviewer behavioural eval (50 user-voiced scenarios).")
    ap.add_argument("--live", action="store_true", help="call the model + LLM judge (needs venv + OPENAI_API_KEY)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the full signal block (offline) / control tag (live)")
    ap.add_argument("--model", default="", metavar="NAME",
                    help="interviewer model for THIS run, e.g. --model gpt-4o. Sets INTERVIEWER_ADAPTIVE_MODEL "
                         "for the process only, so comparing two models never touches the live provider toggle "
                         "and never affects a real user. Results land in eval_result_live_<model>.txt.")
    ap.add_argument("--judge-model", default="", metavar="NAME", help="override JUDGE_MODEL (default gpt-4o)")
    ap.add_argument("--workers", type=int, default=5, metavar="N",
                    help="scenarios in flight at once (default 5). Lower it if you hit rate limits.")
    ap.add_argument("--best-of", type=int, default=3, metavar="N",
                    help="judge votes per scenario, majority wins (default 3; 1 = the old single call)")
    ap.add_argument("--runs", type=int, default=1, metavar="N",
                    help="repeat the full live eval N times and report mean/range/always-fail/intermittent. "
                         "Each run is independent. Use --runs 3 to separate real bugs from temperature noise.")
    args = ap.parse_args()

    # Set BEFORE the engine is imported inside run_live, so the override is in place when the engine
    # reads it. This is what makes an A/B one command instead of a provider toggle in production.
    if args.model:
        os.environ["INTERVIEWER_ADAPTIVE_MODEL"] = args.model
    if args.judge_model:
        os.environ["JUDGE_MODEL"] = args.judge_model

    # Full run is saved to a .txt right next to this script — open it or paste it back.
    _mode = ("live_" + os.getenv("INTERVIEWER_ADAPTIVE_MODEL", "gpt-4o-mini").replace("/", "-")) if args.live else "offline"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"eval_result_{_mode}.txt")
    tee = _Tee(out_path)
    sys.stdout = tee
    rc = 0
    try:
        if args.live:
            num_runs = max(1, args.runs)
            if num_runs == 1:
                run_live(args.verbose, model=args.model, workers=args.workers, best_of=args.best_of)
            else:
                # Multi-run mode: repeat the eval N times, collect per-run scores,
                # then report mean/range and always-fail vs intermittent.
                from collections import Counter
                scores, totals = [], []
                all_fails: list[list[str]] = []
                for run_idx in range(1, num_runs + 1):
                    print(f"\n{'#' * 68}")
                    print(f"# RUN {run_idx} of {num_runs}")
                    print(f"{'#' * 68}")
                    quiet = run_idx > 1  # only print scenario detail on first run
                    p = run_live(args.verbose, model=args.model, workers=args.workers,
                                 best_of=args.best_of, quiet=quiet)
                    scored = getattr(run_live, "last_scored", len(SCENARIOS))
                    fails = list(getattr(run_live, "last_fails", []))
                    scores.append(p)
                    totals.append(scored)
                    all_fails.append(fails)

                # Summary across runs
                pcts = [s / max(1, t) * 100 for s, t in zip(scores, totals)]
                mean_pct = sum(pcts) / len(pcts)
                lo, hi = min(pcts), max(pcts)
                spread = hi - lo

                # Classify failures
                fail_counts = Counter(f for run_fails in all_fails for f in run_fails)
                always = [f for f, c in fail_counts.items() if c == num_runs]
                intermittent = [(f, c) for f, c in fail_counts.items() if 0 < c < num_runs]

                print(f"\n{'=' * 68}")
                print(f"{num_runs} runs | interviewer={args.model or os.getenv('INTERVIEWER_ADAPTIVE_MODEL', 'gpt-4o-mini')}")
                print(f"  mean  {mean_pct:.1f}%")
                print(f"  range {lo:.0f}% - {hi:.0f}%   (spread {spread:.0f}%)")
                print(f"\n  Treat any difference smaller than {spread:.0f}% between two models as noise.")
                if always:
                    print(f"\n  failed in EVERY run ({len(always)}) - these are real; fix these first:")
                    print(f"      {', '.join(always)}")
                if intermittent:
                    print(f"\n  intermittent ({len(intermittent)}) - the model is inconsistent here, not simply wrong:")
                    for f, c in sorted(intermittent, key=lambda x: -x[1]):
                        print(f"      {f} ({c}/{num_runs})")
                if not always and not intermittent:
                    print(f"\n  No failures across any run. 100% consistent.")
        else:
            rc = 0 if run_offline(args.verbose) else 1
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print(f"\nFull output saved to: {out_path}")
    sys.exit(rc)
