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
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.session_signals import compute_signals, build_signal_block  # noqa: E402
from tools.scenario_variants import VARIANTS  # noqa: E402


def _pick_message(s: dict, vary: bool) -> str:
    """The candidate message to test. vary=True samples one of the scenario's paraphrase
    variants at random, so each run exercises DIFFERENT phrasings -- the score then reflects
    whether the router generalises, not a memorised pass over 50 fixed strings."""
    if not vary:
        return s["new"]
    return random.choice([s["new"]] + list(VARIANTS.get(s["id"], [])))

SCENARIOS = [
    # ---- STATE / INTENT ----
    {"id":"help_when_stuck_with_work","case_type":"profitability",
     "transcript":[{"role":"user","content":"so if 0.95S - 2 = 0.80(S-2) then S comes to like 2.67 cr i think"},
                   {"role":"assistant","content":"How did you get to 2.67? Walk me through the steps."},
                   {"role":"user","content":"arre i already solved it na, you're not even saying whats wrong. guide me or leave it"},
                   {"role":"assistant","content":"I hear you — but what would your very first step be?"}],
     "new":"help na i'm not getting this at all",
     "expect_signals":{"intent":"asking_for_help","repair_due":True,"frustration":"high","has_work":True},
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
     "expect_signals":{"intent":"wants_solution"},
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
     "expect_signals":{"intent":"scope_question","is_scope_question":True},
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
     "expect_signals":{"intent":"answering"},
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
     "expect_signals":{"intent":"answering"},
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
     "expect_signals":{"intent":"scope_question"},
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
     "expect_signals":{"intent":"wants_solution"},
     "required":"Nearly done + coached: acceptable to give the spine / more of the answer. Do NOT stonewall this late."},
    {"id":"explain_why_meta","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"tell me about the value chain."}],
     "new":"why are you asking about the value chain?",
     "expect_signals":{"intent":"scope_question"},
     "required":"One-line bridge ('it's the link from cost to the loss'), then hand back. Don't reveal the answer."},
    {"id":"product_ux_question","case_type":"guesstimate",
     "transcript":[{"role":"assistant","content":"go ahead."}],
     "new":"wait where do i see my results after this?",
     "expect_signals":{"intent":"scope_question"},
     "required":"Answer plainly in role (results page). Do NOT loop the closing line or treat it as a case question."},

    # ---- OWN THE FACTS ----
    {"id":"own_facts_growth_rate","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"go on."}],
     "new":"what's the market growth rate?",
     "expect_signals":{"intent":"scope_question","is_scope_question":True},
     "required":"Give a specific realistic number (e.g. ~8%), own it, hand back. Never 'not specified'."},
    {"id":"own_facts_competitors","case_type":"profitability",
     "transcript":[{"role":"assistant","content":"go on."}],
     "new":"how many competitors are there in this market?",
     "expect_signals":{"intent":"scope_question","is_scope_question":True},
     "required":"Invent a defensible number ('three main players, ~30% each'). Don't say 'isn't given'."},

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
     "expect_signals":{"intent":"answering"},
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
     "expect_signals":{"intent":"asking_for_help","has_work":True},
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
     "expect_signals":{"intent":"answering","has_work":True},
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


_JUDGE_RUBRIC = (
    "You grade ONE interviewer reply in a case-interview practice app. Judge whether the reply "
    "accomplishes the INTENT of the required behaviour below -- NOT whether it matches example "
    "wording. Apply this calibration (it overrides a naive reading):\n"
    "- Any DEFENSIBLE specific number satisfies an 'own the facts' requirement. If the requirement "
    "gives an example like 'three players, ~30% each', then 'five players, ~20% each' is EQUALLY "
    "fine -- do not fail it for using different but plausible numbers.\n"
    "- A brief, earned acknowledgement ('Good', 'Fair point', 'That's a solid recommendation') is "
    "natural and GOOD; do NOT fail it as praise. Only fail GUSHING or rubber-stamping of weak/wrong "
    "work (e.g. 'Amazing! Brilliant structure!' on a vague answer).\n"
    "- Giving a thinking or self-correcting candidate space ('Take your time.') SATISFIES a "
    "'do not interrupt / give space' requirement; do not fail it for 'not engaging'.\n"
    "- Judge the behaviour, not length or spelling. One short question is acceptable unless the "
    "requirement explicitly says to ask nothing.\n"
    'Return strict JSON {"pass": true|false, "why": "<one line>"}.'
)


def _judge_once(judge, model, s, msg, reply):
    jp = (f"{_JUDGE_RUBRIC}\n\nREQUIRED interviewer behaviour: {s['required']}\n\n"
          f'Candidate said: "{msg}"\nInterviewer replied: "{reply}"')
    jr = judge.chat.completions.create(
        model=model, temperature=0.4,
        messages=[{"role": "user", "content": jp}],
        response_format={"type": "json_object"})
    return json.loads(jr.choices[0].message.content or "{}")


def _judge(judge, model, s, msg, reply, n=3):
    """Best-of-n majority vote to de-noise a single flaky LLM judgement."""
    votes, why = [], ""
    for _ in range(max(1, n)):
        try:
            v = _judge_once(judge, model, s, msg, reply)
        except Exception as e:  # noqa: BLE001
            return {"pass": None, "why": f"judge error: {e}", "votes": "err"}
        votes.append(bool(v.get("pass")))
        why = v.get("why") or why
    p = sum(votes)
    return {"pass": p > len(votes) / 2, "why": why, "votes": f"{p}/{len(votes)}"}


def run_offline(verbose: bool = False, vary: bool = True) -> bool:
    print(f"OFFLINE signal eval — {len(SCENARIOS)} scenarios (deterministic, no API)"
          + ("  [varied phrasings]" if vary else ""))
    print("=" * 68)
    fails = []
    for i, s in enumerate(SCENARIOS, 1):
        msg = _pick_message(s, vary)
        sig = compute_signals(s["transcript"], msg, s.get("policy", "coached"))
        bad = []
        for k, v in s["expect_signals"].items():
            got = sig.get(k)
            if got != v:
                bad.append(f"      x {k}={got!r} (want {v!r})")
                fails.append((s["id"], k, got, v))
        head = "PASS" if not bad else "FAIL"
        print(f"\n[{i:>2}/{len(SCENARIOS)}] {head}  {s['id']}  ({s['case_type']})")
        print(f'      user: "{msg}"')
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


def run_live(verbose: bool = False, vary: bool = True, judge_n: int = 3) -> int:
    # Load .env the same way the app does, so the eval sees the key prod would use.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    if not os.getenv("OPENAI_API_KEY"):
        print("LIVE needs OPENAI_API_KEY - put it in .env (OPENAI_API_KEY=sk-...) or the shell,")
        print("and run with the backend venv active. Then re-run --live.")
        return 0
    try:
        os.environ.setdefault("ADAPTIVE_INTERVIEWER", "true")
        from services.interview_engine import complete_interviewer_reply
        from services.interviewer_decision import detect_violations
        from openai import OpenAI
    except Exception as e:  # noqa: BLE001
        print(f"LIVE could not import the engine ({e}).")
        print("Run from the backend folder with the app's venv active (the one that has openai installed).")
        return 0
    judge = OpenAI()
    judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")
    print(f"LIVE eval — {len(SCENARIOS)} scenarios (adaptive model + judge={judge_model} x{judge_n})"
          + ("  [varied phrasings]" if vary else ""))
    print("=" * 68)
    passed = 0
    det_clean = 0        # replies with no deterministic violation (gush / extra question)
    judge_only_fail = 0  # deterministically clean but the judge still failed it
    for i, s in enumerate(SCENARIOS, 1):
        ctl: dict = {}
        msg = _pick_message(s, vary)
        try:
            reply = complete_interviewer_reply(
                case_content=f"A {s['case_type']} case for practice on MECE.",
                case_type=s["case_type"], transcript=s["transcript"],
                new_user_message=msg, teaching_policy=s.get("policy", "coached"),
                control_out=ctl,
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            print(f"\n[{i:>2}/{len(SCENARIOS)}] ERROR {s['id']}: {msg}")
            if any(t in msg for t in ("401", "invalid_api_key", "Incorrect API key")):
                print("\n" + "!" * 68)
                print("AUTH ERROR: OpenAI is rejecting the key. This is NOT the eval.")
                print("Most likely an OPENAI_API_KEY env var is OVERRIDING your .env")
                print("(python-dotenv won't replace an env var that is already set). In PowerShell:")
                print("  echo $env:OPENAI_API_KEY        # if this differs from your .env key, that's it")
                print("  Remove-Item Env:OPENAI_API_KEY  # clear the session override, then re-run")
                print("If it still fails, the key in .env is itself revoked - mint a new one.")
                print("!" * 68)
                return passed
            continue
        det = detect_violations(reply)
        verdict = _judge(judge, judge_model, s, msg, reply, judge_n)
        ok = bool(verdict.get("pass")) and not det
        passed += 1 if ok else 0
        if not det:
            det_clean += 1
            if verdict.get("pass") is False:
                judge_only_fail += 1
        head = "PASS" if ok else "FAIL"
        print(f"\n[{i:>2}/{len(SCENARIOS)}] {head}  {s['id']}  ({s['case_type']})")
        print(f'      user:  "{msg}"')
        print(f'      reply: "{reply.strip()[:200]}"')
        print(f"      judge: {verdict.get('why')}  [{verdict.get('votes')}]"
              + (f"   VIOLATIONS={det}" if det else ""))
        if verbose:
            print(f"      tag: {ctl.get('tag')}")
    n = len(SCENARIOS)
    print("\n" + "=" * 68)
    print(f"LIVE: {passed}/{n} pass ({passed / max(1, n):.0%}).")
    print(f"  deterministically clean (no gush / no stray question): {det_clean}/{n} ({det_clean / max(1, n):.0%})")
    print(f"  clean but judge failed (judge disagreements to review): {judge_only_fail}")
    print("  -> the gap between 'clean' and 'pass' is the JUDGE, not the interviewer. "
          "Review those, and see --judge-n / the rubric before raising your bar.")
    return passed


class _Tee:
    """Write everything printed to BOTH the console and a text file."""
    def __init__(self, path: str):
        self._f = open(path, "w", encoding="utf-8")
        self._console = sys.__stdout__

    def write(self, s):
        self._console.write(s)
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
    ap.add_argument("--fixed", action="store_true",
                    help="use the original fixed phrasings (default: a random paraphrase per scenario "
                         "each run, so the score reflects generalisation not memorised strings)")
    ap.add_argument("--seed", type=int, default=None, help="seed the paraphrase sampling for a reproducible run")
    ap.add_argument("--judge-n", type=int, default=3,
                    help="LLM-judge votes per scenario, majority wins (default 3, de-noises the judge; 1 = fast)")
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    vary = not args.fixed

    # Full run is saved to a .txt right next to this script — open it or paste it back.
    _mode = ("live_" + os.getenv("INTERVIEWER_ADAPTIVE_MODEL", "gpt-4o-mini").replace("/", "-")) if args.live else "offline"
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"eval_result_{_mode}.txt")
    tee = _Tee(out_path)
    sys.stdout = tee
    rc = 0
    try:
        if args.live:
            run_live(args.verbose, vary, args.judge_n)   # behavioural eval: varies phrasing each run
        else:
            rc = 0 if run_offline(args.verbose, False) else 1   # signal unit-test: fixed inputs
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print(f"\nFull output saved to: {out_path}")
    sys.exit(rc)
