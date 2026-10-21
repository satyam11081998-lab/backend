"""V12.1 follow-up: the admin arena's adversarial check (Oct 21 2026, 50 scenarios,
gpt-4o judge) on candidate @ 653f2a0 -- New 34/50, Current 30/50, 5 regressions.

Every scenario New still failed is pinned here by what the build must now DO with
it: which response function the turn gets, what the control packet tells the
model (computed in code), and what may ship. The judge itself needs the network;
these tests do not.

Runs with pytest; no network, no keys.
"""
from __future__ import annotations

import os
import random
import re
import time
import types

import pytest

for _k, _v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
               "SUPABASE_SERVICE_ROLE_KEY": "test"}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from services import interview_engine as ie  # noqa: E402
from services import numbers_check as nc  # noqa: E402
from services.logic_check import check_logic  # noqa: E402
from services.interviewer_decision import (  # noqa: E402
    StreamReplyGuard, StreamShapeGuard, enforce_mode, detect_violations, shape_lists, cap_sentences,
    get_fast_lane_event, _FAST_PHRASES,
)
from tools.eval_interviewer_behavior import SCENARIOS  # noqa: E402
from routes.arena import scenario_case_content  # noqa: E402

NO_LLM = {"assessor": lambda *a, **k: {}, "numbers_auditor": lambda *a, **k: {"issues": [], "verified": []}}
BY_ID = {s["id"]: s for s in SCENARIOS}


def U(c):
    return {"role": "user", "content": c}


def A(c):
    return {"role": "assistant", "content": c}


def plan_scenario(sid, overrides=None, prior=None):
    s = BY_ID[sid]
    return ie._plan_adaptive_turn(scenario_case_content(s["case_type"]), s["case_type"], s["transcript"],
                                  s["new"], False, s.get("policy", "coached"), prior or {}, "text", False,
                                  prompt_overrides=overrides or NO_LLM)


def pk(p):
    return p["packet"]["interviewer_control"]


# --------------------------------------------------------------- 1. routes

@pytest.mark.parametrize("sid,fn", [
    ("help_when_stuck_with_work", "MICRO_HINT"),        # was TEACH_STEP (generic profit tree, too long)
    ("double_count_error", "CORRECT_AND_CONTINUE"),
    ("factor_1000_unit_error", "CORRECT_AND_CONTINUE"),
    ("confidently_wrong_penetration", "SANITY_CHECK"),
    ("contradicts_established_fact", "CORRECT_AND_CONTINUE"),  # was DATA_REVEAL / "interesting insight"
    ("explain_why_meta", "EXPLAIN_WHY"),                 # was REPAIR_AND_RESET in the arena
    ("product_ux_question", "PRODUCT_HELP"),             # was MICRO_HINT in the arena
    ("too_many_questions_back_off", "HAND_BACK"),        # was REPAIR_AND_RESET (more guidance)
    ("productive_thinking_pause", "HAND_BACK"),
    ("sanity_check_absent", "SANITY_CHECK"),             # was FINAL_REVIEW ("0.5 per person, plausible")
    ("asks_same_thing_twice", "TEACH_STEP"),
    ("semantic_repetition_guard", "TEACH_STEP"),
    ("wants_solution_outright", "DELIVER_SOLUTION"),
    ("own_facts_competitors", "ANSWER_DIRECT"),
    ("good_answer_should_advance", "REFLECT_PROGRESS"),
    ("correct_but_uncertain_do_not_derail", "FOLLOW_UP_ANSWER"),
])
def test_each_failed_scenario_gets_the_right_function(sid, fn):
    assert plan_scenario(sid)["function"] == fn


@pytest.mark.parametrize("sid,fn", [
    ("explain_why_meta", "EXPLAIN_WHY"), ("product_ux_question", "PRODUCT_HELP"),
    ("contradicts_established_fact", "CORRECT_AND_CONTINUE"), ("too_many_questions_back_off", "HAND_BACK"),
])
@pytest.mark.parametrize("assessed", [
    {"intent": "asking_for_help", "progress": "stuck"}, {"intent": "answering", "progress": "progressing"},
    {"intent": "meta", "materiality": "none"},
])
def test_the_context_assessor_cannot_reroute_them(sid, fn, assessed):
    # In the arena the assessor (an extra model call) turned these into a repair, a hint
    # and an acknowledgement. They are decided in code now, before the assessor.
    p = plan_scenario(sid, overrides={"assessor": lambda *a, **k: dict(assessed),
                                      "numbers_auditor": NO_LLM["numbers_auditor"]})
    assert p["function"] == fn


def test_the_fast_lines_fit_the_turn():
    think = {o["text"] for o in _FAST_PHRASES["THINK"]}
    space = {o["text"] for o in _FAST_PHRASES["SPACE"]}
    p = plan_scenario("productive_thinking_pause")
    assert get_fast_lane_event(p["function"], p["signals"])["data"]["text"] in think
    p = plan_scenario("too_many_questions_back_off")
    line = get_fast_lane_event(p["function"], p["signals"])["data"]["text"]
    assert line in space and "?" not in line
    p = plan_scenario("thinking_out_loud_self_correct")
    assert get_fast_lane_event(p["function"], p["signals"])["data"]["text"] in think


def test_after_asking_for_space_the_next_turn_asks_nothing():
    hist = [A("and then? and what about frequency?"), U("you're asking too many questions man, just let me solve"),
            A("Fair enough. It's all yours; I'll hold my questions.")]
    p = ie._plan_adaptive_turn("A guesstimate case.", "guesstimate", hist, "i think 40% of them order online",
                               False, "coached", {}, "text", False, prompt_overrides=NO_LLM)
    if p["decision"]["render"] == "model":
        assert p["allow_questions"] is False


@pytest.mark.parametrize("msg", [
    "just let me know the population", "where do i see the growth coming from?", "why are we losing money",
    "let me try 30% penetration",
    # adversarial review (Oct 21): case questions that looked like app / meta questions
    "where can i see the results of the survey?", "how should i get the solution, top down or bottom up",
    "how do i find the answer to the market size? should i use population",
    "how can i check my answer is right? maybe per capita", "can i see the feedback report from customers",
    "is there a time limit for the offer the competitor is running", "how do i finish this calculation",
    "why is it important for the client to grow in north india",
    "what's the use of this product for rural customers",
    "why does this matter? because cost is the driver, so i'll check raw material",
    "let me solve", "ok let me solve it", "just let me think if population works",
    "too many questions in the survey would reduce response rate",
])
def test_lookalikes_keep_their_routes(msg):
    p = ie._plan_adaptive_turn("A guesstimate case.", "guesstimate", [A("Walk me through the value chain.")], msg,
                               False, "coached", {}, "text", False, prompt_overrides=NO_LLM)
    assert p["function"] not in ("PRODUCT_HELP", "EXPLAIN_WHY") and p["reason"] != "candidate_wants_space"


@pytest.mark.parametrize("last,msg,fn", [
    ("tell me about the value chain.", "why is the value chain relevant here?", "EXPLAIN_WHY"),
    ("tell me about the value chain.", "why does the value chain matter here", "EXPLAIN_WHY"),
    ("tell me about the value chain.", "why value chain?", "EXPLAIN_WHY"),
    ("Go on.", "why are you asking?", None),   # nothing of yours to explain after "Go on."
    ("go ahead.", "will i get feedback at the end?", "PRODUCT_HELP"),
    ("go ahead.", "how do i end this case", "PRODUCT_HELP"),
])
def test_paraphrases_reach_the_new_functions(last, msg, fn):
    p = ie._plan_adaptive_turn("A profitability case.", "profitability", [A(last)], msg, False, "coached", {},
                               "text", False, prompt_overrides=NO_LLM)
    if fn:
        assert p["function"] == fn
    else:
        assert p["function"] != "EXPLAIN_WHY"


@pytest.mark.parametrize("msg,pool", [
    ("can you stop asking questions and let me think", "SPACE"),
    ("stop asking me questions", "SPACE"),
    ("give me a minute", "THINK"),
    ("ek minute soch raha hu", "THINK"),
])
def test_space_and_thinking_paraphrases(msg, pool):
    p = ie._plan_adaptive_turn("A guesstimate case.", "guesstimate", [A("and the segments?")], msg, False,
                               "coached", {}, "text", False, prompt_overrides=NO_LLM)
    assert p["decision"]["render"] == "fast"
    assert get_fast_lane_event(p["function"], p["signals"])["data"]["text"] in {
        o["text"] for o in _FAST_PHRASES[pool]}


@pytest.mark.parametrize("msg,want", [("help karo", "help"), ("bhai approach bata do", "solution"),
                                      ("pata nahi", "stuck")])
def test_hinglish_help_is_heard(msg, want):
    from services.session_signals import compute_signals
    s = compute_signals([A("how would you size it?")], msg, "coached", channel="text")
    assert {"help": s["help_requested"], "solution": s["solution_requested"], "stuck": s["stuck_blank"]}[want]


def test_a_hedged_100_percent_is_not_a_confident_claim():
    from services.session_signals import compute_signals
    s = compute_signals([A("and the number?")], "not 100% sure but 20 lakh", "coached", channel="text")
    assert not s["confident_unsupported_claim"]


# --------------------------------------------------------------- 2. what the packet says

def test_help_after_working_is_a_hint_on_that_working():
    c = pk(plan_scenario("help_when_stuck_with_work"))
    st = c["candidate_state"]
    assert "0.95S - 2 = 0.80(S-2)" in st["latest_work"]
    assert st["latest_work_check"].startswith("verified:") and "2.67" in st["latest_work_check"]
    assert c["interviewer_decision"]["may_confirm_correctness"] is True  # the algebra really holds
    assert c["interviewer_decision"]["question_allowed"] is False


def test_a_second_help_ask_after_a_failed_hint_on_working_is_taught():
    s = BY_ID["help_when_stuck_with_work"]
    p = ie._plan_adaptive_turn("A profitability case.", "profitability", s["transcript"], s["new"], False,
                               "coached", {"recent_functions": ["MICRO_HINT"]}, "text", False,
                               prompt_overrides=NO_LLM)
    assert p["function"] == "TEACH_STEP"
    assert "0.95S" in pk(p)["candidate_state"]["latest_work"]  # ...built on their working


def test_a_recap_with_figures_is_not_working():
    # real session [9]: "I don't know actually." after a recap full of figures -> still taught
    hist = [U("Right, so a PE firm looking at 60 car washes, 14 times EBITDA, $90 million revenue with 38% "
              "margin, asking price of $479 million."), A("That's correct."), U("the first question would be"),
            A("Mm-hm.")]
    p = ie._plan_adaptive_turn("A PE firm considers buying car washes.", "m&a", hist, "I don't know actually.",
                               False, "coached", {}, "text", False, prompt_overrides=NO_LLM)
    assert p["function"] == "TEACH_STEP"


@pytest.mark.parametrize("sid,kind,needle", [
    ("factor_1000_unit_error", "units", "5 crore ml is 50,000 litres"),
    ("double_count_error", "double_count", "families, singles and office-goers"),
    ("contradicts_established_fact", "contradiction", "volumes are flat"),
    ("confidently_wrong_penetration", "extreme_assumption", "every single person buys it every day"),
])
def test_the_slip_is_named_exactly_in_code(sid, kind, needle):
    st = pk(plan_scenario(sid))["candidate_state"]
    assert st["issue_kind"] == kind and needle in st["detected_issue"]


def test_corrections_and_sanity_checks_may_ask_one_targeted_question():
    for sid in ("factor_1000_unit_error", "double_count_error", "contradicts_established_fact",
                "confidently_wrong_penetration", "sanity_check_absent"):
        assert plan_scenario(sid)["allow_questions"] is True, sid


def test_an_implausible_final_gets_one_sanity_check_with_the_numbers():
    c = pk(plan_scenario("sanity_check_absent"))
    ctx = c["case_context"]
    assert ctx["final_answer"] == "90000"
    assert "50,000" in ctx["implied_check"] and "0.45 to 1.8 per person" in ctx["implied_check"]
    assert ctx["implied_verdict"].startswith("looks high")


def test_the_final_is_reviewed_after_its_sanity_check():
    s = BY_ID["sanity_check_absent"]
    p = ie._plan_adaptive_turn("A guesstimate case.", "guesstimate", s["transcript"], s["new"], False,
                               "coached", {"recent_functions": ["SANITY_CHECK"]}, "text", False,
                               prompt_overrides=NO_LLM)
    assert p["function"] == "FINAL_REVIEW"
    assert pk(p)["case_context"]["implied_verdict"].startswith("looks high")  # still said honestly


@pytest.mark.parametrize("hist,msg", [
    ([U("delhi population 2 cr")], "my final answer is 65 lakh rides per day"),
    ([U("india population 140 cr")], "final answer: about 50 crore transactions per day"),
    ([U("chennai population 1 cr")], "final answer 30 lakh tickets per day"),
    ([U("mumbai population 2 cr")], "final answer 5.5 crore passengers a year"),
    ([U("goa population 15 lakh")], "final answer 80 lakh people visit Goa each year"),
    ([U("mumbai population 2 cr")], "final answer 2.2 crore phones in use"),
    ([U("a bus is 12 m long, 2.5 m wide")], "final answer 1.75 lakh balls"),
    ([U("one store, 500 customers a day")], "final answer Rs 6.3 crore a year"),
])
def test_plausible_or_unbased_finals_close_without_a_looks_high(hist, msg):
    p = ie._plan_adaptive_turn("A guesstimate case.", "guesstimate", hist, msg, False, "coached", {}, "text",
                               False, prompt_overrides=NO_LLM)
    assert p["function"] == "FINAL_REVIEW"
    assert "implied_verdict" not in pk(p)["case_context"]


def test_exam_policy_never_carries_the_corrected_figure():
    s = BY_ID["factor_1000_unit_error"]
    p = ie._plan_adaptive_turn("A guesstimate case.", "guesstimate", s["transcript"], s["new"], False, "exam", {},
                               "text", False, prompt_overrides=NO_LLM)
    issue = pk(p)["candidate_state"]["detected_issue"]
    assert "50,000" not in issue and "5 crore ml" in issue
    hist = [U("so if 0.95S - 2 = 0.80(S-2) then S = 3 cr"), A("How did you get to 3?")]
    p = ie._plan_adaptive_turn("A profitability case.", "profitability", hist, "help, i'm not getting this", False,
                               "exam", {}, "text", False, prompt_overrides=NO_LLM)
    check = pk(p)["candidate_state"]["latest_work_check"]
    assert check.startswith("issue:") and "2.67" not in check


def test_the_first_i_dont_know_is_not_an_already_tried_pointer():
    p = ie._plan_adaptive_turn("Estimate the number of cars in Delhi.", "guesstimate",
                               [A("Estimate the number of cars in Delhi. Where would you like to start?")],
                               "i dont know", False, "coached", {}, "text", False, prompt_overrides=NO_LLM)
    assert p["function"] == "TEACH_STEP" and "already_tried" not in pk(p)["case_context"]


@pytest.mark.parametrize("recap", [
    "So to recap: revenue = $90 million, margin = 38%, asking price = $479 million",
    "Given: population 1.2 cr, households 30 lakh, 2024-25 base year",
])
def test_a_recap_with_equals_signs_is_not_working(recap):
    p = ie._plan_adaptive_turn("A case.", "m&a", [U(recap), A("Right.")], "I don't know actually.", False,
                               "coached", {}, "text", False, prompt_overrides=NO_LLM)
    assert p["function"] == "TEACH_STEP"


def test_after_a_space_request_every_function_is_told_the_same_thing():
    hist = [A("and then? and what about frequency?"), U("you're asking too many questions man, just let me solve"),
            A("Fair enough. Over to you.")]
    p = ie._plan_adaptive_turn("A guesstimate case.", "guesstimate", hist, "so 5 crore ml which is 5 crore litres",
                               False, "coached", {}, "text", False, prompt_overrides=NO_LLM)
    assert p["function"] == "CORRECT_AND_CONTINUE" and p["allow_questions"] is False
    system = "\n".join(m["content"] for m in p["messages"] if m["role"] == "system")
    assert "no question at all this turn" in system


def test_a_plausible_final_with_a_base_closes_directly():
    hist = [U("Mumbai population 2 cr, 4 per household so 50 lakh households"), A("Go on."),
            U("10% buy a laptop each year so 5 lakh")]
    p = ie._plan_adaptive_turn("Estimate laptops sold in Mumbai.", "guesstimate", hist,
                               "so my final answer is 5 lakh laptops a year", False, "coached", {}, "text", False,
                               prompt_overrides=NO_LLM)
    assert p["function"] == "FINAL_REVIEW"
    assert "implied_verdict" not in pk(p)["case_context"]


def test_teaching_switches_method_when_the_last_pointer_did_not_land():
    ctx = pk(plan_scenario("asks_same_thing_twice"))["case_context"]
    assert ctx["already_tried"] == "start by anchoring the population"
    # a first teach after open questions keeps the top-down default
    assert "already_tried" not in pk(plan_scenario("semantic_repetition_guard"))["case_context"]


def test_why_and_product_turns_carry_what_they_need():
    assert pk(plan_scenario("explain_why_meta"))["case_context"]["your_question_they_ask_about"] == \
        "tell me about the value chain."
    facts = " ".join(pk(plan_scenario("product_ux_question"))["case_context"]["product_facts"])
    assert "top right" in facts and "results page" in facts


def test_new_instructions_say_the_new_things():
    from services.interviewer_mode import get_function_instruction, ALLOW_QUESTIONS
    assert "ONE short, targeted question" in get_function_instruction("CORRECT_AND_CONTINUE", "CORRECT_MATERIAL",
                                                                      "coached", "")
    assert "Never a bare 'Does that number look right to you?'" in get_function_instruction(
        "SANITY_CHECK", "RETHINK_CUE", "coached", "")
    assert "already_tried" in get_function_instruction("TEACH_STEP", "TEACH", "coached", "")
    assert "latest_work" in get_function_instruction("MICRO_HINT", "HINT", "coached", "")
    assert "do NOT ask them to justify" in get_function_instruction("FOLLOW_UP_ANSWER", "FOLLOW_UP", "coached", "")
    assert "share they hold" in get_function_instruction("ANSWER_DIRECT", "ANSWER_DIRECT", "coached", "")
    assert "no evaluative words" in get_function_instruction("REFLECT_PROGRESS", "ACKNOWLEDGE", "coached", "")
    assert ALLOW_QUESTIONS["PRODUCT_HELP"] is False and ALLOW_QUESTIONS["EXPLAIN_WHY"] is False


# --------------------------------------------------------------- 3. the checks themselves

@pytest.mark.parametrize("msg,issue", [
    ("so total is 5 crore ml which is like 5 crore litres for the whole city, done", "50,000 litres"),
    ("2 hours is 100 minutes", "120 minutes"),
    ("300 g is 3 kg", "0.3 kg"),
    ("10 crore litres i.e. 1000 crore ml", "10,000 crore ml"),
    ("if 2x + 4 = 10 then x = 4", "x ≈ 3"),
    ("0.95S - 2 = 0.80(S-2) so S = 3", "S ≈ 2.67"),
    ("4N = 1.2 crore, N = 3 lakh households", "N ≈ 30 lakh"),
    ("5P = 25,000 so P = 50,000", "P ≈ 5,000"),
    ("0.8C + 20 = 100 so C = 90", "C ≈ 100"),   # the equation's verdict, not a chain fragment
])
def test_unit_and_equation_slips_are_caught(msg, issue):
    r = nc.check_turn([], msg, "", {})
    assert r["math_issue"] and issue in r["math_issue"]["message"], r


@pytest.mark.parametrize("msg", [
    "so total is 5 crore ml which is 50,000 litres", "2 hours is 120 minutes", "1.5 kg is 1500 g",
    "500 ml a day is 15 litres a month", "it takes 30 min or 1 hr", "2 lakh people x 30 minutes = 1 lakh hours",
    "20 lakh people x 500 ml = 10 lakh litres a day", "each bottle is 500 ml so 2 crore bottles is 1 crore litres",
    "100 ml per cup, 2 cups a day is 200 ml a day which is 0.2 litres a day",
    "so if 0.95S - 2 = 0.80(S-2) then S comes to like 2.67 cr i think", "if 2x + 4 = 10 then x = 3",
    "P - 0.2P = 80 so P = 100", "4 x 5 = 20", "revenue R = 500 x 2 = 1000", "3m = 3 million",
    "x = 40% of 2 cr = 80 lakh", "5k + 5k = 10k",
    # adversarial review (Oct 21): scales, comma groups and unit letters
    "4N = 1.2 crore, N = 30 lakh households", "2x = 1 cr so x = 50 lakh", "1000p = 5 lakh so p = 500",
    "3x = 1.5 bn so x = 500 mn", "5P = 25,000 so P = 5,000", "2x = 1,00,000 so x = 50,000", "3x = 1,500 so x = 500",
    "1 M = 10 L, so 5 M = 50 L", "1 B = 100 cr, so 2 B = 200 cr", "1 m = 100 cm so 2 m = 200 cm",
    "1 h = 60 min so 2 h = 120 min", "1 T = 1000 kg so 3 T = 3000 kg", "100 k = 1 L so 500 k = 5 L",
    "0.8C + 20 = 100 so C = 100", "S(1-0.2) = 80 then S = 100",
])
def test_correct_conversions_and_algebra_stay_quiet(msg):
    assert nc.check_turn([], msg, "", {})["math_issue"] is None


_VOL = [U("and you told me volumes have stayed flat this whole period"), A("correct, volumes are flat. go on.")]


@pytest.mark.parametrize("hist,msg", [
    (_VOL, "so since volumes are flat, the revenue drop must be price"),
    (_VOL, "it can't be fewer units, so it must be price"),
    (_VOL, "we should push volumes up with a promotion"),
    (_VOL, "are we selling fewer units in some regions?"),
    (_VOL, "so we need more customers to fix this"),
    ([A("Assume prices are unchanged.")], "we should cut prices by 10%"),
    ([A("Costs rose 10% while prices stayed flat.")], "our prices are lower than competitors"),
    ([], "so the revenue drop must be because we're selling fewer units"),   # nothing established
    ([], "i'll take the whole city population and then split into families and singles"),
    ([U("families 3 times, singles 5 times")], "instead of adding the whole population i'll just add the segments"),
    ([U("families 3 times a week, singles 5 times")], "now i'll add the total population of singles and families"),
    ([], "not everyone buys it daily, maybe 30%"), ([], "everyone in my family drinks tea daily"),
    ([], "almost everyone uses a toothbrush daily"), ([], "everyone uses salt every day so 100% of households"),
    ([], "penetration is definitely 80%, almost everyone uses premium cars"),
    # adversarial review (Oct 21): correct reasoning the first version flagged
    (_VOL, "so selling fewer units is not the reason"),
    (_VOL, "my hypothesis was lower volumes, but you said flat, so its price"),
    (_VOL, "in the north region we are selling fewer units but the south made up for it"),
    (_VOL, "competitors are selling fewer units but we are not"),
    (_VOL, "last year volumes went up 10%, this year flat"),
    (_VOL, "our goal is higher volumes next year"),
    (_VOL, "higher volumes would help us cover fixed costs"),
    ([A("Raw material costs have risen 15%.")], "packaging costs dropped 2%"),
    ([A("Costs rose 10%.")], "competitors have lower costs"),
    ([A("Costs rose 10%.")], "lower costs are the main lever"),
    ([A("Prices have gone up 10%.")], "competitor prices are lower"),
    ([A("Industry volumes are down 8%. Ours are flat.")], "so our volumes grew relative to the market"),
    ([U("families 3 times a week, singles 5 times, offices 2")],
     "I will add up the segments and cross check against the entire city population"),
    ([U("families 3 times a week, singles 5 times, offices 2")],
     "adding them up gives 30 lakh, which is 25% of the total population, seems fine"),
    ([U("families 3 times a week, singles 5 times, offices 2")], "then add the tourists on top of the whole city population"),
    ([], "100% of restaurants purchase vegetables daily"), ([], "i'll take 100% of chai stalls ordering milk every day"),
    ([], "100% of metro riders buy a ticket per day"), ([], "100% of the people buy groceries weekly, not daily"),
    ([], "100% of households buy milk daily"),
])
def test_reasoning_lookalikes_stay_quiet(hist, msg):
    assert check_logic(hist, msg, "A case for practice.") == {}


@pytest.mark.parametrize("hist,msg,kind", [
    ([A("Assume prices are unchanged.")], "so the drop is because prices fell", "contradiction"),
    (_VOL, "the revenue decline is due to fewer units sold", "contradiction"),
    (_VOL, "so volumes must have dropped", "contradiction"),
    ([], "100% of people buy it every day", "extreme_assumption"),
])
def test_reasoning_slips_are_caught(hist, msg, kind):
    assert check_logic(hist, msg, "A case.")["logic_issue"]["kind"] == kind


def test_the_brief_counts_as_an_established_fact():
    r = check_logic([A("go on.")], "so the revenue fell because we are selling fewer units",
                    "Revenue is down 12% this year even though volumes have stayed flat.")
    assert r["logic_issue"]["kind"] == "contradiction"


@pytest.mark.parametrize("payload", ["a" * 20000, "S=" * 5000, "(S" * 3000 + ")" * 3000,
                                     "volumes are flat " * 1000, "add the whole population plus " * 800,
                                     "2 hours is 100 minutes " * 800, "1x+" * 5000 + "=1"])
def test_the_new_checks_are_fast_and_never_raise(payload):
    t0 = time.time()
    nc.check_turn([U(payload)], payload, "x")
    check_logic([A(payload), U(payload)], payload, payload)
    assert time.time() - t0 < 2.0


# --------------------------------------------------------------- 4. what ships: shape and length

REPLY_A = ("Sure, let's break it down. We'll use a top-down approach to analyze growth opportunities.\n\nFirst, we "
           "can look at:\n\n1. Total market size\n2. Market segments\n3. Customer needs\n4. Current company "
           "performance\n\nFrom there, we can identify growth drivers and potential strategies.")
REPLY_B = ("Let's approach it step-by-step. We can start by estimating the total population in the area, then the "
           "share who are relevant. \n\nFor example: \nPopulation of the area \n→ share who are potential "
           "consumers \n→ frequency of use \n= total consumption.\n\nLet's estimate the population first. For "
           "this case, let's assume an urban region of about 1 crore. What percentage of that might be "
           "relevant for our product?")
REPLY_C = ("Let's break it down together. In a profitability case, we can start with revenue and costs. Profit is "
           "revenue minus costs. \n\nNow, let's structure this step by step: \n1. Identify total revenue \n2. "
           "Identify total costs \n3. Calculate profit \n\nFor our current context, let's focus first on total "
           "revenue and total costs. Could you share what you have for either of those?")


def _stream_text(text, seed):
    rnd = random.Random(seed)
    parts, i = [], 0
    while i < len(text):
        n = rnd.randint(1, 7)
        parts.append(text[i:i + n])
        i += n
    rg, sg, out = StreamReplyGuard(), StreamShapeGuard(4), []
    for p in parts:
        for o in rg.feed(p):
            out += list(sg.feed(o))
    t = rg.flush()
    if t:
        out += list(sg.feed(t))
    out.append(sg.flush())
    return re.sub(r"[ \t]+\n", "\n", "".join(out)).strip()


@pytest.mark.parametrize("text", [REPLY_A, REPLY_B, REPLY_C])
def test_the_three_too_long_replies_now_ship_within_five_sentences(text):
    streamed = {_stream_text(text, s) for s in range(25)}
    assert len(streamed) == 1, "the stream guard must not depend on chunk boundaries"
    for out in list(streamed) + [enforce_mode(text, "TEACH", True)]:
        assert "too_long" not in detect_violations(out), out
        assert not re.search(r"(?m)^\s*\d\.\s", out), "no numbered list survives"
    if "?" in text:
        assert list(streamed)[0].rstrip().endswith("?"), "the final question still goes out"


@pytest.mark.parametrize("text,must_have", [
    # adversarial review (Oct 21): the cap cut at "Rs." and dropped the figure
    ("Rent is fixed. Power is fixed. Staff cost is fixed too. The ticket is Rs. 300 on average. Then multiply. "
     "What's the daily revenue?", "What's the daily revenue?"),
    ("There are 1 crore people. Half are adults. Of these 30% work. That gives 15 lakh. Would you assume "
     "Rs. 300 per month for each of them?", "Would you assume Rs 300 per month for each of them?"),
    ("Start with households, e.g. 4 people each. Then the share who buy. Then how often. Then price. "
     "What share would you take?", "What share would you take?"),
    ("Which segment would you start with?\n1. Students\n2. Working professionals\n3. Homemakers", "→ Homemakers"),
    ("Which segment first?\n- Students\n- Professionals", "· Professionals"),
])
def test_abbreviations_and_lists_survive_shaping(text, must_have):
    outs = {_stream_text(text, s) for s in range(30)}
    assert len(outs) == 1, outs
    out = outs.pop()
    assert must_have in out, out
    assert "too_long" not in detect_violations(out)
    assert "1." not in out.replace("Rs. ", "")
    e = enforce_mode(text, "TEACH", True)
    assert "too_long" not in detect_violations(e)


def test_a_long_sentence_is_never_cut_mid_word():
    long_q = ("Now think about how many of these " + "households in the larger metro region " * 8
              + "would actually buy it each week?")
    text = "Let's do it top-down. Population first. Then the share who buy. Then frequency. " + long_q
    outs = {_stream_text(text, s) for s in range(30)}
    assert len(outs) == 1
    out = outs.pop()
    assert out.endswith("each week?") and "households in the larger" in out


def test_sanitize_keeps_eg_urls_and_ellipses():
    from services.interviewer_decision import sanitize_reply
    t = "Take a base, e.g. 4 people per home. See app.mece.in for the brief... then go on."
    assert sanitize_reply(t) == t


def test_lists_become_steps_and_dots():
    assert shape_lists("Look at:\n1. Market size\n2. Segments\n3. Needs") == \
        "Look at:\nMarket size\n→ Segments\n→ Needs"
    assert shape_lists("Buckets:\n- Students\n- Professionals") == "Buckets:\n· Students\n· Professionals"
    assert shape_lists("First, look at: 1. Market size 2. Segments") == "First, look at:\nMarket size\n→ Segments"
    assert shape_lists("It grew 2.5x in 2024. Then 3 more.") == "It grew 2.5x in 2024. Then 3 more."


def test_the_cap_keeps_the_point_the_structure_and_the_hand_back():
    t = ("One. Two. Three.\nPopulation\n→ share\n→ frequency\nFour. Five. Six. What share would you take?")
    out = cap_sentences(t, 5)
    assert out.startswith("One.") and out.endswith("What share would you take?")
    assert "→ share\n→ frequency" in out
    assert len(re.split(r"[.!?]+", out.strip().rstrip("?"))) <= 5


def test_a_teaching_reply_with_a_numbered_list_streams_as_arrows():
    class _LLM:
        def __init__(self):
            outer = self

            class _C:
                @staticmethod
                def create(**kw):
                    if kw.get("stream"):
                        chunks = [REPLY_C[i:i + 3] for i in range(0, len(REPLY_C), 3)]
                        return iter([types.SimpleNamespace(
                            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content=c))],
                            usage=None, id="s") for c in chunks])
                    msg = types.SimpleNamespace(content=REPLY_C)
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None, id="c")
            self.chat = types.SimpleNamespace(completions=_C())
    s = BY_ID["semantic_repetition_guard"]
    ctl = {}
    out = "".join(ie.stream_interviewer_reply(
        scenario_case_content(s["case_type"]), s["case_type"], s["transcript"], s["new"], control_out=ctl,
        prior_state={}, llm_override=(_LLM(), "fake", "openai"), prompt_overrides=NO_LLM))
    assert ctl["function"] == "TEACH_STEP"
    assert "→ Identify total costs" in out and "too_long" not in detect_violations(out)
