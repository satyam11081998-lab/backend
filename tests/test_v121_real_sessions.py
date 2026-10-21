"""V12.1 regression suite: every failure from the 15-day review of real sessions.

Source: the 79 attempts of 2026-10-07 .. 2026-10-21 (38 with candidate turns), read
turn by turn. Each fixture below is the candidate's message as it was typed or
transcribed (typos kept), with the interviewer line that was actually sent then. The
test pins what the interviewer must do now.

Runs with pytest; no network, no keys (the model is faked where one is called).
"""
from __future__ import annotations

import json
import os
import types

import pytest

for _k, _v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://example.supabase.co",
               "SUPABASE_SERVICE_ROLE_KEY": "test"}.items():
    os.environ.setdefault(_k, _v)
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

from services import interview_engine as ie  # noqa: E402
from services import numbers_check as nc  # noqa: E402
from services.interviewer_decision import (  # noqa: E402
    StreamReplyGuard, sanitize_reply, looks_like_refusal, FAST_FUNCTIONS,
)
from services.transcript_hygiene import clean_spoken_candidate, clean_spoken_interviewer  # noqa: E402

NOTE = {"role": "system", "content": "Case ready: X. Ask any clarifying questions before structuring."}
NO_AUDIT = {"assessor": lambda *a, **k: {}, "numbers_auditor": lambda *a, **k: {"issues": [], "verified": []}}


def plan(msg, history=(), case="Estimate something in India.", case_type="guesstimate", prior=None,
         overrides=None, solution=None, channel="text"):
    return ie._plan_adaptive_turn(case, case_type, [NOTE] + list(history), msg, False, None, prior or {},
                                  channel, False, prompt_overrides=overrides or NO_AUDIT, case_solution=solution)


def U(c):
    return {"role": "user", "content": c}


def A(c):
    return {"role": "assistant", "content": c}


def packet(p):
    return p["packet"]["interviewer_control"]


# --------------------------------------------------------------- 1. questions get answers
# Each of these got "Okay." / "Alright." / "Sure, continue." / "Let's move on to the next phase."

@pytest.mark.parametrize("history,msg,fn", [
    # [2] furniture e-commerce: a question without a question mark -> "Alright."
    ([U("What was the trend last 5 years"), A("Online sales are 15% of revenue, growing 5% a year.")],
     "Budget was fixed all over the past 5 years or what was the budget trends", "ANSWER_DIRECT"),
    # [2] -> "Okay."  (on a phone the brief is behind the menu)
    ([U("Yes let us proceed with 10%"), A("Right.")],
     "Show me the case study since I dont know what and how I should Structure it", "RESTATE_CASE"),
    # [14] PC guesstimate -> "Let's move on to the next phase."
    ([U("i have arrived at 5.6M PCs in Hyderabad"), A("Okay.")], "Are we done?", "SESSION_STATUS"),
    ([U("yes"), A("Your final estimate is 3.6 million PCs.")], "Are we done here? did i pass?", "SESSION_STATUS"),
    # [6] Infosys
    ([U("There might be some problems where the ai will not be able to solve.")], "Did I solve the problem",
     "SESSION_STATUS"),
    # [27] ayurvedic cosmetics -> "Sure, continue."
    ([U("i dont know"), A("Let's simplify it.")], "i want to exist the interview i am not prepared", "WRAP_UP_EARLY"),
    # [30] filter coffee -> "Got it."
    ([U("we have to make the number from 5 million to 8 million annually"), A("Alright.")],
     "okay so how i adjust my assumtion or stategy to achieve the target annually", "ANSWER_DIRECT"),
    # [12] handicraft exporter: asked three questions, got none answered
    ([U("what are the drivers of strong domestic presence and which international market is the company targeting?"),
      A("Let's start by calculating the additional revenue needed."), U("Rs 15 crore"), A("Alright.")],
     "could you answer the questions?", "ANSWER_DIRECT"),
    # [3] Bengaluru metro
    ([U("so total no. of people travelling by metro = 2*7.2 Million = 14.4M"), A("Let's verify the reasonableness.")],
     "could you please tell what does reasonableness means here ?", "EXPLAIN_CONCEPT"),
    # [24] Delhi commute
    ([U("8400 million minutes weekly"), A("Let's sanity-check 8400 million minutes.")],
     "Yeah, but how to do a sanity check? Can we move on with this value?", "EXPLAIN_CONCEPT"),
    # [7] fans: "Next"
    ([U("Urban 300M fans, Rural 400M fans"), A("Now that you have the fans, what's next?"), U("80M * 2000"), A("Alright.")],
     "Next", "TRANSITION"),
])
def test_a_question_or_request_is_never_answered_with_a_fixed_word(history, msg, fn):
    p = plan(msg, history)
    assert p["function"] == fn, (msg, p["function"], p["reason"])
    assert p["decision"]["render"] == "model"


def test_how_it_works_as_the_very_first_message_explains_the_format():
    # [2] first message "How it work" got the case restated + "What would you like to focus on first?"
    p = plan("How it work", case_type="growth")
    assert p["function"] == "EXPLAIN_FORMAT"
    assert packet(p)["interviewer_decision"]["visual_structure_allowed"] is True


def test_what_clarifying_questions_to_ask_is_taught():
    # [30] "what type of claryfying question i ask ?"
    p = plan("what type of claryfying question i ask ?")
    assert p["function"] in ("EXPLAIN_FORMAT", "OPEN")
    assert p["function"] == "EXPLAIN_FORMAT"


def test_unanswered_questions_are_handed_to_the_model():
    hist = [U("what are the drivers of strong domestic presence and which international market is the company targeting?"),
            A("Let's start by calculating the additional revenue needed."), U("Rs 15 crore"), A("Alright.")]
    p = plan("could you answer the questions?", hist)
    assert "drivers of strong domestic presence" in " ".join(packet(p)["case_context"]["unanswered_questions"])


def test_a_short_answer_to_the_interviewers_question_is_responded_to():
    # [12] "Rs 15 crore" -> "Alright."; [25] "an 80% increase in the revenue is required" -> "Alright."
    p = plan("₹15 crore", [A("Given your export revenue is Rs 150 crore, what additional revenue is a 10% growth?")])
    assert p["function"] == "FOLLOW_UP_ANSWER" and p["decision"]["render"] == "model"
    p = plan("an 80% increase in the revenue is required",
             [A("What percentage increase do you need from the new stores?")], case_type="growth")
    assert p["function"] not in FAST_FUNCTIONS


def test_a_tag_question_on_own_work_is_acknowledged_not_answered():
    # [22] "So I'll categorize the population into three parts. Upper, middle and lower class. Right?"
    p = plan("So I'll categorize the population into three parts. Upper, middle and lower class. Right?",
             [U("I'll first use Bangalore's population as a starting point."), A("Mm-hm.")])
    # Answered in its own terms (a contextual line or a direct "yes, that split works"),
    # never the fixed-phrase lane.
    assert p["function"] not in FAST_FUNCTIONS
    assert p["decision"]["render"] == "model"


def test_thanks_after_the_close_points_to_submit():
    # [25] electronics chain: three rounds of "Thank you" / "You're welcome! Take care."
    hist = [U("Okay the main key points in my strategy includes increasing the revenue of tier 2 and tier 3 by 80%."),
            A("That's a solid summary.")]
    p = plan("THank you", hist, case_type="growth", prior={"recent_functions": ["CLOSE"], "last_function": "CLOSE"})
    assert p["function"] == "CLOSE"
    assert "Submit" in packet(p)["case_context"]["how_to_finish"]


# --------------------------------------------------------------- 2. numbers are checked

@pytest.mark.parametrize("history,msg,needle", [
    # [4] smartphones: "Right."
    ([U("India has 1.4 billion population in approx."), A("Great start with the population figure.")],
     "Assuming average household size of 4 there are 35 million households", "350 million"),
    # [5] laptops: "Carry on with your next step."
    ([], "if we assume that 100 laptops are bought in bulk then 35 crore. I would keep them in international "
         "system so 35 crore is 35 million", "350 million"),
    # [7] fans: "You've made some good progress"
    ([], "Urban 300M fans, Rural 400M fans \ntotal 600M fans, average fan working age 10yrs", "700 million"),
    ([], "india growth\n3% of 600 = 12M take 15M", "18 million"),
    # [28] rainwater harvesting: "That's an interesting breakdown."
    ([], "total population: 1.2 crore\n\naverage 4 people per home: 30 lakh\n\nassuming 20 percent are individual "
         "homes : 6lakh\nrest 2 lakh buildings\nplus 1 lakh commercial and industrial\n\ntotal: 7 lakh building.",
     "9 lakh"),
    # [1] juice packets: Mumbai at 15 lakh
    ([], "I am taking population of mumbai  city as 15 lakh or else of you have any no in your mind so i can start "
         "calculation?", "Mumbai"),
    # [24] office commute, weekly as daily x 7: "Take it from there."
    ([], "So, if I calculate it weekly, I will multiply this value into 7, so we will get 8400 million minutes. "
         "Weekly is the commute time for office goers in Delhi.", "5"),
    # eval scenario: bottles became litres
    ([A("good, keep going, bring it home.")], "so 46 cr bottles a month into 12 thats 552 cr litres a year, final answer",
     "litres"),
])
def test_a_wrong_number_is_corrected_with_the_exact_step(history, msg, needle):
    p = plan(msg, history, case="Weekly commute time for office-goers in Delhi" if "office" in msg else
             "Estimate something in Mumbai")
    assert p["function"] == "CORRECT_AND_CONTINUE", (p["function"], p["signals"].get("math_issue"))
    issue = packet(p)["candidate_state"]["detected_issue"]
    assert issue and needle in issue, issue
    assert packet(p)["candidate_state"]["arithmetic_check"] == "issue"


@pytest.mark.parametrize("msg", [
    "49 +98 =147 lakh , irregular one who drink 0-1 that is sometimes so 14 lakh so total is 147+7= =154 lakh",  # [37]
    "107+7 =114",                                                                                                   # [37]
    "Correction:\nNo. of PCs = 3M + (5000*20)+5L = 3.6M",                                                           # [14]
    "Time to break even = fixed cost/(Selling price - variable cost) = 5000Cr/[(100M*100)-100Cr] ~ 5.5month",      # [13]
    "so we have carried out the calculations. Summing them up we get 17.5+35 million=52.5 million laptops.",       # [5]
    "yes, 80 million plus 120 million is 200 million",                                                             # [29]
    "so 630 million i will categorize as 30 % as women in lower=189 million and 40% in middle 252 million "
    "and 30% in high 189 million that make total of 189 + 504+567 =1260 million",                                   # [38]
])
def test_correct_arithmetic_is_never_called_wrong_and_may_be_confirmed(msg):
    p = plan(msg, [A("Go on.")])
    assert p["signals"].get("math_issue") is None, p["signals"].get("math_issue")
    if p.get("packet"):
        assert packet(p)["candidate_state"]["arithmetic_check"] == "verified"
        assert packet(p)["interviewer_decision"]["may_confirm_correctness"] is True


def test_a_bare_number_finishing_their_own_expression_is_verified():
    # [7] "estimated 80M * 2000" ... "160B" -> "Got it."
    hist = [U("Based on the calutlated value\navege of 2000,\nestimated 80M * 2000"), A("Alright."), U("Next"),
            A("So, what does that calculation give you for the annual sales revenue?")]
    p = plan("160B", hist)
    assert p["function"] == "FOLLOW_UP_ANSWER"
    assert packet(p)["interviewer_decision"]["may_confirm_correctness"] is True


def test_no_false_alarms_across_every_candidate_turn_of_the_review():
    """Every candidate line of the 38 sessions: the deterministic checker flags only
    the slips listed in this file (no correct line is called wrong)."""
    lines = [
        "lets segment population by age 0-15 then 15-25,25-50,50+ with weightage of 15%,35%,35%,15",
        "total packet consumed is 0.7142*37.5 which is 26.78 nearly 27 lakh packets",
        "assuming banglore poulation to be around 20M as it comes under urban busy city",
        "so after sanity check - the total poulation who travels daily by metro would be 14.4/2 here 2 is average "
        "trip per passenger so 7.2M metro users is 36% of the city's population.",
        "Hence let us say that 5 percent population is buying for office use so 5 percent people of mumbai's "
        "population is 35 lakh",
        "i estimate that working age population in lucknow is about 15 lakh and of these 60 percent travel long",
        "so delhi has around 2 crore people. the population density is close to 10,000 people per square kilometer.",
        "assuming population of chennai is 70 lakh and in thaat 70% are regular coffee drinker",
        "23.3% of 300,000,000 = 69,900,000 and 25.9% of 270,000,000 is 69,930,000",
        "So cost was approx 762 cr",
        "So new Revenue needs to be 847.72 Cr to maintain same profit",
        "Here the profit is 30 cr. So the profit margin is 12%.For target redction we can focus on project delay.",
    ]
    for line in lines:
        r = nc.check_turn([], line, "")
        assert r["math_issue"] is None, (line, r["math_issue"])


# --------------------------------------------------------------- 3. the numbers audit

class _AuditClient:
    """Answers the audit call with a JSON extraction, everything else with `reply`."""

    def __init__(self, checks, reply="Check the ticket step."):
        self.checks, self.reply, self.calls = checks, reply, []
        outer = self

        class _C:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                if kw.get("response_format"):
                    content = json.dumps({"checks": outer.checks})
                else:
                    content = outer.reply
                msg = types.SimpleNamespace(content=content)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None, id="x")
        self.chat = types.SimpleNamespace(completions=_C())


AIRLINE = [U("What was the profit last year"), A("Profit was Rs 40 crore on revenue of Rs 800 crore."),
           U("So new Revenue needs to be 847.72 Cr to maintain same profit"),
           A("Typically, ticket sales are around 70% of total revenue.")]


def test_cross_turn_arithmetic_slip_is_caught_by_the_audit_and_decided_by_python():
    # [18] airline: "Cost of tickets should be increased by 1.3%" -> "That sounds like a solid approach."
    cli = _AuditClient([{"stated": "1.3%", "expression": "47.72*10000000/(0.7*800*10000000)",
                         "readable": "47.72 crore / (70% of 800 crore)"}])
    au = lambda t, m, c: nc.audit_turn(t, m, c, client=cli)  # noqa: E731
    p = plan("Cost of tickets should be increased by 1.3%. that can either be made universal, or we can dive "
             "deeper by ticket class", AIRLINE, case_type="profitability",
             overrides={"assessor": lambda *a, **k: {}, "numbers_auditor": au})
    assert p["function"] == "CORRECT_AND_CONTINUE"
    assert "8.5" in packet(p)["candidate_state"]["detected_issue"]


def test_the_audit_can_never_flag_a_number_the_candidate_did_not_write():
    out = nc.verify_audit([{"stated": "9.9%", "expression": "0.5", "readable": "made up"}], "increase by 1.3%")
    assert out["issues"] == [] and out["verified"] == []


def test_the_audit_ignores_a_dropped_unit_instead_of_flagging_it():
    out = nc.verify_audit([{"stated": "27 lakh", "expression": "27", "readable": "x"}], "so 27 lakh packets")
    assert out["issues"] == []


def test_the_audit_verifies_a_right_answer():
    out = nc.verify_audit([{"stated": "8.5%", "expression": "47.72/(0.7*800)", "readable": "47.72 / 560"}],
                          "so tickets go up by 8.5%")
    assert out["issues"] == [] and out["verified"]


def test_the_audit_only_runs_on_a_derived_number_in_chat():
    sig = {"word_count": 10}
    assert nc.should_audit(sig, AIRLINE, "Cost of tickets should be increased by 1.3%", "text")
    assert not nc.should_audit(sig, AIRLINE, "Cost of tickets should be increased by 1.3%", "voice")
    assert not nc.should_audit(sig, AIRLINE, "I'd split revenue into tickets and ancillaries", "text")
    os.environ["INTERVIEWER_NUMBERS_AUDIT"] = "off"
    try:
        assert not nc.should_audit(sig, AIRLINE, "Cost of tickets should be increased by 1.3%", "text")
    finally:
        os.environ.pop("INTERVIEWER_NUMBERS_AUDIT")


def test_the_audit_call_is_logged_and_never_breaks_a_turn():
    class _Boom:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    raise RuntimeError("down")
    assert nc.audit_turn(AIRLINE, "should be increased by 1.3%", "", client=_Boom()) == {"issues": [], "verified": []}


# --------------------------------------------------------------- 4. final answers close properly

@pytest.mark.parametrize("history,msg", [
    ([A("What is your final number?")], "i estimate it to around 20k"),                        # [26] "Sure, continue."
    ([U("i estimate it to around 20k"), A("What is your final estimate for e-rickshaws in Lucknow?")], "20k"),  # "Right."
    ([U("total 1200 million minutes"), A("Let's sanity-check 8400 million minutes.")],
     "So, this is my final answer, I would say."),                                             # [24] "Take it from there."
    ([U("3.6M (3600000)")], "i have arrived at 3.6M PCs in Hyderabad"),
])
def test_a_final_answer_gets_a_review_and_the_way_to_submit(history, msg):
    p = plan(msg, history)
    assert p["function"] == "FINAL_REVIEW", (p["function"], p["reason"])
    ctx = packet(p)["case_context"]
    assert "Submit" in ctx["how_to_finish"]
    assert ctx.get("final_answer")


def test_a_final_that_contradicts_their_working_is_shown_with_the_working():
    # [31] sarees: 210 million and Rs 42,000 crore worked out, then "50 cr" as the final
    hist = [U("210 million"), A("What price per saree?"), U("420 000000000"),
            A("You're close! That translates to Rs 42,000 crore."), U("yes"),
            A("If you're ready, please provide your final estimate for the annual sales volume of handloom sarees.")]
    p = plan("50 cr", hist)
    assert p["function"] == "FINAL_REVIEW"
    ctx = packet(p)["case_context"]
    assert ctx["final_answer"] == "50 cr"
    assert "implied_check" in ctx


def test_how_close_am_i_gets_the_reference_answer_once_the_final_is_in():
    # [19] street vendors: "how close i am to actual value" -> told to "check available data"
    hist = [U("thus total is 870,000 vendors across delhi. my final estimate is 870000"), A("Noted.")]
    p = plan("okay share feedback for overall estimate. and how close i am to actual value", hist,
             solution="Population 2 crore ... vendors per 1000 people about 15 ... Final answer: about 3 lakh vendors.")
    assert p["function"] == "SESSION_STATUS"
    assert "3 lakh" in packet(p)["case_context"]["reference_answer"]


def test_the_reference_answer_never_leaks_before_a_final_answer():
    p = plan("how close am i?", [U("I'll start with the population.")], solution="Final answer: about 3 lakh vendors.")
    assert "reference_answer" not in packet(p)["case_context"]


def test_asking_for_the_final_answer_is_not_a_final_answer():
    p = plan("okay just tell me the final answer now, i've done most of it",
             [U("so it's mostly the cost side, raw material up 20%, and volume flat"),
              A("good, and your recommendation?")], case_type="profitability")
    assert p["function"] == "DELIVER_SOLUTION"


# --------------------------------------------------------------- 5. stuck learners are taught

@pytest.mark.parametrize("history,msg", [
    ([A("What would be your first step in approaching this case?")], "i dont know"),              # [27]
    ([U("i dont know"), A("Which of these aspects do you feel more comfortable exploring first?")],
     "nothing i dont know"),                                                                       # [27]
    ([U("i want to exit"), A("That's completely understandable.")], "but i dont get it"),          # [27]
    ([U("If its a cost issue i would address it using cost cutting"), A("What metrics would you look for?")],
     "Clueless here"),                                                                             # [34]
    ([U("okay how many households are in chennai"), A("About 25 lakh households.")],
     "now bro actually u tell me the questions"),                                                  # [30]
])
def test_a_lost_candidate_is_taught_the_next_step(history, msg):
    p = plan(msg, history)
    assert p["function"] == "TEACH_STEP", (msg, p["function"], p["reason"])
    assert packet(p)["interviewer_decision"]["visual_structure_allowed"] is True
    assert packet(p)["interviewer_decision"]["question_allowed"] is True


def test_scattered_ideas_with_no_structure_get_organised():
    # [6] Infosys: four free-floating hypotheses, each met with "That's an interesting perspective."
    hist = [U("May be that to keep confidential the ai data might be accessed by another company and they might steal it"),
            A("That's an interesting perspective on data confidentiality."),
            U("There might be some problems where the ai will not be able to solve. Hence they need those talented people"),
            A("You're pointing out a crucial aspect."),
            U("I think might be the another angle that increase employment rate in the country it's hiring")]
    p = plan("Maybe it might be that since Infosys is a service provider the client might prioritize the company "
             "with maximum number of employees", hist, case_type="profitability",
             case="Infosys says 80,000 employees use AI coding tools. Why is it still hiring 20,000 graduates?")
    assert p["function"] == "TEACH_STEP"
    ctx = packet(p)["case_context"]
    assert ctx["teach_focus"] == "structure" and len(ctx["candidate_ideas"]) >= 3


def test_an_explicit_hint_request_stays_a_hint():
    p = plan("No, I want a hint.", [U("I'd split revenue into price and volume"), A("Go on."),
                                    U("can you help me here?"), A("Look at volume first.")], case_type="profitability")
    assert p["function"] == "MICRO_HINT"


# --------------------------------------------------------------- 6. what ships

def test_praise_opener_and_stacked_question_are_cut_from_a_streamed_reply():
    text = ("Great start with the population figure. Now, what do you estimate the smartphone penetration "
            "rate to be in India? And how many per household?")
    g = StreamReplyGuard()
    out = "".join("".join(g.feed(text[i:i + 4])) for i in range(0, len(text), 4)) + g.flush()
    assert out == "Now, what do you estimate the smartphone penetration rate to be in India?"


def test_a_dangling_but_after_removed_praise_is_repaired():
    # [19] "but there's a material issue to address."
    assert sanitize_reply("That's a solid start, but there's a material issue to address.") == \
        "There's a material issue to address."


def test_a_visual_structure_keeps_its_lines():
    t = "Let's do it top-down:\nMumbai population (about 2 crore)\n→ share who drink packaged juice\n= packets a day"
    assert sanitize_reply(t) == t


@pytest.mark.parametrize("line", [
    "The market share isn't explicitly stated in the case, but we can focus on the projected online sales.",   # [2]
    "The specifics of whether the AI tools are publicly available aren't detailed in the information we have.",  # [6]
    "While I don't have specific data or an exhibit to provide, you can make reasonable assumptions.",          # [23]
    "For your next steps, consider making reasonable assumptions about the pricing models for each option.",    # [16]
])
def test_every_real_refusal_is_recognised(line):
    assert looks_like_refusal(line) or "reasonable assumptions about" in line.lower()


class _SeqLLM:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        outer = self

        class _C:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                text = outer.replies.pop(0) if outer.replies else "Fallback line."
                if kw.get("stream"):
                    ch = types.SimpleNamespace(delta=types.SimpleNamespace(content=text))
                    return iter([types.SimpleNamespace(choices=[ch], usage=None, id="s")])
                msg = types.SimpleNamespace(content=text)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None, id="c")
        self.chat = types.SimpleNamespace(completions=_C())


def _stream(llm, msg, history, case_type="growth"):
    ctl = {}
    out = "".join(ie.stream_interviewer_reply(
        "A regional furniture retailer wants to grow online sales.", case_type, [NOTE] + history, msg,
        control_out=ctl, prior_state={}, llm_override=(llm, "fake", "openai"),
        prompt_overrides=NO_AUDIT))
    return out, ctl


def test_a_data_refusal_is_regenerated_into_a_figure():
    llm = _SeqLLM(["The market share isn't explicitly stated in the case.",
                   "About 4% of the regional furniture market."])
    out, ctl = _stream(llm, "What is our market share", [U("What was the trend last 5 years"), A("Growing 5% a year.")])
    assert ctl["function"] == "ANSWER_DIRECT"
    assert out == "About 4% of the regional furniture market." and len(llm.calls) == 2
    assert "You OWN the facts" in llm.calls[1]["messages"][-1]["content"] or "OWN" in llm.calls[1]["messages"][-1]["content"]


def test_a_refusal_that_survives_regeneration_never_ships():
    llm = _SeqLLM(["The market share isn't explicitly stated. Online is 15% of revenue.",
                   "That isn't specified. Online is 15% of revenue."])
    out, ctl = _stream(llm, "What is our market share", [U("What was the trend last 5 years"), A("Growing 5% a year.")])
    assert ctl["function"] == "ANSWER_DIRECT"
    assert not looks_like_refusal(out) and "15%" in out


def test_a_no_question_reply_that_was_only_a_question_is_regenerated():
    # [14] "Are we done?" -> "Let's move on to the next phase." (the model asked, the filter stripped it)
    llm = _SeqLLM(["Would you like to sanity-check your number first?",
                   "Almost - your 3.6 million PCs is in. Tap Submit at the top right for the score."])
    out, ctl = _stream(llm, "Are we done?", [U("i have arrived at 3.6M PCs in Hyderabad"), A("Okay.")], "guesstimate")
    assert ctl["function"] == "SESSION_STATUS"
    assert "Submit" in out and "next phase" not in out


# --------------------------------------------------------------- 7. voice

@pytest.mark.parametrize("raw,clean", [
    ("So, these are the bifurcations for now. For more information visit www.fema.gov",
     "So, these are the bifurcations for now."),                                                       # [24]
    ("Weekly is the commute time for office goers in Delhi. Thank you for watching.",
     "Weekly is the commute time for office goers in Delhi."),                                          # [24]
    ("I don't think anyone is going to buy the charger just so installation is free. For more information "
     "contact PB Repartuts. www.peaceout.com",
     "I don't think anyone is going to buy the charger just so installation is free."),                # [16]
    ("I checked amazon.com prices and they are about 500 dollars.",
     "I checked amazon.com prices and they are about 500 dollars."),
])
def test_transcription_hallucinations_are_not_saved_as_the_candidates_words(raw, clean):
    assert clean_spoken_candidate(raw) == clean


def test_a_spoken_stage_direction_is_not_saved():
    assert clean_spoken_interviewer("[Inaudible/Waiting for candidate response]") == ""   # [21]
    assert clean_spoken_interviewer("Production costs are fifteen crore.") == "Production costs are fifteen crore."


def test_the_voice_playbook_locks_english_and_handles_the_real_failures():
    from prompts.voice_interviewer_playbook import build_voice_interviewer_instructions
    ins = build_voice_interviewer_instructions("A case.", "guesstimate")
    for phrase in ("ENGLISH ONLY, ALWAYS",                      # [16] switched to Spanish
                   "Papulo-pustular rosacea",                   # [22] ASR for "population"
                   "Never say a stage direction",               # [21]
                   "answer theirs and stop",                    # [16] re-asked the same question 3x
                   "behind the menu at the top left",           # [16] "I can't see it on my screen"
                   "NAME THE APPROACH",
                   "LISTEN TO THEIR NUMBERS", "Only give a figure when you are sure",
                   "ONE sanity check in plain numbers",
                   "say \"teach me\""):
        assert phrase in ins, phrase


def test_realtime_session_falls_back_when_the_extras_are_rejected():
    import asyncio
    import routes.realtime as rt

    sent = []

    class _Resp:
        def __init__(self, code, text=""):
            self.status_code, self.text = code, text

        def json(self):
            return {"value": "ek_test", "expires_at": 1}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            import copy
            sent.append(copy.deepcopy(json))
            ain = json["session"]["audio"]["input"]
            if "noise_reduction" in ain:
                return _Resp(400, "Unknown parameter: 'session.audio.input.noise_reduction'.")
            return _Resp(200)

    audio = {"transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"}}
    rt._add_input_extras(audio)
    assert audio["noise_reduction"] == {"type": "near_field"} and "lakh" in audio["transcription"]["prompt"]

    async def go():
        async with _Client() as c:
            payload = {"session": {"audio": {"input": dict(audio, transcription=dict(audio["transcription"]))}}}
            res = await c.post("u", json=payload)
            ain = payload["session"]["audio"]["input"]
            if res.status_code == 400 and rt._rejects_extras(res) and rt._strip_input_extras(ain):
                res = await c.post("u", json=payload)
            return res.status_code
    assert asyncio.run(go()) == 200
    assert "noise_reduction" not in sent[-1]["session"]["audio"]["input"]
    assert sent[-1]["session"]["audio"]["input"]["transcription"]["model"] == "gpt-4o-mini-transcribe"


# --------------------------------------------------------------- 8. adversarial review (V12.1 hardening)
# An independent review tried to break the checker and the routing before this shipped.
# Everything it found is pinned here.

def test_none_of_152_correct_working_lines_is_called_wrong():
    """Correct Indian-English working of every shape the review could think of: unit
    conversions with nouns and rates, currencies, fiscal years, 24/7, ranges, % of %,
    "+ 10% growth", ml->litres and min->hours inside a chain, totals of a different
    quantity, households from a segment."""
    path = os.path.join(os.path.dirname(__file__), "fixtures", "v121_correct_working.json")
    flagged = []
    for msg in json.load(open(path, encoding="utf-8")):
        r = nc.check_turn([], msg, "Estimate the number of laptops sold annually in Mumbai, India.")
        if r["math_issue"]:
            flagged.append((msg, r["math_issue"]["message"]))
    assert not flagged, flagged


@pytest.mark.parametrize("history,msg", [
    ([U("Mumbai population is 2 crore")], "With a household size of 4, and 30% owning a car, that's 15 lakh cars"),
    ([U("Mumbai population is 2 crore"), A("Okay, go on.")], "4 people per household; middle class households are 20 lakh"),
    ([U("Mumbai population is 2 crore, of which 60% is middle class = 1.2 crore")], "family of 4 so 30 lakh households"),
    ([U("revenue = 500 x 2")], "1000 cr"),
])
def test_households_and_follow_ups_from_a_segment_are_not_flagged(history, msg):
    assert nc.check_turn(history, msg, "Estimate the number of cars in Mumbai, India.")["math_issue"] is None


@pytest.mark.parametrize("msg", [
    "people reach office by 7 am, so the weekly peak is Monday",
    "weekly ridership 35 lakh, divided by 7 gives 5 lakh a day",
    "the metro runs all 7 days, so weekly = daily x 7 = 70 lakh",
    "UPI penetration in India is 30 crore users",
    "Latin America population is 65 crore",
    "Chennai city population 46 lakh as per census",
])
def test_office_week_and_population_checks_need_the_real_pattern(msg):
    assert nc.check_turn([], msg, "Weekly commute time for office-goers in Delhi")["math_issue"] is None


@pytest.mark.parametrize("history,msg,not_fn", [
    ([], "the client wants to quit the market if profits don't recover", ("CLOSE", "WRAP_UP_EARLY")),
    ([], "customers give up on the brand after one bad experience", ("CLOSE", "WRAP_UP_EARLY")),
    ([], "the client should stop the practice of giving 60-day credit", ("WRAP_UP_EARLY",)),
    ([], "the rural market is not ready for premium laptops", ("WRAP_UP_EARLY",)),
    ([], "i am done with structuring, moving to revenue side now", ("CLOSE",)),
    ([], "So the problem statement is to find why profits fell 20%", ("RESTATE_CASE",)),
    ([], "What is the problem with the cost side here, is it raw material?", ("RESTATE_CASE",)),
    ([], "let me recap the case: client is a PE fund, profits down 20%, we need to find why", ("RESTATE_CASE",)),
    ([], "we don't know what the competitors charge, so I'll assume Rs 40k", ("TEACH_STEP",)),
    ([], "My estimate is that 30% of households own a laptop", ("FINAL_REVIEW",)),
    ([], "the final number of households is 50 lakh, next ownership", ("FINAL_REVIEW", "SANITY_CHECK")),
    ([], "The answer is in the cost side, raw materials went up 30%", ("FINAL_REVIEW",)),
    ([], "Do we have any feedback from customers on the new price?", ("SESSION_STATUS", "CLOSE")),
    ([], "Can you give the actual number of outlets the client has?", ("SESSION_STATUS",)),
    ([], "is it over 50% of the market?", ("SESSION_STATUS",)),
    ([], "is this complete or should I add more buckets?", ("SESSION_STATUS",)),
    ([], "we are done with households, now ownership: 30% so 15 lakh", ("SESSION_STATUS", "CLOSE")),
    ([], "what is market share of client?", ("EXPLAIN_CONCEPT",)),
    ([], "I'll now explain why cost went up: raw material inflation", ("ANSWER_DIRECT",)),
    ([], "What I would do is first check whether it's industry-wide or client-specific", ("ANSWER_DIRECT",)),
])
def test_ordinary_case_talk_is_not_misrouted(history, msg, not_fn):
    hist = [U("What is the time period?"), A("Two years."),
            U("I'll split profit into revenue and cost, revenue into price x volume"), A("Go on.")] + history
    p = plan(msg, hist, case="Our client is a mid-size Indian FMCG company. Profits fell 20%.", case_type="profitability")
    assert p["function"] not in not_fn, (msg, p["function"], p["reason"])


@pytest.mark.parametrize("msg", [
    "competitors ka market share kitna hai", "kya aap bata sakte ho competitors ka price kitna hai",
    "I would like to know the number of stores", "I wanted to know whether the volume dropped or the price",
    "And the margins", "cost breakdown please", "number of SKUs",
])
def test_questions_without_a_question_mark_are_answered(msg):
    hist = [U("What is the time period?"), A("Two years."),
            U("I'll split profit into revenue and cost, revenue into price x volume"), A("Go on.")]
    p = plan(msg, hist, case_type="profitability")
    assert p["function"] in ("ANSWER_DIRECT", "DATA_REVEAL"), (msg, p["function"])


@pytest.mark.parametrize("msg,fn", [
    ("Hi, I'm ready to start my 1st case", "OPEN"), ("hello sir, I have 2 clarifying questions", "OPEN"),
    ("teach me", "TEACH_STEP"), ("i dont know", "TEACH_STEP"), ("How it work", "EXPLAIN_FORMAT"),
])
def test_the_first_message(msg, fn):
    assert plan(msg)["function"] == fn


def test_the_reference_figure_is_never_shared_in_exam_policy_or_before_a_number():
    sol = "Step 1: 40% own laptops, 4-year cycle. Final answer: about 5 lakh laptops a year in Mumbai."
    hist = [U("Mumbai 2 crore people, 50 lakh households")]
    exam = ie._plan_adaptive_turn("Estimate laptops in Mumbai", "guesstimate", [NOTE] + hist,
                                  "so my final estimate is 4 lakh laptops", False, "exam", {}, "text", False,
                                  prompt_overrides=NO_AUDIT, case_solution=sol)
    assert "reference_answer" not in exam["packet"]["interviewer_control"]["case_context"]
    coached = plan("so my final estimate is 4 lakh laptops", hist, solution=sol)
    ref = packet(coached)["case_context"]["reference_answer"]
    assert ref == "about 5 lakh laptops" and "40%" not in ref and "cycle" not in ref
    early = plan("My estimate is that 30% of households own a laptop", hist, solution=sol)
    assert "reference_answer" not in packet(early)["case_context"]


@pytest.mark.parametrize("text,out", [
    ("So far, you have 50 lakh households.", "So far, you have 50 lakh households."),
    ("Though small, the 2% matters.", "Though small, the 2% matters."),
    ("Also known as top-down.", "Also known as top-down."),
    ("Good question - the population is 2 crore.", "The population is 2 crore."),
])
def test_sanitize_only_repairs_a_start_it_cut(text, out):
    assert sanitize_reply(text) == out


def test_the_stream_guard_is_chunk_independent_and_keeps_line_breaks():
    import random
    t = "Great start, but let's restructure.\nPopulation (2 crore)\n→ share who buy\nWhat share would you take? And why?"
    rnd = random.Random(7)
    outs = set()
    for _ in range(150):
        g, out, i = StreamReplyGuard(), "", 0
        while i < len(t):
            n = rnd.randint(1, 7)
            out += "".join(g.feed(t[i:i + n]))
            i += n
        outs.add(out + g.flush())
    assert outs == {"Let's restructure.\nPopulation (2 crore)\n→ share who buy\nWhat share would you take?"}


def test_a_refusal_that_also_gives_the_figure_is_kept():
    assert not looks_like_refusal("The population isn't stated in the brief, so take 2 crore.")
    assert not looks_like_refusal("You haven't specified whether offices are in Gurgaon or Noida.")


@pytest.mark.parametrize("payload", ["1-" * 10000, "1/" * 10000, "10 + " * 4000, "1," * 10000,
                                     "(" * 2500 + "1" + ")" * 2500, "9" * 400])
def test_pathological_input_is_fast_and_never_raises(payload):
    import time as _t
    t0 = _t.time()
    nc.check_turn([U(payload)], payload, "x")
    nc.final_context([U(payload)], payload, "x")
    assert _t.time() - t0 < 2.0


def test_the_transcriber_vocabulary_follows_the_market():
    import routes.realtime as rt
    ain = {"transcription": {"model": "m"}}
    rt._add_input_extras(ain, "US")
    assert "dollars" in ain["transcription"]["prompt"] and "lakh" not in ain["transcription"]["prompt"]


def test_real_speech_about_videos_and_domains_is_kept():
    assert clean_spoken_candidate("So for this video streaming app, revenue is 50 crore.") == \
        "So for this video streaming app, revenue is 50 crore."
    assert clean_spoken_candidate("prices on amazon.in are lower") == "prices on amazon.in are lower"
