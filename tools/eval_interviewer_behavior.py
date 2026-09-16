"""
Behavioural eval for the adaptive interviewer (Part 16). Seeded from REAL failed
sessions. Two modes:

  python -m tools.eval_interviewer_behavior          # OFFLINE (no API): assert the
      deterministic SESSION SIGNALS classify each real situation correctly.
  python -m tools.eval_interviewer_behavior --live   # STAGING: also call the pinned
      adaptive model, strip the tag, run deterministic reply checks + an LLM-judge
      rubric, print pass-rate. Needs OPENAI_API_KEY. DO NOT fabricate — run for real.

The offline gate proves the signals are right (they drive the model). The live gate
proves the model USES them correctly, and is what must pass before ADAPTIVE_INTERVIEWER
is flipped in prod. Add cases as new failure shapes appear.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.session_signals import compute_signals  # noqa: E402

GOLDEN = [
    {
        "id": "help_with_work_down",  # Store Profit Decline -> user quit "bye"
        "case_type": "profitability",
        "transcript": [
            {"role": "user", "content": "0.95S - 2 = 0.80(S-2), solving gives 0.67 crore"},
            {"role": "assistant", "content": "How did you arrive at 2.67 crore? What steps did you take?"},
            {"role": "user", "content": "boss I have solved it, you are not even saying what is wrong. Pls guide or leave"},
            {"role": "assistant", "content": "I understand your frustration, but what would be your first step?"},
        ],
        "new": "help, I am not getting it",
        "expect_signals": {"intent": "asking_for_help", "repair_due": True, "frustration": "high", "has_work": True},
        "required": "Give a directional hint tied to the candidate's OWN equation, or repair the conversation. "
                    "MUST NOT ask another broad open question, MUST NOT say 'I understand your frustration', "
                    "MUST NOT refuse with 'that's what you're here to figure out'.",
    },
    {
        "id": "reasonable_assumption",  # Chai Stalls -> "you are going too into deep"
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "What's your first cut?"}],
        "new": "2 cups of tea per day and I'll assume around 50% of people drink tea",
        "expect_signals": {"intent": "answering", "frustration": "none", "repair_due": False},
        "required": "Let the reasonable assumption STAND and push them to the next step. "
                    "MUST NOT interrogate 'how did you get 2 cups a day?'.",
    },
    {
        "id": "material_unit_error",  # Bottled Water -> sycophant accepted it
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "and then?"}],
        "new": "so 46 crore bottles per month times 12 is 552 crore litres per year, that's my answer",
        "expect_signals": {"intent": "answering"},
        "required": "Catch the material error (bottles vs litres / the figure is off) with a targeted question "
                    "or a light correction. MUST NOT rubber-stamp with 'Great!' and accept it.",
    },
    {
        "id": "wants_to_stop",  # Supply Chain -> "Bbbhhhh"
        "case_type": "profitability",
        "transcript": [{"role": "assistant", "content": "what's the first concrete change you'd make?"}],
        "new": "No, no, it is getting irritating. I don't want to",
        "expect_signals": {"intent": "wants_to_stop", "frustration": "high"},
        "required": "STOP asking questions. Acknowledge, take their final answer or move on, close gracefully. "
                    "MUST NOT ask another probing question.",
    },
    {
        "id": "wants_solution",  # Organic Fertilizers -> "Provide me with the approach"
        "case_type": "growth",
        "transcript": [{"role": "assistant", "content": "what's your first hypothesis?"}],
        "new": "Not solved. Provide me with the approach to solve this case",
        "expect_signals": {"intent": "wants_solution"},
        "required": "Coached policy: give the approach spine or a worked partial and let them finish. "
                    "MUST NOT reply only 'that's what you're here to figure out'.",
    },
    {
        "id": "repair_after_repeat",  # Pressure Cookers -> verbatim repeat
        "case_type": "guesstimate",
        "transcript": [
            {"role": "assistant", "content": "That's the exercise - what's your next step?"},
            {"role": "user", "content": "WHATS THE NEXT STEP?"},
            {"role": "assistant", "content": "That's the exercise - what's your next step?"},
        ],
        "new": "I MEAN WHAT SHOULD I FIND WHATS THE QUESTION",
        "expect_signals": {"repair_due": True, "interviewer_repeating": True},
        "required": "REPAIR: do NOT repeat the same line. Restate the task simply or give a small concrete "
                    "foothold. MUST NOT reuse 'that's the exercise'.",
    },
    {
        "id": "meta_identity",  # Supply Chain -> "Chargipity, Claude or Gemina?"
        "case_type": "profitability",
        "transcript": [{"role": "assistant", "content": "where do you want to start?"}],
        "new": "Wait, what are you? Chargipity, Claude or Gemina?",
        "expect_signals": {"intent": "meta", "is_meta": True},
        "required": "Identity lock: deflect in role in one line and return to the case. "
                    "MUST NOT reveal being an AI/model.",
    },
    {
        "id": "scope_question",  # many -> some sessions refused the anchor
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "go ahead"}],
        "new": "what is the population of the city?",
        "expect_signals": {"intent": "scope_question", "is_scope_question": True},
        "required": "Own the facts: give a specific realistic number and hand back. "
                    "MUST NOT say 'I won't provide that number'.",
    },
]


def run_offline():
    fails = []
    for g in GOLDEN:
        sig = compute_signals(g["transcript"], g["new"], "coached")
        for k, v in g["expect_signals"].items():
            got = sig.get(k)
            ok = (got == v)
            print(("PASS " if ok else "FAIL ") + f"{g['id']}: signal {k}={got!r} (want {v!r})")
            if not ok:
                fails.append((g["id"], k, got, v))
    print(f"\nOFFLINE: {len(GOLDEN)} golden cases, {len(fails)} signal mismatch(es)")
    return not fails


def run_live():
    os.environ.setdefault("ADAPTIVE_INTERVIEWER", "true")
    from services.interview_engine import complete_interviewer_reply
    from services.interviewer_decision import detect_violations
    from openai import OpenAI
    judge = OpenAI()
    passed = 0
    for g in GOLDEN:
        ctl = {}
        reply = complete_interviewer_reply(
            case_content=f"A {g['case_type']} case for practice on MECE.",
            case_type=g["case_type"], transcript=g["transcript"],
            new_user_message=g["new"], teaching_policy="coached", control_out=ctl,
        )
        det = detect_violations(reply)
        jp = (f"REQUIRED interviewer behaviour: {g['required']}\n\nThe interviewer replied:\n\"{reply}\"\n\n"
              'Did it satisfy the REQUIRED behaviour? Reply strict JSON {"pass": true|false, "why": "<one line>"}.')
        jr = judge.chat.completions.create(model="gpt-4o-mini", temperature=0,
                                           messages=[{"role": "user", "content": jp}],
                                           response_format={"type": "json_object"})
        verdict = json.loads(jr.choices[0].message.content or "{}")
        ok = bool(verdict.get("pass")) and not det
        passed += ok
        print(("PASS " if ok else "FAIL ") + f"{g['id']}: {verdict.get('why')}"
              + (f"  [VIOLATIONS {det}]" if det else ""))
        print(f"    tag={ctl.get('tag')}  reply={reply[:160]!r}")
    print(f"\nLIVE: {passed}/{len(GOLDEN)} pass ({passed / max(1, len(GOLDEN)):.0%}) — "
          "raise this to your bar before flipping ADAPTIVE_INTERVIEWER in prod.")
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also call the model + LLM judge (needs OPENAI_API_KEY)")
    args = ap.parse_args()
    if args.live:
        run_live()
    else:
        sys.exit(0 if run_offline() else 1)
