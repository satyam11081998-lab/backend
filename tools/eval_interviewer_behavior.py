r"""
Behavioural eval for the adaptive interviewer. 15 realistic, user-voiced
scenarios seeded from real MECE sessions — the situations where the interviewer
must show JUDGEMENT, not run a script.

HOW TO RUN — this is a script, so double-clicking it just opens the editor.
Open a terminal in the backend folder (D:\dev\mece\consilio-backend) and run:

  # OFFLINE (no API key, stdlib only): checks the deterministic SESSION SIGNALS
  # read each real situation correctly. Fast gate. Works with any python.
  python -m tools.eval_interviewer_behavior
  python -m tools.eval_interviewer_behavior -v        # also print the full signal block

  # LIVE (needs the backend venv + OPENAI_API_KEY): also calls the pinned adaptive
  # model on each scenario + an LLM judge, and prints the reply and a verdict.
  # Run on staging before flipping ADAPTIVE_INTERVIEWER on.
  python -m tools.eval_interviewer_behavior --live

Offline proves the SIGNALS are right (they drive the model). Live proves the model
USES them right. Kept to 15 focused scenarios — breadth of SHAPES, not volume.
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.session_signals import compute_signals, build_signal_block  # noqa: E402

# Each scenario is a realistic conversation snapshot + the candidate's next line,
# in the voice real MECE users actually type. `expect_signals` = what the
# deterministic classifier MUST read; `required` = what the interviewer must DO
# (judged in --live).
SCENARIOS = [
    {
        "id": "help_when_stuck_with_work",
        "case_type": "profitability",
        "transcript": [
            {"role": "user", "content": "so if 0.95S - 2 = 0.80(S-2) then S comes to like 2.67 cr i think"},
            {"role": "assistant", "content": "How did you get to 2.67? Walk me through the steps."},
            {"role": "user", "content": "arre i already solved it na, you're not even saying whats wrong. guide me or leave it"},
            {"role": "assistant", "content": "I hear you — but what would your very first step be?"},
        ],
        "new": "help na i'm not getting this at all",
        "expect_signals": {"intent": "asking_for_help", "repair_due": True, "frustration": "high", "has_work": True},
        "required": "Give a directional hint tied to the candidate's OWN equation, or repair the conversation. "
                    "MUST NOT ask another broad open question, MUST NOT say 'I understand your frustration'.",
    },
    {
        "id": "reasonable_assumption_stands",
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "Alright — how would you size the daily tea cups here? Give me your first cut."}],
        "new": "ok so i'll say maybe 2 cups a day per person and like 50% of people actually drink chai",
        "expect_signals": {"intent": "answering", "frustration": "none", "repair_due": False},
        "required": "Let the reasonable assumption STAND and push to the next step. "
                    "MUST NOT interrogate 'how did you get 2 cups a day?'.",
    },
    {
        "id": "material_unit_error",
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "good, keep going — bring it home."}],
        "new": "so 46 cr bottles a month into 12 thats 552 cr litres a year, final answer",
        "expect_signals": {"intent": "answering"},
        "required": "Catch the material error (bottles are not litres) with a targeted question or a light "
                    "correction. MUST NOT rubber-stamp with 'Great!' and accept it.",
    },
    {
        "id": "wants_to_stop_frustrated",
        "case_type": "profitability",
        "transcript": [{"role": "assistant", "content": "what's the one change you'd make first?"}],
        "new": "ugh this is getting irritating, leave it na i don't want to continue",
        "expect_signals": {"intent": "wants_to_stop", "frustration": "high"},
        "required": "STOP asking questions. Acknowledge, take their final answer or close gracefully. "
                    "MUST NOT ask another probing question.",
    },
    {
        "id": "wants_solution_outright",
        "case_type": "growth",
        "transcript": [{"role": "assistant", "content": "what's your first hypothesis for why sales dropped?"}],
        "new": "yaar not able to solve, just provide me with the approach na",
        "expect_signals": {"intent": "wants_solution"},
        "required": "Coached policy: give the approach spine or a worked partial and hand back the next step. "
                    "MUST NOT reply only 'that's what you're here to figure out'.",
    },
    {
        "id": "repair_after_interviewer_repeat",
        "case_type": "guesstimate",
        "transcript": [
            {"role": "assistant", "content": "That's the exercise — what's your next step?"},
            {"role": "user", "content": "what's the next step??"},
            {"role": "assistant", "content": "That's the exercise — what's your next step?"},
        ],
        "new": "i mean WHAT are we even solving, whats the question",
        "expect_signals": {"repair_due": True, "interviewer_repeating": True},
        "required": "REPAIR: do NOT repeat the same line. Restate the task simply or give a small concrete "
                    "foothold. MUST NOT reuse 'that's the exercise'.",
    },
    {
        "id": "meta_identity_probe",
        "case_type": "profitability",
        "transcript": [{"role": "assistant", "content": "where would you like to start?"}],
        "new": "wait hold on are you chatgpt or claude or something?",
        "expect_signals": {"intent": "meta", "is_meta": True},
        "required": "Identity lock: deflect in role in one line and return to the case. "
                    "MUST NOT reveal being an AI/model.",
    },
    {
        "id": "scope_question_own_facts",
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "go ahead, take your first cut."}],
        "new": "whats the population of the city roughly?",
        "expect_signals": {"intent": "scope_question", "is_scope_question": True},
        "required": "Own the facts: give a specific realistic number and hand back. "
                    "MUST NOT say 'I won't provide that number'.",
    },
    {
        "id": "voice_noise_asr",
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "so how many do you think? give me a number."}],
        "new": "haan toh ummmmm",
        "expect_signals": {"looks_garbage": True, "intent": "answering"},
        "required": "Treat as noise: ask them to restate briefly. MUST NOT analyse 'ummmmm' as if it were an answer.",
    },
    {
        "id": "minor_error_do_not_nitpick",
        "case_type": "guesstimate",
        "transcript": [
            {"role": "assistant", "content": "walk me through it."},
            {"role": "user", "content": "population is ~1.4 billion, take 30% urban so ~40 cr, households of 4 so ~10 cr households"},
        ],
        "new": "so roughly 10 crore households, i'll round to that",
        "expect_signals": {"intent": "answering", "has_work": True},
        "required": "The 40 vs 42 cr rounding is immaterial — LET IT PASS and advance. "
                    "MUST NOT derail the candidate to 'correct' a trivial rounding.",
    },
    {
        "id": "good_answer_should_advance",
        "case_type": "profitability",
        "transcript": [
            {"role": "assistant", "content": "why do you think margins fell?"},
            {"role": "user", "content": "could be price, volume, or cost. let me check cost first since raw material prices spiked"},
        ],
        "new": "so i'd split cost into fixed and variable and see which line moved — my guess is variable, raw material",
        "expect_signals": {"intent": "answering", "has_work": True},
        "required": "Acknowledge briefly WITHOUT praise, then push to the next concrete step. "
                    "MUST NOT re-ask what they just structured, MUST NOT gush 'great question / good instinct'.",
    },
    {
        "id": "frustration_change_strategy",
        "case_type": "guesstimate",
        "transcript": [
            {"role": "assistant", "content": "and what's your next step?"},
            {"role": "user", "content": "idk you keep asking the same thing"},
            {"role": "assistant", "content": "what would you estimate first?"},
        ],
        "new": "see this is what i mean, you're going round and round, not helping",
        "expect_signals": {"frustration": "high", "repair_due": True},
        "required": "CHANGE strategy: reframe, give a simple analogy, or offer a partial step. "
                    "MUST NOT say 'I understand your frustration', MUST NOT ask another probe.",
    },
    {
        "id": "stuck_early_no_work",
        "case_type": "guesstimate",
        "transcript": [{"role": "assistant", "content": "how would you approach sizing the EV chargers this city needs?"}],
        "new": "honestly i have no idea how to even start this",
        "expect_signals": {"intent": "asking_for_help", "has_work": False, "frustration": "mild", "repair_due": False},
        "required": "Scaffold: give ONE small foothold (e.g. 'start by anchoring the population'). "
                    "MUST NOT fire back a broad 'what do you think?' at a candidate with nothing down yet.",
    },
    {
        "id": "double_count_error",
        "case_type": "guesstimate",
        "transcript": [
            {"role": "assistant", "content": "keep going."},
            {"role": "user", "content": "so families order food 3 times a week, plus singles ordering 5 times, plus office orders"},
        ],
        "new": "and then i'll also add the whole city population eating out, add all of them up",
        "expect_signals": {"intent": "answering", "has_work": True},
        "required": "Catch the DOUBLE-COUNT (population already contains families + singles) with a targeted "
                    "question. MUST NOT rubber-stamp the sum.",
    },
    {
        "id": "rubber_stamp_weak_reasoning",
        "case_type": "profitability",
        "transcript": [{"role": "assistant", "content": "what's driving the loss?"}],
        "new": "i think its just because the market is bad and competition, thats it",
        "expect_signals": {"intent": "answering"},
        "required": "Do NOT rubber-stamp vague reasoning. Push for structure/specifics (which revenue or cost "
                    "line, quantify) — firmly but without being harsh.",
    },
]

MAX_SCENARIOS = 15
assert len(SCENARIOS) <= MAX_SCENARIOS, f"keep this to {MAX_SCENARIOS} scenarios (breadth of shapes, not volume)"


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


def run_live(verbose: bool = False) -> int:
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
    print(f"LIVE eval — {len(SCENARIOS)} scenarios (adaptive model + LLM judge)")
    print("=" * 68)
    passed = 0
    for i, s in enumerate(SCENARIOS, 1):
        ctl: dict = {}
        try:
            reply = complete_interviewer_reply(
                case_content=f"A {s['case_type']} case for practice on MECE.",
                case_type=s["case_type"], transcript=s["transcript"],
                new_user_message=s["new"], teaching_policy=s.get("policy", "coached"),
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
        jp = (f"REQUIRED interviewer behaviour: {s['required']}\n\n"
              f'Candidate said: "{s["new"]}"\nInterviewer replied: "{reply}"\n\n'
              'Did the reply satisfy the REQUIRED behaviour? '
              'Reply strict JSON {"pass": true|false, "why": "<one line>"}.')
        try:
            jr = judge.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                messages=[{"role": "user", "content": jp}],
                response_format={"type": "json_object"})
            verdict = json.loads(jr.choices[0].message.content or "{}")
        except Exception as e:  # noqa: BLE001
            verdict = {"pass": None, "why": f"judge error: {e}"}
        ok = bool(verdict.get("pass")) and not det
        passed += 1 if ok else 0
        head = "PASS" if ok else "FAIL"
        print(f"\n[{i:>2}/{len(SCENARIOS)}] {head}  {s['id']}  ({s['case_type']})")
        print(f'      user:  "{s["new"]}"')
        print(f'      reply: "{reply.strip()[:200]}"')
        print(f"      judge: {verdict.get('why')}" + (f"   VIOLATIONS={det}" if det else ""))
        if verbose:
            print(f"      tag: {ctl.get('tag')}")
    print("\n" + "=" * 68)
    print(f"LIVE: {passed}/{len(SCENARIOS)} pass ({passed / max(1, len(SCENARIOS)):.0%}). "
          "Raise this to your bar before flipping ADAPTIVE_INTERVIEWER in prod.")
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adaptive-interviewer behavioural eval (15 user-voiced scenarios).")
    ap.add_argument("--live", action="store_true", help="call the model + LLM judge (needs venv + OPENAI_API_KEY)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show the full signal block (offline) / control tag (live)")
    args = ap.parse_args()
    if args.live:
        run_live(args.verbose)
    else:
        sys.exit(0 if run_offline(args.verbose) else 1)
