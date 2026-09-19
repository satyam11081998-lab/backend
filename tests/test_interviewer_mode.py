"""Offline gate for the interviewer MODE selector.

Every adversarial eval scenario must map to its intended interviewer move, and the
mode-aware reply gate must honour each mode's question budget. Pure deterministic --
NO model call, NO API key -- so it runs in CI and on the local VM.

Run:  python -m tests.test_interviewer_mode
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.session_signals import compute_signals          # noqa: E402
from services.interviewer_mode import select_mode              # noqa: E402
from services.interviewer_decision import enforce_mode         # noqa: E402
from tools.eval_interviewer_behavior import SCENARIOS          # noqa: E402  (lazy live imports)

# Intended interviewer move per scenario id, read off each scenario's `required` behaviour.
EXPECTED = {
    "help_when_stuck_with_work": "HINT",
    "reasonable_assumption_stands": "ACK_ADVANCE",
    "material_unit_error": "SANITY_CHECK",
    "wants_to_stop_frustrated": "CLOSE",
    "wants_solution_outright": "DELIVER_SOLUTION",
    "repair_after_interviewer_repeat": "REPAIR",
    "meta_identity_probe": "DEFLECT_META",
    "scope_question_own_facts": "ANSWER_DIRECT",
    "voice_noise_asr": "NOISE",
    "minor_error_do_not_nitpick": "ACK_ADVANCE",
    "good_answer_should_advance": "ACK_ADVANCE",
    "frustration_change_strategy": "REPAIR",
    "stuck_early_no_work": "HINT",
    "double_count_error": "CORRECT_MATERIAL",
    "rubber_stamp_weak_reasoning": "ACK_ADVANCE",
    "factor_1000_unit_error": "CORRECT_MATERIAL",
    "tiny_arithmetic_slip_ignore": "ACK_ADVANCE",
    "confidently_wrong_penetration": "CHALLENGE_CLAIM",
    "correct_but_uncertain_do_not_derail": "ACK_ADVANCE",
    "contradicts_established_fact": "ACK_ADVANCE",
    "broken_mece_buckets": "CORRECT_MATERIAL",
    "wrong_denominator": "CORRECT_MATERIAL",
    "over_help_risk_still_working": "HOLD_SPACE",
    "repeated_stuckness_escalate": "HINT",
    "recovers_after_hint_step_back": "RELEASE",
    "candidate_re_asks_answered_fact": "ANSWER_DIRECT",
    "semantic_repetition_guard": "REPAIR",
    "asks_same_thing_twice": "HINT",
    "explicit_hint_request": "HINT",
    "solution_request_late": "DELIVER_SOLUTION",
    "explain_why_meta": "ANSWER_DIRECT",
    "product_ux_question": "ANSWER_DIRECT",
    "own_facts_growth_rate": "ANSWER_DIRECT",
    "own_facts_competitors": "ANSWER_DIRECT",
    "valid_alt_customer_journey": "ACK_ADVANCE",
    "valid_alt_supply_first": "ACK_ADVANCE",
    "valid_alt_top_down_sizing": "ACK_ADVANCE",
    "productive_thinking_pause": "HOLD_SPACE",
    "thinking_out_loud_self_correct": "HOLD_SPACE",
    "keyboard_mash_noise": "NOISE",
    "done_take_final_answer": "CLOSE",
    "small_talk_after_close": "CLOSE",
    "too_many_questions_back_off": "HOLD_SPACE",
    "wants_to_move_on": "ACK_ADVANCE",
    "help_after_good_work": "HINT",
    "challenge_confident_wrong_80pct": "CHALLENGE_CLAIM",
    "accept_well_defended_assumption": "ACK_ADVANCE",
    "sanity_check_absent": "SANITY_CHECK",
    "greeting_kickoff": "OPEN",
    "strong_synthesis_close": "CLOSE",
}

# A 0-budget mode's reply must never contain a question after the gate runs.
_ZERO_BUDGET = {"CLOSE", "HOLD_SPACE", "RELEASE", "DELIVER_SOLUTION"}


def test_mode_mapping() -> int:
    bad = 0
    for s in SCENARIOS:
        sig = compute_signals(s["transcript"], s["new"], "coached")
        mode, _instr, _q = select_mode(sig, "coached", s["new"])
        want = EXPECTED.get(s["id"])
        if want is None:
            print(f"NO-EXPECTATION  {s['id']}  (got {mode})")
            bad += 1
        elif mode != want:
            print(f"MISMATCH  {s['id']:<34} got {mode:<16} want {want}")
            bad += 1
    print(f"mode mapping: {len(SCENARIOS)} scenarios, {bad} mismatch(es)")
    return bad


def test_gate_budgets() -> int:
    """The gate must strip every question/solicitation from a zero-budget reply and
    must never blank a reply (fallback), and must keep <=1 question otherwise."""
    bad = 0
    probes = [
        ("CLOSE", "Let's wrap up here. Please share your final thoughts on the case."),
        ("CLOSE", "Could you break down the impact on profitability first?"),
        ("HOLD_SPACE", "Sure, take your time. What comes to mind first?"),
        ("RELEASE", "Exactly! Now what would you estimate the total orders to be?"),
        ("DELIVER_SOLUTION", "First, what actions could the company take to improve profitability?"),
    ]
    for mode, reply in probes:
        out = enforce_mode(reply, mode, 0)
        if "?" in out or not out.strip():
            print(f"GATE 0-budget FAIL [{mode}] -> {out!r}")
            bad += 1
    # a 1-budget mode keeps a single question
    out1 = enforce_mode("Now, can you quantify the impact on profitability?", "ACK_ADVANCE", 1)
    if out1.count("?") != 1:
        print(f"GATE 1-budget FAIL -> {out1!r}")
        bad += 1
    print(f"gate budgets: {bad} failure(s)")
    return bad


if __name__ == "__main__":
    total = test_mode_mapping() + test_gate_budgets()
    print("\nRESULT:", "ALL PASS" if total == 0 else f"{total} FAILURE(S)")
    sys.exit(1 if total else 0)
