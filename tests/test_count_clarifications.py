"""
C9 v2 — clarification counting, text vs voice.

This is the only part of talk mode that can silently make an interview unfair,
so it gets a table rather than a smoke test. Run:

    python -m tests.test_count_clarifications

No pytest dependency on purpose — the backend has no test runner wired up and
this must be runnable in any environment that can import the service.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imported from the PURE module, not from services.interview_engine. That module
# builds an OpenAI client at import time and raises without OPENAI_API_KEY, so
# importing it here would mean a real API key, fastapi and supabase were all
# required to check whether "does that make sense?" counts as a clarification —
# making the one piece of logic that can silently make an interview unfair the
# hardest piece to run. This test needs nothing but the standard library.
# interview_engine re-exports the same function, so callers are unaffected.
from services.clarification_counter import count_clarifications  # noqa: E402


# (label, text, kind, expected)
CASES = [
    # --- VOICE: the whole reason C9 v2 exists ----------------------------
    (
        "spoken structure with rhetorical marks spends nothing",
        "So I'd size this top-down, does that make sense? I'll assume urban "
        "households only, is that fair? Then I'd split by income band, right?",
        "voice", 0,
    ),
    (
        "one genuine data request spends one",
        "What's the client's current market share?",
        "voice", 1,
    ),
    (
        "genuine request plus two confirmations still spends one",
        "What's their current market share? Does that make sense? Shall I continue?",
        "voice", 1,
    ),
    (
        "five stacked requests are clamped to one",
        "What's the market share? What's the price point? Who are the competitors? "
        "What's the cost base? What's the time horizon?",
        "voice", 1,
    ),
    (
        "pure narration with no questions spends nothing",
        "I'd break revenue into volume times price, then look at the cost side.",
        "voice", 0,
    ),
    (
        "trailing confirmation tacked onto a statement spends nothing",
        "Let me take you through the structure, does that make sense",
        "voice", 0,
    ),
    (
        "flat-intonation request with no question mark still counts",
        "What is the client's cost base",
        "voice", 1,
    ),
    (
        "bare acknowledgement spends nothing",
        "Right?",
        "voice", 0,
    ),
    (
        "empty turn spends nothing",
        "",
        "voice", 0,
    ),

    # --- TEXT: legacy behaviour must be byte-for-byte unchanged ----------
    (
        "text: every question mark still counts (anti-packing)",
        "What's the market share? What's the price point? Who competes?",
        "text", 3,
    ),
    (
        "text: rhetorical marks still count — typing them is deliberate",
        "So I'd size this top-down, does that make sense? Is that fair?",
        "text", 2,
    ),
    (
        "text: interrogative opener with no mark counts as one",
        "What is the client's cost base",
        "text", 1,
    ),
    (
        "text: plain structure spends nothing",
        "I'd break revenue into volume times price.",
        "text", 0,
    ),
    (
        "text: default kind argument behaves as text",
        "What's the market share? What's the price?",
        None, 2,
    ),
]


def main() -> int:
    failures = []
    for label, text, kind, expected in CASES:
        got = count_clarifications(text) if kind is None else count_clarifications(text, kind)
        ok = got == expected
        print(f"{'PASS' if ok else 'FAIL'}  [{kind or 'default'}] {label}: expected {expected}, got {got}")
        if not ok:
            failures.append(label)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        return 1
    print(f"All {len(CASES)} cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
