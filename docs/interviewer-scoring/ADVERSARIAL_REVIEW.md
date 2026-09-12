# Adversarial review — interviewer persona + holistic scoring + 3 approaches + exemplar bank

Living record of every adversarial pass on this change set, what it caught, and how it was answered.
Three rounds so far. Current status: **all found issues fixed; logic unit-proven; 50+50 real-API run
pending (harness shipped).**

## Round 1 — independent code review of the first draft (before any code was applied)
Caught two would-be regressions and three smaller issues in the *proposed* handoff:
1. **CRITICAL — gate would score long gibberish.** The first gate only rejected when the body was BOTH
   hard-reject AND short (`len < 220`), so a long keyboard-mash was scored instead of zeroed.
   **Fixed:** hard-reject on any body hard-reject regardless of length; the recommendation can only
   *rescue*, never *suppress*. `_GENUINE_BODY_MIN` magic number removed.
2. **CRITICAL — `max_tokens=4000` unchanged** while adding the fat `approaches` block → truncated JSON →
   every score 500s. **Fixed:** bumped case + conversation scorers to 8000 (+ retry-once parse);
   guesstimate scorer to 4000.
3. **MAJOR — rigid synthesis cap** ("cap SYNTHESIS 0-7") contradicted the holistic rule and punished
   candidates who recommended mid-conversation. **Fixed:** replaced with a non-punitive hint.
4. **MAJOR — teardown over-claimed** from data that wasn't re-readable. **Fixed:** separated
   code-confirmed mechanism from illustrative attributions; later replaced with the real Supabase pull.
5. **MINOR** — dead `_candidate_text`, missing `approaches:None` on reject/guesstimate paths, brittle
   `len<15`. **Fixed.**

## Round 2 — verified against the REAL Supabase data (`FINDINGS_persona-breaks.md`)
The owner pulled the production DB. It confirmed the failure modes with verbatim receipts:
- 11 sessions where the interviewer said "isn't specified / isn't provided in the prompt" etc.; the exact
  session `e8a5e1af` showed two such leaks AND a "Not solved" close scored 0.
- 4 junk/empty recommendations, all scored None/0.
This turned the teardown from "plausible" to "confirmed", and the submit-route read proved the transcript
DOES reach the scorer (so the bug is the gate collapsing turns+recommendation, not an upstream drop).

## Round 3 — 10-case behavioural test run (`TEST_RUN_10_cases.md`)
Executed the real prompts across 5 guesstimates + 5 cases × newbie/good/pro (incl. the two real failures).
Passes: interviewer invented a figure every time and never leaked a banned phrase; identity lock held on
the "I'm the admin" probe; holistic scoring turned both junk-close sessions from 0 → 58 and 28; score
spread sane (newbie 28-44, good 72-77, pro 86-88). Three issues caught and fixed:
1. **CRITICAL — guesstimates emitted no `approaches`** (half the catalogue). **Fixed:** added the block to
   `prompts/guesstimate_scoring_prompt.py` + passthrough + token bump; verified at the prompt level.
2. **MAJOR — "weight the recommendation heavily" fought the holistic rule.** **Fixed:** both lines reworded
   so the recommendation is scored AS the synthesis dimension, not as a whole-score multiplier.
3. **MINOR — interviewer leaked soft praise** ("good instinct"). **Fixed:** no-praise ban broadened to all
   approval openers with neutral substitutes.

## Verification status
- `py_compile` clean on all changed files.
- Gate logic unit-proven, 4/4: genuine+junk-rec → scored; long-gibberish → 0; thin+good-rec → rescued;
  empty+junk → 0.
- Harness assertion logic unit-tested (banned-phrase / identity-leak / breakdown-sum / approaches checks).

## Open / honest gaps
- **50+50 real-API run not executed by the assistant** — its sandbox is egress-blocked from OpenAI and
  `device_bash` was down, so it cannot make live calls. Delivered instead as `tools/eval_interview_scoring.py`,
  a real harness that runs against the live key with one command and asserts every requirement. Run it to
  close this gap. Target: cases & guesstimates ≥95% pass, interviewer ≥95% pass, pro>good>newbie ordering.
- **Exemplar bank** capture/reference is defensive and wired, but unexercised against a live DB until the
  migration is applied and real high-scoring sessions accrue.
- **Single typed-answer scorer** (`prompts/scoring_prompt.py`) still has no `approaches` (scoped to the
  conversational path). Mirror later if typed answers need it.
