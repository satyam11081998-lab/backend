# MECE Interviewer — State Machine

Two machines run per attempt. The **candidate-state** machine classifies each completed turn; the
**interviewer machine** (phase, assistance ladder, frustration, loop guards) is what persists.

## 1. Candidate state per turn (first match wins)

| Order | Condition (deterministic unless noted) | State | Default intervention |
|---|---|---|---|
| 0 | `is_partial` | VOICE_PARTIAL | NO_OUTPUT (state untouched) |
| 1 | identity / injection / rubric / system-prompt request | META | DEFLECT |
| 2 | garbled text (voice: ≥3 words garbled; <3 → NO_OUTPUT) | UNINTELLIGIBLE | ANSWER_DIRECT ("didn't catch that") |
| 3 | explicit frustration | FRUSTRATED | REPAIR (+ next ladder rung of concrete help) |
| 4 | explicit solution / approach request | ASKING_FOR_SOLUTION | DELIVER_SOLUTION (level: approach / full) |
| 5 | explicit help / hint request, or "I don't know / no idea" | ASKING_FOR_HELP | next ladder rung |
| 6 | stuck language without a request, 2nd time in an episode | REPEATEDLY_STUCK | next ladder rung |
| 7 | information request the interviewer owns (scope, data, figures, timeframe, objective) | ASKING_CLARIFICATION | ANSWER_DIRECT or DATA_REVEAL (C9-exhausted → assume-and-continue line) |
| 8 | "can I assume X?" with a numeric anchor check | ASKING_CLARIFICATION | VALIDATE (reasonable) / DIRECT_CORRECTION (gross anchor error) |
| 9 | arithmetic claim off by ≥1.5x, or anchor off by ≥2x beyond range | MATERIAL_ERROR | DIRECT_CORRECTION (step) / RETHINK_CUE (final estimate) |
| 10 | explicit move-on request | TRANSITIONING | TRANSITION |
| 11 | final recommendation / final estimate | FINAL_RECOMMENDATION | CLOSE |
| 12 | first turn / greeting | TRANSITIONING (open) | OPEN |
| 13 | recovery language after assistance / error | RECOVERING | NO_OUTPUT (text) · HAND_BACK for a bare "got it" in voice |
| 14 | thinking aloud / hold ("let me think", "wait", "hmm", trailing "…") | PROGRESSING | NO_OUTPUT |
| 15 | floor yield ("shall I proceed?", "is that okay?", "right?") | UNCERTAIN / COMPLETING_STEP | HAND_BACK or VALIDATE (fixed) |
| 16 | structure presented / validation of own work requested (assessor, optional) | COMPLETING_STEP (or MATERIAL_ERROR / MINOR_ERROR) | REFLECT_PROGRESS · ACKNOWLEDGE_AND_CONTINUE (model-worded; verdict only if vetted) · TARGETED_PROBE · DIRECT_CORRECTION |
| 16b | a stage of the candidate's own plan is finished, plan on record, next part not yet named | COMPLETING_STEP | ACKNOWLEDGE_AND_ORIENT (model-worded) |
| 17 | arithmetic claim off by 1.2–1.5x | MINOR_ERROR | NO_OUTPUT (let it stand) |
| 18 | hedged number ("maybe 30%") | UNCERTAIN | NO_OUTPUT |
| 19 | anything else | PROGRESSING | NO_OUTPUT (voice: ACKNOWLEDGE_AND_CONTINUE, worded from their content, after a long completed step if no presence in the last 2 actions) |

`RECOVERING` is only valid after an assistance/correction/stuck turn; otherwise it is normalised
to `PROGRESSING`. `FINAL_RECOMMENDATION` moves the phase to `closed`; later turns are treated as
post-close (UX questions answered, no new case threads).

## 2. Interviewer machine (persisted `BrainState`)

```
phase:   opening → clarifying → structuring → analysis → synthesis → closed
         (monotonic, except clarifying can recur from structuring/analysis without moving phase back)

hint_level (assistance ladder, per episode):
   0 none → 1 MICRO_HINT → 2 TARGETED_HINT → 3 STRUCTURAL_HINT → 4 DEMONSTRATION → 5 DELIVER_SOLUTION
   up:    help request (+1), insistence "no, I want a hint / still stuck" (+2), repeated stuck (+1),
          frustration with help (max(level+1, 2))
   down:  RECOVERING (−1), each PROGRESSING/COMPLETING_STEP turn after that (−1), to 0 = episode closed
   a new help request after the episode closed restarts at 1

frustration: 0..2   explicit frustration → 2; −1 per non-frustrated turn
repairs_in_row: consecutive REPAIRs; ≥2 → the next repair delivers the current step's solution
```

## 3. Loop guards (checked on every decision)
| Guard | Rule |
|---|---|
| presence metronome | never two presence beats in a row unless both answered an explicit floor-yield |
| question streak | if the last two interviewer turns ended in a question, the next move asks none |
| repeated question | a generated question ≥0.8 similar to any of the last 6 interviewer questions is regenerated once, then dropped |
| hint loop | at level 5 the next help request re-delivers the solution for the current step (never "think harder") |
| repair loop | third REPAIR in a row is replaced by DELIVER_SOLUTION(step) |
| frustration | while frustration ≥1: no questions in generated moves, no ACKNOWLEDGE beats |
| oscillation | the same (intervention, reason) pair three times in the last four actions forces the next rung |
| stale state | state carries `turns`; a decision whose state `turns` is older than the stored one is discarded (turn dedupe + serialized decisions make this rare) |

## 4. Invalid transitions (rejected / normalised)
- `VOICE_PARTIAL` never changes persisted state and never yields output.
- `RECOVERING` without prior assistance/error/stuck → `PROGRESSING`.
- `REPEATEDLY_STUCK` requires a prior STUCK/ASKING_FOR_HELP in the open episode → else `STUCK`.
- `closed` phase is absorbing for case threads: a hint/probe after close becomes a short CLOSE-scope answer.
- hint_level can only rise through an assistance intervention; it can never exceed 5 or go below 0.

All of the above are asserted by `tests/test_interviewer_brain_state_machine.py` and exercised by
the chaos simulation (thousands of random sequences; zero invalid transitions allowed).
