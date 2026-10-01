# Interviewer V12 — function first, language second

Date: 2026-10-01. Built on `main` as the local files at `8c6530f` (V10.2), the last version of
the four interviewer files that starts. It replaces the GitHub web edits of 2026-10-01
(`aa8aa43` … `3001a72`), which did not import (`get_modality_instruction` /
`determine_response_function` missing).

```
touches:  services/session_signals.py, services/interviewer_decision.py,
          services/interviewer_mode.py, services/interview_engine.py,
          routes/attempts.py (voice-decision lane for model-worded presence, `function` field,
          fold the function into session_state, telemetry),
          tests/test_response_functions.py (NEW), tests/test_v11_voice_integration.py (1 check),
          tools/try_interviewer.py (NEW, terminal tryout, writes nothing to Supabase)
breaking: no. /voice-decision gains `function`; PRESENCE may now carry a model-worded line
          (event.code "CTX"); the frontend only reads `say`. C9, scoring, plans, billing,
          quotas, auth: untouched.
```

## What changed

Before: `has_work` → presence gate → `ACKNOWLEDGE` → `_FAST_PHRASES` → "Got it." / "Right." for
every turn with work in it, however long.

Now:

```
candidate turn
  → deterministic signals (+ turn shape: turn_type, is_substantive_reasoning, step_completed,
                             asks_to_proceed, numbers_stated, candidate_plan)
  → V10.2 gates, unchanged (help, solution, material error, frustration, meta, clarification,
                            voice partial, silence)
  → policy layer: decide_response() → RESPONSE FUNCTION
  → JSON INTERVIEWER CONTROL PACKET
  → model words the turn → validation → (one regeneration) → plain hand-back if still unusable
```

| Candidate turn | Function | Worded by |
|---|---|---|
| voice partial, mid-calculation in voice, presence cool-down in voice | `NO_OUTPUT` | — |
| "okay", "yeah", "100.", "I'll split households first." | `SHORT_ACK` / `HAND_BACK` | fixed short line (0 tokens) |
| long plan / calculation / estimate | `ACKNOWLEDGE_AND_CONTINUE` | model |
| structure, strategic distinction, hypothesis | `REFLECT_PROGRESS` | model |
| finished a step with a result | `ACKNOWLEDGE_AND_ORIENT` (toward the candidate's OWN plan) | model |
| recovery after a correction | `VALIDATE_AND_HAND_BACK` | model |
| plan + "shall I proceed?" | `ACKNOWLEDGE_AND_CONTINUE` (not a clarification answer) | model |
| help / solution / material error / frustration / meta / question / final number / close | `MICRO_HINT` / `DELIVER_SOLUTION` / `CORRECT_AND_CONTINUE` / `REPAIR_AND_RESET` / `DEFLECT_META` / `ANSWER_DIRECT` / `SANITY_CHECK` / `CLOSE` (same V10.2 routes and instructions) | model |

A substantive turn never reaches the fixed-phrase lane (tested over the 50 behaviour scenarios
and 4,500 fuzzed turns).

## Control packet

`interviewer_decision.build_interviewer_control_packet()` — `interviewer_control` with
`response_function`, `conversational_objective`, `candidate_state` (turn type, gist in the
candidate's own words, numbers, reasoning stage, progress, quality, confidence, working, needs
space, help/solution request, material error, frustration), `interviewer_decision`
(intervention level, question / hint / correction / solution / silence allowed,
`may_confirm_correctness: false`, new case facts allowed), `response_generation` (medium,
length, sentence count, content specificity, must reference the candidate, no over-praise),
`conversation_memory` (recent functions and lines, repetition avoidance, the candidate's own
plan), `alerts`. It holds no sentence to copy and no case solution.

## Validation (layer C)

Model-worded presence lines: control metadata → rejected; questions, solicitations, gush and
"that's correct / you're right / on track" (nothing verified it) → removed; must pick up
something the candidate said; 1 sentence in voice, 2 in text. One regeneration with the reason,
then a plain hand-back (`ctl.fallback = True`, logged `[interviewer] contextual_fallback`).
A provider failure on a presence beat gives the hand-back line, never an error. Substantive
replies additionally pass `scrub_control_leak` (a JSON or field-name reply is reduced to its
spoken line; a streamed reply that opens as JSON is held and scrubbed).

## Cost and latency

Presence beats on substantive turns now make one small model call (`gpt-4o-mini`,
`max_tokens` 90) where they used to cost 0 tokens. Minimal turns, silence and the cool-down
stay at 0 tokens. Live latency and wording quality were NOT measured (no API access from the
build environment) — try it with `python -m tools.try_interviewer` before relying on it.

## Kill switch and rollback

- `INTERVIEWER_CONTEXTUAL_PRESENCE=off` on Render (restart): presence beats go back to the
  fixed short lines; everything else stays V12. Default `on`.
- Full rollback to V10.2 (do NOT `git revert` this commit — that restores the broken web edits):
  `git checkout 8c6530f -- services/session_signals.py services/interviewer_decision.py services/interviewer_mode.py services/interview_engine.py routes/attempts.py tests/test_v11_voice_integration.py`
  then commit and push.

## Gates (run from the backend folder, dummy env is enough)

```
python -m py_compile routes/attempts.py services/interview_engine.py services/interviewer_decision.py services/interviewer_mode.py services/session_signals.py
python -c "import main"
python -m tests.test_response_functions        # ALL PASS
python -m tests.test_v11_voice_integration     # ALL PASS
python -m tests.test_count_clarifications      # 14/14
python -m tests.test_learning_model            # ALL PASS
python -m tests.test_interviewer_mode          # 24 failures — identical output to 8c6530f (pre-existing)
python -m tests.test_session_signals           # 5 failures — identical output to 8c6530f (pre-existing)
```
