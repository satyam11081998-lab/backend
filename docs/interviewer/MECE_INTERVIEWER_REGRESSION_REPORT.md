# MECE Interviewer — Regression Report

Date: 2026-09-30. Branch `feat/unified-interviewer-brain`, rebased onto backend `main` `9e009c5`
(which includes four V11 updates pushed on 2026-09-30: `session_signals.py`, `interviewer_decision.py` ×2,
`interviewer_mode.py`) and frontend `main` `a73be90`.
Labels: **VERIFIED** = reproduced by a test or a run in this work; **PARTIALLY VERIFIED** = verified
with fakes/mocks, not against the live provider; **UNVERIFIED** = not measurable here.

## 1. Baseline vs new brain on the same inputs (decision layer)

Same 17 inputs, same opening transcript ("Hi, take me through how you'd size this."), guesstimate
case. V11 = baseline engine (`compute_signals` + `evaluate_intervention_gate`) on current `main`
`9e009c5`; the probe was also run on `12308ec` and the V11 decisions differ only where noted.
Brain = `services/interviewer` (flag ON). Probe scripts: `baseline_probe.py` / `after_probe.py`
(reproduce with the snippet in §6). **VERIFIED.**

| Candidate turn | Channel | V11 decision (what the candidate got) | Brain decision | Why the change is correct |
|---|---|---|---|---|
| leave 20% for rural | text | SUBSTANTIVE / TRANSITION ("leave" read as "wants to stop") | NO_OUTPUT | a reasonable assumption, not a request to stop (F8) |
| Can you help me? | text | SUBSTANTIVE / HINT | SUBSTANTIVE / MICRO_HINT | unchanged intent; brain adds the ladder |
| help | text | SUBSTANTIVE / HINT | SUBSTANTIVE / MICRO_HINT | unchanged |
| 3 | text | PRESENCE "Right." | NO_OUTPUT | short numeric turn is a valid turn; no mechanical ack (F1, F3) |
| 50% | text | PRESENCE "Right." ("Understood." at `12308ec`) | NO_OUTPUT | same |
| 1.4 billion / 3 is about 0.46B | text | PRESENCE "Got it." ("Alright." at `12308ec`) | NO_OUTPUT | correct rough arithmetic accepted silently |
| What population should I use? | text | SUBSTANTIVE / ANSWER_DIRECT | SUBSTANTIVE / ANSWER_DIRECT | unchanged |
| Are we talking annual sales? | text | SUBSTANTIVE / ANSWER_DIRECT | SUBSTANTIVE / ANSWER_DIRECT | unchanged |
| This is irritating, you're going in circles | text | SUBSTANTIVE / REPAIR | SUBSTANTIVE / REPAIR (0 questions) | unchanged move; brain forbids a question in repair |
| Oh right, so I just divide by household size | text | PRESENCE "Right." | NO_OUTPUT | recovery restraint (57F) |
| I'll start with revenue because it is easier | text | SUBSTANTIVE / DATA_REVEAL | NO_OUTPUT | process rationale is not a business hypothesis (contextual hypothesis) |
| Revenue fell because volume declined | text (guesstimate) | SUBSTANTIVE / DATA_REVEAL | NO_OUTPUT | in a guesstimate there is no case data to reveal; in a profitability case the brain returns DATA_REVEAL (verified separately) |
| This will help us reduce costs | text | SUBSTANTIVE / HINT (substring "help") | NO_OUTPUT | not a help request (F8) |
| Show me the correct approach | text | SUBSTANTIVE / DELIVER_SOLUTION | SUBSTANTIVE / DELIVER_SOLUTION (approach level) | unchanged; level now explicit |
| I think urban households are about 35% | text | PRESENCE "Alright, continue." | NO_OUTPUT | hedged but reasonable; no hand-back needed |
| let me think | voice | PRESENCE "Alright." ("Got it." at `12308ec`) | NO_OUTPUT | thinking aloud (57G) |
| Urban share is maybe 35% so that's 49 crore people | voice | SUBSTANTIVE / ANSWER_DIRECT ("clarification") — at `12308ec`: PRESENCE "Makes sense. Go ahead." | NO_OUTPUT | a correct step, not a question; the candidate keeps the floor |

Profitability-case check (same brain): "Revenue fell because volume declined" → DATA_REVEAL;
"My hypothesis is that costs rose." → DATA_REVEAL; "I'll start with revenue because it is easier"
→ NO_OUTPUT. **VERIFIED.**

## 2. Named failure modes (brief §57 A–L)

| Id | Failure mode | Test(s) | Result |
|---|---|---|---|
| A | Help request refused / turned into "what's your next step?" | `test_A_B_help_is_never_refused_or_turned_into_next_step` (10 phrasings), `test_A_refusing_model_is_blocked_even_when_it_misbehaves` (model forced to refuse → validator drops it → regeneration) | PASS — VERIFIED (with scripted model) |
| B | Hint request → another question | same as A (asserts 0 questions after a hint) + adversarial "No, I want a hint", "hint?" | PASS — VERIFIED |
| C | Reasonable numbers challenged | `test_C_reasonable_numbers_are_not_challenged` + adversarial numeric set (18 formats) | PASS — VERIFIED |
| D | Assumptions interrogated across a run | `test_D_reasonable_assumptions_not_interrogated_across_a_run` | PASS — VERIFIED |
| E | Frustration → more questions | `test_E_frustration_changes_strategy_and_stops_questions` (repair, 0 questions, 2nd repair escalates to solution step) | PASS — VERIFIED |
| F | Recovery met with more teaching | `test_F_recovery_means_restraint` | PASS — VERIFIED |
| G | Acknowledgement on every turn / rhythm | `test_G_no_acknowledgement_on_every_turn_in_any_channel`, `test_no_turn_count_modulo_logic_in_brain_source` | PASS — VERIFIED |
| H | NO_OUTPUT creates an empty bubble / row / TTS | `test_H_no_output_never_creates_an_assistant_message` (routes) + STT browser E2E "TTS requested only for real interviewer lines" + route test `test_no_output_sends_explicit_silence_and_writes_no_assistant_row` | PASS — VERIFIED (browser E2E with fake mic + fake transcribe) |
| I | Realtime model answers on its own | `test_I_realtime_session_cannot_auto_answer` (session config) + browser E2E "session: create_response=false…" + "NO_OUTPUT: the voice model is never asked to speak" | PASS — PARTIALLY VERIFIED (mock realtime peer; live OpenAI behaviour UNVERIFIED) |
| J | Interviewer talks over the candidate / no barge-in | frontend `realtime-turns.test.cjs` (barge-in, held line, stale drop) + browser E2E barge-in and held-line checks | PASS — PARTIALLY VERIFIED (mock peer) |
| K | Interviewer hears itself (echo → turn) | frontend echo tests + browser E2E "echo: the interviewer's own words never reach the brain" | PASS — PARTIALLY VERIFIED (synthetic echo transcript; real acoustic echo UNVERIFIED) |
| L | One turn → two responses (retry, reconnect, duplicate event) | `test_L_same_turn_cannot_produce_two_responses_voice_or_text` + frontend duplicate-item tests + browser E2E "duplicate item: no second decision, no second response" + chaos sim duplicate checks + migration 0071 unique index | PASS — VERIFIED (backend/browser with mock), DB index verified on real Postgres |

## 3. Invariants (brief §48, 1–10)

| # | Invariant | Evidence | Result |
|---|---|---|---|
| 1 | Every channel uses the same brain | routes call `services.interviewer.engine` for text/stt/voice; unit `test_same_turn_same_decision_on_every_channel_except_rendering`; property `test_one_brain_same_decision_on_every_channel` | VERIFIED |
| 2 | NO_OUTPUT ≠ empty model output ≠ error | route tests `test_no_output_sends_explicit_silence_and_writes_no_assistant_row` vs `test_provider_failure_is_an_error_not_silence_and_not_v11`, `test_voice_regenerates_once_then_fails_loudly` | VERIFIED |
| 3 | Explicit help is honoured | A/B tests, adversarial linguistic set, property `test_invariant3_help_is_never_silence_or_refusal` (hypothesis-generated prefixes × phrasings × channels) | VERIFIED |
| 4 | At most one question; zero is valid | property test on validator + `apply_question_rules` tests | VERIFIED |
| 5 | No praise / refusal phrases reach the candidate | validator tests + route test with a praising/refusing model | VERIFIED |
| 6 | Voice partials never speak | `test_voice_partial_never_produces_speech` | VERIFIED |
| 7 | Silence / presence never touch meters | `test_silence_and_presence_never_touch_meters` (no ai_usage_log rows) | VERIFIED |
| 8 | C9 counting unchanged | `test_c9_consumption_is_identical_to_the_baseline_counter` (parametrised, compares with `count_clarifications`) | VERIFIED |
| 9 | Old static "no hints" prompts unreachable | `test_old_prompts_unreachable` (grep of the tree for the removed phrases in reachable code) + copilot stub | VERIFIED |
| 10 | Hidden metadata never visible | `test_control_metadata_never_becomes_visible_text`, debug-only `mode`/`reason`, bundle scan (security report) | VERIFIED |

## 4. Pre-existing bugs found and what was done

| Bug | Where | Action |
|---|---|---|
| Noise guard dropped real short answers: "50%", "3", "yes", "no" were classified as noise in the voice clients, so the brain never saw them | `lib/voice/noise-guard.ts` (baseline) — found by the STT browser E2E | Fixed: digits are never noise; short answer words (yes/no/yep/nope/nah/sure/haan/nahi) kept. Unit test added. VERIFIED |
| V11 unit suites fail on baseline (`test_session_signals` 5 failures, `test_interviewer_mode` 24 failures) | backend tests at `12308ec` | Not fixed (V11 files are frozen by the brief and stay byte-identical). Same counts on this branch — no regression. VERIFIED |
| Canned substantive fallback text when the model returns nothing (F5) | `interviewer_decision.py` | Not reachable with the flag ON (brain raises `EMPTY_MODEL_OUTPUT`); still present in V11 with the flag OFF by design |
| Identity lock "you are a human interviewer" (F10) | `prompts/interview_prompts_v2.py` | Not used by the brain (honest AI identity). V11 prompt left unchanged for flag OFF |

## 5. Baseline suites, flag OFF

| Suite | Baseline (`12308ec`, and re-run on `9e009c5`) | This branch (rebased on `9e009c5`) | Result |
|---|---|---|---|
| `python -m tests.test_v11_voice_integration` | 74 PASS lines, ALL PASS | 74 PASS lines, ALL PASS | no regression — VERIFIED |
| `python -m tests.test_count_clarifications` | 14/14 | 14/14 | VERIFIED |
| `python -m tests.test_learning_model` | ALL PASS | ALL PASS (18 checks) | VERIFIED |
| `python -m tests.test_session_signals` | 5 FAILED | 5 FAILED (same names) | unchanged — VERIFIED |
| `python -m tests.test_interviewer_mode` | 24 FAILURE(S) | 24 FAILURE(S) | unchanged — VERIFIED |
| V11 engine files (`interview_engine.py`, `interviewer_decision.py`, `session_signals.py`, `interviewer_mode.py`, `learning_model.py`, `prompts/interview_prompts_v2.py`) | — | `git diff origin/main HEAD -- <files>` empty (this branch never edits them; the 2026-09-30 upstream V11 updates come in unchanged) | byte-identical to `main` — VERIFIED |

## 6. Reproduce
```
cd consilio-backend
export OPENAI_API_KEY=sk-test SUPABASE_URL=https://x.supabase.co SUPABASE_SERVICE_ROLE_KEY=t
python -m pytest -q tests/test_interviewer_brain_regression.py      # 47 passed
python -m tests.test_v11_voice_integration                          # flag OFF path
cd ../consilio && node --require ./qa/ts-register.cjs --test qa/voice/*.test.cjs
```

## 7. 2026-10-01: contextual presence (function first, language second)

Change: a finished structure, an "is my approach okay?", a hypothesis with a floor-yield and a
long finished step in voice no longer get a stock line ("Okay. Take it from there.", "That works.
Go ahead.", "Okay." / "Right."). The policy chooses one of three functions
(`REFLECT_PROGRESS`, `ACKNOWLEDGE_AND_CONTINUE`, `ACKNOWLEDGE_AND_ORIENT`); the model words it from a
JSON control packet; the validator rejects stock, unrelated or unverified lines (one
regeneration, then the plain hand-back). Fixed lines remain for context-free beats.

| Turn | Before (2026-09-30 branch) | After |
|---|---|---|
| "I'd split by urban and rural, then income, then ownership. That's my structure." | HAND_BACK "Okay. Take it from there." | REFLECT_PROGRESS, model-worded from their branches |
| "Is my structure okay?" with the assessor unavailable | VALIDATE "That works. Go ahead." (a verdict nothing had checked) | ACKNOWLEDGE_AND_CONTINUE, no verdict allowed |
| long finished step, voice | ACKNOWLEDGE "Okay." / "Right." / "Mm-hm." | ACKNOWLEDGE_AND_CONTINUE from their content; "it holds" only for arithmetic the checker verified |
| "So that's the urban side." (their structure on record) | NO_OUTPUT | ACKNOWLEDGE_AND_ORIENT: names the next part of their plan |
| "Shall I proceed?", "50%", "let me think" | "Yes, go ahead." / silence / silence | unchanged |

Invariant 7 now reads: silence and **fixed** presence never touch meters; a contextual beat makes
one model call (two if regenerated), logged like any model call. All other invariants and A–L are
unchanged and still pass (375 tests). VERIFIED (scripted model); wording quality with a live model
UNVERIFIED.

