# MECE Interviewer — Test Report (summary of all evidence)

Date: 2026-09-30. Branches `feat/unified-interviewer-brain` in both repos, rebased onto
backend `main` `9e009c5` and frontend `main` `a73be90`. Everything below was run on the rebased
code unless marked otherwise.

Labels: **VERIFIED** — reproduced here by a test or run; **PARTIALLY VERIFIED** — verified with
fakes or mocks standing in for a provider or the browser's far end; **UNVERIFIED** — could not be
measured here (no network route to OpenAI/Groq/Supabase, no keys, no physical devices).

## 1. Suites and results

| # | Suite | What it exercises | Command | Result | Label |
|---|---|---|---|---|---|
| 1 | Brain unit | normalisation, numbers, classification, policy, ladder, state machine, validator, presence, assessor contract, cross-turn scale-slip trigger | `pytest tests/test_interviewer_brain_unit.py` | 173 passed | VERIFIED |
| 2 | Route contracts | real FastAPI routes ↔ brain ↔ fake provider ↔ fake DB: silence SSE, presence rows, streaming, refusal blocking, provider failure, dedupe, C9, namespaced state, voice contract, regeneration, early-fold, `/realtime-turn` idempotency (in-process + DB-level + pre-migration), telemetry, flag OFF → V11, allowlist, scoring input | `pytest tests/test_interviewer_brain_routes.py` | 24 passed | VERIFIED |
| 3 | Named regressions (§57 A–L, invariants 1–10) | see the regression report | `pytest tests/test_interviewer_brain_regression.py` | 47 passed | VERIFIED |
| 4 | Property-based (hypothesis, 400 examples each) | well-formed decisions for any text; invariants 2–6; state machine for any sequence; validator contract | `pytest tests/test_interviewer_brain_properties.py` | 9 passed | VERIFIED |
| 5 | Diversified / adversarial | 17 case types, 6 personas, linguistic/numeric/emotional/noise diversity, 50-turn drift, transition graph, 14 injection attacks, hidden-solution leak, duplication | `pytest tests/test_interviewer_brain_adversarial.py` | 101 passed | VERIFIED |
| | **Backend brain total** | | all five files | **354 passed, 0 failed** | VERIFIED |
| 6 | Chaos simulation | 3,000 random 25-turn sessions through the real routes with fault injection | `python -m tools.interviewer_chaos_sim --sequences 3000 --turns 25 --seed 20260929` | 75,000 requests (8,560 duplicate turn ids, 7 injected model fault modes), **0 violations**, 230 s | VERIFIED (fakes) |
| 7 | V11 (flag OFF) | the unchanged baseline engine | `python -m tests.test_v11_voice_integration` · `test_count_clarifications` · `test_learning_model` · `test_session_signals` · `test_interviewer_mode` | ALL PASS (74 checks) · 14/14 · ALL PASS · 5 FAILED · 24 FAILURES — **the same failures as on `main`** (pre-existing, F14; V11 files are byte-identical to `main`) | VERIFIED |
| 8 | Frontend unit | realtime turn controller (12), live-transcription assembler + SSE parsing (7) | `node --require ./qa/ts-register.cjs --test qa/voice/*.test.cjs` | 19/19 | VERIFIED |
| 9 | Realtime browser E2E | headless Chromium + fake mic + werift WebRTC mock peer + real routes | `node qa/e2e-voice/run.cjs` | 24/24 (incl. 30 repeated barge-ins) | PARTIALLY VERIFIED |
| 10 | STT browser E2E | headless Chromium + fake mic + real VAD + real routes + fake `/transcribe` and `/speak` | `node qa/e2e-voice/run-stt.cjs` | 13/13 | PARTIALLY VERIFIED |
| 11 | Typecheck | whole frontend | `tsc --noEmit -p .` | EXIT 0 | VERIFIED |
| 12 | Production build | whole frontend | `next build` (Google Fonts mocked via `NEXT_FONT_GOOGLE_MOCKED_RESPONSES`; the build host cannot reach Google) | EXIT 0 | VERIFIED |
| 13 | Bundle scan | built `.next/static` | grep for key-shaped / JWT-shaped values and prompt markers | 0 / 0 / 0 | VERIFIED |
| 14 | Migration 0071 | real local Postgres (pgserver) on top of 0002 | apply twice; duplicate insert; NULL keys | idempotent; duplicate rejected; NULL rows unaffected | VERIFIED (not Supabase) |
| 15 | Python compile | every changed backend module | `python -m py_compile …` | EXIT 0 | VERIFIED |
| 16 | Brain overhead | decision + route overhead, N = 200 / 60 per condition | `python -m tools.brain_overhead_bench` | decision P95 ≤ 0.52 ms; route to first event P95 ≤ 4.9 ms | VERIFIED (in process) |
| 17 | Load / concurrency | real uvicorn worker in its own process, simulated 650 ms model, 5–100 concurrent candidates | `python -m tools.interviewer_load_test --users 5 10 25 50 100 --turns 10` | 0 errors, isolation OK, 0 duplicate rows at every level; saturates at ~76–88 req/s on 2 vCPU (see load report) | VERIFIED (local) |
| 18 | Cost model | scripted 10/20-turn interviews, call and token counts | `python -m tools.interviewer_cost_model` | see cost report | PARTIALLY VERIFIED |
| 19 | Runtime trace | 11 requests through the real routes with telemetry | trace script | see runtime trace | VERIFIED (fakes) |

## 2. Not verified (and why)

| Item | Why not | Instrument provided |
|---|---|---|
| Live model wording quality (hints, answers, repairs, solutions) | no provider access | human review of real sessions; `[interviewer.turn]` violations field |
| End-to-end latency on real providers, networks and devices (P50/P90/P95, N≥30) | no provider access, no devices | `tools/voice_latency_report.py`, `tools/text_latency_probe.py`, client telemetry |
| OpenAI Realtime honouring `create_response:false`, out-of-band audio playback, `response.cancel` timing | no provider access | realtime report §4 checklist |
| `gpt-live-transcribe` transport | no provider access | STT report §5 |
| iOS Safari / Android / Bluetooth / real acoustic echo | no devices | device matrix in the latency report |
| `cases.solution` readability through Supabase RLS (F16) | no DB access | SQL in the security report |
| Real spend | no provider access | `ai_usage_log` by endpoint/model |
| Production load | not authorised | load report §4 |

## 3. Bugs found by the testing and fixed

| Found by | Bug | Fix |
|---|---|---|
| STT browser E2E | noise guard dropped "50%", "3", "yes", "no" (pre-existing) | digits and short answer words are never noise |
| Load test | a candidate asking for the same hint twice got an error (verbatim repeat dropped by the validator → empty output) | repeated line is recorded, not dropped; regression test added |
| Chaos | a retried turn whose user row already existed was counted against C9 twice | no second C9 count when the row already exists |
| Route tests | a failed hint still persisted its decision-time state | error path restores the pre-turn state |
| Property tests | "ok help" not detected; validator let "?0" through | lead-in prefix; count every "?" |
| Adversarial | "provide a hint", "wat … shud", "Still not helping." missed; "How would you solve this?" → full solution; "What population should I use?" → assumption check | classifier fixes, each with a test |
| Leak test (strengthened) | (no bug) test now builds the prompt from a real `cases` row that carries a solution | — |
| Chaos harness | TestClient kept every per-request event loop alive (harness only) | one context-managed client per sequence |
| Load harness | server and client sharing a GIL distorted the 50/100 levels | server in its own process |


## 4. Final human-quality walk-through (decision layer)

A 20-turn spoken guesstimate ("new cars sold in India per year"), decided by the shipped brain on
the VOICE channel. Deterministic lines are shown verbatim; model-worded lines are shown as the move
the model is given, because no live model was reachable (their wording is UNVERIFIED). The
assessor is a stub that answers like a correct reviewer: it flags turn 16 and passes everything
else.

| # | Candidate (spoken) | Brain decision | What the candidate hears |
|---|---|---|---|
| 1 | Hi | SUBSTANTIVE / OPEN (`session_open`) | Hi. Take a moment with the prompt, then tell me how you'd go about the estimate. |
| 2 | What population should I use? | SUBSTANTIVE / ANSWER_DIRECT (`clarification`) | *(model-worded ANSWER_DIRECT, 0 question max)* |
| 3 | Can I assume India only? | SUBSTANTIVE / ANSWER_DIRECT (`assumption_check`) | *(model-worded ANSWER_DIRECT, 0 question max)* |
| 4 | let me think | NO_OUTPUT / NO_OUTPUT (`thinking_aloud`) | — (silence; keeps listening) |
| 5 | I'd split by urban and rural, then income, then ownership. That's my structure. | PRESENCE / HAND_BACK (`structure_complete`) | Okay. Take it from there. |
| 6 | Shall I proceed? | PRESENCE / HAND_BACK (`floor_yield`) | Yes, go ahead. |
| 7 | Urban is 35%, so about 11 crore households. | NO_OUTPUT / NO_OUTPUT (`progressing`) | — (silence; keeps listening) |
| 8 | Can you give me a hint? | SUBSTANTIVE / MICRO_HINT (`help_requested`) | *(model-worded MICRO_HINT, 0 question max)* |
| 9 | Oh right, affordability. | PRESENCE / HAND_BACK (`recovered_short`) | Go ahead. |
| 10 | So maybe 25% of urban households. | NO_OUTPUT / NO_OUTPUT (`hedged_progressing`) | — (silence; keeps listening) |
| 11 | 50% | NO_OUTPUT / NO_OUTPUT (`progressing`) | — (silence; keeps listening) |
| 12 | Hmm | NO_OUTPUT / NO_OUTPUT (`filler_only`) | — (silence; keeps listening) |
| 13 | That's 2.7 crore households. | NO_OUTPUT / NO_OUTPUT (`progressing`) | — (silence; keeps listening) |
| 14 | Do we have data on replacement cycles? | SUBSTANTIVE / DATA_REVEAL (`data_requested`) | *(model-worded DATA_REVEAL, 0 question max)* |
| 15 | Okay, 7 years. | NO_OUTPUT / NO_OUTPUT (`progressing`) | — (silence; keeps listening) |
| 16 | So 4 lakh a year from urban. | SUBSTANTIVE / DIRECT_CORRECTION (`assessor_arithmetic`) | *(model-worded DIRECT_CORRECTION, 0 question max)* |
| 17 | Rural adds maybe 10 lakh. | NO_OUTPUT / NO_OUTPUT (`hedged_progressing`) | — (silence; keeps listening) |
| 18 | I'm not sure about first-time buyers. | SUBSTANTIVE / MICRO_HINT (`stuck_signal`) | *(model-worded MICRO_HINT, 0 question max)* |
| 19 | Say 25 lakh. | NO_OUTPUT / NO_OUTPUT (`progressing`) | — (silence; keeps listening) |
| 20 | So my final estimate is 40 lakh cars a year. | SUBSTANTIVE / CLOSE (`final_recommendation`) | Okay, noted. If that's your final estimate, submit it and the debrief will walk through the numbers. |

Read as a human interviewer would:

- **It stays quiet while the candidate works** (11 of 20 turns silent: thinking aloud, "50%",
  "Hmm", correct steps). V11 on the same script spoke on all 20 turns (14 presence beats).
- **It answers what it owns** (population, scope, replacement-cycle data) directly, with no
  question back.
- **It helps when asked or visibly stuck** (turns 8 and 18) and then gets out of the way — the
  candidate's "Oh right, affordability." gets a two-word hand-back, not more teaching.
- **Floor-yields get one short beat** ("Yes, go ahead."), never a probe.
- **It catches the one material slip** (turn 16: 2.7 crore households on a 7-year cycle is ~39
  lakh a year, not 4 lakh) — found during this review. The first version of the brain let it pass
  because the arithmetic was spread across three turns; the cross-turn scale-slip trigger was added
  for it (it asks the assessor, and only a "material" verdict corrects). Regression tests added.
- **It closes neutrally** on the final estimate, without praise and without opening a new thread.
- Weak spots a reviewer might still debate: turn 18 ("I'm not sure about first-time buyers")
  gets a micro-hint where some interviewers would wait one more turn; turn 5's "Okay. Take it from
  there." is a presence beat a text interviewer would not need (voice only).
