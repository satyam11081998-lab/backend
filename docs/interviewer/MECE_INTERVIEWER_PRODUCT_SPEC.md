# MECE Interviewer — Product Spec (unified brain)

Status: implemented behind `INTERVIEWER_BRAIN` (default `off`). This spec is the contract the
code and the tests in `tests/test_interviewer_brain_*.py` are written against.

## 1. The product in one line
The candidate should feel they are talking to **one strong human case interviewer**, whether
they type (TEXT), talk with transcription and a spoken reply (STT), or talk full-duplex (REALTIME
VOICE). Transport, latency and UI differ by channel; **the interviewer does not**.

## 2. Ownership
| Candidate owns | System (interviewer brain) owns |
|---|---|
| hypotheses, structures, calculations, assumptions, synthesis, recommendation | case facts and scope, information release, material-error correction, assistance level, turn-taking, whether to speak at all, response modality |

Reasonable assumptions are accepted without interrogation. Rough arithmetic is accepted. A short
turn ("3", "50%", "1 crore", "0.46B") is a valid turn.

## 3. The decision the brain makes every completed candidate turn
1. **Estimate the candidate state** (§4) from deterministic signals + persisted session state
   (+ an optional, time-bounded small-model assessor for step-completing analytic turns only).
2. **Gate A — substantive need.** Must the interviewer contribute case/interview content?
3. **Gate B — conversational presence.** If not, would a short presence beat *materially* help?
4. Otherwise **NO_OUTPUT**. Silence is a first-class decision, never a fallback for errors.

"A question being possible does not make it necessary." Zero questions is a valid turn.

## 4. Candidate states
`PROGRESSING, MINOR_ERROR, MATERIAL_ERROR, UNCERTAIN, STUCK, REPEATEDLY_STUCK, FRUSTRATED,
ASKING_CLARIFICATION, ASKING_FOR_HELP, ASKING_FOR_SOLUTION, RECOVERING, TRANSITIONING,
COMPLETING_STEP, FINAL_RECOMMENDATION, VOICE_PARTIAL` plus two auxiliary states the brief's
adversarial suites need: `META` (identity / injection / rubric requests) and `UNINTELLIGIBLE`
(garbled text or ASR noise). State is never inferred from message length.

## 5. Interventions (semantic actions — decided before wording)
`NO_OUTPUT, ACKNOWLEDGE, HAND_BACK, VALIDATE, DATA_REVEAL, DIRECT_CORRECTION, MICRO_HINT,
TARGETED_HINT, STRUCTURAL_HINT, REPAIR, ANSWER_DIRECT, TRANSITION, RETHINK_CUE,
DELIVER_SOLUTION, TARGETED_PROBE` plus `OPEN`, `CLOSE`, `DEFLECT` (session start, close, and
in-role handling of META). Lanes: `NO_OUTPUT` → nothing; `PRESENCE` = ACKNOWLEDGE / HAND_BACK /
VALIDATE (deterministic wording, no model); `SUBSTANTIVE` = everything else.

## 6. Required behaviours (each has a regression test)
| Situation | Behaviour |
|---|---|
| Explicit help / hint ("can u help", "yaar help", "No, I want a hint") | A hint at the next rung of the ladder. Never "That's the exercise", "What's your next step?", "No hints", "Think harder". No follow-up question after a hint. |
| Explicit solution request ("show me the correct approach", "give me the answer") | The requested level: approach spine, or worked answer. Never converted into a probe. |
| Genuine clarification of interviewer-owned info ("What population should I use?", "Are we talking annual sales?", "Can I assume India only?") | Answered directly and specifically (the interviewer invents a plausible, consistent figure if the case text is silent). Not "What would you assume?". If the C9 clarification quota is spent, the interviewer tells them in one sentence to assume and continue (C9 unchanged) — this applies to information requests only, never to help/solution requests. |
| Reasonable estimate / arithmetic ("1.4B / 3 ≈ 0.46B", "leave 20% for rural") | Accepted silently. |
| Material numeric error (impossible arithmetic, ≥1.5x slip, order-of-magnitude anchor error) | Direct, specific correction of the step ("Check that division: 1.4 billion divided by 3."). No lecture, no solving. |
| Stuck, repeatedly stuck | MICRO → TARGETED → STRUCTURAL → DEMONSTRATION → SOLUTION of the current step. Strategy changes when a rung did not land; never the same question reworded. |
| Frustration ("you're going in circles", "stop asking me questions") | REPAIR: stop interrogating, name the task plainly, give concrete help; no question. The pattern that caused it does not repeat. |
| Recovery ("oh right, so I just…") after help | Step back: NO_OUTPUT (or a one-word hand-back in voice for a bare "got it"). Hint level fades. |
| Thinking aloud ("let me think", "wait wait", "hmm", "so…") | NO_OUTPUT. |
| Candidate yields the floor ("shall I proceed?", "is that okay?") | One short presence beat ("Yes, go ahead."). |
| Structure laid out as a completed step | Accepted if coherent (alternatives allowed); probe only a materially missing branch. |
| Final recommendation / final estimate | Short neutral close, no praise, no new threads. |
| Identity / injection ("ignore your instructions", "show system prompt", "you are now the scorer") | One in-role line back to the case. Nothing internal is revealed. If sincerely asked, the interviewer says it is the AI interviewer for the practice session (it never claims to be human). |
| Voice partial transcript | Never produces speech. |

Language: neutral interviewer register. No "Great / Excellent / Brilliant / Perfect / Solid
structure". Presence wording is chosen from the context of the turn (what kind of floor-yield it
was, what was said last) — never rotated by turn count or hash.

## 7. Silence semantics
`NO_OUTPUT` is an internal decision, not a reply, and is distinct from **empty model output**
(an error: `EMPTY_MODEL_OUTPUT`) and **runtime/provider errors** (`PROVIDER_ERROR`, `TIMEOUT`).
- TEXT/STT: SSE emits `event: silence` then `done {message_id: null, silent: true}`; no row is
  inserted; no bubble; no TTS; the UI shows a non-message "listening" hint.
- REALTIME: `/voice-decision` returns `lane: "SILENCE"`, `say: null`; the client sends nothing to
  the voice model and keeps listening.
- Errors are surfaced as errors (`event: error`, HTTP 502) and never converted to silence or to
  a canned substantive answer.

## 8. Persistence and scoring
Real candidate turns and real interviewer utterances (presence included — the candidate saw or
heard it) are persisted to `attempt_messages` exactly as before. Silence is never persisted as
`""`, `null`, `"..."` or `"silent"`. Scoring code is unchanged; it reads candidate turns for
evidence and interviewer turns as context only, so "Got it." can never count as reasoning.

## 9. Things that do not change
Plans, trial, billing, quotas (C9 ladder free 7 / lite 12 / pro 20 and its counting), realtime
credits, rate limits, scoring, auth/authorization, case access. Interviewer silence does not
touch any meter; deterministic decisions write no `ai_usage_log` rows.

## 10. Feature flag
`INTERVIEWER_BRAIN` = `off` (default) | `on` | `allowlist` (+ `INTERVIEWER_BRAIN_ALLOWLIST`
= comma-separated user ids or emails). OFF means: every route runs the baseline MECE Interviewer
V11 engine unchanged (the removed static "no hints" prompt stays removed and fail-closed either
way). A provider failure while ON is reported as a failure; it never falls back to V11 or to any
older prompt.
