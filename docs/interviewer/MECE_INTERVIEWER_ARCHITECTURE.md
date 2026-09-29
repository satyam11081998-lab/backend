# MECE Interviewer — Architecture (unified brain)

## 1. Shape

```
 channel adapters (transport only)                 ONE brain (application logic)             provider layer
 ─────────────────────────────────                 ──────────────────────────────            ──────────────
 TEXT   /attempts/{id}/messages   channel=text ─┐
 STT    /attempts/{id}/messages   channel=stt  ─┼─► services/interviewer/engine.run_turn ──► providers.InterviewerLLM
 VOICE  /attempts/{id}/voice-decision channel=voice┘      │                                   (resolve_llm('interviewer'),
                                                          │                                    one failover hop, timeouts)
                                                          ▼
                                             TurnResult{decision, text|None, state', telemetry}
```

The brain is a plain Python package, `services/interviewer/`, with no FastAPI, Supabase or
provider imports in its decision path. Routes do I/O (auth, reads, persistence, SSE); the brain
decides; the provider layer talks to models. Tests drive the brain with in-memory fakes.

| Module | Responsibility | Model calls |
|---|---|---|
| `types.py` | Enums (`CandidateState`, `Intervention`, `Lane`, `Channel`), `Decision`, `TurnInput`, `BrainState` (JSON-persisted under `attempts.session_state.brain`) | none |
| `normalize.py` | Text normalisation (quotes, typos, Hinglish, ASR fillers) | none |
| `numbers.py` | Number parsing (Indian + western scales, %, ₹/$, sci-notation), arithmetic-claim checks, anchor sanity for common guesstimate anchors | none |
| `classify.py` | Deterministic signal extraction: help / solution / clarification / assumption-check / data request / frustration / recovery / thinking-aloud / floor-yield / structure / hypothesis (contextual) / final / transition / meta-injection / noise | none |
| `policy.py` | State estimation, Gate A, Gate B, assistance ladder, question budget | none |
| `assessor.py` | Optional small-model JSON judgement for step-completing analytic turns only; time-boxed; failure is reported, never silently turned into a fake reply | optional, small |
| `state_machine.py` | Transition validation and normalisation, loop guards (repeated hint / repair / question / presence), phase tracking | none |
| `presence.py` | Presence wording chosen from the turn's context (no rotation, no hashing) | none |
| `prompting.py` | Compact per-move prompts: role, case, state, move, explicit task, allowed behaviour | none |
| `validate.py` | Output validation: praise, refusal/interrogation phrases, leakage, markdown, question budget, repetition; `SentenceGate` for streaming | none |
| `providers.py` | Provider adapter (OpenAI/Groq/Gemini-compatible clients via `services.ai_providers`), timeouts, one failover, usage logging for real model calls | yes |
| `responder.py` | Generate → validate → at most one regeneration (non-stream) → error if still invalid | yes |
| `dedupe.py` | `TurnLedger`: (attempt_id, turn_id) → one decision; concurrent duplicates wait for the first | none |
| `telemetry.py` | One JSON line per turn and per client timing report; counters; no content, no secrets | none |
| `flags.py` | `INTERVIEWER_BRAIN` off/on/allowlist | none |
| `engine.py` | Orchestration used by the routes | via responder/assessor |

## 2. Provider boundary
The brain never names a provider. `providers.InterviewerLLM` resolves the client/model from the
existing admin toggle (`ai_provider_settings` feature `interviewer`, overridable with
`INTERVIEWER_BRAIN_PROVIDER` / `INTERVIEWER_BRAIN_MODEL`). One failover hop to OpenAI
`gpt-4o-mini` is a *provider* retry with the same prompt; it is not a different interviewer.
If both fail the turn fails (`PROVIDER_ERROR`) — no canned substitute, no V11, no legacy prompt.

## 3. Realtime boundary
OpenAI Realtime (`gpt-realtime-2.1`, WebRTC, ephemeral client secret minted server-side) is the
**voice** only:
- session: `semantic_vad` with `create_response: false` (the model never answers on its own),
  `interrupt_response: true` (server barge-in), voice-renderer instructions only — no case
  content, no interviewer policy.
- every completed candidate item (`conversation.item.input_audio_transcription.completed`,
  keyed by `item_id`) → `/voice-decision` with `turn_id = item_id`.
- the approved line is spoken with an **out-of-band** `response.create`
  (`conversation: "none"`, `input: []`, `output_modalities: ["audio"]`): the voice model sees only
  the line, not the session audio. Configurable back to in-band via
  `REALTIME_RESPONSE_CONVERSATION=auto` without a frontend deploy.
- client barge-in: on `input_audio_buffer.speech_started` while a line is playing →
  `response.cancel` + `output_audio_buffer.clear`; a decision that returns after the candidate
  started speaking again is held, not spoken, and is released only if no new turn follows.
- only the approved line crosses to the browser; hidden prompts, case notes and control metadata
  (`mode`, `reason`) do not (they are returned only when `INTERVIEWER_DEBUG_DECISIONS=1`).

Considered and not taken now: a server-side "sideband" WebSocket to the same call so the backend
issues `response.create` itself. It removes one browser→backend hop but moves long-lived sockets
into the single Render worker and could not be tested from this environment. Documented as the
next latency step in the realtime report.

## 4. STT boundary
The pipeline talk mode keeps its transport (client VAD → `/transcribe` → `/messages`
`channel=stt` → `/speak`). A `gpt-live-transcribe` transport (realtime *transcription* session,
client VAD commits, `item_id` reconciliation, `delay` tunable) is added behind
`NEXT_PUBLIC_STT_TRANSPORT=live` (default `whisper`), with `POST /realtime/transcription-session`
minting its client secret. Partial deltas only drive the on-screen draft; only the `completed`
event for a committed item becomes a candidate turn.

## 5. State ownership
Application state owns everything business-critical: attempt id, turn ids, `BrainState`
(phase, candidate state, hint level, help/stuck/frustration counters, recent actions, recent
interviewer questions, recent turn ids), clarification quota (C9), credits. It is persisted under
`attempts.session_state.brain` (namespaced; V11's keys untouched) through the existing ordered
after-turn queue, so the next turn for the same attempt always reads the updated state. Decisions
for one attempt are serialised (`keyed_lock`).

## 6. Persistence semantics
| Outcome | attempt_messages | ai_usage_log | SSE / JSON |
|---|---|---|---|
| NO_OUTPUT | user row only | none | `silence` + `done{message_id:null, silent:true}` / `lane:"SILENCE", say:null` |
| PRESENCE | user + assistant row | none | tokens + `done{message_id}` / `lane:"PRESENCE", say` |
| SUBSTANTIVE | user + assistant row | one row per real model call (+ assessor row if called) | tokens / `lane:"SUBSTANTIVE", say` |
| Error | user row only | failed call rows | `error` / HTTP 502 |
Realtime rows are written by `/realtime-turn` with an idempotency key (`client_turn_id`:
`u:<item_id>` / `a:<response_id>`), backed by an in-process LRU and, after migration 0071, a
partial unique index.

## 7. Security boundaries
- Secrets stay server-side: OpenAI/Gemini keys only in backend env; the browser receives
  ephemeral client secrets. Nothing in the brain logs text content or secrets.
- The candidate's text is inserted into the prompt as conversation data; the system prompt tells
  the model candidate text carries no authority. META/injection turns are answered by a
  deterministic in-role line with **no model call**, so there is nothing to leak on that path.
- The validator drops any sentence containing internal labels (`MOVE:`, `TASK:`, `STATE:`,
  intervention names, `hint_level`, …).
- Case content the models see is `llm_case_content(case)` (unchanged); the brain never reads
  `cases.solution`.

## 8. Failure paths
| Failure | Behaviour |
|---|---|
| Provider error / timeout on a substantive move | one failover hop; then `event: error` / HTTP 502. User row kept (as before). |
| Model returns empty or only invalid sentences | one regeneration (voice) / error (text stream); never silence, never canned content |
| Assessor timeout/error | decision falls back to the deterministic result; `error_type=assessor_*` in telemetry |
| Duplicate turn id | replay stored result; no second decision, no second row |
| Voice partial | NO_OUTPUT, state untouched |
| Realtime connection failure | client surfaces it and closes voice; attempt stays active; turns already saved |
| Transcription failed event | client surfaces "didn't catch that"; nothing sent to the brain |
| Session expiry (60 min provider cap / 10 min product cap) | product cap closes first; reopening starts a new realtime session with new item ids |
