# MECE Interviewer — Call Graph (audit, before and after)

Audit date: 2026-09-29. Baseline commits: backend `12308ec` (main), frontend `f9841b6` (main).
Audit was read-only; no code was modified during it.

## 1. As found (baseline) — who decides what the interviewer says

```
                                   ┌───────────────────────────────────────────────┐
 TEXT (typed / dictated)           │ backend routes/attempts.py                    │
 ConversationalSolve.send('text')  │  POST /attempts/{id}/messages  (SSE)          │
   └─ lib/interview-api            │   _enter_turn/_settle_deferred                │
      postMessageStream ──────────►│   _gather(load attempt, budget, transcript)   │
                                   │   count_clarifications (C9)                   │
                                   │   INSERT user row                             │
                                   │   stream_interviewer_reply(channel='text') ───┼──► services/interview_engine.py (V11)
                                   │   INSERT assistant row unless V11 SILENCE     │      compute_signals        (session_signals.py)
                                   │   _after_turn(_fold_session_state)            │      needs_contextual_assessment
                                   └───────────────────────────────────────────────┘      assess_context_with_llm (gpt-4o-mini, serial)
                                                                                            evaluate_intervention_gate (interviewer_decision.py)
 STT / "pipeline talk mode"                                                                    SILENCE  -> yield ""  (+ ai_usage row model=local)
 VoiceInterview.tsx                                                                            PRESENCE -> canned phrase by md5(msg) % n
   Vad (lib/voice/vad.ts) ─► MediaRecorder ─► POST /transcribe (Groq whisper / whisper-1)       SUBSTANTIVE -> build_adaptive_interviewer_messages
   └─► parent send('voice') ─► same /messages path as TEXT (channel 'text')                                 (prompts/interview_prompts_v2.py)
   └─► token sink ─► TtsQueue ─► POST /speak (OpenAI tts / Google WaveNet)                                 gpt-4o-mini (INTERVIEWER_ADAPTIVE_MODEL)
                                                                                                          enforce_mode -> _MODE_FALLBACK canned text
 REALTIME VOICE (OpenAI)
 VoiceInterviewRealtime.tsx
   startRealtimeSession ─► POST /realtime/session (routes/realtime.py)
        mint client secret: gpt-realtime-2.1, VOICE_RENDERER_INSTRUCTIONS,
        semantic_vad create_response=false interrupt_response=true, whisper-1 transcription
   WebRTC ─► api.openai.com/v1/realtime/calls   (audio browser<->OpenAI)
   data channel 'oai-events':
     conversation.item.input_audio_transcription.completed
        └─► handleUserTurn ─► POST /attempts/{id}/voice-decision ─► stream_interviewer_reply(channel='voice') (V11)
               └─► say(line) ─► response.create {instructions: "SAY: <line>"}  (DEFAULT conversation)
     response.output_audio_transcript.done ─► POST /attempts/{id}/realtime-turn (assistant row)
     user row ─► POST /attempts/{id}/realtime-turn (after decision; no idempotency key)

 REALTIME VOICE (Gemini Live, admin option)
 VoiceInterviewGemini.tsx ─► POST /realtime-gemini/session ─► WS to Google
   GeminiTurnGate discards Gemini's own replies; V11 decides via /voice-decision (early/deferred fold);
   approved line sent as realtimeInput.text "SAY: <line>"; turns persisted via /realtime-turn
```

Scoring (unchanged by this work): `POST /attempts/{id}/submit` → `score_conversation` →
`_score_case_conversation` (gpt-4o, CONVERSATION_SCORING_SYSTEM_PROMPT) or
`score_guesstimate_answer` (+ deterministic backstop). The scorer reads `attempt_messages`.

## 2. Findings from the audit (evidence: file:line at baseline)

| # | Finding | Where | Brief section violated |
|---|---------|-------|------------------------|
| F1 | Text channel can never be silent: the gate's fallback returns `PRESENCE/ACKNOWLEDGE` "text_fallback_prevent_blank". | `services/interviewer_decision.py:221-223` | §5 Gate B, §6, §33 |
| F2 | Any `?` in a non-substantive turn → `HAND_BACK` ("Go ahead.") — a genuine question can get "Go ahead." | `interviewer_decision.py:182-183` | §17 |
| F3 | `has_work` (any number or any turn > 60 chars, ever) → `ACKNOWLEDGE` → presence on almost every turn. | `interviewer_decision.py:191-192`, `session_signals.py:249` | §5 "Do NOT use candidate has work -> always acknowledge", §57G |
| F4 | Presence wording picked by `md5(message) % n` — mechanical, context-free. | `interviewer_decision.py:253-285` | §25, §46 |
| F5 | When the model returns an empty/over-stripped reply, a canned, case-agnostic "answer" is substituted (`"The data confirms that."`, `"In short: work the cost side first..."`). | `interviewer_decision.py:407-432` | §40 "Never fabricate a substantive interviewer answer" |
| F6 | Silence represented as `yield ""`; route infers it from `control_out`. Also logs an `ai_usage_log` row (`model="local"`) for every silent/presence turn. | `interview_engine.py:113-153` | §6 explicit event, §39 |
| F7 | A contextual-assessor LLM call runs serially before every "hypothesis"/"clarification" turn (any `?` or first word interrogative). | `interview_engine.py:106-108`, `session_signals.py:362-372` | §39, §63 |
| F8 | `detect_intent` treats substring `"help"` (e.g. "this will help margins") and `" stuck"` as help requests; `"leave"`/`"skip"` as wanting to stop — so the brief's own example "leave 20% for rural" is classified `wants_to_stop` and routed to a TRANSITION. | `session_signals.py:11-36`, `interviewer_decision.py:148-149` | §3, §16, §47A |
| F9 | Clarification-exhausted directive (C9) is attached to ANY turn containing `?`, including hint requests ("can you give me a hint?") → help can be declined. | `routes/attempts.py:788-801`, `interview_prompts_v2.py:176-181` | §8 |
| F10 | V11's adaptive prompt has an "IDENTITY LOCK: You are a human interviewer. Never confirm... that you are an AI". | `prompts/interview_prompts_v2.py:65-69` | honesty; §37 |
| F11 | Realtime `response.create` runs in the DEFAULT conversation: the voice model sees the whole session audio on every line (cost grows with session length; model can react to candidate audio). | `lib/voice/realtime-session.ts:223-227` | §23, §39 |
| F12 | Barge-in relies only on server VAD; no client `response.cancel`/`output_audio_buffer.clear`; a decision that returns after the candidate started speaking again is still spoken (only a turn-sequence check). | `realtime-session.ts:158-162`, `VoiceInterviewRealtime.tsx:176-183` | §26, §57J |
| F13 | `/realtime-turn` has no idempotency key; a retried or reconnect-duplicated save lands twice. Transcription `item_id` is ignored. | `routes/attempts.py:1081-1186`, `realtime-session.ts:138-142` | §32, §57L |
| F14 | V11's own unit suites fail on baseline: `tests.test_session_signals` 5 failures, `tests.test_interviewer_mode` 24 failures (run with dummy env). | tests | §47 |
| F15 | Prep Copilot v2 keeps an isolated copy of the engine whose NON-adaptive branch still builds the "EXAMINE, DO NOT TEACH / NO HINTS / That's the exercise - what's your next step?" prompt. Unreachable while `ADAPTIVE_INTERVIEWER` is pinned on in `main.py`, but reachable code. | `services/copilot/engine/prompts_interview.py:33-185` | §61, invariant 9 |
| F16 | `cases` table may be world-readable including the `solution` column if `supabase/daily-read-policies.sql` ("cases readable ... using (true)") was applied. Not verifiable from here (no DB access). | `supabase/daily-read-policies.sql:13-14` | §36 |
| F17 | Realtime input transcription uses `whisper-1` (batch after commit) — the largest avoidable contributor to T1−T0 on the realtime path. | `routes/realtime.py:213` | §30 |

Duplicate interviewer logic: `services/copilot/engine/*` (isolated copy for Prep Copilot, flag-gated).
Dead/unreachable: `prompts/interview_prompts.build_interviewer_messages` (fail-closed stub);
`interviewer_mode.select_mode` (legacy shim). Hidden dependency: `main.py` force-sets
`ADAPTIVE_INTERVIEWER=true` at import, which the copilot engine also reads.

## 3. After this change (flag `INTERVIEWER_BRAIN=on` or allow-listed user)

```
 TEXT  ─► POST /attempts/{id}/messages {content, kind, channel:'text', turn_id?}
 STT   ─► POST /attempts/{id}/messages {content, kind:'voice', channel:'stt', turn_id?}
 VOICE ─► POST /attempts/{id}/voice-decision {content, turn_id=item_id, is_partial}
              │
              ▼
   routes/attempts.py  (auth, rate limit, budget, cap, C9 count, persistence, SSE)
              │   brain_enabled(user) ?
              ▼
   services/interviewer/engine.py  run_turn()          ◄── ONE decision system
       dedupe.TurnLedger (attempt_id, turn_id) ──► replay, never re-decide
       classify.extract_signals()   (deterministic, no model)
       numbers.check_claims()       (deterministic arithmetic / anchor sanity)
       policy.decide()              Gate A (substantive) → Gate B (presence) → NO_OUTPUT
           └─ assessor (optional small-model JSON call, only for step-completing analytic turns, timeout-bounded)
       state_machine.advance()      (validated transitions, hint ladder, loop guards)
       presence.wording()           (deterministic, context-chosen, no model)
       responder.generate()         (only SUBSTANTIVE moves that need case-aware wording)
           providers.InterviewerLLM (resolve_llm('interviewer'), one provider failover, timeouts)
           validate.OutputValidator (praise/refusal/leak/question-budget/markdown; never fabricates)
       telemetry.emit()             (one JSON line per turn, no content, no secrets)
              │
              ▼
   Decision.lane ∈ {NO_OUTPUT, PRESENCE, SUBSTANTIVE}
     TEXT/STT: NO_OUTPUT → SSE `event: silence` + `done {message_id:null, silent:true}`; no row, no TTS
               PRESENCE/SUBSTANTIVE → tokens → assistant row
     VOICE:    NO_OUTPUT → {lane:'SILENCE', say:null}; client sends nothing to the voice model
               otherwise → {lane, say}; client: out-of-band response.create (conversation:'none')
```

Flag OFF: every route runs exactly the baseline V11 code path (byte-identical V11 files).
