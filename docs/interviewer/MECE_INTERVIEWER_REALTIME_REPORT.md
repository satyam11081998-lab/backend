# MECE Interviewer — Realtime Voice Report

Date: 2026-09-29. Channel: REALTIME VOICE (OpenAI Realtime over WebRTC, model from
`REALTIME_MODEL`, expected `gpt-realtime-2.1`). No live OpenAI session was possible from this
environment (no network egress to api.openai.com, no key); everything below that depends on the
live provider is labelled UNVERIFIED.

## 1. Design (what changed and why)

| Concern | Baseline (`f9841b6` / `12308ec`) | Now | Status |
|---|---|---|---|
| Who decides a turn | V11 via `/voice-decision` | unified brain via `/voice-decision` (flag ON); V11 when OFF | VERIFIED (routes + browser E2E) |
| Model auto-answer | `semantic_vad` `create_response:false` | unchanged; asserted in regression test and in the browser E2E session config | VERIFIED (config) / live UNVERIFIED |
| How a line is spoken | in-band `response.create` in the DEFAULT conversation: the voice model saw the whole session audio every line (F11) | **out-of-band** `response.create` with `conversation:"none"`, `input:[]`, `output_modalities:["audio"]`, `metadata.mece_turn=<item_id>`; backend can switch back with `REALTIME_RESPONSE_CONVERSATION=auto` (sent to the browser in the session response, no frontend deploy) | VERIFIED (payload captured in E2E) / live audio playback of out-of-band responses UNVERIFIED |
| Turn identity | turn sequence counter; `item_id` ignored (F13) | `item_id` is the turn id end to end: decision (`turn_id`), candidate row (`u:<item_id>`), interviewer row (`a:<item_id>`) | VERIFIED |
| Duplicate / replayed transcription events | could double-decide | controller ignores a seen `item_id`; backend ledger replays a seen `turn_id`; DB unique index after 0071 | VERIFIED |
| Barge-in | server `interrupt_response` only (F12) | + client `response.cancel` (by response id) + `output_audio_buffer.clear` when `speech_started` arrives while interviewer audio is active; interruption latency measured and reported | VERIFIED (mock peer) |
| Decision returns after the candidate started talking again | spoken anyway (only a sequence check) | **held**; released only if the candidate stops and no new transcript follows within 1.2 s; dropped if a newer turn arrives or it is older than 8 s | VERIFIED (unit + E2E) |
| Echo (interviewer's own words transcribed as a turn) | `isEchoOfLine` in the Gemini path only | same filter applied in the realtime controller before any decision request | VERIFIED (synthetic) / real acoustic echo UNVERIFIED |
| NO_OUTPUT | V11 SILENCE → nothing said | `lane:"SILENCE", say:null` → nothing sent to the model; candidate row saved; no assistant row | VERIFIED |
| Persistence order | save after speaking | unchanged ("speak first, save after"), via `SaveQueue` with idempotency keys | VERIFIED |
| Input transcription model | `whisper-1` hard-coded (F17) | `REALTIME_TRANSCRIBE_MODEL` env (default still `whisper-1`; switch only after a measured A/B on accuracy + latency) | code VERIFIED / benefit UNVERIFIED |

## 2. Event handling (client, `lib/voice/realtime-session.ts` + `lib/voice/realtime-turns.ts`)

```
input_audio_buffer.speech_started ─► controller.onSpeechStarted ─► if interviewer audio active: bargeIn
                                                                     (response.cancel + output_audio_buffer.clear)
input_audio_buffer.speech_stopped ─► T0 (per-turn clock)
conversation.item.input_audio_transcription.completed {item_id, transcript}
   ─► controller.onTranscriptCompleted: drop if seen / noise / echo; supersede older pending turn
   ─► POST /attempts/{id}/voice-decision {content, turn_id:item_id}                        (T1 → T2)
   ─► controller.onDecision: SILENCE → nothing | superseded → drop | candidate talking → hold | else speak
   ─► say(line, item_id): response.create (out-of-band)                                     (T3)
response.created / output_audio_buffer.started ─► interviewer speaking; analyser detects first audible sample (T4)
response.done ─► assistant row a:<item_id> = the approved line, saved only if audio started
                 (a line cancelled before any audio played is not persisted: the candidate never heard it)
output_audio_buffer.stopped | .cleared ─► speaking ends; interruption_ms reported if a barge-in was pending
conversation.item.input_audio_transcription.failed ─► "didn't catch that" (nothing sent to the brain)
error (response_cancel_not_active, conversation_already_has_active_response ignored) ─► surfaced
```

## 3. Tests

### 3.1 Unit (`qa/voice/realtime-turns.test.cjs`, node:test, controller with an injected clock)
12 tests: one item → one decision (duplicate/replayed transcription ignored); nothing spoken without
an application decision (NO_OUTPUT speaks nothing); barge-in (cancel + clear); a decision arriving
while the candidate talks again is held; held line released after a cough; stale held line dropped;
decision for an older turn never spoken; the interviewer's own line never becomes a turn; Whisper
silence artefacts never become turns; reconnect replay does not duplicate the response; timing
report carries T1..T4 and never text; short numeric turns and yes/no are turns, not noise. Part of
the 19/19 frontend unit run (with 7 STT tests). **VERIFIED.**

### 3.2 Browser end-to-end (`qa/e2e-voice/run.cjs`) — 24/24 PASS (re-run 2026-10-01: 6 consecutive runs, 24/24 each)
Real headless Chromium (`chrome-headless-shell` 141.0.7390.37) with a fake microphone
(`--use-file-for-fake-audio-capture` on a generated speech WAV), the real
`VoiceInterviewRealtime` code bundled with esbuild, a **mock realtime peer** built on `werift`
(real WebRTC: ICE, DTLS, SCTP data channel, RTP audio) standing in for api.openai.com, and the
**real FastAPI routes** (`tools/e2e_voice_backend.py`) with the brain on, an in-memory DB and a
scripted model.

| Check | Result |
|---|---|
| mic permission, real WebRTC session, data channel open; UI reaches Listening | PASS |
| session config: `semantic_vad`, `create_response:false`, `interrupt_response:true` | PASS |
| fake-mic audio reaches the far end (RTP packets received) | PASS |
| NO_OUTPUT ("let me think"): one decision request, **no response.create**, candidate row `u:item_think`, no assistant row | PASS |
| help ("Can you give me a hint?"): exactly one `response.create`, out-of-band (`conversation:"none"`, `input:[]`, audio only), sent after the decision, line is a hint (not a refusal), persisted once as `a:item_help` | PASS |
| duplicate `item_id`: no second decision, no second response | PASS |
| echo of the interviewer's own line: never reaches the brain | PASS |
| barge-in: client sends `response.cancel` + `output_audio_buffer.clear` | PASS |
| barge-in round trip (speech_started sent by the mock → response.cancel received by the mock), **N = 30**: P50 267 / P90 270 / P95 277 ms | PASS — loopback through werift's pure-JS SCTP stack; excludes audio devices, network and the provider. Not a production number. |
| client reaction (event received → `response.cancel` sent), **N = 30**: P50 0.0 / P90 0.1 / P95 0.2 ms; every one of the 30 lines cancelled | PASS |
| harness note (2026-10-01): the mock peer sends no interviewer audio and used to end each response ~0.25 s in, so on a busy box the client's silence detector sometimes (correctly) decided a line had finished before the scripted interruption arrived (28–29/30 in 3 of 6 runs). The mock now holds `response.done` open during the interruption loop, as a real provider streaming a long line would; 6/6 runs then measured 30/30 | fixed (harness) |
| held line: nothing spoken over a talking candidate; stale line dropped for the newer turn | PASS |
| telemetry: per-turn T1/T2 and interruption reports received; no transcript text in any record | PASS |
| no uncaught page errors | PASS |

Run: `E2E_DEPS=<dir with werift, esbuild, playwright-core> BACKEND_DIR=<consilio-backend> PYTHON=<python> CHROME=<chrome-headless-shell> node qa/e2e-voice/run.cjs`.
Results are written to `qa/e2e-voice/last-run.json` (git-ignored).

## 4. What is NOT verified (needs a live run)

| Item | Why it matters | How to verify |
|---|---|---|
| OpenAI honours `create_response:false` for every turn in `gpt-realtime-2.1` | invariant "no auto-answer" | 30+ turns with the flag ON; count `response.created` events without a preceding `response.create` (telemetry `lane` vs model output) |
| Out-of-band (`conversation:"none"`) responses play audio on the WebRTC track and emit `response.output_audio_transcript.*` | lines must be audible and persisted | one session; if silent, set `REALTIME_RESPONSE_CONVERSATION=auto` (no deploy) |
| `response.cancel` + `output_audio_buffer.clear` stop audio within ~150–300 ms on real devices | barge-in feel | `interruption_ms` from `[interviewer.timing]` via `tools/voice_latency_report.py` |
| Real acoustic echo (laptop speakers, phone speaker) | self-transcription | test with speakers, not headphones; count `drop reason=echo` |
| iOS Safari / Android Chrome audio unlock, Bluetooth routing | playback reliability | device matrix, see latency report |
| Transcription model A/B (`whisper-1` vs `gpt-4o-mini-transcribe` / `gpt-live-transcribe`) | T1 dominates the realtime path | `REALTIME_TRANSCRIBE_MODEL`, N≥30 turns each, compare T1 P50/P95 and WER on a fixed script |

## 5. Next latency step (not implemented)
Server-side "sideband" control (backend joins the same call over WebSocket and issues
`response.create` itself) would remove one browser→backend→browser hop from T2→T3. It moves
long-lived sockets into the single Render worker and could not be tested here, so it is documented,
not shipped.
