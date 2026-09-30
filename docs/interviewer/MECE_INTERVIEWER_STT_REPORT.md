# MECE Interviewer — STT (talk mode) Report

Date: 2026-09-29. Channel: STT — the candidate speaks, the words are transcribed, the unified
brain decides, and an interviewer line (if any) is streamed as text and spoken with TTS.
No live provider call was possible from this environment; provider-dependent items are UNVERIFIED.

## 1. Transports

| Transport | Build flag | Path | Status |
|---|---|---|---|
| Whisper (default, unchanged transport) | `NEXT_PUBLIC_STT_TRANSPORT` unset or `whisper` | client VAD (`lib/voice/vad.ts`) → MediaRecorder blob → `POST /transcribe` (Groq whisper / whisper-1, as before) → `POST /attempts/{id}/messages {kind:'voice', channel:'stt', turn_id}` (SSE) → `TtsQueue` → `POST /speak` | VERIFIED (browser E2E, fake transcribe + fake TTS) |
| Live transcription (opt-in) | `NEXT_PUBLIC_STT_TRANSPORT=live` | `POST /realtime/transcription-session` (server mints a 600 s client secret; `gpt-live-transcribe`, `turn_detection:null`, `delay`, `languages`, `keywords`) → WebRTC transcription session; client VAD sends `input_audio_buffer.commit`; `…transcription.delta` drives the on-screen draft only; `…transcription.completed` (keyed by `item_id`) is the turn → same `/messages` call. If the live session cannot start, the client falls back to Whisper. Streamed seconds are reported to `/realtime/transcription-usage` and metered under `/transcribe` so the existing daily voice-minute quota applies. | unit-VERIFIED (event reducer); live UNVERIFIED |

The session config follows the realtime-transcription guide as read on 2026-09-29
(`gpt-live-transcribe` does not accept server/semantic VAD; completion order across items is not
guaranteed, so results are reconciled by `item_id`).

## 2. What the brain changes for STT

- The STT channel is decided by the same brain as TEXT and VOICE (`channel:'stt'` is a rendering
  hint only: voice-length limits for spoken lines).
- **NO_OUTPUT**: SSE `event: silence` + `done {message_id:null, silent:true}`. The client shows a
  non-message "listening" hint (`onSilence`), sends **nothing to TTS**, inserts no bubble, and the
  mic reopens. VERIFIED.
- **Presence** ("Yes, go ahead.") and **substantive** lines stream tokens and are spoken; each is
  persisted once. VERIFIED.
- Each turn carries a `turn_id` (client-generated, unique); a retried POST replays the stored
  result instead of deciding twice. VERIFIED (routes).
- Per-turn client timing is posted to `/attempts/{id}/voice-telemetry`: `t1_ms` (speech end →
  final transcript), `t2_ms` (→ first reply token), `t4_ms` (→ first TTS audio), or `lane:"SILENCE"`.

## 3. Pre-existing bug fixed: short answers were thrown away as noise
The STT browser E2E exposed that `lib/voice/noise-guard.ts` classified "50%", "3", "yes" and
"no" as noise, so these turns never reached the interviewer (brief §3/§14: a short numeric turn
is a valid turn). Fixed: anything containing a digit is never noise; a short list of answer words
(yes, no, yep, nope, nah, sure, haan, nahi) is kept. Whisper silence artefacts ("Thank you.",
"you", "Bye.") remain noise. Unit test `short numeric turns and explicit yes/no are candidate
turns, not noise`. VERIFIED.

## 4. Tests

### 4.1 Unit (`qa/voice/live-transcribe.test.cjs`, 7 tests)
Partials drive the draft only and `completed` is the boundary; out-of-order completions reconciled
by `item_id`; duplicate completion and late deltas ignored; empty commit / failed item resolve to
an empty turn; `/messages` SSE parsing: NO_OUTPUT → explicit silence, no text, `message_id:null`;
an interviewer reply streams and is not silent; `newTurnId` unique. VERIFIED.

### 4.2 Browser end-to-end (`qa/e2e-voice/run-stt.cjs`) — 13/13 PASS
Headless Chromium with a fake microphone playing a real speech WAV; the real `VoiceInterview`
component and VAD; the real FastAPI routes with the brain on; `/transcribe` returns queued texts
(the recorded audio blob is really uploaded: 29–32 KB per turn); `/speak` returns a short WAV.

| Check | Result |
|---|---|
| mic opens, VAD calibrates, UI reaches Listening | PASS |
| the VAD endpointed 4 spoken turns from the fake mic | PASS |
| every turn went through `/transcribe` with a real recorded blob | PASS |
| "let me think" → explicit silence, no reply text | PASS |
| "50%" → explicit silence | PASS |
| "Shall I proceed?" → "Yes, go ahead." | PASS |
| "Can you give me a hint?" → a hint, not a refusal | PASS |
| TTS requested only for real interviewer lines (never for silence, never empty) | PASS |
| persistence: 4 candidate rows, 2 interviewer rows, no empty rows | PASS |
| mic reopens after the last turn | PASS |
| telemetry: per-turn T1 reported; T4 for spoken turns, `SILENCE` lane for silent ones | PASS |
| no uncaught page errors | PASS |

Harness timings in that run (T1 8–15 ms, T2 14–15 ms, T4 26 ms) measure only loopback and the
fakes (instant transcription, instant model, instant TTS). **They are not STT latency.** Note that
T0 in STT is the local VAD's end-of-turn, which fires after the configured ~750 ms silence window:
real speech end is ~750 ms earlier than T0.

## 5. Not verified (needs live runs)
| Item | How to verify |
|---|---|
| End-to-end STT latency on real providers (Groq whisper / whisper-1 / `gpt-live-transcribe`), P50/P90/P95 | N≥30 turns per transport; `tools/voice_latency_report.py` on the Render log |
| Transcription accuracy on Indian-English numbers ("1.4 crore", "35 percent", "point four six") | fixed 30-utterance script; compare transcripts |
| `gpt-live-transcribe` `delay` setting trade-off | `STT_LIVE_DELAY` low vs default, same script |
| Mobile Safari mic + TTS unlock | device matrix |
