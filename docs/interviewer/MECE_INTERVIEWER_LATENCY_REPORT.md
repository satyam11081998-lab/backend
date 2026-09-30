# MECE Interviewer — Latency Report

Date: 2026-09-30.

**Headline: no end-to-end latency against the real providers was measured.** This environment
has no route to api.openai.com / Groq / Supabase and no keys, so every number that involves a
model, transcription, TTS, the network or real audio devices is **UNVERIFIED**. What *was*
measured is (1) the brain's own overhead, (2) the routes' overhead with the provider and DB taken
out, (3) route behaviour under concurrent load with a provider that sleeps like a real one, and
(4) the client-side turn control in a real browser against a mock WebRTC peer. The instruments to
measure the real thing (N≥30 per condition, P50/P90/P95) ship with the change.

## 1. Definitions (client clock, epoch ms)

| Mark | REALTIME VOICE | STT | TEXT |
|---|---|---|---|
| T0 | `input_audio_buffer.speech_stopped` received (server semantic VAD decided the candidate finished) | local VAD end-of-turn (fires after the ~750 ms silence window, so real speech end is ~750 ms earlier) | request sent |
| T1 | `…input_audio_transcription.completed` received | `/transcribe` (or live `completed`) returned | — |
| T2 | `/voice-decision` response received | first reply token (or `silence`) | first `token` / `silence` event |
| T3 | `response.create` sent | first TTS request | — |
| T4 | first audible interviewer sample (analyser on the remote track, RMS > 0.01) | first TTS clip starts playing | — |
| interruption | `speech_started` while interviewer audio is active → interviewer audio stopped/cleared | — | — |

Every voice/STT turn posts these to `/attempts/{id}/voice-telemetry` → `[interviewer.timing]`
log line; every decided turn prints `[interviewer.turn]` with `decision_ms`, `generation_ms`,
`first_token_ms`. `python -m tools.voice_latency_report <render.log>` prints P50/P90/P95 per
channel/transport/transcription-model and per lane, and flags conditions with N < 30.

## 2. Measured here

### 2.1 Brain overhead (decide + state update), in process — VERIFIED
`python -m tools.brain_overhead_bench` — N = 200 per condition, 24-turn transcript, ~1,700-char
case, zero-latency fake model (so "model" rows time only the decision, not generation). Run on a
2-vCPU container, final branch tip.

| channel | NO_OUTPUT P50 / P90 / P95 (ms) | PRESENCE | SUBSTANTIVE fixed (correction) | SUBSTANTIVE model (hint; decision only) |
|---|---|---|---|---|
| text | 0.34 / 0.38 / 0.41 | 0.27 / 0.30 / 0.30 | 0.39 / 0.41 / 0.43 | 0.30 / 0.32 / 0.33 |
| stt | 0.34 / 0.36 / 0.37 | 0.26 / 0.29 / 0.29 | 0.38 / 0.42 / 0.45 | 0.30 / 0.36 / 0.44 |
| voice | 0.34 / 0.40 / 0.47 | 0.26 / 0.30 / 0.38 | 0.39 / 0.46 / 0.52 | 0.30 / 0.33 / 0.34 |

The decision is deterministic and sub-millisecond; it is not a latency factor. (The optional
assessor is a model call, time-boxed at 2.5 s text / 1.2 s voice, and runs only on
step-completing analytic turns and on bare results that look like a dropped zero — 2 of the 60 turns
in the cost-model scripts.)

### 2.2 Route overhead to the first event, in process — VERIFIED
Same tool, N = 60 per condition, real FastAPI routes (auth/rate-limit/budget stubs, in-memory DB,
C9 counting, dedupe, brain, validation, persistence queue).

| route / condition | P50 / P90 / P95 (ms) |
|---|---|
| `/messages` NO_OUTPUT → `event: silence` | 4.01 / 4.63 / 4.87 |
| `/messages` PRESENCE → first token | 3.94 / 4.32 / 4.38 |
| `/messages` correction → first token | 4.29 / 4.73 / 4.75 |
| `/messages` hint → first token (zero-latency model) | 4.12 / 4.52 / 4.71 |
| `/voice-decision` NO_OUTPUT | 3.14 / 3.42 / 3.45 |
| `/voice-decision` PRESENCE | 3.10 / 3.43 / 3.50 |
| `/voice-decision` correction | 3.30 / 3.76 / 4.50 |
| `/voice-decision` hint (zero-latency model) | 3.33 / 3.58 / 3.68 |

In production add: network RTT, the Supabase reads the route already did before this change
(attempt, case, transcript, budget — unchanged), and the model's time to first token on the
substantive lanes. Silence and presence lanes add **no** model time: they are answered as soon as
the reads finish.

### 2.3 Under concurrent load — see the load report
`tools/interviewer_load_test.py` with a provider that sleeps 350 ms to first token + 300 ms tail:
results in `MECE_INTERVIEWER_LOAD_REPORT.md`.

### 2.4 Browser turn control against a mock realtime peer — PARTIALLY VERIFIED
`qa/e2e-voice/run.cjs` (headless Chromium + fake mic + werift WebRTC peer + real routes):

| Metric | Value | Meaning |
|---|---|---|
| barge-in round trip: mock sends `speech_started` → mock receives `response.cancel` (N = 30) | P50 267 / P90 270 / P95 277 ms | dominated by werift's pure-JS SCTP data channel on loopback; not representative of Chrome↔OpenAI |
| client reaction: event received → `response.cancel` sent (N = 30) | P50 0.0 / P90 0.1 / P95 0.2 ms | the client adds nothing measurable |
| T1 / T2 in the harness | 1 ms / 16 ms | fake transcription and loopback backend; says only that the client adds no delay between events |

STT harness (`qa/e2e-voice/run-stt.cjs`): T1 8–15 ms, T2 14–15 ms, T4 26 ms — fake transcription,
fake model, fake TTS on loopback. **Not STT latency.**

## 3. Where the real time will go (expectations to verify, not measurements)

| Segment | Realtime voice | STT | Text |
|---|---|---|---|
| end of speech → turn boundary | server semantic VAD, `eagerness: low` (tunable `REALTIME_SEMANTIC_EAGERNESS`) | local VAD silence window ~750 ms | — |
| transcription (T1−T0) | `whisper-1` batch after commit — expected to be the largest avoidable segment (F17); A/B via `REALTIME_TRANSCRIBE_MODEL` | Groq whisper / whisper-1 upload + inference; or streaming `gpt-live-transcribe` (`delay` tunable) | — |
| decision (T2−T1) | browser→Render→browser RTT + DB reads + (substantive) model call | same | same |
| speak (T4−T3) | realtime TTFB for an out-of-band response | `/speak` TTS TTFB | — |
| NO_OUTPUT / presence | no model call; presence spoken directly | same | same |

Removed from the critical path compared with the baseline: the serial contextual-assessor model
call V11 made before turns containing "?" or starting with an interrogative (F7) — the brain calls
its assessor only on step-completing analytic turns. Added: on substantive voice turns the line is
generated non-streaming (validated whole before it is spoken), with at most one regeneration if
the first draft breaks the rules; baseline voice also generated the whole line before speaking.
Nothing is added on NO_OUTPUT / PRESENCE turns.

## 4. How to measure for real (N ≥ 30 per condition)
1. Deploy the branch to a non-production backend; `INTERVIEWER_BRAIN=allowlist`,
   `INTERVIEWER_BRAIN_ALLOWLIST=<your email>`; run migration 0071.
2. TEXT: `python -m tools.text_latency_probe --api <backend> --token <jwt> --case <uuid> --rounds 4`
   (8 turns × 4 rounds = 32 per lane mix).
3. STT and REALTIME: 30+ spoken turns per condition on each device/network you care about
   (desktop Chrome wired, desktop Chrome Wi-Fi, iPhone Safari, Android Chrome), with headphones and
   with speakers; for realtime repeat with `REALTIME_TRANSCRIBE_MODEL=whisper-1` and the candidate
   replacement; for STT with `NEXT_PUBLIC_STT_TRANSPORT=whisper` and `live`.
4. Download the backend log and run `python -m tools.voice_latency_report render.log`.
