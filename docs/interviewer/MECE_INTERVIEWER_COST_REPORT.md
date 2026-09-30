# MECE Interviewer — Cost Report

Date: 2026-09-30. Tool: `python -m tools.interviewer_cost_model` (scripted interviews through the
real brain with a counting fake model). **Every dollar figure here is a model estimate, not a
measured bill** (PARTIALLY VERIFIED: call counts are exact for the scripts; tokens are
characters/4; audio seconds are characters/15; prices as listed below). Real spend must be read
from `ai_usage_log` after real sessions.

## 1. Prices used

| Item | Price | Source |
|---|---|---|
| gpt-4o-mini (interviewer default via `resolve_llm('interviewer')`, and the assessor) | $0.15 in / $0.60 out per 1M tokens | OpenAI pricing page, checked 2026-09-29; same values in `services/ai_usage.py` |
| llama-3.3-70b-versatile (Groq, if selected in the admin provider toggle) | $0.59 / $0.79 per 1M | `services/ai_usage.py` |
| Realtime audio out / in (gpt-realtime) | $64 / $32 per 1M audio tokens (~1 token per 50 ms of speech) | `services/ai_usage.py` (`REALTIME_AUDIO_*`) |
| Realtime text input (the approved line sent as instructions) | $4 per 1M | provider list price used in the model |
| whisper-1 | $0.006 / min | provider list price |
| Groq whisper | $0.000667 / min | provider list price |
| gpt-live-transcribe | $0.017 / min | as configured in `services/ai_usage.py` for the new transport — confirm on the pricing page before enabling |
| tts-1 | $15 per 1M characters | provider list price |

## 2. Model calls and estimated text-model spend per interview

Output tokens are priced at the `max_tokens` cap (an upper bound): the fake model's replies are
shorter than real ones, so pricing them would understate spend. The scripted case text is short
(~300 characters); real cases carry up to ~2,600 characters of case text in the prompt, so real
input tokens per call are roughly 2–3× the figures below (≈1,000–1,500 per call).

| interview / channel | turns | NO_OUTPUT / PRESENCE / SUBSTANTIVE | generation calls | assessor calls | model calls per turn | V11 model calls per turn (same script) | est. input tokens | output cap | text model $ (gpt-4o-mini) | $ if Groq llama-70b | assessor $ | interviewer lines spoken |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strong_10 / text | 10 | 7 / 0 / 3 | 1 | 0 | 0.1 | 0.3 | 444 | 180 | 0.00017 | 0.00040 | 0 | 3 |
| strong_10 / voice | 10 | 7 / 0 / 3 | 1 | 0 | 0.1 | 0.3 | 458 | 140 | 0.00015 | 0.00038 | 0 | 3 |
| average_10 / text | 10 | 5 / 2 / 3 | 1 | 0 | 0.1 | 0.6 | 439 | 180 | 0.00017 | 0.00040 | 0 | 5 |
| average_10 / voice | 10 | 5 / 2 / 3 | 1 | 0 | 0.1 | 0.6 | 453 | 140 | 0.00015 | 0.00038 | 0 | 5 |
| weak_10 / text | 10 | 2 / 0 / 8 | 5 | 0 | 0.5 | 0.7 | 2,495 | 1,100 | 0.00103 | 0.00234 | 0 | 8 |
| weak_10 / voice | 10 | 1 / 1 / 8 | 5 | 0 | 0.5 | 0.7 | 2,566 | 860 | 0.00090 | 0.00219 | 0 | 9 |
| mixed_20 / text | 20 | 11 / 2 / 7 | 5 | 2 | 0.35 | 0.45 | 2,414 | 900 | 0.00090 | 0.00214 | 0.00012 | 9 |
| mixed_20 / voice | 20 | 10 / 3 / 7 | 5 | 2 | 0.35 | 0.45 | 2,459 | 700 | 0.00079 | 0.00200 | 0.00012 | 10 |

Why substantive turns outnumber generation calls: OPEN, CLOSE, DIRECT_CORRECTION and DEFLECT
lines are worded deterministically (no model), as are all PRESENCE lines.

**Silence and presence bill nothing.** NO_OUTPUT and PRESENCE turns make no model call and write
no `ai_usage_log` row (route test `test_silence_and_presence_never_touch_meters`). Baseline V11
wrote an `ai_usage_log` row with `model="local"` for every silent/presence turn (F6) — no dollars,
but ledger noise. VERIFIED.

The weak-candidate script is the most expensive because it asks for help and a solution: every
rung of help is a model call by design (the brief requires help to be honoured). Even so it stays
around **$0.001–0.003 per 10 turns** on gpt-4o-mini with realistic case lengths (estimate).

## 3. The dominant cost is speech, not the brain

| Channel (20-turn mixed interview, estimate) | Brain (text model + assessor) | Transcription | Speech out | Total (est.) |
|---|---|---|---|---|
| TEXT | ≈ $0.001–0.002 | — | — | ≈ $0.002 |
| STT (Whisper transport) | ≈ $0.001–0.002 | ~2.7 min candidate audio: Groq $0.002 / whisper-1 $0.016 / live-transcribe $0.046 | tts-1, ~1,100 chars: ≈ $0.017 | ≈ $0.02–0.07 |
| REALTIME VOICE | ≈ $0.001–0.002 | whisper-1 (realtime input transcription) ≈ $0.016 | gpt-realtime audio out, 10 lines ≈ $0.088 (+ ≈ $0.00002 text-in for the line instructions) | ≈ $0.10 |

Two design choices cut speech cost directly:

1. **Fewer spoken lines.** On the same scripts V11 speaks on every turn in voice (0 silent turns;
   10/10/10/20 lines), the brain on 3/5/9/10. Presence lines are short (~1–2 s), so the saving is
   smaller than the line count suggests, but a spoken "Got it." every turn is also the mechanical
   rhythm the brief rules out. VERIFIED (counts); dollar effect PARTIALLY VERIFIED.
2. **Out-of-band realtime responses** (`conversation:"none"`, `input:[]`). In-band (the baseline),
   every spoken line is generated with the whole session conversation as input — all candidate
   audio so far is billed as audio input tokens again on each response (mostly at the cached rate
   after the first read, $0.40/1M, with each new turn's audio at $32/1M), and that context grows
   with the session. Out-of-band, each response's input is just the line text (~60–100 tokens at
   $4/1M). Rough estimate for a 10-minute session with ~5 minutes of candidate speech and 10
   interviewer lines: in-band ≈ $0.10–0.12 of audio input on top of output; out-of-band ≈ $0.0004.
   **UNVERIFIED** — confirm with the `usage` object of `response.done` (logged via `onUsage`) in a
   live A/B using `REALTIME_RESPONSE_CONVERSATION=auto|none`.

## 4. Cost controls in the code

| Control | Where | Status |
|---|---|---|
| No model for NO_OUTPUT, PRESENCE, OPEN/CLOSE, DIRECT_CORRECTION, DEFLECT | `policy.py`, `presence.py`, `numbers.py` | VERIFIED |
| Assessor only on step-completing analytic turns and on bare results that sit ~10× off the candidate's own numbers (a likely dropped zero) — not on every "?" as V11 did (F7); 2.5 s text / 1.2 s voice time box | `policy.py`, `numbers.scale_slip`, `assessor.py` | VERIFIED |
| Compact prompt: case text capped at 2,600 chars, last 12 turns, per-move task | `prompting.py` | VERIFIED |
| `max_tokens` per move and channel (voice lines shorter) | `prompting.max_tokens` | VERIFIED |
| At most one regeneration per voice line; none on text streams | `responder.py` | VERIFIED |
| One provider failover hop (never a retry storm) | `providers.py` | VERIFIED |
| Duplicate turn ids never re-bill (ledger replay) | `dedupe.py` | VERIFIED |
| Every real provider call logged to `ai_usage_log` with endpoint `/attempts/messages` or `/attempts/assess` | `providers.py` | code VERIFIED; live rows UNVERIFIED |

## 5. Not measured
Real token counts per call, real provider bills, cache hit rates, the in-band vs out-of-band
difference, and the transcription-model A/B. Use `ai_usage_log` grouped by endpoint and model after
a week of allow-listed traffic.
