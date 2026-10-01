# MECE Interviewer — Cost Report

Date: 2026-10-01 (re-run after the contextual-presence change and the merge of `main`). Tool: `python -m tools.interviewer_cost_model` (scripted interviews through the
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
| strong_10 / text | 10 | 7 / 0 / 3 | 1 | 0 | 0.1 | 0.3 | 697 | 180 | 0.00021 | 0.00055 | 0 | 3 |
| strong_10 / voice | 10 | 7 / 0 / 3 | 1 | 0 | 0.1 | 0.3 | 706 | 140 | 0.00019 | 0.00053 | 0 | 3 |
| average_10 / text | 10 | 5 / 2 / 3 | 1 | 0 | 0.1 | 0.6 | 692 | 180 | 0.00021 | 0.00055 | 0 | 5 |
| average_10 / voice | 10 | 5 / 2 / 3 | 1 | 0 | 0.1 | 0.6 | 702 | 140 | 0.00019 | 0.00052 | 0 | 5 |
| weak_10 / text | 10 | 2 / 0 / 8 | 5 | 0 | 0.5 | 0.7 | 3,854 | 1,100 | 0.00124 | 0.00314 | 0 | 8 |
| weak_10 / voice | 10 | 1 / 1 / 8 | 5 | 0 | 0.5 | 0.7 | 3,874 | 860 | 0.00110 | 0.00297 | 0 | 9 |
| mixed_20 / text | 20 | 11 / 2 / 7 | 6 | 2 | 0.4 | 0.55 | 4,451 | 1,040 | 0.00129 | 0.00345 | 0.00012 | 9 |
| mixed_20 / voice | 20 | 10 / 3 / 7 | 6 | 2 | 0.4 | 0.45 | 4,499 | 800 | 0.00115 | 0.00329 | 0.00012 | 10 |

Why substantive turns outnumber generation calls: OPEN, CLOSE, DIRECT_CORRECTION and DEFLECT
lines are worded deterministically (no model), as are the fixed PRESENCE lines.

**Silence and fixed presence bill nothing.** NO_OUTPUT turns and fixed PRESENCE beats ("Shall I
proceed?" → "Yes, go ahead.") make no model call and write no `ai_usage_log` row (route test
`test_silence_and_presence_never_touch_meters`). Baseline V11 wrote an `ai_usage_log` row with
`model="local"` for every silent/presence turn (F6) — no dollars, but ledger noise. VERIFIED.

**Contextual presence costs one call (two if regenerated).** Since 2026-10-01 a finished
structure, an "is my approach okay?" and a long finished step in voice get a line worded from the
candidate's content instead of a stock phrase. In the 20-turn mixed script that is one extra call
(5 → 6 generation calls). Each call now carries the JSON control packet, which adds roughly 250
input tokens per call (≈ +$0.00004 per call on gpt-4o-mini) — the input-token column grew about
1.5× for that reason. Net effect on a 20-turn interview: ≈ +$0.0004 on the text model. VERIFIED
(counts) / PARTIALLY VERIFIED (dollars).

The weak-candidate script is the most expensive because it asks for help and a solution: every
rung of help is a model call by design (the brief requires help to be honoured). Even so it stays
around **$0.001–0.003 per 10 turns** on gpt-4o-mini with realistic case lengths (estimate).

## 3. The dominant cost is speech, not the brain

| Channel (20-turn mixed interview, estimate) | Brain (text model + assessor) | Transcription | Speech out | Total (est.) |
|---|---|---|---|---|
| TEXT | ≈ $0.0015–0.0025 | — | — | ≈ $0.002 |
| STT (Whisper transport) | ≈ $0.0015–0.0025 | ~2.7 min candidate audio: Groq $0.002 / whisper-1 $0.016 / live-transcribe $0.046 | tts-1, ~1,100 chars: ≈ $0.017 | ≈ $0.02–0.07 |
| REALTIME VOICE | ≈ $0.0015–0.0025 | whisper-1 (realtime input transcription) ≈ $0.016 | gpt-realtime audio out, 10 lines ≈ $0.10 (+ ≈ $0.00002 text-in for the line instructions) | ≈ $0.12 |

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
