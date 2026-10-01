# Realtime voice interviewer — live, prompt-led (default since 2026-10-02)

**Nothing to set.** Both live voice transports are live by default: whichever the admin voice mode
is — `gemini` (Gemini Live) or `realtime` (OpenAI Realtime) — the speech model is the interviewer
and talks to the candidate directly, speech to speech. No Render variable is needed.

## What it is

In realtime voice the speech model (`gpt-realtime-2.1`, the ChatGPT-voice family) **is** the
interviewer. It hears the candidate, decides when to speak and how much, and answers immediately.
Nothing sits between the candidate and the reply: no rule engine, no tool call, no per-turn
backend request. Everything it needs is in one prompt built when the session starts
(`prompts/voice_interviewer_playbook.py`):

1. **THE CASE** — on top: type + case text (US cases get the US market note).
2. **Private interviewer notes** — the stored starting hint and model solution: used to judge,
   to shape hints, and for the answer rule; never read out.
3. **Conversation so far** — the last 14 turns of the attempt, so switching from chat to voice
   picks up where it was (the old `SAY:` label is stripped).
4. **The common playbook** — how to talk in real time (short turns, reuse the candidate's words,
   react to meaning not punctuation, don't talk over thinking), how a structured thinker works
   any case (clarify → MECE structure / hypothesis → quantify → sanity-check → synthesise),
   frameworks by case type (guesstimate top-down/bottom-up + stock vs flow, profitability,
   market entry, growth, pricing, M&A, operations, issue tree), how to react (specific "you're on
   track", no questioning every step, at most one question, challenge only material issues, real
   talk when asked for a suggestion, own the facts), the hint ladder (cue → hint → framework or
   analogy → one step together; frustration skips to a framework), the answer rule (first ask: a
   way of thinking + "full worked answer on your results page"; insisting: the answer with an
   honest note that their results will show it), closing, boundaries.

Session settings: `create_response: true` (answers by itself), barge-in on, semantic VAD
eagerness `medium` (`REALTIME_MODEL_LED_EAGERNESS`), the interviewer **opens the call** itself.
The browser shows the candidate's words live when the transcription model streams them.

The only client-side check is a guardrail on the spoken text: an answer volunteered **before the
candidate has asked for it** is cut (`response.cancel` + `output_audio_buffer.clear`) and steered
back to a framework. Once they have asked, it never cuts (their transcript can land after the
model has started speaking).

**Gemini Live** (`routes/realtime_gemini.py`, `components/solve/VoiceInterviewGemini.tsx`): the
same prompt is pinned into the session token as `system_instruction` (never sent to the browser);
the client opens the call with one text turn, plays Gemini's own audio as it streams, shows and
saves both transcripts in speaking order (`lib/voice/gemini-live.ts`), stops the voice on
barge-in, and applies the same answer guardrail (cut, then steer once the cut turn ends).

Unchanged: transcript saving (`/realtime-turn`, same rows for scoring), C9 counting, credits and
metering, scoring and the results page. Gemini and the standard (pipeline) voice are not affected.

## Switches (Render env; restart, no deploy)

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_INTERVIEWER` | `model_led` | applies to OpenAI Realtime AND Gemini Live. `renderer` = back to V11/V12 deciding every turn (the old transcribe → decide → read-out flow); `allowlist` = live only for `VOICE_INTERVIEWER_ALLOWLIST` |
| `REALTIME_MODEL_LED_EAGERNESS` | `medium` | `high` answers sooner after the candidate stops; `low` waits longest (more room for thinking pauses) |
| `REALTIME_TRANSCRIBE_MODEL` | `gpt-4o-mini-transcribe` (live OpenAI sessions) / `whisper-1` (renderer) | candidate transcription on OpenAI Realtime; streams the words as they are spoken. It does NOT affect reply speed (the model hears the audio). If the API rejects the model, the session retries with `whisper-1` automatically. Gemini Live transcribes natively |
| `VOICE_COACH` | `off` | `on` = after each turn the server's learner read adds notes to the prompt (async) |
| `VOICE_TOOLS` | `off` | `on` = offer `get_hint` / `answer_request` server tools (adds a round trip when used) |

## Verified offline / not verified

- Backend: `python -m tests.test_voice_model_led` (session payload, prompt order and contents,
  history carry-over, coach/tools opt-in, transcription fallback, coach and tool routes).
- Frontend: `node --test qa/voice/model-led.test.cjs`, `npx tsc --noEmit`, `next build`.
- Browser E2E: `qa/e2e-voice/run-model-led.cjs` (OpenAI: real WebRTC to a mock realtime peer) 16/16,
  and `qa/e2e-voice/run-gemini-live.cjs` (Gemini: real WebSocket to a mock Gemini Live server,
  real `/realtime-gemini/session`) 13/13, three runs each.
- Known trade-off: on OpenAI Realtime the browser receives the session instructions (with the
  private notes) in its `session.created` event, so a determined user could read the model
  solution in devtools; Gemini keeps them inside the token. The solution is shown on the results
  page anyway.
- NOT verified: the real model's conversation quality and latency with this prompt.
