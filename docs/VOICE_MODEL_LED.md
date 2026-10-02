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
4. **The common playbook** — a human interviewer who says only what the moment needs:
   - *the golden rule*: on track → a short go-ahead ("Mm-hm.", "Okay, go on."); a sound step →
     "you're on the right track", in their words, then stop; a question → answer it and stop; a
     real mistake → "Are you sure about that?" once; stuck / "I don't know" / "how do I structure
     it?" → food for thought (an analogy or a concrete angle), never a question back; frustrated or
     "you're repeating yourself" → something new and concrete straight away; small slips → let go.
     Most turns under fifteen words.
   - *how it talks*: never apologises, no assistant filler ("Certainly", "No problem", "Does that
     make sense?"), never repeats a question or the case, at most one question and not every turn,
     reuses the candidate's words, understands Hinglish and replies in English, ignores noise.
   - *the case brief* is on screen and may be written in the client's voice ("Our company...") -
     never read out; unknown facts are answered as the client's data.
   - how a structured thinker works any case, frameworks by case type, the hint ladder (cue →
     everyday analogy → framework → one step together; analogies to adapt per case type), the
     answer rule (first ask: "I'd suggest thinking about it this way..." + results page, never "I
     can't"; insisting: the answer with an honest note), closing, boundaries.
   - in the conversation so far, an interviewer line that read the brief aloud is replaced by a
     note, so the model never copies it.

Session settings: `create_response: true` (answers by itself), barge-in on, semantic VAD
eagerness `high` (`REALTIME_MODEL_LED_EAGERNESS`; waits at most ~2 s on a trailing-off sentence),
the interviewer **opens the call** itself. Transcription (English) runs alongside; nothing is
shown while people talk - each finished turn appears in the conversation and is saved.

The only client-side check is a guardrail on the spoken text: an answer volunteered **before the
candidate has asked for it** is cut (`response.cancel` + `output_audio_buffer.clear`) and steered
back to a framework. Once they have asked, it never cuts (their transcript can land after the
model has started speaking).

**Gemini Live** (`routes/realtime_gemini.py`, `components/solve/VoiceInterviewGemini.tsx`): the
same prompt is pinned into the session token as `system_instruction` (never sent to the browser);
the client opens the call with one text turn, plays Gemini's own audio as it streams, saves both
transcripts in speaking order and shows each turn once it is finished (`lib/voice/gemini-live.ts`),
stops the voice on barge-in, and applies the same answer guardrail (cut, then steer once the cut
turn ends).

- **Model:** the newest general live model on the key (`gemini-3.8-live`, Google's low-latency
  default since Sept 2026, ~$0.005/min in + $0.018/min out), never the translate / transcribe /
  extended-thinking variants; the 2.5 native-audio previews are legacy and capacity-managed.
  `GEMINI_LIVE_MODEL` still wins when set and available.
- **Session configs, fastest first** (`_session_configs`): `fast` = candidate transcription pinned to
  English (`en-IN` / `en-US`, no Devanagari), quick end-of-turn (`END_SENSITIVITY_HIGH`, 500 ms),
  harder to trigger by echo or noise (`START_SENSITIVITY_LOW`, 200 ms of speech), and no thinking
  pass on 2.5 native audio (3.x live models reject thinking settings); `tuned` = plain
  transcription + 500 ms; `plain` = the minimum. A config rejected at mint falls through; one
  Google refuses at setup makes the browser ask again with `tier + 1`.
- **Reconnect:** Google ends a Live connection about every 10 minutes (and networks drop). The
  browser reconnects by itself and the new session RESUMES from the saved turns (at most 3
  reconnects in 2 minutes, so it can never loop). Only connected time is metered.

Unchanged: transcript saving (`/realtime-turn`, same rows for scoring), C9 counting, credits and
metering, scoring and the results page. The standard (pipeline) voice mode is not affected.

## Opening, resume and difficulty

- **Fresh call:** the case is already on the candidate's screen, so the interviewer does not explain
  it - it greets, points to the case, and asks for their approach (it explains only if asked).
- **Coming back** (ended voice, reopened it; or switched from chat): the session prompt carries the
  saved conversation and the interviewer RESUMES from where it was - no new greeting, no restart.
- **Easy / Medium / Hard** picker in the voice header (remembered per browser; default = the case's
  own difficulty). Easy = coaching (cues offered unprompted, earlier hints, frameworks named);
  Medium = the standard playbook; Hard = tough final round (rare affirmation, pressure-testing,
  "so what?", minimal hints only on request, sanity check required). Changing it reconnects the
  call with the new style and resumes.

## Latency and transcripts

- The model answers from the audio itself; transcription runs alongside and never holds a reply
  up. Nothing is transcribed on screen while people talk (that read as "transcribe, then answer");
  each finished turn appears in the conversation.
- Gemini: the newest live model, quick end-of-turn and 64 ms mic chunks (see above). The browser
  console prints the model, level, resume flag and config tier
  (`[gemini] interviewer: live ... model=... config=0/3`).
- A reply whose end Gemini never reports is closed after 2.5 s of quiet, so the transcript and the
  saved turns never stall; a save that hangs is skipped after 10 s so later saves still go through.

## Switches (Render env; restart, no deploy)

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_INTERVIEWER` | `model_led` | applies to OpenAI Realtime AND Gemini Live. `renderer` = back to V11/V12 deciding every turn (the old transcribe → decide → read-out flow); `allowlist` = live only for `VOICE_INTERVIEWER_ALLOWLIST` |
| `REALTIME_MODEL_LED_EAGERNESS` | `high` | how soon OpenAI Realtime answers after the candidate stops; `medium` waits up to ~4 s, `low` longest |
| `GEMINI_LIVE_MODEL` | (unset) | pin a Gemini Live model; unset = the newest general live model on the key |
| `REALTIME_TRANSCRIBE_MODEL` | `gpt-4o-mini-transcribe` (live OpenAI sessions) / `whisper-1` (renderer) | candidate transcription on OpenAI Realtime (language `en`). It does NOT affect reply speed (the model hears the audio). If the API rejects the model, the session retries with `whisper-1` automatically. Gemini Live transcribes natively |
| `VOICE_COACH` | `off` | `on` = after each turn the server's learner read adds notes to the prompt (async) |
| `VOICE_TOOLS` | `off` | `on` = offer `get_hint` / `answer_request` server tools (adds a round trip when used) |

## Verified offline / not verified

- Backend: `python -m tests.test_voice_model_led` (session payload, prompt order and contents,
  history carry-over, coach/tools opt-in, transcription fallback, coach and tool routes).
- Frontend: `node --test qa/voice/model-led.test.cjs`, `npx tsc --noEmit`, `next build`.
- Browser E2E: `qa/e2e-voice/run-model-led.cjs` (OpenAI: real WebRTC to a mock realtime peer) 20/20,
  and `qa/e2e-voice/run-gemini-live.cjs` (Gemini: real WebSocket to a mock Gemini Live server,
  real `/realtime-gemini/session`; includes a dropped connection with a refused config on the
  way back) 25/25.
- Known trade-off: on OpenAI Realtime the browser receives the session instructions (with the
  private notes) in its `session.created` event, so a determined user could read the model
  solution in devtools; Gemini keeps them inside the token. The solution is shown on the results
  page anyway.
- NOT verified: the real model's conversation quality and latency with this prompt.
