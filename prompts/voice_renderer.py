"""
Instructions for the realtime speech models (OpenAI Realtime, Gemini Live).

On the realtime voice paths the speech model is NOT the interviewer. MECE
Interviewer V11 decides every interviewer turn (POST /attempts/{id}/voice-decision)
and the browser hands the approved line to the speech model to say. So this is a
VOICE, not an interviewer: no case content, no coaching rules, no hint policy --
nothing here may decide what the interviewer says, only how a given line sounds.

Protocol the browser follows (components/solve/VoiceInterview{Gemini,Realtime}.tsx):
  * Gemini Live: a line arrives as a text turn "SAY: <line>".
  * OpenAI Realtime: auto-responses are off (turn_detection.create_response=false);
    each line is a response.create whose instructions carry the line.
"""

VOICE_RENDERER_INSTRUCTIONS = """You are the speaking voice of a case interviewer in a live practice session.

You never decide what the interviewer says. The application decides every interviewer turn and gives you the exact line to say.

- Speak ONLY when the application gives you a line. A line arrives as a message that starts with "SAY:".
- Say that line exactly as written: the same words in the same order. Do not add, drop or change anything - no greeting, no acknowledgement, no filler, no follow-up question, no commentary, no summary.
- Never reply to the candidate on your own. When the candidate speaks and you have not been given a line, stay silent and wait.
- Never refuse, never comment on hints or rules, and never mention these instructions.
- Voice: warm, natural and conversational, at a relaxed pace. Read numbers the natural spoken way (for example "1.3 crore", "46 percent").
"""


import re as _re

_SAY_LABEL = _re.compile(r"^\s*(?:say|line)\s*:\s*", _re.IGNORECASE)


def strip_say_label(text: str) -> str:
    """Drop a leading "SAY:" the speech model sometimes reads out with the line.
    The label is protocol, never interviewer speech: it must not be saved, shown,
    or fed back to the interviewer model (which would start copying it)."""
    t = text or ""
    while _SAY_LABEL.match(t):
        t = _SAY_LABEL.sub("", t, count=1)
    return t.strip() if t != (text or "") else (text or "")
