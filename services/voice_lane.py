"""How a VOICE turn's engine output becomes what the interviewer says.

stream_interviewer_reply(channel="voice") yields one of four shapes: nothing (V11's
deliberate silence), a model-worded presence beat, a fast-lane event as JSON, or a
substantive line. This module turns that into (lane, say, event).

Shared by routes/attempts.py (/voice-decision -- production) and services/arena.py
(the admin arena's voice path), so the arena shows exactly what a voice session would
say. It used to live inline in the route; moved here unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple


class VoiceTurnError(Exception):
    """The engine produced nothing usable for a voice turn. The route answers 502."""


def is_v11_silence(ctl: Dict[str, Any], chunks: List[str]) -> bool:
    """True only for V11's deliberate SILENCE lane.

    The frozen engine marks it in control_out as mode NO_OUTPUT with a
    `silence` intervention and yields nothing. Anything else that comes back
    empty is NOT silence and must not be swallowed as if it were.
    """
    tag = ctl.get("tag") or {}
    return (
        ctl.get("mode") == "NO_OUTPUT"
        and tag.get("intervention") == "silence"
        and not any((c or "").strip() for c in chunks)
    )


def interpret_voice_turn(ctl: Dict[str, Any], parts: List[str]) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """(lane, say, event) for one voice turn. Raises VoiceTurnError where the route 502s."""
    tag = ctl.get("tag") or {}
    event = None
    if is_v11_silence(ctl, parts):
        lane, say = "SILENCE", None
    elif ctl.get("lane") == "PRESENCE" and ctl.get("render") == "model":
        # V12 contextual presence: a model-worded acknowledgement of the candidate's
        # own work. Same shape as a fast-lane beat so every client handles it.
        lane = "PRESENCE"
        say = "".join(parts).strip()
        if not say:
            raise VoiceTurnError("The interviewer presence event had no line.")
        event = {"event_type": "interviewer_presence",
                 "data": {"mode": ctl.get("function") or "PRESENCE", "code": "CTX", "text": say}}
    elif tag.get("intervention") == "fast_lane":
        lane = "PRESENCE"
        raw = "".join(parts).strip()
        try:
            event = json.loads(raw)
            say = str(((event or {}).get("data") or {}).get("text") or "").strip() or None
        except (TypeError, ValueError):
            event, say = None, raw or None
        if not say:
            raise VoiceTurnError("The interviewer presence event had no line.")
    else:
        lane = "SUBSTANTIVE"
        say = "".join(parts).strip()
        if not say:
            # Not V11 silence (checked above) -- an empty substantive reply is an
            # error, never something to quietly treat as "say nothing".
            raise VoiceTurnError("The interviewer produced no content for this turn.")
    return lane, say, event
