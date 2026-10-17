"""services/voice_lane.py -- how a voice turn's engine output becomes what is said.

Moved out of routes/attempts.py (/voice-decision) unchanged so the admin arena's voice
path shares it; tests/test_response_functions.py still drives the route end to end.
"""
from __future__ import annotations

import json

import pytest

from services.voice_lane import VoiceTurnError, interpret_voice_turn, is_v11_silence


def test_silence_is_only_the_deliberate_lane():
    ctl = {"mode": "NO_OUTPUT", "tag": {"intervention": "silence"}}
    assert interpret_voice_turn(ctl, [""]) == ("SILENCE", None, None)
    assert is_v11_silence(ctl, [""]) is True
    # words came back: not silence, whatever the mode says
    assert is_v11_silence(ctl, ["hello"]) is False


def test_model_worded_presence_beat_becomes_an_event_with_the_same_line():
    ctl = {"lane": "PRESENCE", "render": "model", "function": "ACKNOWLEDGE_AND_ORIENT", "tag": {}}
    lane, say, event = interpret_voice_turn(ctl, ["Right, 3 crore a day ", "on 240 outlets."])
    assert lane == "PRESENCE" and say == "Right, 3 crore a day on 240 outlets."
    assert event["data"]["text"] == say and event["data"]["mode"] == "ACKNOWLEDGE_AND_ORIENT"


def test_fast_lane_json_event_is_unpacked_and_raw_text_is_tolerated():
    ctl = {"tag": {"intervention": "fast_lane"}}
    ev = {"event_type": "interviewer_presence", "data": {"mode": "HAND_BACK", "code": "HB_02", "text": "Sure, continue."}}
    assert interpret_voice_turn(ctl, [json.dumps(ev)]) == ("PRESENCE", "Sure, continue.", ev)
    assert interpret_voice_turn(ctl, ["Go ahead."]) == ("PRESENCE", "Go ahead.", None)


def test_substantive_line_and_the_three_502s():
    assert interpret_voice_turn({"tag": {}}, ["What's the split?"]) == ("SUBSTANTIVE", "What's the split?", None)
    with pytest.raises(VoiceTurnError, match="produced no content"):
        interpret_voice_turn({"tag": {}}, ["  "])
    with pytest.raises(VoiceTurnError, match="presence event had no line"):
        interpret_voice_turn({"tag": {"intervention": "fast_lane"}}, [""])
    with pytest.raises(VoiceTurnError, match="presence event had no line"):
        interpret_voice_turn({"lane": "PRESENCE", "render": "model", "tag": {}}, [""])


def test_the_route_uses_this_module():
    src = open("routes/attempts.py", encoding="utf-8").read()
    assert "interpret_voice_turn(ctl, parts)" in src
    assert "from services.voice_lane import is_v11_silence" in src
