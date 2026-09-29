"""
Structured telemetry for the interviewer brain.

One JSON line per decided turn ("[interviewer.turn] {...}") and one per client
timing report ("[interviewer.timing] {...}"). Lines carry ids, lanes, modes,
timings, token counts and error types - never candidate or interviewer text,
never secrets. In-process counters back the load and cost reports.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
COUNTERS: Counter = Counter()
_ENABLED = os.getenv("INTERVIEWER_TELEMETRY", "on").strip().lower() not in ("0", "off", "false", "no")

TURN_FIELDS = (
    "session_id", "attempt_id", "turn_id", "channel", "turn_complete", "interviewer_state", "interviewer_lane",
    "interviewer_mode", "decision_reason", "provider", "model", "llm_called", "assessor_called",
    "speech_end_timestamp", "transcript_final_timestamp", "decision_timestamp", "response_start_timestamp",
    "first_audio_timestamp", "interruption_timestamp", "error_type", "decision_ms", "generation_ms",
    "first_token_ms", "tokens_in", "tokens_out", "hint_level", "phase", "duplicate", "violations",
)

_FORBIDDEN_KEYS = {"content", "text", "say", "transcript", "prompt", "messages", "authorization", "token",
                   "api_key", "client_secret", "case_content"}


def now_ms() -> int:
    return int(time.time() * 1000)


def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in _FORBIDDEN_KEYS and v is not None}


def emit_turn(event: Dict[str, Any]) -> Dict[str, Any]:
    ev = _clean({k: event.get(k) for k in TURN_FIELDS if k in event})
    with _LOCK:
        COUNTERS["turns"] += 1
        COUNTERS[f"lane:{ev.get('interviewer_lane')}"] += 1
        COUNTERS[f"mode:{ev.get('interviewer_mode')}"] += 1
        COUNTERS[f"channel:{ev.get('channel')}"] += 1
        if ev.get("llm_called"):
            COUNTERS["llm_calls"] += int(ev.get("llm_calls_n") or 1)
        if ev.get("assessor_called"):
            COUNTERS["assessor_calls"] += 1
        COUNTERS["tokens_in"] += int(ev.get("tokens_in") or 0)
        COUNTERS["tokens_out"] += int(ev.get("tokens_out") or 0)
        if ev.get("error_type"):
            COUNTERS[f"error:{ev.get('error_type')}"] += 1
        if ev.get("duplicate"):
            COUNTERS["duplicates"] += 1
    if _ENABLED:
        print("[interviewer.turn] " + json.dumps(ev, separators=(",", ":"), default=str), flush=True)
    return ev


_TIMING_KEYS = ("session_id", "attempt_id", "turn_id", "channel", "speech_end_timestamp",
                "transcript_final_timestamp", "decision_timestamp", "response_start_timestamp",
                "first_audio_timestamp", "interruption_timestamp", "interruption_stop_timestamp",
                "t1_ms", "t2_ms", "t3_ms", "t4_ms", "interruption_ms", "lane", "transport", "model", "stt_model",
                "network", "browser", "error_type")


def emit_timing(report: Dict[str, Any]) -> Dict[str, Any]:
    ev = {}
    for k in _TIMING_KEYS:
        v = report.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            ev[k] = v
        else:
            ev[k] = str(v)[:80]
    with _LOCK:
        COUNTERS["timing_reports"] += 1
    if _ENABLED:
        print("[interviewer.timing] " + json.dumps(ev, separators=(",", ":")), flush=True)
    return ev


def snapshot() -> Dict[str, int]:
    with _LOCK:
        return dict(COUNTERS)


def reset() -> None:
    with _LOCK:
        COUNTERS.clear()
