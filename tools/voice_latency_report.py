"""
Latency report from REAL sessions: parses the structured telemetry the product emits
("[interviewer.timing] {...}" from the browser via /attempts/{id}/voice-telemetry, and
"[interviewer.turn] {...}" from the brain) out of backend logs, and prints P50/P90/P95
per condition.

Nothing here is simulated. Run real interviews (N >= 30 turns per condition) in a real
browser against a deployed backend with INTERVIEWER_BRAIN on for your account, export the
backend log (Render -> Logs -> download, or `render logs`), then:

    python -m tools.voice_latency_report render.log            # markdown tables
    python -m tools.voice_latency_report render.log --json

Definitions (client clock, epoch ms; T0 = candidate stopped speaking as the client sees it):
  REALTIME: T0 = input_audio_buffer.speech_stopped received (server semantic VAD decided the
            candidate finished), T1 = transcription.completed received, T2 = decision received,
            T3 = response.create sent, T4 = first audible interviewer sample (analyser on the
            remote track). interruption_ms = speech_started (while interviewer audio active) ->
            interviewer audio stopped/cleared.
  STT:      T0 = local VAD end-of-turn (NOTE: fires after the configured 750 ms silence window,
            so real speech end is ~750 ms earlier), T1 = final transcript, T2 = first token of the
            reply, T4 = first TTS clip starts playing.
  TEXT:     server-side only: decision_ms and first_token_ms from [interviewer.turn].
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List

_LINE = re.compile(r"\[interviewer\.(timing|turn)\]\s+(\{.*\})")


def pct(xs: List[float], p: float):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return round(xs[k])


def load(paths: List[str]):
    timing, turns = [], []
    for p in paths:
        fh = sys.stdin if p == "-" else open(p, encoding="utf-8", errors="replace")
        for line in fh:
            m = _LINE.search(line)
            if not m:
                continue
            try:
                ev = json.loads(m.group(2))
            except ValueError:
                continue
            (timing if m.group(1) == "timing" else turns).append(ev)
    return timing, turns


def summarize(timing, turns) -> Dict:
    out: Dict = {"client": {}, "server": {}}
    groups = defaultdict(lambda: defaultdict(list))
    for ev in timing:
        key = " / ".join(str(ev.get(k) or "-") for k in ("channel", "transport", "stt_model"))
        for m in ("t1_ms", "t2_ms", "t3_ms", "t4_ms", "interruption_ms"):
            if isinstance(ev.get(m), (int, float)):
                groups[key][m].append(float(ev[m]))
    for key, ms in groups.items():
        out["client"][key] = {m: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90), "p95": pct(v, 95)} for m, v in ms.items()}
    sg = defaultdict(lambda: defaultdict(list))
    for ev in turns:
        key = f"{ev.get('channel')} / {ev.get('interviewer_lane')}"
        for m in ("decision_ms", "generation_ms", "first_token_ms"):
            if isinstance(ev.get(m), (int, float)):
                sg[key][m].append(float(ev[m]))
        sg[key]["_llm"].append(1.0 if ev.get("llm_called") else 0.0)
    for key, ms in sg.items():
        llm = ms.pop("_llm", [])
        out["server"][key] = {m: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90), "p95": pct(v, 95)} for m, v in ms.items()}
        out["server"][key]["llm_call_rate"] = round(sum(llm) / len(llm), 2) if llm else None
    return out


def markdown(s: Dict) -> str:
    lines = ["## Client-measured (from [interviewer.timing])", "",
             "| condition | metric | N | P50 | P90 | P95 |", "|---|---|---|---|---|---|"]
    for key, ms in sorted(s["client"].items()):
        for m, v in ms.items():
            lines.append(f"| {key} | {m} | {v['n']} | {v['p50']} | {v['p90']} | {v['p95']} |")
    lines += ["", "## Server-measured (from [interviewer.turn])", "",
              "| channel / lane | metric | N | P50 | P90 | P95 |", "|---|---|---|---|---|---|"]
    for key, ms in sorted(s["server"].items()):
        for m, v in ms.items():
            if m == "llm_call_rate":
                continue
            lines.append(f"| {key} | {m} | {v['n']} | {v['p50']} | {v['p90']} | {v['p95']} |")
    lines.append("")
    lines.append("Conditions with N < 30 are not statistically meaningful; collect more turns before quoting them.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    timing, turns = load(a.paths)
    s = summarize(timing, turns)
    print(json.dumps(s, indent=1) if a.json else markdown(s))


if __name__ == "__main__":
    main()
