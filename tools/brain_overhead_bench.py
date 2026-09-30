"""
What the interviewer brain itself adds to a turn, with the provider and the database taken out.

  1. decision:  engine.decide_turn + finalize, in process (per channel x lane)
  2. route:     request -> first SSE event (`silence` / first `token`) or JSON decision, through the
                real FastAPI routes with an in-memory DB and a zero-latency scripted model

N per condition is --n (default 200 for decisions, 60 for routes). Reports P50/P90/P95 in ms.
This is the application's own overhead: it excludes network, Supabase, the model, transcription
and audio. Those are measured in production with tools/voice_latency_report.py.

    python -m tools.brain_overhead_bench
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["INTERVIEWER_BRAIN"] = "on"
os.environ["INTERVIEWER_TELEMETRY"] = "off"
os.environ["LOG_TURN_TIMING"] = "0"
for k, v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "t"}.items():
    os.environ.setdefault(k, v)

from tests.interviewer_fakes import FakeDB, FakeLLM, install_route_fakes  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import routes.attempts as att  # noqa: E402
from services.interviewer import dedupe, engine  # noqa: E402
from services.interviewer.assessor import Assessment  # noqa: E402
from services.interviewer.types import CaseContext, Channel, TurnInput  # noqa: E402

CASE = CaseContext(case_type="guesstimate", content="Estimate the number of new cars sold in India per year. " * 30)
TRANSCRIPT = [{"role": "assistant" if i % 2 else "user", "content": "I'll take 1.4 billion people and 4.5 per household."}
              for i in range(24)]
CONDITIONS = {
    "NO_OUTPUT": "I'll take 1.4 billion people.",
    "PRESENCE": "Shall I proceed?",
    "SUBSTANTIVE_fixed": "1.4 billion / 3 = 4.6 billion",
    "SUBSTANTIVE_model": "Can you give me a hint?",
}


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))], 3)


def summary(xs):
    return {"n": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "p95": pct(xs, 95), "max": round(max(xs), 3)}


def bench_decisions(n):
    out = {}
    llm = FakeLLM()
    for ch in (Channel.TEXT, Channel.STT, Channel.VOICE):
        for cond, text in CONDITIONS.items():
            xs = []
            for _ in range(n):
                t = time.perf_counter()
                p = engine.decide_turn(turn=TurnInput(text=text, channel=ch), case=CASE, transcript=TRANSCRIPT,
                                       session_state={"brain": {"opened": True, "turns": 5, "phase": "analysis"}},
                                       assess=lambda **k: Assessment(material=False), llm=llm)
                engine.finalize(p, p.text or ("x" if p.needs_model else None))
                xs.append((time.perf_counter() - t) * 1000)
            out[f"{ch.value} / {cond}"] = summary(xs)
    return out


def bench_routes(n):
    S = {"db": None}
    install_route_fakes(att, lambda: S["db"])
    engine.LLM_FACTORY = lambda plan: FakeLLM()
    engine.default_assessor = lambda uid: (lambda **kw: Assessment(material=False))
    app = FastAPI()
    app.include_router(att.router)
    client = TestClient(app)
    H = {"Authorization": "Bearer t"}
    out = {}
    for route in ("messages", "voice-decision"):
        for cond, text in CONDITIONS.items():
            xs = []
            for i in range(n):
                S["db"] = FakeDB([{"role": "assistant", "kind": "text", "content": "Walk me through it."}],
                                 session_state={"brain": {"opened": True, "turns": 5, "phase": "analysis"}})
                dedupe.LEDGER.__init__()
                dedupe.ROWS.__init__()
                t = time.perf_counter()
                if route == "messages":
                    with client.stream("POST", "/attempts/a1/messages", json={"content": text, "channel": "text",
                                                                              "turn_id": f"b{i}"}, headers=H) as r:
                        first = None
                        for line in r.iter_lines():
                            if first is None and (line.startswith("event: silence") or line.startswith("event: token")):
                                first = (time.perf_counter() - t) * 1000
                else:
                    r = client.post("/attempts/a1/voice-decision", json={"content": text, "turn_id": f"b{i}"}, headers=H)
                    first = (time.perf_counter() - t) * 1000
                att._await_after_turn("a1")
                if first is not None:
                    xs.append(first)
            out[f"{route} / {cond}"] = summary(xs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--route-n", type=int, default=60)
    a = ap.parse_args()
    res = {"decision_ms": bench_decisions(a.n), "route_first_event_ms": bench_routes(a.route_n),
           "note": "in-process, fake DB, zero-latency model: application overhead only"}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
