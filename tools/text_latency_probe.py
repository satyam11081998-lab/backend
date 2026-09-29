"""
TEXT-channel latency probe against a REAL deployed backend (your own account).

Starts (or resumes) an attempt on a case you choose and posts scripted turns, measuring
request -> first visible token (SSE `token`), request -> `silence` for NO_OUTPUT turns,
and total time. Reports P50/P90/P95 by lane. It spends real model calls (only the
SUBSTANTIVE turns do) and writes real rows into that practice attempt.

    # token = your Supabase access token (browser devtools -> Application -> supabase auth)
    python -m tools.text_latency_probe --api https://<backend> --token <jwt> --case <case_uuid> --rounds 3

Run it with INTERVIEWER_BRAIN=on (or your user in INTERVIEWER_BRAIN_ALLOWLIST) on the backend.
Take the numbers from a network you care about; they include your network RTT.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid

import httpx

TURNS = [
    "What population should I use?",          # SUBSTANTIVE (answer)
    "I'll take 1.4 billion people.",           # NO_OUTPUT
    "Shall I proceed?",                        # PRESENCE
    "let me think",                            # NO_OUTPUT
    "Can you give me a hint?",                 # SUBSTANTIVE (hint)
    "50%",                                     # NO_OUTPUT
    "Show me the correct approach.",           # SUBSTANTIVE (solution)
    "1.4 billion / 3 is 4.6 billion",          # SUBSTANTIVE-fixed (deterministic correction)
]


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))]) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    H = {"Authorization": f"Bearer {a.token}"}
    with httpx.Client(timeout=60) as c:
        att = c.post(f"{a.api}/attempts", json={"case_id": a.case}, headers=H)
        att.raise_for_status()
        aid = att.json()["attempt_id"]
        rows = []
        for _ in range(a.rounds):
            for text in TURNS:
                t0 = time.perf_counter()
                first = silence = None
                with c.stream("POST", f"{a.api}/attempts/{aid}/messages",
                              json={"content": text, "channel": "text", "turn_id": uuid.uuid4().hex}, headers=H) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if line.startswith("event: token") and first is None:
                            first = (time.perf_counter() - t0) * 1000
                        if line.startswith("event: silence"):
                            silence = (time.perf_counter() - t0) * 1000
                total = (time.perf_counter() - t0) * 1000
                lane = "NO_OUTPUT" if silence is not None else ("SPOKE" if first is not None else "ERROR")
                rows.append({"text": text, "lane": lane, "first_ms": first, "silence_ms": silence, "total_ms": total})
                time.sleep(0.5)
    by = {}
    for r in rows:
        v = r["silence_ms"] if r["lane"] == "NO_OUTPUT" else r["first_ms"]
        if v is not None:
            by.setdefault(r["lane"], []).append(v)
    print(json.dumps({lane: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90), "p95": pct(v, 95)} for lane, v in by.items()}, indent=1))
    print("attempt:", aid, "(a practice attempt on your account; submit or abandon it as you like)")


if __name__ == "__main__":
    main()
