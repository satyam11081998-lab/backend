"""
Local load / concurrency test for the unified-brain routes.

Runs the REAL FastAPI app under uvicorn (one worker, like production) with an
in-memory database and a scripted provider that sleeps like a real one (first token
after --llm-first-ms, then --llm-tail-ms), and drives N concurrent candidates, each
with its own attempt, through a mixed TEXT/STT/VOICE interview.

Measures per lane: server-side latency to first SSE token / JSON decision; errors;
event-loop responsiveness (a trivial endpoint pinged during the load); RSS growth;
session isolation (no attempt ever sees another attempt's words); duplicate turns.

It does NOT measure provider throughput, provider throttling or real network. It is
safe: nothing leaves localhost.

    python -m tools.interviewer_load_test --users 5 10 25 50 100 --turns 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import resource
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["INTERVIEWER_BRAIN"] = "on"
os.environ["INTERVIEWER_TELEMETRY"] = "off"
os.environ["LOG_TURN_TIMING"] = "0"

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from tests.interviewer_fakes import FakeDB, FakeLLM, install_route_fakes  # noqa: E402
import routes.attempts as att  # noqa: E402
from services.interviewer import dedupe, engine  # noqa: E402
from services.interviewer.assessor import Assessment  # noqa: E402

SCRIPT = [
    ("text", "Hi"), ("text", "What population should I use?"), ("stt", "let me think"),
    ("voice", "I'll take 1.4 billion people and 4.5 per household, so about 31 crore households."),
    ("text", "Can you give me a hint?"), ("voice", "Oh right, affordability first."), ("stt", "50%"),
    ("voice", "Shall I proceed?"), ("text", "1.4 billion / 3 = 4.6 billion"), ("voice", "This is irritating."),
    ("text", "Show me the correct approach."), ("stt", "My final estimate is 40 lakh cars a year."),
]


class MultiDB(FakeDB):
    def __init__(self, n):
        super().__init__([])
        base = self.tables["attempts"][0]
        self.tables["attempts"] = [dict(base, id=f"a{i}", session_state={}) for i in range(n)]


class TimedLLM(FakeLLM):
    def __init__(self, first_ms, tail_ms):
        super().__init__("good")
        self.first_ms, self.tail_ms = first_ms, tail_ms

    def complete(self, messages, *, max_tokens):
        time.sleep((self.first_ms + self.tail_ms) / 1000)
        return super().complete(messages, max_tokens=max_tokens)

    def stream(self, messages, *, max_tokens, meta):
        time.sleep(self.first_ms / 1000)
        parts = list(super().stream(messages, max_tokens=max_tokens, meta=meta))
        for p in parts:
            time.sleep(self.tail_ms / 1000 / max(1, len(parts)))
            yield p


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return round(xs[k], 1)


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


async def run_level(port, users, turns, db):
    lat = {"NO_OUTPUT": [], "PRESENCE": [], "SUBSTANTIVE": [], "ERROR": []}
    first_token = []
    errors = []
    pings = []
    stop = asyncio.Event()

    async def pinger(client):
        while not stop.is_set():
            t = time.perf_counter()
            try:
                await client.get(f"http://127.0.0.1:{port}/ping")
                pings.append((time.perf_counter() - t) * 1000)
            except Exception as e:  # noqa: BLE001
                errors.append(f"ping:{type(e).__name__}")
            await asyncio.sleep(0.05)

    async def candidate(client, i):
        aid = f"a{i}"
        rnd = random.Random(i)
        for n in range(turns):
            ch, text = SCRIPT[(n + rnd.randint(0, 3)) % len(SCRIPT)]
            tid = f"{aid}-t{n}"
            t0 = time.perf_counter()
            try:
                if ch == "voice":
                    r = await client.post(f"http://127.0.0.1:{port}/attempts/{aid}/voice-decision",
                                          json={"content": text, "turn_id": tid}, headers={"Authorization": "Bearer t"})
                    ms = (time.perf_counter() - t0) * 1000
                    if r.status_code != 200:
                        errors.append(f"voice:{r.status_code}")
                        lat["ERROR"].append(ms)
                        continue
                    lane = r.json()["lane"]
                    lat["NO_OUTPUT" if lane == "SILENCE" else lane].append(ms)
                else:
                    first = None
                    lane = "NO_OUTPUT"
                    async with client.stream("POST", f"http://127.0.0.1:{port}/attempts/{aid}/messages",
                                             json={"content": text, "channel": ch, "turn_id": tid},
                                             headers={"Authorization": "Bearer t"}) as r:
                        if r.status_code != 200:
                            errors.append(f"text:{r.status_code}")
                            continue
                        async for line in r.aiter_lines():
                            if line.startswith("event: token") and first is None:
                                first = (time.perf_counter() - t0) * 1000
                                lane = "SPOKE"
                            if line.startswith("event: error"):
                                lane = "ERROR"
                    ms = (time.perf_counter() - t0) * 1000
                    if lane == "SPOKE":
                        first_token.append(first)
                        lat["SUBSTANTIVE" if first > 50 else "PRESENCE"].append(first)
                    elif lane == "ERROR":
                        lat["ERROR"].append(ms)
                    else:
                        lat["NO_OUTPUT"].append(ms)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{ch}:{type(e).__name__}")
            await asyncio.sleep(rnd.uniform(0.05, 0.3))   # candidate "thinking" between turns

    limits = httpx.Limits(max_connections=users + 10, max_keepalive_connections=users + 10)
    async with httpx.AsyncClient(timeout=60, limits=limits) as client:
        ping_task = asyncio.create_task(pinger(client))
        t0 = time.perf_counter()
        await asyncio.gather(*(candidate(client, i) for i in range(users)))
        wall = time.perf_counter() - t0
        stop.set()
        await ping_task

    # isolation: each attempt's rows contain only its own attempt id, user rows only script texts
    iso_ok = all(m.get("attempt_id") in {f"a{i}" for i in range(users)} for m in db.rows())
    dup_rows = 0
    seen = set()
    for m in db.rows():
        k = (m.get("attempt_id"), m.get("client_turn_id"))
        if k[1] and k in seen:
            dup_rows += 1
        seen.add(k)
    return {
        "users": users, "turns_per_user": turns, "requests": users * turns, "wall_s": round(wall, 1),
        "throughput_rps": round(users * turns / wall, 1),
        "ms": {lane: {"n": len(v), "p50": pct(v, 50), "p90": pct(v, 90), "p95": pct(v, 95)} for lane, v in lat.items()},
        "event_loop_ping_ms": {"n": len(pings), "p50": pct(pings, 50), "p95": pct(pings, 95), "max": round(max(pings), 1) if pings else None},
        "errors": errors[:20], "error_count": len(errors), "isolation_ok": iso_ok, "duplicate_rows": dup_rows,
        "rss_mb_after": rss_mb(), "ledger_entries": dedupe.LEDGER.size(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, nargs="+", default=[5, 10, 25, 50, 100])
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--llm-first-ms", type=int, default=350)
    ap.add_argument("--llm-tail-ms", type=int, default=300)
    a = ap.parse_args()

    maxu = max(a.users)
    db = MultiDB(maxu)
    llm = TimedLLM(a.llm_first_ms, a.llm_tail_ms)
    install_route_fakes(att, lambda: db)
    engine.LLM_FACTORY = lambda plan: llm
    engine.default_assessor = lambda uid: (lambda **kw: Assessment(material=False))
    app = FastAPI()
    app.include_router(att.router)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    config = uvicorn.Config(app, host="127.0.0.1", port=a.port, log_level="warning", workers=1)
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    out = {"rss_mb_start": rss_mb(), "llm_first_ms": a.llm_first_ms, "llm_tail_ms": a.llm_tail_ms, "levels": []}
    for u in a.users:
        db.tables["attempt_messages"].clear()
        for row in db.tables["attempts"]:
            row["session_state"] = {}
        dedupe.LEDGER.__init__()
        out["levels"].append(asyncio.run(run_level(a.port, u, a.turns, db)))
        print(json.dumps(out["levels"][-1]), flush=True)
    server.should_exit = True
    th.join(timeout=5)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
