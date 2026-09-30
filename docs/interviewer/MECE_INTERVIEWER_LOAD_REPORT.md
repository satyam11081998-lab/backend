# MECE Interviewer — Load & Concurrency Report

Date: 2026-09-30. Tool: `python -m tools.interviewer_load_test`. **Local only** — nothing was sent
to production, Supabase or any provider (no authorization was given for production load tests,
and none was attempted).

## 1. Set-up

- The **real FastAPI app** (the attempt routes with the brain ON) under **uvicorn, one worker**
  (as on Render), in its **own process** (forked), so the load generator does not share its GIL.
- In-memory database; a scripted provider that **sleeps like a real one**: 350 ms to first
  token, then 300 ms of tail (650 ms per substantive line). The assessor answers instantly.
- N concurrent candidates, each with its own attempt, 10 turns each, a mixed script across
  TEXT (`/messages` SSE), STT (`/messages`, `channel:"stt"`) and VOICE (`/voice-decision`):
  greetings, clarifications, thinking aloud, numbers, help, frustration, corrections, a
  solution request, a final answer. **Think time between turns 50–300 ms** — far more aggressive
  than a real candidate (seconds to tens of seconds per turn), so each level is a stress level,
  not a user count.
- A trivial async `/ping` is polled every 50 ms throughout to measure event-loop responsiveness.
- After each level: every persisted row belongs to one of that level's attempts (isolation) and
  no `(attempt_id, client_turn_id)` appears twice (duplicates).
- Host: 2 vCPU container, shared by the load generator and the server.

## 2. Results — anyio default worker threads (40)

Latency = request → first SSE event (`silence` or first `token`) for TEXT/STT, request → JSON
decision for VOICE. Substantive includes the simulated 650 ms model.

| concurrent candidates | requests | throughput (req/s) | NO_OUTPUT P50 / P95 (ms) | PRESENCE P50 / P95 | SUBSTANTIVE P50 / P95 | event-loop ping P50 / P95 / max | errors | isolation | duplicate rows | server RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 50 | 9.4 | 4.1 / 7.6 | 4.1 / 5.8 | 658 / 680 | 1.5 / 2.4 / 71 | 0 | OK | 0 | 95 MB |
| 10 | 100 | 19.0 | 3.9 / 22.6 | 3.8 / 24.9 | 654 / 678 | 1.5 / 1.8 / 10 | 0 | OK | 0 | 96 MB |
| 25 | 250 | 43.3 | 4.4 / 65.7 | 3.8 / 6.8 | 655 / 706 | 1.8 / 3.0 / 18 | 0 | OK | 0 | 98 MB |
| 50 | 500 | 80.5 | 7.0 / 123 | 5.7 / 14.5 | 656 / 762 | 2.7 / 7.0 / 34 | 0 | OK | 0 | 103 MB |
| 100 | 1,000 | 85.6 | 352 / 2,697 | 62 / 1,075 | 716 / 2,315 | 5.5 / 181 / 2,944 | 0 | OK | 0 | 111 MB |

Final run on the final branch tip (`--port 8794`). Two earlier identical runs (before the rebase,
and after it) gave the same picture within noise (50 candidates: NO_OUTPUT P95 113–123 ms,
SUBSTANTIVE P95 741–747 ms; 100 candidates: NO_OUTPUT P95 2.8–2.9 s).

Same levels with **120 worker threads** (`--threadpool 120`):

| concurrent candidates | throughput | NO_OUTPUT P50 / P95 | PRESENCE P50 / P95 | SUBSTANTIVE P50 / P95 | ping P95 / max | errors |
|---|---|---|---|---|---|---|
| 50 | 78.7 | 6.7 / 260 | 6.1 / 14.4 | 656 / 883 | 6.1 / 96 | 0 |
| 100 | 87.5 | 294 / 2,661 | 88 / 995 | 719 / 2,259 | 364 / 2,046 | 0 |

An earlier run with the server and the load generator in the **same** process (shared GIL)
saturated at 50 candidates (NO_OUTPUT P95 889 ms) and logged one client `ReadError` at 100; moving
the server into its own process removed both. That first run is superseded and kept only as the
reason the tool changed.

## 3. Reading the numbers

- **Up to 50 aggressive concurrent candidates on 2 vCPUs:** silence and presence answer in
  single-digit milliseconds at P50 and ≤ 125 ms at P95; substantive turns add ≤ ~110 ms at P95 on
  top of the model's own time; the event loop stays responsive; zero errors, zero cross-attempt
  leakage, zero duplicate rows. VERIFIED (locally, with a simulated provider).
- **At 100 the box saturates at ~76–88 requests/s** (CPU: both processes share 2 vCPUs). Latency
  then grows for every lane including silence, because every turn — silent or not — first does
  the same route work (auth, reads, C9, persistence) in the same worker. More worker threads do
  not help (CPU-bound, not thread-bound). Still zero errors and correct isolation.
- **What this means for production (inference, not measured):** the saturation point is a
  request rate, not a user count. At the harness rate each candidate sends ~1.2 requests/s; a
  real candidate sends one turn every several seconds. The production worker additionally waits
  on Supabase and the provider over the network (I/O in threads, which frees the CPU but holds a
  worker thread). The shape of the limit is unchanged from the baseline: V11's routes are the
  same sync routes holding a thread for the model call. The brain lowers per-turn cost where it
  answers without a model (NO_OUTPUT / PRESENCE / corrections / deflections).
- **Memory:** server RSS 94 → 110 MB across 1,900 requests with 1,000 turns in the ledger; the
  turn ledger is bounded (5,000 entries, 15-minute TTL) and the row-idempotency cache is bounded
  (20,000). No growth attributable to the brain was observed.

## 4. Not measured
Provider throughput limits and throttling (OpenAI / Groq rate limits per key), Supabase
connection limits under load, Render's CPU (the instance class is not known here), real network.
A production-shaped test needs a staging backend with its own keys and an explicit go-ahead.

## 5. Reproduce
```
python -m tools.interviewer_load_test --users 5 10 25 50 100 --turns 10
python -m tools.interviewer_load_test --users 50 100 --turns 10 --threadpool 120 --port 8792
```
