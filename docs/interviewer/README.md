# Unified interviewer brain — documents

Start with the product spec, then the architecture. The reports carry the evidence; every claim in
them is labelled VERIFIED / PARTIALLY VERIFIED / UNVERIFIED.

| Document | What it answers |
|---|---|
| `MECE_INTERVIEWER_PRODUCT_SPEC.md` | What the interviewer does on every turn, in every channel |
| `MECE_INTERVIEWER_ARCHITECTURE.md` | Modules, boundaries (provider, realtime, STT, state, security), failure paths |
| `MECE_INTERVIEWER_CALL_GRAPH.md` | Audit of the baseline (findings F1–F17) and the call graph after |
| `MECE_INTERVIEWER_STATE_MACHINE.md` | Candidate states, phases, ladder, loop guards, invariants |
| `MECE_INTERVIEWER_TEST_REPORT.md` | Every suite, command and result; the human-quality walk-through |
| `MECE_INTERVIEWER_REGRESSION_REPORT.md` | Baseline vs brain on the same inputs; failure modes A–L; invariants 1–10 |
| `MECE_INTERVIEWER_ADVERSARIAL_REPORT.md` | Diversified, property-based and chaos testing |
| `MECE_INTERVIEWER_REALTIME_REPORT.md` | OpenAI Realtime control: out-of-band lines, barge-in, dedupe, what still needs a live run |
| `MECE_INTERVIEWER_STT_REPORT.md` | Talk mode (Whisper and live-transcription transports) |
| `MECE_INTERVIEWER_LATENCY_REPORT.md` | What was measured, what was not, and how to measure it for real |
| `MECE_INTERVIEWER_LOAD_REPORT.md` | Concurrency on one uvicorn worker |
| `MECE_INTERVIEWER_COST_REPORT.md` | Model calls per turn and estimated spend per interview |
| `MECE_INTERVIEWER_SECURITY_REPORT.md` | Secrets, leakage, authorization, abuse, F16 (`cases.solution` readability) |
| `MECE_INTERVIEWER_RUNTIME_TRACE.md` | 11 real requests through the routes, with telemetry and persisted rows |
| `MECE_INTERVIEWER_ROLLBACK.md` | Git state before/after and how to undo each stage |
| `MECE_INTERVIEWER_FILE_MANIFEST.txt` | Every changed file with its hash |

Switch: `INTERVIEWER_BRAIN=off|on|allowlist` (default `off` = baseline V11).
