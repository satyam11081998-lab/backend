# MECE Interviewer — Security Report

Date: 2026-09-29. Scope: the unified-brain change set (backend `feat/unified-interviewer-brain`
`12308ec..HEAD`, frontend `f9841b6..HEAD`). No production system, database or key was accessed;
nothing here was tested against live infrastructure. Labels as in the other reports.

## 1. Secrets

| Check | Result | Status |
|---|---|---|
| OpenAI / Groq / Gemini keys stay in backend env; the browser receives only ephemeral client secrets (`/realtime/session` as before; new `/realtime/transcription-session`, `expires_after` 600 s) | Code review of `routes/realtime.py`; no new `NEXT_PUBLIC_*` secret | VERIFIED (code) |
| Production client bundle contains no key-shaped value | `next build` output (`.next/static`, 5.6 MB) scanned for `sk-…`/`sk-proj-…` values and JWT-shaped strings: **0** matches. Matches for the words `SUPABASE_SERVICE_ROLE_KEY` / `GROQ_API_KEY` are pre-existing admin help text naming env vars (no values) | VERIFIED (build made with placeholder env) |
| No secrets or candidate text in logs | `[interviewer.turn]` and `[interviewer.timing]` lines carry ids, lanes, timings, error types only; route test `test_voice_telemetry_endpoint_logs_no_content`; browser E2E "telemetry: no transcript text in any telemetry record" | VERIFIED |
| No secrets in git | `.env*` untouched and ignored; tests use `sk-test` dummies | VERIFIED |

## 2. Hidden prompt / control metadata / case solution

| Check | Result | Status |
|---|---|---|
| `mode` / `reason` / intervention names never reach the browser unless `INTERVIEWER_DEBUG_DECISIONS=1` | `/voice-decision` returns `{lane, say, event:null, turn_id, duplicate}`; route test `test_control_metadata_only_in_debug` | VERIFIED |
| Internal labels in a model reply are dropped before the candidate sees them | `validate._LEAK_RE` (MOVE:/TASK:/RULES:/CONTEXT:, intervention names, and since 2026-10-01 the control-packet field names: `response_function`, `may_say_correct`, `verified_claims`, "control packet"…); `test_control_metadata_never_becomes_visible_text`, `test_model_receives_a_json_control_packet_not_a_line` | VERIFIED |
| The fallback line for a contextual beat is never shown to the model, and contextual beats may not introduce case facts | packet `permissions.new_case_facts=false`; test asserts `fallback` is absent from the packet | VERIFIED |
| No unverified "you're right" | contextual beats drop sentences that say or imply the work is correct unless a check verified it (`unverified_claim`) | VERIFIED |
| Prompt-injection / identity / rubric turns | classified META → one deterministic in-role line, **no model call** (nothing to leak on that path). 14 attack strings in `test_injection_never_reaches_a_model_and_stays_in_role` | VERIFIED (decision layer); live model jailbreak resistance on turns that *do* reach a model: PARTIALLY VERIFIED (candidate text is wrapped as data, prompt says it carries no authority; not red-teamed against the live model) |
| The brain never reads `cases.solution` | Case text = `services.markets.llm_case_content(case)` (content + US note), same as V11; `test_prompt_never_contains_hidden_solution_or_secrets` | VERIFIED |
| Realtime session instructions carry no case content or policy | `VOICE_RENDERER_INSTRUCTIONS` only; each line is sent as an out-of-band response containing only the approved line | VERIFIED (code + browser E2E payload check) |
| **F16 — `cases.solution` readability through Supabase RLS** | `supabase/daily-read-policies.sql` in the repo defines `cases readable ... using (true)`. If that file was applied, any client holding the public anon key can `select solution from cases`. Not verifiable here (no DB access). Not changed (outside the interviewer's scope, and changing RLS could break case pages). | **UNVERIFIED — check it:** run in the Supabase SQL editor: `select policyname, cmd, qual from pg_policies where tablename = 'cases';` and, with the anon key, `GET /rest/v1/cases?select=solution&limit=1`. If it returns data, move `solution` to a separate table readable only by the service role, or revoke column access: `revoke select (solution) on public.cases from anon, authenticated;` |

## 3. Authorization and entitlements (unchanged)

- Every new branch reuses the baseline ownership check (`_load_attempt(supabase, attempt_id, user_id)`) and the baseline case loader; no new unauthenticated route. VERIFIED (code).
- Guests are refused on the new transcription-session and usage routes, as on `/transcribe`. VERIFIED (code).
- Plans, trial, billing, C9 quota and its counting, realtime credits, rate limits, scoring: unchanged. C9 parity test compares the brain route with the unchanged `count_clarifications` on 5 representative inputs (floor-yield, double clarification, hint request, number, double rhetorical question); a retried turn whose user row already exists is not counted a second time. VERIFIED.
- The identity behaviour changed: the brain says it is the AI interviewer when sincerely asked; V11's prompt told the model to claim to be human (F10). This is a deliberate honesty fix, not an entitlement change.

## 4. Abuse and cost-safety

| Surface | Control | Residual risk |
|---|---|---|
| `/attempts/{id}/messages`, `/voice-decision` | baseline rate limits, budget checks, per-attempt `keyed_lock`, turn-id ledger (duplicates replayed, not re-decided) | — |
| `/attempts/{id}/voice-telemetry` (new) | auth, 240/min/user, Pydantic model with length-capped string fields and a key whitelist in `telemetry.emit_timing`; nothing stored; no content logged | log volume only; the attempt id is not ownership-checked (a signed-in user could log timing lines against another attempt id — log noise, no data access) |
| `/realtime/transcription-session` (new, used only when the frontend is built with `NEXT_PUBLIC_STT_TRANSPORT=live`; default `whisper`) | auth, no guests, 6/min/user, global daily budget, per-user daily voice-minute quota checked at mint, 600 s secret expiry | **Metering is client-reported** (`/realtime/transcription-usage`, clamped to 10 min per report). A modified client could under-report streamed minutes; each secret is still bounded to 600 s and 6 mints/min. Keep the transport OFF until it is live-tested, or move to server-side metering (sideband) before enabling for all users. |
| `/speak`, `/transcribe` | unchanged | — |

## 5. Data integrity

- Silence is never written as an empty/placeholder row; errors never become silence; empty model output never becomes canned content (F5 path unreachable with the flag ON). VERIFIED.
- Idempotent inserts: in-process ledger + (after migration 0071) a partial unique index `(attempt_id, client_turn_id)`. Migration run on a real local Postgres (pgserver) on top of `0002_conversational_attempts.sql`: applied twice without error, a duplicate `(attempt_id, client_turn_id)` insert rejected by the unique index, rows with NULL `client_turn_id` unaffected. VERIFIED (local Postgres, not Supabase). The app degrades to plain inserts on a pre-migration database (route test `test_pre_migration_database_still_saves`). VERIFIED.

## 6. If the repositories are public
Everything added is safe to publish: no keys, no case solutions, no production data. The prompts
are in the repo, as V11's already were — treat the interviewer's prompt text as readable by anyone;
nothing security-relevant depends on it being hidden.
