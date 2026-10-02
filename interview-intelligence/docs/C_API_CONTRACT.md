# C · API contract

Two contracts: (1) the **MECE → II entitlement assertion** (the only coupling between the
systems; proposed as CONTRACTS.md **C10**), and (2) the **II REST API** used by the MECE
frontend.

## 1. MECE → II entitlement assertion (C10, v1)

**Issuer**: MECE Next.js route `GET /api/interview-intelligence/token` (Node runtime,
`Cache-Control: no-store`). It:
1. verifies the Supabase session server-side (`auth.getUser()`), rejects anonymous guests;
2. reads `users.subscription_tier, subscription_expires_at, is_admin, email` for that id;
3. computes `effectiveTier()` (the existing single helper);
4. signs a compact JWS with **Ed25519 (`alg: EdDSA`)** using `II_ASSERTION_PRIVATE_KEY`
   (PKCS#8 PEM, Vercel server env only — never `NEXT_PUBLIC_`).

**Claims** (all required unless marked):

| claim | type | meaning |
|---|---|---|
| `iss` | `"mece-app"` | issuer |
| `aud` | `"mece-interview-intelligence"` | audience — II rejects anything else |
| `sub` | uuid | MECE user id (II's user key) |
| `email` | string | account email (lower-cased by II for grant matching) |
| `tier` | `free\|lite\|pro` | **effective** tier at issue time |
| `sub_exp` | ISO-8601 \| null | subscription expiry (informational) |
| `ent` | string[] | `["interview_intelligence"]` iff effective tier is `pro` |
| `adm` | bool | `users.is_admin` |
| `iat`, `nbf`, `exp` | int | `exp - iat ≤ 900` (default 300) |
| `jti` | uuid | unique id (logged for audit) |
| `ver` | `1` | contract version |

**II verification** (`auth/assertion.py`): signature with `II_ASSERTION_PUBLIC_KEY`;
`alg` pinned to EdDSA (no `none`, no HS*); `iss`, `aud`, `ver`; `exp`/`nbf` with 30 s
leeway; **max lifetime 900 s** even if signed (defends against a mis-configured issuer);
`sub` must be a UUID. Any failure → `401` with a generic message.

**Entitlement is computed by II**, never accepted as a request field:
`can_use = flag(enabled) ∧ (('interview_intelligence' ∈ ent ∧ flag(enabled_for_pro)) ∨ active_test_grant(email) ∨ is_ii_admin)`.

Rotation: II accepts a comma-separated list of public keys (`II_ASSERTION_PUBLIC_KEYS`),
selected by `kid`, so keys rotate without downtime.

Versioning rule: adding an optional claim is additive; renaming/removing a claim, changing
`aud`/`iss`, or the algorithm is BREAKING (bump `ver`).

## 2. II REST API (v1)

Base: `${NEXT_PUBLIC_II_API_URL}/v1`. Auth: `Authorization: Bearer <assertion>` on every
route except `GET /healthz`. Errors: `{ "error": { "code": "...", "message": "..." } }`.
Resources owned by another user return **404** (never 403) to prevent enumeration.

### Identity & access
| Method | Path | Notes |
|---|---|---|
| GET | `/me` | `{user, access: {allowed, reason, via: pro\|test_grant\|admin}, limits: {max_active, active_count, sessions_today, max_per_day}, flags: {voice, company_intel}, is_admin}` |

### Documents (CV / JD)
| POST | `/documents` | multipart `file`, `kind=cv\|jd`, optional `label`. PDF/DOC/DOCX, ≤ `II_MAX_UPLOAD_MB` (5). Dedupes on `(user, kind, sha256)`. Returns document + analysis status |
| POST | `/documents/text` | JSON `{kind: "jd", text, label?}` for pasted JDs |
| GET | `/documents?kind=` | list own (no file bytes, no Drive ids) |
| GET | `/documents/{id}` | metadata + analysis summary (structured profile) |
| DELETE | `/documents/{id}` | soft delete, purge content, enqueue Drive delete |

### Sessions & interview
| POST | `/sessions` | `{cv_document_id, jd_document_id, config: {mode, depth, difficulty, duration_minutes, focus_areas?, target_competencies?, company?: {name, notes?}}, source_session_id?}` → `201 {session}`; `409 ACTIVE_LIMIT` with the active sessions when 2 slots are taken; `429 DAILY_LIMIT` |
| GET | `/sessions` | own sessions (history list) |
| GET | `/sessions/{id}` | status, config, progress, `pre_interview_summary` once `ready` |
| POST | `/sessions/{id}/start` | `ready → active`; returns the interviewer's opening message |
| POST | `/sessions/{id}/turns` | `{client_turn_id, content, kind: text\|voice, answer_ms?}` → `{messages: [...], session: {status, elapsed_s, remaining_s, section}}`. Idempotent on `client_turn_id`. Never returns scores or evaluation signals |
| POST | `/sessions/{id}/pause` · `/resume` | explicit pause/resume |
| POST | `/sessions/{id}/end` | user ends early → `completed` (`ended_reason=ended_early_by_user`), assessment enqueued |
| POST | `/sessions/{id}/abandon` | frees the slot, no assessment |
| GET | `/sessions/{id}/transcript` | candidate-visible transcript (after the interview only) |

### Reports, history, progress
| GET | `/sessions/{id}/report` | `202 {status: processing, stage}` while assessing; `200 {report}` when ready/partial |
| GET | `/sessions/{id}/questions/{exchange_id}/why` | "Why was I asked this?" |
| GET | `/progress` | comparable competency trajectories + recurring weaknesses |
| POST | `/sessions/{id}/reattempt` | `{target_competencies: [...]}` → new weakness-targeting session reusing CV/JD. 409 `not_finished` unless the source interview finished; 422 `bad_target` for ids that are neither library competencies nor `rs:` ids from the source interview |

### Voice (flag `voice.enabled`, default on; engine `voice.engine`: realtime | standard)
| POST | `/voice/transcribe` | multipart audio → `{text}` (standard voice) |
| POST | `/voice/speak` | `{text, voice?}` → `audio/mpeg` (standard voice, one sentence per call) |
| POST | `/voice/live` | `{session_id, voice?}` → `{client_secret, expires_at, model, voice, max_session_s, say_prefix}`. Mints a short-lived OpenAI Realtime secret for one running interview (ready/active/paused, owned). Session: semantic VAD, `create_response:false`, barge-in on, English transcription; instructions make the model a VOICE only. 403 `live_off` / 503 `live_unconfigured`, `live_failed`, `capacity` = use standard voice. |
| POST | `/voice/live/usage` | `{session_id, usage, kind: line\|ack}` → 204. Meters one spoken response (counted in the daily budget, not the per-interview AI cap). |

### Admin (`is_ii_admin` only)
| GET/POST | `/admin/access-grants` | list / add `{email, note?}` |
| PATCH/DELETE | `/admin/access-grants/{id}` | enable/disable / delete |
| GET/PATCH | `/admin/config` | feature flags + limits |
| GET | `/admin/overview` | users, active interviews, completed, failure rate, AI latency p50/p95, cost today, mode usage, versions, Drive sync counts, anomalies |
| GET | `/admin/sessions` · `/admin/sessions/{id}` | inspection (transcript, events, model runs) |
| GET | `/admin/model-runs` | recent runs, filter by stage/status |
| GET | `/admin/library` | role families, competencies, archetypes, curated bank stats |
| POST | `/admin/drive/retry-failed` | requeue failed syncs |
| GET | `/admin/evaluation-runs` | golden/regression results |

## 3. CORS

II allows `https://mece.in`, `https://www.mece.in`, `http://localhost:3000` and the
preview regex (same pattern the existing backend uses), credentials **not** required
(bearer tokens, no cookies).
