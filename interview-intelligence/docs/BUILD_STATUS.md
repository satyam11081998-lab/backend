# Build status — 2026-10-02

Honest state of Interview Intelligence (II) after the first build. "Tested" means an
automated test or a recorded run exists (see `L_ADVERSARIAL_QA_MATRIX.md`); nothing below is
claimed because it was written, only because it ran.

## 0. Deployment changed: host mode (second pass, 2026-10-02)

II no longer gets its own Render service. To stay on the free plan with the existing keys, it
runs **inside `consilio-backend`**, mounted at `/ii` (`interview_intelligence/host.py`, glue in
`consilio-backend/routes/interview_intelligence.py`). The backend supplies only identity (its
own Supabase session check → user id, confirmed email, effective tier, admin, guest); II keeps
its own code, schema, role, RLS, prompts, routing, admin, audit and spend caps. Standalone
mode (signed assertion, C10) still works and is what the §1 numbers below were first run on.

| Gate (host mode) | Result |
|---|---|
| Offline suite incl. host-mode and voice tests, Postgres 16 | 279 passed (both the backend's pinned versions and newer ones) |
| Offline suite, SQLite | 278 passed, 1 skipped |
| Venv built from consilio-backend's own `requirements.txt` + SQLAlchemy 2.1.1, psycopg 3.3.6, python-docx 1.2.0, olefile 0.47 (Python 3.13) | resolves without conflicts; suite as above |
| Real backend `main.py` with II mounted, II on Postgres as `ii_service` after the migration | backend routes unchanged; II not imported at start-up; dormant → 503; identity errors; single CORS header; admin grants a test user; full interview → report via the background worker; cross-user 404; admin 200 / non-admin 403 |

Host-mode compromises (accepted to avoid a paid instance):
* **Shared process.** II adds ~42 MB RSS when it first loads and ~65 MB after a full
  interview (measured locally); nothing before the first `/ii` call. The backend already runs
  close to Render's 512 MB cap. Switch-off: unset `II_DATABASE_URL`.
* **Shared AI keys.** II falls back to the backend's `OPENAI_API_KEY` etc.; II's own per-session
  cost cap and daily budget still apply and are visible in II's admin.
* **Latency.** Backend in Oregon, database in Tokyo: an interview turn runs ~22 SQL statements
  (~2 s of round trips) on top of the model call. Batching can cut this later.
* **One worker thread**, polling every 20 s when idle and fast while there is work; DB pool 3.
* Run II's tests from inside `interview-intelligence/` (its `tests` package would clash with
  the backend's if pytest were run from the backend root).

## 0b. Voice call stability, Gemini Live, per-step model routing (fourth pass, 2026-10-02)

* **Echo fix.** On laptop speakers the interviewer's own voice leaked into the mic, cut the
  interviewer off mid-sentence (OpenAI's server-side interruption) and could be sent as an
  answer. Server-side interruption is now off (`interrupt_response: false`, Gemini
  `NO_INTERRUPTION`); the browser cuts in only when the words heard are not the line's own
  (two or more foreign words, ≥ a third of what was heard), and drops near-exact echoes of the
  last line (fuzzy, ≥ 85 % of words) for 8 s after it ends. OpenAI lines are spoken with
  `input: []` — out of the conversation context, so a line costs the same at minute 40 as at
  minute 1 and the model never answers the candidate.
* **Cost guards.** One answer is capped at 5 minutes (warning at 4:30, then it is sent and the
  interview moves on). Silence ladder per question: nudge at 1 min, "are you still there?" at
  3, pause + hang-up at 4. Two minutes on another tab (or a locked phone) = pause + hang-up. A
  break closes the voice line and releases the mic; "Resume" redials. Nothing streams or is
  billed while paused.
* **Gemini Live** as a third engine (`voice.engine` = realtime | gemini | standard, Admin →
  Settings). II mints a constrained ephemeral token (model, voice, instructions, transcription,
  no-interruption and a 3 000-token sliding context window pinned server-side; 4 config tiers,
  the browser steps down if Google refuses one at setup). Lines go as `SAY: …`; only the reply
  to a SAY is played, Gemini's own answers are discarded and a SAY waits until one is over; a
  guard cuts a reply that goes off script and asks once more. Mic audio streams only while the
  candidate talks (local VAD + 600 ms pre-roll, `audioStreamEnd` after). Google's ~10-minute
  connection limit and network drops redial the same engine (3 per 2 minutes, then standard).
  Metered per minute (`II_GEMINI_LIVE_PRICES`), inside the daily budget.
* **Per-step model routing** (Admin → AI routing; flag `ai.routes`; `II_MODEL_ROUTES` still
  wins). Defaults: resume/JD parsing, role family and plan check on Gemini (free tier), OpenAI
  as fallback; competency mapping, rubrics, questions, answer analysis, interviewer, evidence,
  scoring and feedback on OpenAI. Free-tier caveat: Google may use free-tier content to improve
  its products; documents are PII-redacted first (contact details, protected attributes), names
  and employers are not.

| Gate (fourth pass) | Result |
|---|---|
| II suite, Postgres 16 (backend-pinned venv and newer) | 287 passed |
| II suite, SQLite | 286 passed, 1 skipped |
| Voice node tests (conductor, OpenAI transport, Gemini transport) | 47 / 47 |
| Frontend `tsc --noEmit` (device) / `next build` (copy) | exit 0 / OK |
| Browser, Gemini call vs a local fake Gemini Live socket | token config pinned (voice, no-interruption, window); opening line sent verbatim as SAY; 2 spoken answers reached II as `kind=voice`; no ack sent; no SAY while Gemini's own answer was open; mic streamed only in speech stretches; break closed the socket and metered usage; resume redialled; a dropped socket redialled by itself and the interview continued; admin AI-routing tab change + reset |
| Browser, OpenAI live call vs mock WebRTC peer | lines sent with `input: []`; echo of the line did not cut in or become an answer; a sound alone did not cut in; real words did (cancel + clear); 20 responses metered |
| Browser, standard voice (fake mic) | 2 voice answers, typed answer, mute, break/resume, end |
| Browser, 2 minutes on a hidden tab | interview paused (server status `paused`), line closed |
| NOT verified | real Gemini Live / OpenAI audio (first heard on the live site); Gemini free-tier concurrency limits |

## 1. Verification run on 2026-10-02

| Gate | Result |
|---|---|
| Offline suite, Postgres 16 | 260 passed |
| Offline suite, SQLite | 259 passed, 1 skipped (the Postgres-only concurrency race) |
| `python -m compileall` (service, tests, qa, scripts) | clean |
| Migration `0001_interview_intel.sql` applied twice to a fresh Postgres 16 DB | idempotent; 26 tables, RLS on 26, 26 policies; a role with a stray SELECT grant sees 0 rows, `ii_service` sees them |
| MECE frontend `tsc --noEmit` (whole project, with the II files) | exit 0 |
| MECE frontend `next build` (copy of the repo; Google Fonts stubbed because the sandbox has no internet) | compiled; all 6 II routes built |
| Browser walk-through (Playwright, real II service with simulated models + the real II components) | free-user gate, admin test-user add, settings, upload → paste JD → role understanding → modes → difficulty → duration → build → room → 5 turns → end → report → mobile; no console errors from II |
| Token interop | TS signer (`lib/interview-intelligence/assertion.ts`) → Python verifier and the live service: accepted |
| `qa.run_golden --simulated` (96 items) | harness and gates work; **not a quality measurement** (see §3) |
| `qa.simulate_interview --all --simulated` (8 families × 7 personas) | 56 interviews completed, 56 reports generated |

## 2. Built

* Independent FastAPI service, own schema/role/RLS, own durable job queue, own audit log.
* Entitlement via Ed25519 assertion (proposed C10); access policy with kill switch, launch
  flag (`ii.enabled_for_pro`, default **off**), admin-managed test users, bootstrap
  placeholders for two test accounts (no addresses in code), lapsed-Pro read-only history,
  PII minimisation for users who never had access.
* Two active interviews per user under a per-user row lock; daily limits; explicit
  lifecycle (`created … failed`) with idle pause → expiry sweeper.
* Documents: PDF, DOCX and best-effort legacy DOC; magic-byte validation, macro/zip-bomb
  rejection; protected-attribute redaction before any model call; Fernet encryption at rest;
  byte and text-level dedupe; analysis cache per `(document, analysis_version)` reused across
  identical text for the same user.
* CV and JD intelligence with deterministic checks (timeline gaps/overlaps/future dates,
  duplicate and implausible claims, JD contradictions, no invented seniority).
* Role taxonomy and competency framework as versioned data: 30 families, 78 competencies,
  24 question archetypes, 113 curated questions; role-specific (`rs:`) competencies from the
  JD; keyword fallback classifier.
* 16 modes × 3 depths × 5 difficulties × durations, composed into a blueprint: section
  allocation, competency coverage **with repair** (a key competency without a question gets a
  redundant slot; gaps that cannot fit the duration are disclosed before the interview),
  frozen hashed rubrics, CV claims to investigate, adaptive branches, exit criteria,
  deterministic + model QA.
* Question engine: curated → archetype → generated → CV-specific → adaptive probes, every
  item with intent/expected evidence/signals/probe tree; quality filters; generated questions
  kept in a cross-user library (CV-derived questions never are).
* Live interviewer: deterministic decision policy (probe, challenge, clarify contradiction,
  move on, adjust difficulty, close), interview memory and claim ledger, neutral
  contradiction clarification, pressure/grill pushback, pause/resume/break, refusal handling,
  idempotent turns, deterministic fallbacks on any model failure, evaluation-leak filter,
  per-session cost cap with graceful close, global daily budget.
* Evidence engine (verbatim-quote verification; instructions to the system never become
  evidence; tested gap vs. untested), measured answer metrics, keyword-stuffing signal.
* Evaluator per competency with anchored bands, evidence-ref validation, deterministic gates,
  separate confidence with its basis, QA judge with re-run on protected-attribute rationale.
* Feedback with the bad-feedback detector (generic, unmeasured numbers, protected language,
  untested competency, contradicting the assessment, hiring predictions) → one regeneration →
  withheld and report marked partial.
* Report: headline facts (no single fit score), role-adaptive health dimensions, competency
  map with evidence drawers, JD→CV→interview→assessment alignment, "what the interviewer
  learned", critical moments, CV claims, communication metrics, coverage map, question-level
  review with "why was I asked this?", next questions, preparation plan, progress deltas
  (comparable points only), recurring patterns, targeted re-attempt, transcript.
* Google Drive storage: opaque per-user folders created once, crash-safe dedupe, retries,
  non-retryable failures surfaced, admin retry, delete propagation, report export.
* **Voice call (default way to take the interview).** Lobby (mic check with level meter, mic picker,
  interviewer voice) → full-screen call: animated voiceprint orb driven by the real audio levels,
  captions, live words while the candidate talks, mute / repeat / type an answer / break / end,
  transcript drawer, keyboard shortcuts, screen wake lock, gentle nudge after 45 s of silence,
  auto-pause after 4 min. Two engines, one brain: **live** (OpenAI Realtime over WebRTC; words-based
  end of turn, barge-in, streaming transcript; the speech model only voices II's lines — auto-replies
  off) and **standard** (own VAD → /transcribe → /turns → /speak sentence by sentence). Live falls
  back to standard by itself; the admin picks the engine (`voice.engine`). A turn-taking conductor
  keeps a thinking pause inside one answer, drops echoes and recogniser hallucinations, closes the
  mic while II thinks, fills the gap with a neutral "Okay.", and sends every answer through /turns
  with an idempotent turn id (retried on network blips).
* Text interview restyled as a conversation thread (dictation mic, typing indicator).
* Admin control center: test users, flags/limits, health (failure rate, AI latency p50/p95,
  cost, mode usage, top questions, evaluation anomalies, Drive sync, versions), sessions with
  audited inspection, model runs, evaluation runs, audit log.
* Observability: every model call (including speech) with prompt id@version, schema version,
  tokens, cost, latency, status and validation problems; unpriced models flagged.
* MECE frontend: token route, hub, guided setup, interview room, report, admin page, one
  admin-nav entry.

## 3. Not built, or not yet proven — read before launch

1. **Assessment quality with real models has not been measured.** This environment has no
   AI keys, so every run above used the deterministic simulator. Before turning on
   `ii.enabled_for_pro`, run `python -m qa.run_golden --repeats 3 --record` with the
   production model routes and read the archetype agreement and gate results.
2. **Golden labels are author drafts.** Spec §69 (independent human interviewers scoring a
   sample, agreement tracking) has not happened. The dataset is a regression and safety
   harness until reviewed.
3. **Voice: live calls verified against a mock peer, not yet against OpenAI from this sandbox.**
   Real Chromium WebRTC to a mock realtime peer (same events OpenAI sends) and the standard path
   with a fake microphone both ran end to end; the real OpenAI Realtime voice quality and latency
   are first heard on the live site. Turn latency = end of speech + II's turn (DB round trips
   Oregon↔Tokyo + one or two fast-model calls), so expect ~2–5 s, covered by the spoken "Okay.".
   Interviewer replies are not streamed token by token.
4. **No coding sandbox.** Spec §33 executable tests are not built
   (`technical.coding_exercises` off). Technical depth is assessed through conversation.
5. **No web research for company intelligence.** Only JD-stated facts and what the user
   types, each with provenance (`company_intel.web_research` off). Nothing is invented.
6. **No OCR.** Image-only CVs are reported as unreadable with a clear message.
7. **No automatic retention job.** Deletion on request works; a time-based purge is not
   scheduled.
8. **Interviewer replies are not streamed** token by token; each turn returns whole
   (≤ 2 fast-model calls on the hot path).
9. **Rate limiting is per process.** Fine for one Render instance; a shared store is needed
   before scaling out. (In host mode the backend must keep running a single worker process.)
10. **Employer mode** is not built (the data model keeps candidate preparation separate from
    any hiring decision).
11. **Navigation link** ("More → Interview Intelligence") is shown only to accounts II lets in
    (admins, test users, and Pro once `ii.enabled_for_pro` is on), checked on every page load.
12. Legacy `.doc` parsing is best-effort; users are asked to re-save as PDF/DOCX when it fails.

## 4. Known compromises

* **Same Supabase project, separate schema.** Isolated by schema, role, RLS and no PostgREST
  exposure; moving to its own project is a one-schema dump/restore plus a new
  `II_DATABASE_URL`.
* **Bearer assertion.** A stolen assertion is usable until it expires (≤ 5 minutes by MECE,
  ≤ 15 enforced by II). Mitigated by HTTPS, short lifetime and no storage in localStorage.
* **MECE admins are II admins** by default (`II_TRUST_MECE_ADMIN=true`); set it to `false`
  and use `II_ADMIN_EMAILS` to separate the two.
* **Simulated models in dev/test.** They make the full flow testable offline but produce
  template-like questions and simple heuristics; production refuses to use them.
* **Report is a snapshot.** Reports are stored as assembled JSON (what the user saw), with
  the underlying assessments, evidence and feedback also stored as rows.
* **MECE frontend lint is broken upstream** (`eslint-config-next@16` with ESLint 8 throws a
  circular-config error); `next build` ignores lint. Not changed here.
