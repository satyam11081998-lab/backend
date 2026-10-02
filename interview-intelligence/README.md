# MECE Interview Intelligence

An independent interview-assessment service for MECE (mece.in). From a CV and a job
description, it builds a role-specific interview plan and runs an adaptive 15–60 minute
interview. Afterwards it produces an evidence-traced assessment: every score points back to
a verbatim quote, untested competencies are reported as untested rather than weak, and
confidence is shown separately from the assessment.

**Where it runs:** inside `consilio-backend` (this folder), mounted at `/ii` by
`routes/interview_intelligence.py` — no separate service. The backend contributes exactly one
thing: *who the caller is* (its own Supabase session check → user id, email, effective tier,
admin, guest). Everything else — the API, database schema, storage, prompts, model routing,
scoring, admin and audit — lives here, and this package imports nothing from the backend.
(It can still run as its own service with signed assertions — contract C10 — if it ever
needs its own box.)

| Read | For |
|---|---|
| `docs/00_AUDIT.md` | What exists in MECE today, and what this service deliberately does not reuse |
| `docs/A_ARCHITECTURE.md` … `docs/L_ADVERSARIAL_QA_MATRIX.md` | Architecture, data model, API contract (incl. proposed C10), AI pipeline, interview state, competency model, question engine, scoring, feedback, admin/access, Drive, QA matrix |
| `docs/M_SELF_REVIEW.md` | The hostile review: what was found and fixed, and what is still open |
| `docs/BUILD_STATUS.md` | What is built and tested, what is not built, and known compromises |

## Layout

```
interview_intelligence/
  api/            REST v1 (candidate + admin)          auth/        assertion verification
  access/         policy, flags, limits, audit         documents/   upload, parse, redact, encrypt
  cv_intelligence/ jd_intelligence/ company_intelligence/          role_taxonomy/ (+ data/*.json)
  competency_engine/ question_engine/ interview_engine/ interview_memory/
  evidence_engine/ evaluation_engine/ feedback_engine/ report_engine/
  ai/             providers, routing, prompt registry, structured outputs, injection guard, simulator
  jobs/           durable DB queue + worker threads    drive_integration/   voice/
migrations/       generated SQL (idempotent, own role, RLS)
qa/               golden dataset, real-model evaluation harness, interview simulator
tests/            offline test suite (Postgres or SQLite)
```

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # II_ENV=dev uses the offline simulated models
python -m scripts.generate_keys # II_ENCRYPTION_KEY (and standalone-only assertion keys)
uvicorn interview_intelligence.main:app --reload --port 8100   # standalone, for development
```

With `II_ENV=dev` and no AI keys, every model stage is served by a deterministic simulator,
so the whole flow works offline. The simulator checks plumbing, not judgement.

## Tests and QA

```bash
# Postgres is recommended (row locks, SKIP LOCKED and the 2-active race are only real there)
II_TEST_DATABASE_URL=postgresql+psycopg://postgres@localhost/ii_test pytest -q
pytest -q                                   # SQLite fallback (one Postgres-only test skipped)
python -m compileall -q interview_intelligence tests qa scripts

python -m qa.run_golden --simulated         # harness smoke test (offline)
python -m qa.run_golden --repeats 3 --record # REAL models: golden dataset, gates, regression vs last run
python -m qa.simulate_interview --all --simulated
python -m qa.simulate_interview --family sales --persona evasive --llm-candidate   # real models both sides
```

`qa.run_golden` exits non-zero if a hard gate fails (fairness gap for non-native phrasing
> 1 point, an off-topic answer scored as "weak", an injection raising a score, or keyword
stuffing rewarded). Run it before changing any prompt, model route or evaluator code.

## Deploy (host mode — the backend's existing Render service)

1. Supabase SQL editor, once: run `migrations/0001_interview_intel.sql` (idempotent), then
   `ALTER ROLE ii_service WITH PASSWORD '<generated>';`.
2. Render → consilio-backend → Environment: `II_DATABASE_URL` (Session pooler URL, user
   `ii_service.<project-ref>`, port 5432) and `II_ENCRYPTION_KEY`. Nothing else is required —
   AI keys are read from the backend's `OPENAI_API_KEY` etc. Until `II_DATABASE_URL` is set,
   `/ii/*` answers 503 and nothing of II is loaded.
3. Frontend: nothing to set — it calls `<NEXT_PUBLIC_API_URL>/ii` with the Supabase token.

The launch flag `ii.enabled_for_pro` starts **off**: only MECE admins (`users.is_admin`) and
test users added in Admin → Interview Intelligence get in until an admin turns it on.
Check: `GET <backend>/ii/healthz` → `{"ok": true, ...}`.
