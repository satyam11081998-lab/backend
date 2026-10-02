# A · Architecture — MECE Interview Intelligence (II)

> **Deployed in host mode (2026-10-02).** II runs inside `consilio-backend`, mounted at `/ii`
> (`interview_intelligence/host.py`); the backend supplies identity from its own Supabase session check
> (and, optionally, recent business headlines). The standalone service with the signed assertion
> described below is still supported (set `NEXT_PUBLIC_II_API_URL`) but not used. The old standalone
> repository `interview-intelligence` is superseded by `consilio-backend/interview-intelligence`.

## 1. What is being built, and where

A standalone interview-assessment subsystem with its **own process, database schema,
storage workflow, AI orchestration, prompts, job queue, admin API and audit trail**.
MECE contributes exactly two facts per request — *who is this* and *are they entitled* —
through one signed assertion.

```
 Browser (mece.in)
   │  1. GET /api/interview-intelligence/token   (same-origin, Supabase cookie)
   ▼
 MECE Next.js server  ── verifies Supabase session, reads users.tier/is_admin,
   │                     signs a 5-minute Ed25519 "II entitlement assertion"
   │  2. { token }
   ▼
 Browser ── 3. Authorization: Bearer <assertion> ──►  II Gateway (FastAPI, own Render service)
                                                        │ verifies signature/iss/aud/exp/lifetime
                                                        │ upserts II-local user, applies II access policy
                                                        ▼
        ┌───────────────────────────── Interview Intelligence service ─────────────────────────────┐
        │ api/ (routers)  →  access/ (policy, flags, limits, 2-active rule)                        │
        │ documents/ (validate, parse PDF/DOCX/DOC, PII redaction, encrypted blobs)                 │
        │ cv_intelligence/ · jd_intelligence/ · company_intelligence/ (provenance)                  │
        │ role_taxonomy/ · competency_engine/ (library + role competency model + rubrics)           │
        │ question_engine/ (archetypes, curated bank, generation, selection, QA)                    │
        │ interview_engine/ (modes, difficulty, blueprint, lifecycle, turn orchestrator, policy)    │
        │ interview_memory/ (state, claims ledger, contradictions)                                  │
        │ evidence_engine/ (exchange evidence + measured answer metrics)                            │
        │ evaluation_engine/ (anchored evaluator, deterministic confidence, guards, QA judge)       │
        │ feedback_engine/ (feedback, bad-feedback detector, prep plan)                             │
        │ report_engine/ (report assembly, role-adaptive dimensions, progress, recurring weakness)  │
        │ ai/ (InterviewAIProvider, routing, prompt registry, structured outputs, injection guard)  │
        │ jobs/ (durable DB queue + worker threads)   drive_integration/   voice/   admin/          │
        └───────────────┬───────────────────────────────────────┬──────────────────────────────────┘
                        ▼                                       ▼
        Postgres schema `interview_intel`                Google Drive: "MECE Interview Intelligence"
        (own role, direct connection, not exposed        / Users / u_<opaque> / {CV, JD, Interviews}
         to PostgREST; can move to its own project
         by changing II_DATABASE_URL)
```

## 2. Isolation guarantees (and how each is enforced)

| Requirement | Enforcement |
|---|---|
| Own API layer | Separate FastAPI app, separate Render service, separate URL (`NEXT_PUBLIC_II_API_URL`) |
| Own DB | Schema `interview_intel`; dedicated DB role `ii_service` with privileges on that schema only; zero grants to `anon`/`authenticated`; not in PostgREST exposed schemas. II code never names a `public.*` table. A test asserts every model lives in `interview_intel`. |
| No MECE business logic | The II package imports nothing from `consilio-backend`. A test greps the source tree for forbidden imports (`services.`, `routes.`, `prompts.` of the old backend). |
| Own AI layer | `interview_intelligence/ai/*`: own provider clients, own keys (`II_OPENAI_API_KEY` …), own prompt registry with versions, own model-run log and budget |
| Entitlement not trusted from the browser | The browser only relays a token signed by MECE's server with a private key the browser never sees; II verifies with the public key. No request field such as `isPro` exists anywhere in the II API. |
| Interview state/scoring not in MECE DB | All tables are in `interview_intel`. MECE's DB holds nothing about interviews. |

## 3. Runtime topology

* **One web process** (`uvicorn interview_intelligence.main:app --workers 1`) plus
  **in-process worker threads** (`II_WORKER_THREADS`, default 2) that drain the durable
  `jobs` table. Jobs are claimed with `FOR UPDATE SKIP LOCKED`, so the same code scales to
  N web instances or a dedicated worker process (`python -m interview_intelligence.jobs.worker`)
  without change.
* **Hot path (live interview turn)**: 1 transaction, ≤ 2 fast-model calls (turn analysis +
  interviewer utterance), deterministic decision policy in between. Heavy evidence
  extraction is deferred to a job that runs as each question exchange closes.
* **Cold path (after the interview)**: `assess_session` job → evaluator per competency →
  QA judge → feedback → feedback QA → report assembly → history rows → Drive export.

## 4. Key design decisions

1. **Separate service, not a router inside consilio-backend** — memory isolation (the
   existing box is near its 512 MB cap), failure isolation, independent deploys, and it
   is what the spec prefers.
2. **Schema isolation now, project isolation later** — the code takes one
   `II_DATABASE_URL`. Recommended: a separate Supabase project. Acceptable today: the
   `interview_intel` schema in the existing project with its own role. Moving is a dump/
   restore of one schema.
3. **Asymmetric entitlement assertion (Ed25519)** — MECE holds the private key, II only
   the public key, so a compromised II cannot mint entitlements and the browser cannot
   forge them.
4. **Deterministic core, LLM at the edges** — time allocation, decision policy, coverage,
   confidence, evidence-count gates, contradiction bookkeeping and articulation metrics are
   plain code (testable, debuggable). LLMs produce language and structured judgements that
   are schema-validated and checked against evidence.
5. **Interviewer ≠ analyst ≠ evaluator ≠ feedback writer ≠ QA judge** — five prompt
   families, five model routes, no prompt both talks to the candidate and scores them.
6. **Rubrics frozen before the first answer** — built with the blueprint, hashed, stored;
   the evaluator only ever receives that rubric.
7. **Library as versioned code data** — role families, competencies, archetypes and the
   curated question bank are JSON in the repo (reviewed like code, versioned, loaded into
   memory). Runtime artefacts (generated questions, role-specific competencies, rubrics)
   are DB rows/JSON. New domains = a data-file change, no code change; unknown roles are
   handled dynamically (role-specific competencies generated from the JD).

## 5. Phasing actually delivered in this build

See `docs/BUILD_STATUS.md` (kept current with what exists, what is tested, what is
designed-but-not-built). Nothing is claimed as done that has not run.
