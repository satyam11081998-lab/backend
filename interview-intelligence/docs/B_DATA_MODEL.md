# B · Data model — schema `interview_intel`

The models in `interview_intelligence/db/models.py` are the single source of truth;
`migrations/0001_interview_intel.sql` is generated from them (`python -m scripts.generate_migration`)
and a test fails if the two drift (`test_isolation::test_migration_matches_models`).
Nothing here references a MECE table: the only link to MECE is `users.id`, which is the MECE
user id copied from the signed assertion.

## 1. Entities and why they are shaped this way

| Group | Tables | Notes |
|---|---|---|
| Identity & access | `users`, `access_grants`, `system_config`, `audit_logs` | `users` is II's own registry (no MECE columns); grants are admin-managed test access by email; flags/limits are rows, not env, so admins change them without deploys; every admin mutation and session inspection is audited. A user with no access is **not** stored just for opening the page. |
| Documents | `documents`, `document_contents`, `document_analyses` | Metadata separate from content: `document_contents` holds Fernet-encrypted original bytes, extracted text and PII-redacted text (purged on delete). Analyses are cached per `(document, analysis_version)` and reused across documents with identical normalised text (`documents.text_sha256`), per user only. Versions per user and kind ("CV v3"). |
| Drive | `drive_folders`, `drive_files` | Folder ids by `(scope_key, kind)` so the tree is created once; one `drive_files` row per target with status, attempts and last error. Drive ids never leave the API. |
| Company | `company_profiles` | Per-user company context; `facts` is a JSON list where every fact carries its provenance (source type, basis, confidence, dates). JD-derived facts outrank everything else. |
| Question library | `questions` | Generated and CV-specific questions persisted with their full metadata (competencies, type, difficulty, intent, expected evidence, probes) so usage, repetition across a user's sessions and quality can be analysed. The curated bank and archetypes are versioned JSON in the repo. |
| Interview | `interview_sessions`, `interview_blueprints`, `interview_states`, `interview_exchanges`, `interview_messages`, `interview_events`, `claims` | Session = lifecycle row with explicit status (CHECK constraint) and the 2-active rule enforced under a per-user row lock. Blueprint = frozen plan + rubric + rubric hash + engine versions. State = one compact JSON document (the deterministic engine's working memory), versioned for optimistic updates. Exchange = one planned question with its follow-ups — the unit of question-level analysis. Messages carry `client_turn_id` for idempotency. Events are the replayable decision log. Claims are the CV/answer claim ledger used for contradiction detection. |
| Assessment | `evidence_items`, `competency_assessments`, `feedback_items`, `reports`, `competency_history` | Evidence rows hold verified verbatim quotes with polarity/strength and the extractor version. Assessments store score, band, evidence state, **separate confidence and its basis**, anomaly flags, evaluator version and rubric hash. Feedback items are queryable for recurring patterns. The report is an assembled JSON snapshot (what the user saw), with its own version. History rows make progress comparable by canonical competency. |
| Operations | `jobs`, `model_runs`, `evaluation_runs` | Durable queue (`FOR UPDATE SKIP LOCKED`, dedupe keys, backoff, dead-letter hooks). Every model call — including speech — with provider, model, prompt id@version, schema version, tokens, cost, latency, status and validation problems. Golden-dataset runs for regression comparison. |

JSON columns are used where the shape is genuinely flexible or a snapshot (analysis results,
blueprint, state, report, metrics); everything that is filtered, joined or aggregated is a
real column (status, competency ids, scores, confidence, costs, timestamps).

## 2. Retention and deletion

* Deleting a document purges its bytes and text immediately and queues Drive deletion;
  interviews that used it keep their transcript and report.
* Abandoned/expired sessions keep their rows (they are history); the transcript is the
  user's data and is only visible to them and, audited, to II admins.
* The default retention period is not enforced by a job yet — see `BUILD_STATUS.md`.

## 3. Isolation

Schema `interview_intel`, role `ii_service`, RLS on every table with one policy scoped to
`ii_service`, no grants to `anon`/`authenticated`, not exposed through PostgREST. Verified by
applying the migration twice to a fresh Postgres 16 database and probing with a role that was
given a stray SELECT grant: it sees zero rows; `ii_service` sees them.

## 4. Table reference (generated)

### `access_grants`
`id` uuid PK; `email_lc` varchar(320); `status` varchar(16); `grant_type` varchar(16); `note` Text; `granted_by` varchar(320); `created_at` timestamptz; `updated_at` timestamptz; `expires_at` timestamptz

Unique: (email_lc)

### `audit_logs`
`id` uuid PK; `created_at` timestamptz; `actor_user_id` uuid; `actor_email` varchar(320); `action` varchar(64); `target_type` varchar(64); `target_id` varchar(128); `meta` jsonb

### `drive_folders`
`id` uuid PK; `scope_key` varchar(64); `kind` varchar(16); `folder_id` varchar(128); `parent_folder_id` varchar(128); `name` varchar(128); `created_at` timestamptz

Unique: (scope_key, kind)

### `evaluation_runs`
`id` uuid PK; `created_at` timestamptz; `kind` varchar(24); `evaluator_version` varchar(32); `prompt_versions` jsonb; `dataset_version` varchar(32); `metrics` jsonb; `results` jsonb; `baseline_run_id` uuid; `notes` Text

### `jobs`
`id` uuid PK; `kind` varchar(48); `payload` jsonb; `status` varchar(16); `attempts` Integer; `max_attempts` Integer; `run_after` timestamptz; `locked_at` timestamptz; `locked_by` varchar(64); `last_error` Text; `dedupe_key` varchar(160); `created_at` timestamptz; `updated_at` timestamptz; `finished_at` timestamptz

Unique: (dedupe_key)

### `model_runs`
`id` uuid PK; `created_at` timestamptz; `user_id` uuid; `session_id` uuid; `stage` varchar(48); `provider` varchar(32); `model` varchar(96); `prompt_id` varchar(64); `prompt_version` varchar(32); `schema_version` varchar(32); `input_tokens` Integer; `output_tokens` Integer; `cost_usd` Numeric; `latency_ms` Integer; `status` varchar(24); `attempt` Integer; `error` Text; `validation` jsonb

### `questions`
`id` uuid PK; `origin` varchar(16); `role_family` varchar(64); `competency_ids` jsonb; `question_type` varchar(32); `difficulty` Integer; `text` Text; `text_hash` varchar(64); `meta` jsonb; `quality_status` varchar(16); `times_selected` Integer; `created_at` timestamptz

Unique: (text_hash)

### `system_config`
`key` varchar(128) PK; `value` jsonb; `updated_by` varchar(320); `updated_at` timestamptz

### `users`
`id` uuid PK; `email` varchar(320); `email_lc` varchar(320); `last_tier` varchar(16); `last_entitled` Boolean; `last_access_via` varchar(32); `drive_folder_key` varchar(64); `created_at` timestamptz; `last_seen_at` timestamptz; `deleted_at` timestamptz

Unique: (drive_folder_key)

### `company_profiles`
`id` uuid PK; `user_id` uuid → users; `company_name` varchar(200); `company_name_lc` varchar(200); `facts` jsonb; `created_at` timestamptz; `updated_at` timestamptz

### `documents`
`id` uuid PK; `user_id` uuid → users; `kind` varchar(8); `source` varchar(16); `label` varchar(200); `file_name` varchar(255); `mime_type` varchar(128); `ext` varchar(8); `size_bytes` Integer; `sha256` varchar(64); `text_sha256` varchar(64); `version` Integer; `page_count` Integer; `text_chars` Integer; `text_quality` jsonb; `parse_status` varchar(16); `parse_error` Text; `injection_flags` jsonb; `storage_status` varchar(16); `created_at` timestamptz; `deleted_at` timestamptz

### `drive_files`
`id` uuid PK; `user_id` uuid → users; `target_kind` varchar(16); `target_id` varchar(64); `drive_file_id` varchar(128); `drive_folder_id` varchar(128); `file_name` varchar(255); `mime_type` varchar(128); `sha256` varchar(64); `storage_status` varchar(16); `attempts` Integer; `last_error` Text; `uploaded_at` timestamptz; `created_at` timestamptz; `updated_at` timestamptz

Unique: (target_kind, target_id)

### `document_analyses`
`id` uuid PK; `document_id` uuid → documents; `user_id` uuid → users; `kind` varchar(8); `sha256` varchar(64); `analysis_version` varchar(32); `prompt_version` varchar(32); `model` varchar(96); `status` varchar(16); `result` jsonb; `quality` jsonb; `error` Text; `created_at` timestamptz; `updated_at` timestamptz

Unique: (document_id, analysis_version)

### `document_contents`
`document_id` uuid PK → documents; `blob_enc` bytea; `text_enc` bytea; `redacted_text_enc` bytea; `key_id` varchar(32); `purged_at` timestamptz

### `interview_sessions`
`id` uuid PK; `user_id` uuid → users; `status` varchar(16); `status_reason` Text; `cv_document_id` uuid → documents; `jd_document_id` uuid → documents; `company_profile_id` uuid → company_profiles; `source_session_id` uuid; `config` jsonb; `mode` varchar(32); `difficulty` varchar(16); `duration_minutes` Integer; `role_family` varchar(64); `role_title` varchar(200); `company_name` varchar(200); `pre_interview_summary` jsonb; `created_at` timestamptz; `started_at` timestamptz; `last_activity_at` timestamptz; `paused_at` timestamptz; `ended_at` timestamptz; `ended_reason` varchar(32); `active_seconds` Integer; `assessment_status` varchar(16); `assessment_confidence` varchar(16); `cost_usd` Numeric; `versions` jsonb; `deleted_at` timestamptz

### `claims`
`id` uuid PK; `session_id` uuid → interview_sessions; `claim_key` varchar(16); `source` varchar(16); `text` Text; `claim_type` varchar(32); `slots` jsonb; `priority` Integer; `verification_status` varchar(32); `exchange_id` uuid; `created_at` timestamptz

Unique: (session_id, claim_key)

### `competency_assessments`
`id` uuid PK; `session_id` uuid → interview_sessions; `competency_id` varchar(96); `canonical_competency_id` varchar(96); `name` varchar(200); `category` varchar(32); `importance` varchar(16); `evidence_state` varchar(32); `score` Integer; `band` varchar(32); `confidence` varchar(16); `confidence_basis` jsonb; `rationale` Text; `evidence_refs` jsonb; `strengths` jsonb; `gaps` jsonb; `missing_evidence` jsonb; `anomaly_flags` jsonb; `evaluator_version` varchar(32); `rubric_hash` varchar(64); `created_at` timestamptz

Unique: (session_id, competency_id)

### `competency_history`
`id` uuid PK; `user_id` uuid → users; `session_id` uuid → interview_sessions; `competency_id` varchar(96); `canonical_competency_id` varchar(96); `role_family` varchar(64); `score` Integer; `evidence_state` varchar(32); `confidence` varchar(16); `created_at` timestamptz

Unique: (session_id, competency_id)

### `feedback_items`
`id` uuid PK; `session_id` uuid → interview_sessions; `kind` varchar(16); `category` varchar(48); `severity` varchar(16); `competency_ids` jsonb; `title` varchar(300); `body` jsonb; `qa_status` varchar(16); `created_at` timestamptz

### `interview_blueprints`
`id` uuid PK; `session_id` uuid → interview_sessions; `role_profile` jsonb; `competency_model` jsonb; `rubric` jsonb; `rubric_hash` varchar(64); `blueprint` jsonb; `qa_report` jsonb; `engine_versions` jsonb; `created_at` timestamptz

Unique: (session_id)

### `interview_events`
`id` uuid PK; `session_id` uuid → interview_sessions; `type` varchar(48); `payload` jsonb; `created_at` timestamptz

### `interview_exchanges`
`id` uuid PK; `session_id` uuid → interview_sessions; `seq` Integer; `planned_qid` varchar(32); `section_id` varchar(32); `question_text` Text; `question_meta` jsonb; `competency_ids` jsonb; `difficulty` Integer; `status` varchar(16); `probes` Integer; `live_signals` jsonb; `metrics` jsonb; `evidence_status` varchar(16); `analysis` jsonb; `started_at` timestamptz; `closed_at` timestamptz

Unique: (session_id, seq)

### `interview_messages`
`id` uuid PK; `session_id` uuid → interview_sessions; `seq` Integer; `role` varchar(16); `kind` varchar(16); `content` Text; `exchange_id` uuid; `action` varchar(32); `client_turn_id` varchar(64); `meta` jsonb; `created_at` timestamptz

Unique: (session_id, seq), (session_id, client_turn_id)

### `interview_states`
`session_id` uuid PK → interview_sessions; `version` Integer; `state` jsonb; `updated_at` timestamptz

### `reports`
`id` uuid PK; `session_id` uuid → interview_sessions; `status` varchar(16); `report` jsonb; `report_version` varchar(32); `created_at` timestamptz

Unique: (session_id)

### `evidence_items`
`id` uuid PK; `session_id` uuid → interview_sessions; `exchange_id` uuid → interview_exchanges; `ref` varchar(16); `competency_id` varchar(96); `sub_competency` varchar(160); `evidence_type` varchar(32); `polarity` varchar(16); `strength` varchar(16); `quote` Text; `quote_verified` Boolean; `interpretation` Text; `ownership` varchar(16); `cv_consistency` varchar(16); `confidence` Float; `extractor_version` varchar(32); `created_at` timestamptz
