# J · Admin, access control and test users

## 1. Who can use II
`allowed = flag(ii.enabled) ∧ ( is_ii_admin ∨ active grant(email) of type test | ultra ∨ tier = ultra ∨
           ('interview_intelligence' ∈ assertion.ent ∧ flag(ii.enabled_for_pro)) ∨
           ((grant of type trial ∨ flag(plans.trial_open)) ∧ free interview not used up) )`

Plans (`access/plans.py`, Admin → Plans; shown, not charged):
* **Free interview** (`via=trial`): one per account, ever, `plans.trial_minutes` long (any requested
  length is replaced); allowed before it starts and while it is in progress; afterwards the account is
  read-only (its report stays). At most `plans.trial_max_prepared` interviews may be prepared before one
  is started. Optional cheaper voice: `plans.trial_voice_engine`.
* **Ultra** (`via=ultra`): an "ultra" grant, or tier "ultra" on the identity (no MECE plan yet; accepted
  so nothing changes in II when it ships). Fair use: `plans.ultra_monthly_interviews` started per 30 days.
* Plans page visibility `plans.visibility`: `access` (people with II, and accounts that used a free
  interview) or `off` (admins only). Everyone else gets 404.

* `ii.enabled` — master kill switch (admins keep access for debugging).
* `ii.enabled_for_pro` — launch flag; **default false**, so on first deploy only admins
  and test users get in.
* Pro is asserted by MECE (effective tier at token time, 5-minute tokens).
* Guests (anonymous) never receive a token.
* Read-only access to one's own past reports survives a lapsed subscription (the data is
  theirs); creating or continuing interviews does not.

## 2. Admins
`is_ii_admin = email ∈ II_ADMIN_EMAILS (env) ∨ (II_TRUST_MECE_ADMIN ∧ assertion.adm)`.
Default `II_TRUST_MECE_ADMIN=true` (MECE `users.is_admin` is a guarded column). No email
addresses are hard-coded anywhere in source.

## 3. Test users (spec §4)
Table `access_grants (email_lc unique, status enabled|disabled, grant_type, note,
granted_by, expires_at)`, managed from **Admin → Interview Intelligence → Test users**:
add (email + type: Full access | Free interview | Ultra), change type, enable/disable, delete, see status. Bootstrap placeholders:
`II_BOOTSTRAP_TEST_EMAILS=` (empty by default) — inserted once if absent, never
re-enabling a grant an admin disabled. Every change writes an `audit_logs` row.

## 4. Limits
* Max active sessions per user: `limits.max_active_sessions = 2` (backend-enforced, see E).
* Sessions per user per day: `limits.max_sessions_per_day = 5` (cost control; admins and
  test users configurable separately via `limits.max_sessions_per_day_test`).
* Durations allowed: 15/30/45/60 (default 45).
* Session cost cap and global daily budget (see D).

## 5. Control center (GET `/admin/overview`)
Users, active interviews, completed, failure rate (failed ÷ started, 7 days), AI latency
p50/p95 per stage, model usage and cost today, scoring/prompt versions in use, Drive sync
status counts, mode usage, question usage (top items), evaluation anomalies, recent
errors. Admin can inspect any session (transcript, events, model runs, assessment) — every
inspection is audit-logged.

## 6. Feature flags (`system_config`)
`ii.enabled`, `ii.enabled_for_pro`, `voice.enabled`, `company_intel.enabled`,
`company_intel.web_research`, `technical.advanced_mode`, `technical.coding_exercises`,
`admin.test_access`, `limits.*`, `drive.export_reports`, `ocr.enabled`, `plans.*` (visibility,
trial_open, trial_minutes, trial_max_prepared, trial_voice_engine, ultra_price_inr,
ultra_monthly_interviews). Env provides
defaults; DB overrides; 30 s cache.
