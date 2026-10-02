# L · Adversarial QA matrix

Status legend: **A** = automated test in `tests/` (offline, runs in CI against Postgres and
SQLite) · **H** = harness with a real model (`qa/`), to run before enabling a prompt or
evaluator change · **B** = browser walk-through (Playwright, recorded in the handoff) ·
**M** = manual check listed in the handoff.

The "A" column names the file and test. Everything marked A passed on 2026-10-02
(Postgres 16: 260 passed; SQLite: 259 passed, 1 Postgres-only race test skipped).

| Stage | Attack / edge case | Expected behaviour | How verified |
|---|---|---|---|
| A · CV parsing | malformed PDF, zip-bomb DOCX, macro DOCX, extension ≠ magic bytes, empty, over size cap | rejected with a specific message, nothing stored | A `test_documents::test_validation_rejections`, `test_docx_macro_bomb_and_fake` |
| | image-only PDF | `unreadable` + "scanned image" message, never guessed | A `test_documents::test_image_only_pdf_is_reported_not_guessed` |
| | garbage `.doc` | unreadable with message | A `test_documents::test_garbage_doc_is_unreadable_with_message` |
| | very long CV | text capped, warning recorded | A `test_documents::test_very_long_cv_truncated_with_warning` |
| | tables / multi-section DOCX, real PDF | parsed in reading order | A `test_documents::test_pdf_and_docx_parse_including_tables` |
| | inconsistent, overlapping, future, reversed dates | deterministic `timeline_issues` | A `test_postprocess::test_gap_overlap_future_and_reversed_dates` |
| | duplicate claims, unrealistic metrics, scale vs seniority | flagged and prioritised for probing | A `test_postprocess::test_claim_flags_duplicates_unrealistic_and_seniority` |
| | prompt injection in CV | `injection_flags` set; wrapped as untrusted; cannot raise scores | A `test_documents::test_injection_in_cv_is_flagged`, `test_units::test_injection_patterns_detected` |
| | protected attributes (DOB, religion, marital status…) | redacted before any model call | A `test_documents::test_pii_redaction_cv_vs_jd`, `test_upload_flow_dedupe_encryption_delete` |
| | same CV re-saved (different bytes, same text) | new version, analysis reused, never across users | A `test_documents::test_resaved_file_with_same_text_reuses_the_analysis` |
| B · JD parsing | short / vague JD | rejected or `specificity=low`; seniority not invented | A `test_documents::test_short_jd_rejected_and_wrong_kinds`, `test_postprocess::test_jd_dedupe_specificity_and_contradictions` |
| | duplicated requirements, contradictions ("entry level… 8 years") | de-duplicated; contradiction listed; confidence lowered | A same |
| C · Role classification | model unavailable / unknown title | keyword fallback with whole-word, specificity-ranked matching; `other` when unclear | A `test_units::test_keyword_classifier_fallback` (20 titles), `test_classifier_ignores_common_words_in_prose` |
| D · Competency mapping | invented ids, missing mode-mandatory competencies | dropped / `rs:`-namespaced; injected | A `test_role_families` (family defaults always present) |
| | targeted re-attempt with a made-up competency | 422 `bad_target` (targets reach prompts as names) | A `test_progress::test_reattempt_progress_and_recurring_patterns` |
| E · Questions | protected topic, leading, answer leak, stacked, markdown, too short | rejected by QA | A `test_units::test_question_quality_rejects` (7 cases) |
| | curated bank quality | every curated question passes QA | A `test_units::test_curated_bank_contains_no_question_that_fails_qa` |
| | repeats / near-duplicates / one competency monopolising | selector spreads and de-duplicates; earlier sessions avoided | A `test_units::test_selection_spreads_competencies_and_avoids_repeats` |
| | wrong seniority / family | filtered | A `test_units::test_curated_candidates_respect_family_and_seniority` |
| | key competency with no planned question | coverage repair swaps a redundant item; unfixable gaps disclosed before start | A `test_role_families` (8 families × different modes) |
| | question loops | never re-asks a closed item | A `test_policy_and_memory::test_probe_until_limit_then_advance` |
| F · Interview adaptation | control intents (end, break, repeat, thinking, clarify, meta, off-topic) | correct action per intent | A `test_policy_and_memory::test_control_intents` |
| | refusal | moves on; competency *not tested*, never weak | A `test_policy_and_memory::test_refusal_moves_on_and_marks_declined`, `test_interview_adversarial::test_end_early_gives_limited_confidence_and_untested_is_not_weak` |
| | silence / one-word answers | nudge once, then move on | A `test_policy_and_memory::test_non_answer_nudges_once_then_moves_on` |
| | 5-minute answer | accepted, capped | A `test_interview_adversarial::test_five_minute_answer_is_accepted_and_capped` |
| | contradiction vs CV / earlier answer; self-correction; small differences | clarified once, neutrally; corrections and tolerance respected | A `test_policy_and_memory::test_contradiction_anchoring_and_once_only`, `test_small_differences_and_self_corrections_are_not_contradictions` |
| | model timeout / API failure mid-interview | deterministic line, interview continues | A `test_interview_adversarial::test_model_failures_never_stall_the_interview` |
| | interviewer leaks evaluation ("great answer", "score") | filtered | A `test_interview_adversarial::test_interviewer_evaluation_leaks_are_filtered` |
| | live payload carries evaluation | never | A `test_interview_adversarial::test_live_payload_never_contains_evaluation` |
| | duplicate submit, refresh, pause/resume, break by chat | idempotent; resumable | A `test_interview_adversarial::test_idempotent_turn_replay`, `test_pause_resume_and_refresh` |
| | injection as an answer | redirected, logged as `injection_attempt` | A `test_interview_adversarial::test_unrelated_and_injection_messages_are_redirected` |
| | per-session cost cap; daily budget | degrade and close; block new sessions only | A `test_interview_adversarial::test_session_cost_cap_switches_to_degraded_and_closes`, `test_daily_budget_blocks_new_sessions_only` |
| G · Evidence | hallucinated quotes | dropped by verbatim-quote verification | A `test_scoring_integrity::test_hallucinated_quotes_never_become_evidence` |
| | instruction to the evaluator quoted as "evidence" | dropped (`dropped_manipulation_quotes`) | A `test_qa_harness::test_harness_scores_through_the_real_pipeline_and_gates_work` |
| | generic reply after a follow-up | recorded as a tested gap (negative), not "untested" — prompt `evidence_extractor@2` | A (simulator parity) + **H** |
| H · Scoring | 12 answer archetypes × 8 families (spec §58): fluent-shallow, jargon-wrong, stuffing, non-native-strong, hedged-correct, rambling-strong, injection… | within drafted ranges; fairness gaps ≤ 1.0; injection never raises a score; off-topic never "weak" | **H** `python -m qa.run_golden` (gates fail the run); harness itself A `test_qa_harness` |
| | evaluator returns 10/10 with unknown refs | evaluator retried, then `evaluator_failed`, no score | A `test_scoring_integrity::test_compromised_evaluator_cannot_award_unsupported_scores` |
| | high score without strong evidence; low score without negative evidence; min-evidence gate; polarity mismatch | capped / nulled / insufficient | A `test_scoring_integrity::test_*guards*` (7 unit tests) |
| | rationale mentions accent / grammar | QA judge flags and re-runs | A `test_scoring_integrity::test_assessment_qa_flags_protected_attribute_rationale` |
| I · Feedback | generic, unmeasured numbers, protected-attribute language, untested competency, contradicts assessment, bad refs, hiring predictions | rejected → regenerated once → withheld + report marked partial | A `test_scoring_integrity::test_*feedback*` (6 tests) |
| J · Report | role-adaptive dimensions; every quote verbatim from the candidate; protected words absent; untested = no score | per family | A `test_role_families::test_role_family_run[8 families]` |
| | progress compares only comparable points; recurring patterns need ≥ 2 interviews | | A `test_progress` |
| K · Access control | no token, expired, wrong aud/iss, `alg=none`, HS256 confusion, tampered, lifetime > 900 s, bad sub/version, future iat, rotation, no keys | 401 | A `test_auth` (15 tests) |
| | fake `isPro` fields; free/lite; launch flag; kill switch; test grant lifecycle; lapsed Pro reads own history | 403 / allowed as specified | A `test_access` (11 tests) |
| | cross-user document / session / report / transcript / why | 404 | A `test_isolation::test_cross_user_access_is_404` |
| | 3rd active session; parallel creates | 409; never 3 active | A `test_concurrency` (race test Postgres-only) |
| | unset `II_ENV` in production | treated as production: no simulated AI, no ephemeral key, no /docs | A `test_ai_runner::test_unset_environment_fails_safe_to_production` |
| | browser-side signer ↔ II verifier | interoperable (EdDSA) | B (signer compiled from `lib/interview-intelligence/assertion.ts`, verified by `auth/assertion.py` and the live service) |
| L · Storage | dedupe, delete purges content, ciphertext at rest | | A `test_documents::test_upload_flow_dedupe_encryption_delete` |
| | schema isolation, RLS denies other roles even with a stray grant | | A `test_isolation::test_every_table_is_in_the_isolated_schema`; migration applied twice + RLS probe on Postgres 16 (handoff) |
| M · Drive sync | folder tree once, opaque names, no ids to the browser | | A `test_drive_sync::test_documents_land_in_an_opaque_per_user_tree_created_once` |
| | crash after upload before DB write | appProperties lookup, no duplicate | A `test_drive_sync::test_crash_after_upload_does_not_create_a_duplicate` |
| | 5xx / token errors; 403 | retried; not retried + failed + admin retry | A `test_drive_sync::test_transient_errors_are_retried`, `test_permission_errors_are_not_retried_and_marked_failed` |
| | delete; report export; not configured | | A `test_drive_sync` (3 tests) |
| Voice | on by default and switchable; no access; bad audio; live secret minted server-side with auto-replies off and no interview content; foreign/finished interview refused; engine off / no key / OpenAI error → standard fallback; usage metered outside the interview cap; speech calls logged and costed | | A `test_voice`; turn-taking + live transport: consilio `qa/interview-intelligence/voice` (25 node tests); browser runs with fake mic and a mock realtime peer |
| UI | gate for free user, admin test users + settings, setup (upload, paste, role understanding, modes, difficulty, duration, build), room, end, report, mobile | renders, no console errors from II | B (16 screenshots, handoff) |
| | real mic / speaker permissions, Safari, accidental close mid-answer | | M |
