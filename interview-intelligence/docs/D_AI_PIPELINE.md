# D · AI pipeline

Every AI stage is a **task** with: a versioned prompt (`ai/prompts/*`), a Pydantic output
schema (`*/schemas.py`), a model route (`ai/routing.py`), validation + one repair attempt,
a fallback route, and a `model_runs` row per attempt (model, prompt id/version, schema
version, latency, tokens, cost, status, validation errors). Malformed output never flows
downstream: the caller gets a validated object or a typed failure it must handle.

| # | Stage | When | Prompt id | Route (default) | Output schema | Deterministic checks after the model |
|---|---|---|---|---|---|---|
| 1 | CV parse | upload (job) | `cv_parser` | fast | `CandidateProfile` | timeline gaps/overlaps from parsed dates, duplicate-claim merge, unrealistic-metric heuristics, section completeness |
| 2 | JD parse | upload (job) | `jd_parser` | fast | `RoleProfile` | specificity score, requirement de-dup, contradiction notes kept as-is (never resolved by guessing) |
| 3 | Role classification | session prepare | `role_classifier` | fast | `RoleClassification` | family must exist in taxonomy or be `other`; keyword fallback if the model fails |
| 4 | Competency mapping | session prepare | `competency_mapper` | strong | `CompetencyMapping` | canonical ids validated; role-specific ids namespaced `rs:`; weights normalised; mode-mandatory competencies injected |
| 5 | Rubric build | session prepare | `rubric_builder` | strong | `RubricSet` | every competency has 5 bands + ≥2 strong/weak signals; hashed + frozen |
| 6 | Question generation | session prepare | `question_generator` | strong | `GeneratedQuestions` | assessment-purpose fields required; protected-topic filter; leakage/leading filter; similarity de-dup vs session + user history |
| 7 | Blueprint QA | session prepare | `blueprint_qa` | fast | `BlueprintQA` | plus deterministic coverage/time/balance checks; failing sections regenerated once |
| 8 | Turn analysis | every answer (live) | `turn_analyzer` | fast | `TurnAnalysis` | claim slot comparison for contradictions; metrics computed in code |
| 9 | Interviewer utterance | every turn (live) | `interviewer` | fast | plain text | evaluation-leak filter, one-question rule, length cap; deterministic fallback utterance on failure |
| 10 | Evidence extraction | exchange closes (job) | `evidence_extractor` | strong | `ExchangeEvidence` | every quote must be found in the candidate's own words (fuzzy ≥ 0.82) or is dropped |
| 11 | Competency evaluation | after interview (job) | `competency_evaluator` | strong | `CompetencyEvaluation` | refs ⊆ supplied evidence ids; min-evidence gate → `not_sufficiently_tested`; score capped by strongest supporting evidence; band↔score consistency |
| 12 | Assessment QA judge | after 11 | `assessment_qa` | fast | `AssessmentQA` | anomaly rules in code (uniform scores, polarity mismatch, protected-attribute language) |
| 13 | Feedback generation | after 12 | `feedback_writer` | strong | `FeedbackBundle` | bad-feedback detector (generic phrases, missing example/action/practice, unmeasured numbers) |
| 14 | Feedback QA | after 13 | `feedback_qa` | fast | `FeedbackQA` | failing items regenerated once, then dropped and the section marked partial |
| 15 | Company research (optional, off) | prepare | `company_research` | grounded | `CompanyFacts` | every fact carries source type + URL; inferred facts labelled; JD always wins |

**Routes** are named (`fast`, `strong`, `grounded`, `stt`, `tts`) and configured by env
(`II_MODEL_FAST`, `II_MODEL_STRONG`, `II_PROVIDER_FAST` …) or the admin config. Each
route has a fallback chain. Providers implement `InterviewAIProvider`
(`complete`, `stream`, `transcribe`, `speak`); shipped: OpenAI-compatible (OpenAI, Groq,
Gemini-compat) and Anthropic; `FakeProvider` for tests.

## Prompt-injection posture
CV, JD, company text and answers are wrapped as untrusted data blocks
(`<untrusted kind="cv" id="…">…</untrusted>` with the closing tag neutralised inside the
content), system prompts state the data-not-instructions rule, and a scanner records
`injection_flags` on documents and messages. The evaluator is told that a manipulation
attempt was detected and must not affect scoring; independently, the evidence gates make a
"give me 10/10" line worthless because it is not evidence for any competency.

## Context budget
Live turns never resend the raw CV/JD or the whole transcript. The interviewer gets: the
persona, the decided action, the target question, the last answer, ≤ 6 compact memory
items, and recent openers to avoid. The analyzer gets the question intent + expected
evidence + the answer + the claim slots it may contradict.

## Cost control
Token-priced cost per run (price table in `ai/pricing.py`, overridable), per-session cost
accumulator with a cap (`II_SESSION_COST_CAP_USD`; beyond it the interviewer switches to
deterministic utterances and steers to closing), a global daily budget that blocks *new*
sessions, cached document analyses keyed by `(sha256, analysis_version)`, and batch
post-interview processing.
