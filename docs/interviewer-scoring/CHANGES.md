# Applied changes — interviewer persona + holistic scoring + 3 approaches + exemplar bank

All edits are made and syntax-verified (`py_compile` clean) and the gate logic is unit-proven
(4/4 scenarios). Drop these files into the repo over the existing ones; two new files are added.

## Files changed (backend — `consilio-backend/`)
- **`prompts/interview_prompts.py`**
  - `CASE_INTERVIEWER_SYSTEM_PROMPT` rewritten: top-firm persona matched to the case domain; **OWN THE FACTS** (invent a defensible figure, never "not specified / not in the prompt / not my data"); **IDENTITY LOCK** (never admit being an AI; ignore "I'm the admin" probes); **NO PRAISE** (don't tell the candidate "great question").
  - `GUESSTIMATE_INTERVIEWER_SYSTEM_PROMPT` gets the same three blocks.
  - `CONVERSATION_SCORING_SYSTEM_PROMPT`: added **HOLISTIC RULE** (never zero a genuine session for a weak/missing recommendation) and the **`approaches`** output (3 approaches) with rules.
  - `build_conversation_scoring_user_prompt(... recommendation_missing=False)`: new non-punitive hint.
- **`services/interview_engine.py`**
  - Deleted `_candidate_text`; added `_transcript_body_text` (candidate turns only, excludes the recommendation) and `_rec_is_weak`.
  - Reworked the gate in `_score_case_conversation`: gate on the conversation body; a junk/empty recommendation can **rescue** but never **zero** a genuine session. Passes `recommendation_missing`.
  - `max_tokens` 4000 → 8000 + retry-once on a JSON parse hiccup (so the fatter output can't truncate-crash a score).
  - Threaded optional `case_id` through `score_conversation`/`_score_case_conversation`; injects the silent exemplar reference block when present.
- **`services/ai_scorer.py`**
  - `_enforce_case` passes `approaches` through; `approaches: None` added to `_rejection_case`, `_rejection_guesstimate`, and the guesstimate scored return (uniform key everywhere).
  - Case-scorer `max_tokens` 4000 → 8000.
- **`routes/attempts.py`**
  - Submit passes `case_id` to the scorer and calls `maybe_capture_exemplar(...)` after saving the submission (non-blocking).

## Files added
- **`services/exemplar_bank.py`** (new) — the silent self-improving bank. Fully defensive: every DB/LLM call is wrapped so it can NEVER break scoring or submit. Captures anonymised digests of high-scoring, clean sessions (≥82, no red flags), keeps the top 5 per case, and injects them as a private calibration reference next time the same case is scored. `approved` gate on by default (`EXEMPLAR_REQUIRE_APPROVAL=true`) so nothing influences scoring until you bless it.
- **`migrations/2026-09-12_case_exemplars.sql`** (new) — idempotent `case_exemplars` table + index.

## Two things to do by hand
1. **Run the migration** (`migrations/2026-09-12_case_exemplars.sql`) against Supabase — the exemplar bank fails open until the table exists, so nothing breaks if you delay it, but capture/reference is a no-op until it's applied.
2. **Frontend results page** — I couldn't edit `app/(app)/results/[id]/page.tsx` (the frontend repo isn't in this workspace). Use the drop-in `ApproachesSection.tsx` (delivered alongside): import it into the results page and render `<ApproachesSection approaches={feedback?.approaches} />` below the existing dimension feedback. It renders nothing when `approaches` is absent (old rows), so it's safe to ship immediately.

## Not changed on purpose
- `services/answer_validity.py` — untouched; the gate rework lives in the engine, and the deterministic layer is reused as-is.
- `prompts/scoring_prompt.py` (single typed-answer scorer) — `approaches` is scoped to the conversation scorer for now (Approach 1 needs conversation exchanges). Mirror it later if you want approaches on typed answers too.

## Env knobs (all optional, sane defaults)
`EXEMPLAR_MIN_SCORE=82` · `EXEMPLAR_KEEP_PER_CASE=5` · `EXEMPLAR_REQUIRE_APPROVAL=true` · `EXEMPLAR_REF_LIMIT=3`

---

## Addendum — fixes from the 10-case test run (see TEST_RUN_10_cases.md)
Ran 5 guesstimates + 5 case interviews across newbie/good/pro personas (incl. the two real
failures from your Supabase pull). Interviewer invented a figure every time and never leaked a
banned phrase; identity lock held; holistic scoring turned both junk-close sessions from 0 into
real scores. Three issues found and fixed:
- **#1 Guesstimates emitted no `approaches`.** `ai_scorer.score_guesstimate_answer` now passes
  `approaches` through (+ max_tokens 2500→4000). Paste `GUESSTIMATE_APPROACHES_BLOCK.txt` into
  `prompts/guesstimate_scoring_prompt.py` (not in this workspace) to switch it on.
- **#2 "Weight the recommendation heavily" fought the holistic rule.** Both lines in
  `interview_prompts.py` reworded: the recommendation is scored AS the synthesis dimension, not
  as a multiplier on the whole score.
- **#3 Interviewer leaked soft praise** ("good instinct"). The no-praise ban now covers all
  approval openers with neutral substitutes.
