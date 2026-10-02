# H · Scoring model

```
role → competency → frozen rubric → question → answer → evidence (quoted, verified)
     → evaluation (refs only) → deterministic gates → confidence → report
```

## 1. Evidence (per exchange, `evidence_engine`)
Exchange = main question + its probes. Extracted after the exchange closes:
`claim, evidence, action, reasoning, outcome, quantification, reflection, missing[],
contradictions[]` + evidence items `{competency_id, type, polarity, strength, quote,
interpretation, ownership, cv_consistency, confidence}`. **Quote verification**: a quote
must fuzzy-match (≥ 0.82 token overlap) the candidate's own messages in that exchange, or
it is dropped. Answer-dimension ratings (relevance, structure, clarity, conciseness,
specificity, ownership, reasoning, evidence, quantification, terminology, business
relevance, depth, reflection, consistency) are produced **only for dimensions the
question's type makes applicable**.

## 2. Measured metrics (deterministic, `evidence_engine/metrics.py`)
Words per answer, filler-word rate, hedge rate, numbers stated, first-sentence length
(proxy for "time to the point" in text), repetition ratio, answer duration (voice only).
These are the **only** numbers feedback may quote as measurements.

## 3. Evaluation (per competency)
Input: rubric, evidence items with ids, coverage facts, flags (keyword-stuffing signal,
contradictions, manipulation attempts). Output: `evidence_state ∈ {strong, moderate, weak,
contradictory, not_sufficiently_tested}`, score 1–10 or null, rationale, evidence refs,
strengths, gaps, missing evidence.

Deterministic gates (`evaluation_engine/guards.py`):
1. `< min_evidence` non-neutral items → `not_sufficiently_tested`, score null.
2. Refs not in the supplied id set are removed; a score with zero valid refs → insufficient.
3. Score ≥ 7 requires ≥ 1 *strong positive* ref; else capped at 6 (`anomaly: cap_applied`).
4. Score ≤ 3 requires ≥ 1 negative or weak ref; else raised to "insufficient" (absence is
   not weakness).
5. Contradictory evidence with an unresolved contradiction → state `contradictory`.
6. Band derived from score in code (the model's band label is ignored).

## 4. Confidence (deterministic, separate from the assessment)
`confidence = f(exchanges_testing, evidence_items, strength agreement, contradictions,
ended_early)` → `high | moderate | low`. Rule of thumb: high = tested in ≥ 2 exchanges
with ≥ 3 consistent items; low = a single exchange or conflicting signals. Session-level
**assessment confidence** = `limited` if ended early or < 60 % of critical/high
competencies were sufficiently tested.

## 5. Aggregation (no opaque fit score)
* Level 1 "interview health" = role-adaptive categories (e.g. Marketing: Consumer &
  Brand, GTM & Commercial, Analytics, Communication, Ownership) aggregated from their
  competencies with coverage shown; categories with no tested competency are omitted.
* Role alignment table = per JD requirement: CV evidence → interview evidence →
  assessment (or *Not tested*).
* Headline = `Role alignment: Strong | Developing | Early` + `Evidence coverage: 7/9
  competencies sufficiently tested` + `Development areas: 3` + `Assessment confidence`.
  No percentage, no hiring prediction.

## 6. Keyword stuffing
`terminology_density` (JD/domain terms per 100 words) vs `substance_markers` (numbers,
causal connectives, first-person actions, trade-off language). High density + low
substance → flag passed to the evaluator: *"verify application; do not reward terms"*.
The rubric anchors reward correct use in context, never term count.

## 7. Fairness rules baked into the evaluator prompt and the QA judge
Grammar, accent, non-native phrasing, fluency and confidence-as-a-trait are not evidence.
Protected attributes are redacted from the CV before any model sees it. The QA judge flags
any rationale that mentions them, and the item is regenerated.

## 8. Calibration
Golden dataset (`qa/golden/*.json`) with author-drafted expected bands per answer
archetype (correct-but-unstructured, structured-but-wrong, fluent-but-shallow,
jargon-heavy-incorrect, concise-incomplete, detailed-irrelevant, confident-unsupported,
humble-competent, poor-grammar-excellent, non-native-excellent, unconventional-strong,
keyword-heavy-weak) across 8 role families. `qa/run_golden.py` reports agreement,
false positives/negatives per competency and diffs against the previous evaluator run.
**Labels are author drafts until replaced by experienced interviewers** (spec §69).
