# I · Feedback model

## 1. Output (`FeedbackBundle`)
* `executive_assessment` — 3–5 sentences, every claim backed by refs.
* `strengths[]` — `{title, evidence_refs, why_it_matters}`.
* `development_areas[]` — `{category, severity, title, observed_problem, example_refs,
  why_it_matters, what_to_do, practice, measured_basis?}`.
* `interviewer_learned` — `{strong_signals, weak_signals, unproven_claims,
  missing_evidence, potential_concerns}`, each item with refs (spec §47).
* `next_questions[]` — "If this were a real interview, I'd ask next…", each tied to the
  gap that caused it (spec §45).
* `question_reviews[]` — for important exchanges: what worked, what was missing, what the
  interviewer was looking for, better answer direction, the follow-up that exposed the gap.
* `preparation_plan` — sessions of concrete drills linked to development areas, plus
  functional topics to revise (spec §41), and a targeted re-attempt suggestion (spec §42).

Categories (only those that apply): knowledge, functional depth, technical depth, role
understanding, structured thinking, problem solving, analytical reasoning, commercial
reasoning, communication, articulation, conciseness, specificity, ownership,
quantification, evidence, behavioral examples, leadership, stakeholder management,
situational judgment, domain terminology, CV defense, motivation, company understanding,
answer consistency, case skills.

## 2. Bad-feedback detector (deterministic, `feedback_engine/quality.py`)
An item is rejected if:
* it matches a generic pattern ("work on your communication", "be more confident",
  "study more", "use STAR", "practice more", "improve your articulation") **and** lacks a
  specific observed problem + example;
* it has no valid example ref (exchange id that exists) for a development area;
* `what_to_do` or `practice` is missing or under 8 words;
* it states a number (%, seconds, count) that is not one of the measured metrics passed in;
* it mentions accent, grammar-as-competence, personality, intelligence, mental state,
  honesty, or any protected attribute;
* it contradicts the assessment (e.g. calls a `strong` competency a weakness).

Then the LLM QA pass checks support, relevance, actionability, certainty. Failing items
get one regeneration with the reasons; still failing → dropped and the section flagged
`partial`. The report never shows fabricated certainty.

## 3. Longitudinal
* **Progress**: only canonical competencies tested with ≥ moderate confidence in both
  sessions are compared; role-specific competencies roll up to their parent.
* **Recurring weakness**: a development category (or canonical competency) flagged in
  ≥ 2 of the user's last 3 assessed interviews where it was tested → "Recurring pattern
  detected: ownership specificity", with links to each occurrence and a targeted drill.
