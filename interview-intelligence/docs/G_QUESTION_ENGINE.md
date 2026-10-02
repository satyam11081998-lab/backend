# G · Question engine

## 1. Sources, in priority order
1. **Curated bank** (`data/question_bank.json`) — hand-written questions per role family /
   competency with full metadata.
2. **Archetypes** (`data/question_archetypes.json`) — structured patterns (e.g.
   `ownership_deep_dive`, `quantified_impact_probe`, `prioritization_tradeoff`,
   `concept_explain_apply`, `debugging_walkthrough`, `system_design_lite`, `mini_case`,
   `metric_definition`, `stakeholder_conflict`, `failure_reflection`,
   `cv_claim_verification`, `contradiction_clarify`, `motivation_role_fit`,
   `estimation`, `pressure_pushback`, `company_role_simulation`, …).
3. **Generated role-specific questions** — the generator instantiates archetypes against
   *this* JD/CV; every generated question must carry the same metadata as a curated one.
4. **CV-specific questions** — one per prioritized claim, with a probe ladder
   (surface → depth → ownership → evidence → outcome → reflection).
5. **Adaptive follow-ups** — decided live by the policy, phrased by the interviewer,
   anchored to the item's `probe_tree`.

Unrestricted generation never controls the interview: the blueprint fixes *which
competencies, which claims, which requirements, how much time*; the generator only fills
slots inside that plan, and the policy decides when to move.

## 2. Question metadata (every planned item)
`qid, origin, archetype_id, role_family, competency_ids, sub_competency, question_type,
difficulty (1–5), seniority, technical_depth, ambiguity, expected_duration_s, intent (what
we are testing), expected_evidence[], strong_signals[], weak_signals[], red_flags[],
must_not_infer[], probe_tree[], follow_ups[], prerequisites[], selection_reason {req_ids,
claim_ids, explanation}`.

## 3. Selection
Score per candidate item for a section =
`0.40·competency_need + 0.25·requirement_link + 0.15·difficulty_fit + 0.10·cv_link
 + 0.10·source_quality − repetition_penalty(session + user's last 5 interviews)`.
Competency need rises when a competency is under-covered and falls once its minimum
evidence is met (spec §17 "stop probing if sufficient evidence obtained").

## 4. Quality control (blueprint QA, spec §52)
Deterministic: every `critical/high` competency has ≥ 1 item; total planned time within
±15 % of the budget; no near-duplicate items (token Jaccard ≥ 0.6); ≥ 1 follow-up per
item at depth ≥ deep; CV items reference existing claim ids; technical items only when the
role is technical or the mode asks for it; protected-topic filter (age, marital status,
family plans, religion, caste, health, nationality, politics…); leakage filter (questions
that contain their own answer or lead: "Wouldn't you agree…").
LLM judge: relevance to the JD, answerability, seniority fit, balance. Failing sections
are regenerated once; residual warnings are stored in `qa_report` and visible to admins.
