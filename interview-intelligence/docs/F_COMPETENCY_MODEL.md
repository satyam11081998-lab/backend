# F · Role taxonomy + competency model

## 1. Three layers

1. **Library (versioned data, `interview_intelligence/data/`)**
   * `role_families.json` — ~30 families (consulting, finance sub-families, marketing,
     sales, product, strategy, operations, supply chain, HR, general management, BD, data
     analytics/science, software/product engineering, IT, cybersecurity, cloud/DevOps,
     AI/ML, research, legal, procurement, customer success, project/program management,
     design, healthcare clinical, pharma/life sciences, …) + `other`. Each: aliases,
     default competency weights, typical functional areas (examples, **not** a closed
     list), case archetypes, preferred structure for answers, Level-1 report dimensions,
     `technical` flag.
   * `industries.json` — modifiers (FMCG, retail, e-commerce, manufacturing, real estate,
     media, BFSI, healthcare, pharma, …) that add domain-knowledge expectations.
   * `competencies.json` — canonical competencies (core, behavioral, functional,
     technical, domain), each with definition, sub-competencies, strong/weak signals,
     red flags, appropriate answer structure, must-not-infer list, band anchors.
2. **Role profile (per JD)** — from the JD parser + classifier: family, sub-family,
   industry, seniority, requirements (R-ids), responsibilities, keywords, implied
   competencies. Hybrid roles carry two families.
3. **Role competency model (per session)** — the union of
   (a) family defaults, (b) competencies implied by JD requirements (mapped to canonical
   ids where possible), (c) **role-specific competencies** generated from the JD when
   nothing canonical fits — id `rs:<slug>`, each with a `parent` canonical id (e.g.
   `rs:derivatives_pricing → functional_knowledge`) so history stays comparable, and
   (d) mode-mandatory competencies (e.g. HR mode forces motivation/ownership/conflict).
   Each entry: importance (critical/high/medium/low), weight, linked R-ids, CV evidence
   strength (strong/partial/none with claim ids), expected depth for the seniority,
   sub-areas to probe (role-derived, e.g. *working capital, cost of capital* for a
   treasury analyst — never a hard-coded universe).

## 2. Why this satisfies "don't hardcode, don't limit to consulting"
* Families are data; adding one is a JSON edit + version bump.
* An unseen role ("Clinical Data Manager, oncology trials") classifies as the nearest
  family or `other`, and its specific knowledge areas arrive as `rs:` competencies built
  from the JD itself.
* Functional depth is decided per role: the mapper must justify every sub-area with a
  requirement or responsibility id (or the seniority expectation) — no generic lists.

## 3. Pre-interview analysis (the setup "aha" moment)
From the model: *"We identified N major competencies for this role. K are strongly
represented in your CV. M need deeper validation."* plus the list, each linked to the JD
requirement and the CV claim that drove it. Pure computation over the model — no extra
LLM call.

## 4. Rubrics (frozen before any answer)
For each competency in the model the rubric builder writes role- and seniority-specific
expectations **inside the canonical band anchors**:

| Score | Band (display) | Canonical anchor |
|---|---|---|
| 1–2 | Needs development | Tested; little or no relevant evidence; or answers incorrect |
| 3–4 | Needs development / Below expected | Some relevant evidence, materially below the role's expected level |
| 5–6 | Moderate | Acceptable to solid for the role; gaps in depth, specificity or ownership |
| 7–8 | Strong | Clear, specific, owned, reasoned evidence at or above the expected level |
| 9–10 | Exceptional | Deep, repeatedly demonstrated, handles challenge, unusually insightful |
| — | **Not sufficiently tested** | Too little evidence to assess. Never shown as weak. |

The rubric set is hashed (`rubric_hash`) and stored on the blueprint; assessments record
which hash and evaluator version produced them.
