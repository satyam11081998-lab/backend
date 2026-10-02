# E · Interview state model

## 1. Session lifecycle (explicit state machine, `interview_engine/lifecycle.py`)

```
created ──► uploading ──► analyzing ──► ready ──► active ◄──► paused
   │            │             │           │         │            │
   └────────────┴─────────────┴───────────┴─────────┴────────────┴──► abandoned
                │             │           │                      │
                └─────────────┴─► failed  └─► expired ◄──────────┘
                                                active/paused ──► completed
```
* Slot-occupying (count toward the **2 active sessions** rule): `created, uploading,
  analyzing, ready, active, paused`. Terminal: `completed, abandoned, expired, failed`.
* Any transition not in the table raises `InvalidTransition` (unit-tested exhaustively).
* Timeouts (swept lazily on every create/list and by a periodic job):
  `created/uploading/analyzing` > 2 h → `failed`; `ready` > 24 h → `expired`;
  `active` idle > 20 min → `paused`; `paused` > 24 h → `expired` (if ≥ 2 substantive
  answers exist the assessment still runs, marked limited).
* The 2-slot check runs inside a transaction holding a row lock on the II user record,
  so two simultaneous "start" clicks cannot both pass.

## 2. Live interview state (`interview_states.state`, versioned JSON)

```jsonc
{
  "schema": 1,
  "current_section": "functional",            // id from the blueprint
  "current_exchange_id": "…",                 // open question exchange
  "current_item_qid": "Q7",
  "pending_action": null,                     // what the interviewer just did
  "difficulty": {"level": 3, "by_competency": {"valuation": 4}},
  "time": {"elapsed_s": 1042, "section_spent_s": {"cv": 400}, "remaining_s": 1658},
  "coverage": {                               // live, approximate (the evaluator decides later)
    "commercial_judgment": {"asked": 2, "signals": {"strong": 1, "weak": 0}, "status": "sufficient"}
  },
  "asked_qids": ["Q1","Q3"], "skipped_qids": ["Q4"],
  "claims": [ {"id":"C2","src":"cv","text":"Led a team of 12","slots":{"team_size":12},"status":"probing","ladder_step":2} ],
  "interview_claims": [ {"id":"IC1","text":"worked with four teammates","slots":{"team_size":4},"exchange":"…"} ],
  "contradictions": [ {"id":"K1","a":"C2","b":"IC1","slot":"team_size","status":"open|clarified|dismissed"} ],
  "open_probes": [ {"exchange":"…","focus":"quantification","attempts":1} ],
  "memory": [ {"ref":"M3","exchange":"…","summary":"Reduced churn 8% via win-back flow; owned design, not rollout"} ],
  "strengths_seen": ["…"], "weaknesses_seen": ["…"],          // live hints for adaptation only
  "recent_openers": ["Okay.", "Right —"],                       // avoid canned repetition
  "consecutive": {"strong": 0, "weak": 1, "non_answers": 0},
  "cost_usd": 0.041, "degraded": false
}
```

Why both a JSON state and normalized tables: the JSON is the compact, single-row input
for the live decision (one read, one write per turn); the tables (`interview_messages`,
`interview_exchanges`, `claims`, `interview_events`) are the durable, queryable record used
by the evidence engine, the report and the admin inspector. The state carries a `version`
column; a turn that loads version *n* writes *n+1* under the session row lock, so turns
are strictly serialized and a retried request with the same `client_turn_id` returns the
stored result instead of re-running.

## 3. Turn pipeline (`interview_engine/orchestrator.py`)

1. Lock session row; idempotency check on `client_turn_id`.
2. Classify intent deterministically (repeat / give-me-a-second / break / end /
   clarification / empty) — falls back to the analyzer's `intent` for subtler cases
   (off-topic question, refusal, meta questions about scoring).
3. For answers: persist message, compute **measured metrics** (words, filler words,
   hedges, numbers, time-to-answer if voice), run **turn analysis** (fast model).
4. Update memory: claims (slot extraction), contradiction check (deterministic slot
   comparison + analyzer flags), coverage, consecutive counters, difficulty.
5. **Decision policy** (pure function `decide(state, blueprint, analysis, clock) → Action`):
   `PROBE(focus) | CLARIFY_CONTRADICTION | INVESTIGATE_CLAIM | CHALLENGE | ADVANCE(item) |
   TRANSITION(section) | REPEAT | CLARIFY_QUESTION | WAIT | REDIRECT | ACK_REFUSAL_MOVE_ON |
   PAUSE | CLOSE`. Deterministic, logged as an event with its reasons.
6. Generate the interviewer utterance for that action (fast model, persona from mode ×
   difficulty); evaluation-leak filter; deterministic fallback if the model fails.
7. Persist interviewer message + events; close the exchange on ADVANCE/TRANSITION/CLOSE and
   enqueue `extract_exchange_evidence`.

## 4. Failure behaviour (spec §57/§91)
* Model timeout/error in analysis → treat answer as "adequate, unknown gaps", continue.
* Model failure in utterance → deterministic utterance from the planned question text.
* Budget cap reached → deterministic utterances, steer to closing, mark `degraded`.
* Client refresh / disconnect → `GET /sessions/{id}` returns the transcript and the
  last interviewer message; the room resumes.
* Speech recognition failure → client falls back to text; nothing server-side changes.
