-- Silent per-case exemplar bank (retrieval-augmented calibration).
-- Idempotent: safe to run more than once. Read/written only by services/exemplar_bank.py.
-- Captures anonymised digests of strong answers per case; the scorer references the
-- top few next time the same case is scored. No PII is stored (digests are the
-- scorer's structured read, not raw transcript).

create table if not exists case_exemplars (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references cases(id) on delete cascade,
  submission_id uuid references submissions(id) on delete set null,
  score int not null,
  structure_digest text not null,      -- anonymised: the MECE structure used
  key_moves text not null,             -- 2-4 strong moves (frameworks applied, sharp Q, sanity check)
  recommendation_digest text,          -- anonymised closing recommendation
  frameworks text[] default '{}',
  approved boolean default false,      -- human gate before an exemplar influences scoring
  created_at timestamptz default now()
);

create index if not exists idx_case_exemplars_case_score
  on case_exemplars(case_id, score desc);

-- The service uses the service-role key (same as the rest of the backend), so RLS is
-- bypassed server-side. If you later expose this table to the anon/authenticated
-- roles, add an RLS policy first — it should never be candidate-readable.
alter table case_exemplars enable row level security;
