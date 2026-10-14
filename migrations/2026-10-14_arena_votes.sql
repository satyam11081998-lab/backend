-- Admin interviewer arena: the recorded preference log.
-- Idempotent: safe to run more than once. Written only by routes/arena.py.
--
-- The arena WORKS WITHOUT this table: /admin/arena/config reports
-- votes_persisted=false and every vote comes back "not saved" with this filename.
-- Running it is what buys the running tally. (The exemplar bank has sat dormant
-- since September because its migration was never run -- hence the loud warning.)
--
-- Why the fingerprints: a vote is only meaningful against a KNOWN baseline. The
-- SHA256 of each frozen variant file is stored with the vote, so if a baseline is
-- ever re-snapshotted you can tell which rows were judged against which text
-- instead of silently pooling two different experiments.
--
-- No PII: candidate_message is the ADMIN's own typed test input, never a learner's.

create table if not exists public.arena_votes (
  id uuid primary key default gen_random_uuid(),
  admin_id uuid references public.users(id) on delete set null,
  session_key text not null,                -- opaque per-session id (not a DB session)
  case_id uuid references public.cases(id) on delete set null,
  case_type text,
  turn_index int,
  -- which variant rendered on which side (the side order is randomised per session)
  left_variant text not null,
  right_variant text not null,
  left_fingerprint text,                    -- sha256 of the frozen variant file, or null for `live`
  right_fingerprint text,
  choice text not null check (choice in ('left', 'right', 'tie', 'both_bad')),
  chosen_variant text,                      -- null for tie / both_bad
  model_mode text,
  teaching_policy text,
  candidate_message text,
  note text,
  created_at timestamptz default now()
);

create index if not exists idx_arena_votes_variant_created
  on public.arena_votes(chosen_variant, created_at desc);
create index if not exists idx_arena_votes_session
  on public.arena_votes(session_key, turn_index);
create index if not exists idx_arena_votes_case
  on public.arena_votes(case_id, created_at desc);

-- Service-role only: RLS on with NO policy, so anon/authenticated see nothing via
-- PostgREST. The arena page reads this through the admin-gated backend, never
-- directly from the browser. Eval notes are not candidate-readable.
alter table public.arena_votes enable row level security;
