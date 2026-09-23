-- ============================================================================
-- Prep Copilot v2 - ISOLATED corpus + practice tables.
-- Fully separate from cases / submissions / attempts. Service-role only:
-- RLS is ENABLED with NO policy, so PostgREST/anon/authenticated see nothing and
-- only the backend service key (which bypasses RLS) reads/writes. The frontend
-- never queries these directly - routes/copilot.py mediates everything.
-- Idempotent: safe to run more than once.
-- ============================================================================

-- The self-improving corpus: one row per (role_key, company_key). company_key ''
-- means a role-only pack. `pack` holds the full Pack JSON (rubric/frameworks/
-- assessment/scenarios/sources).
create table if not exists public.copilot_packs (
  id              uuid primary key default gen_random_uuid(),
  role_key        text not null,
  company_key     text not null default '',
  display_role    text,
  display_company text,
  pack            jsonb not null default '{}'::jsonb,
  confidence      text not null default 'low',
  status          text not null default 'ready',
  version         int  not null default 1,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);
create unique index if not exists copilot_packs_role_company_uq
  on public.copilot_packs (role_key, company_key);
create index if not exists copilot_packs_updated_idx
  on public.copilot_packs (updated_at desc);

-- Provenance / self-improvement audit: every build or refresh writes one row.
create table if not exists public.copilot_research_log (
  id          uuid primary key default gen_random_uuid(),
  role_key    text not null,
  company_key text not null default '',
  user_id     uuid,
  ok          boolean not null default false,
  confidence  text,
  n_sources   int not null default 0,
  note        text,
  created_at  timestamptz not null default now()
);
create index if not exists copilot_research_log_role_idx
  on public.copilot_research_log (role_key, created_at desc);

-- One practice run (isolated from case_attempts). scenario holds the cast Scenario JSON.
create table if not exists public.copilot_runs (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null,
  pack_id      uuid,
  role_key     text not null,
  company_key  text not null default '',
  display_role text,
  display_company text,
  scenario     jsonb not null default '{}'::jsonb,
  status       text not null default 'active',  -- active|submitted|scored
  created_at   timestamptz not null default now(),
  submitted_at timestamptz
);
create index if not exists copilot_runs_user_idx
  on public.copilot_runs (user_id, created_at desc);

-- Conversation turns (isolated from attempt_messages).
create table if not exists public.copilot_messages (
  id         uuid primary key default gen_random_uuid(),
  run_id     uuid not null references public.copilot_runs(id) on delete cascade,
  role       text not null,                    -- user|interviewer|system
  content    text not null default '',
  kind       text not null default 'text',
  created_at timestamptz not null default now()
);
create index if not exists copilot_messages_run_idx
  on public.copilot_messages (run_id, created_at asc);

-- Isolated scores (isolated from submissions).
create table if not exists public.copilot_scores (
  id         uuid primary key default gen_random_uuid(),
  run_id     uuid not null references public.copilot_runs(id) on delete cascade,
  user_id    uuid not null,
  score      int,
  breakdown  jsonb not null default '{}'::jsonb,
  feedback   jsonb not null default '{}'::jsonb,
  model      text,
  created_at timestamptz not null default now()
);
create index if not exists copilot_scores_run_idx
  on public.copilot_scores (run_id);
create index if not exists copilot_scores_user_idx
  on public.copilot_scores (user_id, created_at desc);

-- Lock everything to the service role (RLS on, no policy => only service key sees rows).
alter table public.copilot_packs        enable row level security;
alter table public.copilot_research_log enable row level security;
alter table public.copilot_runs         enable row level security;
alter table public.copilot_messages     enable row level security;
alter table public.copilot_scores       enable row level security;

-- Make PostgREST pick up the new tables immediately.
notify pgrst, 'reload schema';

-- Rollback (manual):
--   drop table if exists public.copilot_scores, public.copilot_messages,
--     public.copilot_runs, public.copilot_research_log, public.copilot_packs cascade;
