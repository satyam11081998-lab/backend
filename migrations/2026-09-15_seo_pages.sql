-- 2026-09-15_seo_pages.sql
-- Growth Agent (programmatic SEO) storage.
--
-- Backing table for:
--   * services/growth/seo_writer.py  -> inserts ONE draft row per generate run
--   * routes/seo.py                  -> /seo/generate, /seo/candidates (service role)
--   * app/(app)/admin/growth         -> review list + publish/reject (service client)
--   * /insights/<slug> + sitemap     -> public read of PUBLISHED rows (anon, via RLS)
--
-- Its absence is what produced:
--   PGRST205 "Could not find the table 'public.seo_pages' in the schema cache".
--
-- Columns mirror exactly what the writer inserts and the admin page selects.
-- Idempotent: safe to run more than once.

create extension if not exists pgcrypto;  -- gen_random_uuid() (built-in on PG13+, this is a no-op safety net)

create table if not exists public.seo_pages (
  id                 uuid        primary key default gen_random_uuid(),
  slug               text        not null,
  kind               text,                                   -- writer sets 'news_case'
  title              text        not null,
  meta_description   text,
  dek                text,
  content            jsonb       not null default '{}'::jsonb,   -- structured article body
  source_refs        jsonb       not null default '[]'::jsonb,   -- [{label,url}]
  topic              text,
  keywords           jsonb       not null default '[]'::jsonb,   -- ["phrase", ...]
  status             text        not null default 'draft',       -- draft|approved|published|rejected|archived
  quality_score      integer,                                    -- 0..100 self-critique (nullable)
  quality_notes      text,
  model              text,
  agent_meta         jsonb       not null default '{}'::jsonb,   -- {provider,category}
  source_headline_id text,                                       -- news_headlines.id it was written from
  created_by         text,                                       -- auth uid of the admin who generated it
  published_at       timestamptz,                                -- set when status flips to 'published'
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);

-- One page per slug. seo_writer._unique_slug() relies on this unique index as its backstop.
create unique index if not exists seo_pages_slug_key           on public.seo_pages (slug);
create index        if not exists seo_pages_status_idx          on public.seo_pages (status);
create index        if not exists seo_pages_published_at_idx    on public.seo_pages (published_at);
create index        if not exists seo_pages_source_headline_idx on public.seo_pages (source_headline_id);
create index        if not exists seo_pages_created_at_idx      on public.seo_pages (created_at desc);

-- Keep updated_at fresh on every UPDATE (publish/reject flips status).
create or replace function public.seo_pages_set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists seo_pages_set_updated_at on public.seo_pages;
create trigger seo_pages_set_updated_at
  before update on public.seo_pages
  for each row execute function public.seo_pages_set_updated_at();

-- Row Level Security.
-- The backend (service_role key) and the admin pages (service client) BYPASS RLS,
-- so generation, the review list, and publish/reject all keep working untouched.
-- The ONLY thing RLS must grant is the PUBLIC read of PUBLISHED pages for
-- /insights/<slug> and the sitemap (anon key) — exactly the design the publish
-- action documents ("stamps published_at, which is what makes it visible via RLS").
alter table public.seo_pages enable row level security;

drop policy if exists "public reads published seo pages" on public.seo_pages;
create policy "public reads published seo pages"
  on public.seo_pages
  for select
  to anon, authenticated
  using (published_at is not null);

-- Ask PostgREST to reload its schema cache immediately (otherwise the new table can
-- take up to ~1 min to appear and you'd still see PGRST205 briefly).
notify pgrst, 'reload schema';
