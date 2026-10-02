# 00 · Audit of the existing MECE codebase (2026-10-02)

Scope: what Interview Intelligence (II) must integrate with, and what it must NOT
depend on. Read from `D:\dev\mece\consilio` (frontend) and `D:\dev\mece\consilio-backend`
(backend) on 2026-10-02. Latest `.brain/STATE.md` sync: 2026-10-01 16:55 UTC
(frontend `e6cf01d`, backend `ec79aed` — model-led realtime voice interviewer, flag off).

## 1. Stack as found

| Layer | What it is | Where |
|---|---|---|
| Frontend | Next.js 14.2 (App Router), React 18, Tailwind + shadcn/radix, Supabase SSR auth, Razorpay | `consilio/` on Vercel |
| Backend | FastAPI single process, `uvicorn --workers 1`, Render (Oregon, 512 MB class), ~25 routers all imported at startup | `consilio-backend/` |
| Database | One Supabase Postgres project (Tokyo). Backend uses the **service-role** key through PostgREST (`supabase-py`), RLS bypassed | `services/supabase_client.py` |
| Auth | Supabase Auth (email, Google, LinkedIn OIDC, anonymous guests). Frontend forwards the Supabase access token as `Authorization: Bearer`; backend verifies with `supabase.auth.get_user()` (optional local JWKS fast path, 60 s verified-token cache) | `services/auth.py` |
| AI | OpenAI + Groq + Gemini via one `ai_providers.py` resolver with admin-toggled per-feature provider and a fallback chain; spend logged to `ai_usage_log`; global daily budget kill switch | `services/ai_providers.py`, `services/ai_usage.py` |
| Storage | Supabase buckets + Google Drive (OAuth refresh token or service account) for the Deck Vault, `gdrive:` path prefix contract (C8) | `services/gdrive.py`, `lib/google-drive.ts` |

## 2. Identity and Pro entitlement — how it works today

* **Identity**: Supabase session cookie (`sb-*`) in the browser; server components call
  `createClient().auth.getUser()`; the browser obtains the access token with
  `supabase.auth.getSession()` and sends it to FastAPI.
* **Tier**: columns on `public.users` — `subscription_tier ∈ {free, lite, pro}` and
  `subscription_expires_at`. Effective tier = tier unless expired → `free`.
  Computed in two mirrored places: `lib/tier-core.ts::effectiveTier` (frontend) and
  `services/access_guard.py::_effective_tier_from_row` (backend, authoritative).
* **Writers of the tier**: Razorpay `verify` + `webhook` routes (service role), admin
  membership grants. `0006` column guard stops users writing their own tier.
* **Admin**: `users.is_admin` (guarded column). `/admin/*` layout checks it server-side.
* **Guests**: anonymous Supabase users (`is_anonymous`), `users.is_guest`.
* **Existing "isolated feature" precedent**: Prep Copilot v2 (`routes/copilot.py`,
  `services/copilot/*`, `lib/copilot/*`, `components/copilot/*`) — own tables, own engine
  copy, Pro-gated server-side, allowlist preview by email env var. II follows the same
  spirit but goes further (separate process, separate database schema, no shared code).

## 3. What II may and may not reuse

| Existing asset | II decision | Why |
|---|---|---|
| Supabase session (identity) | **Used indirectly** — the MECE Next.js server verifies it and signs an II entitlement assertion | Identity is the one thing MECE is allowed to provide |
| `users.subscription_tier/expires_at`, `is_admin` | **Read only by MECE** when minting the assertion; II never reads MECE tables | "Do not depend on MECE tables for internal operation" |
| `services/ai_providers.py`, prompts, `ai_usage_log` | **Not used.** II has its own provider abstraction, routing, prompt registry, model-run log, budget | Spec §1, §113 |
| `services/interview_engine.py`, `interviewer_decision.py`, scorer | **Not used** | Different product (case interviews); spec forbids hidden coupling |
| `services/gdrive.py` | **Not imported.** II has its own Drive client (same auth options, different root folder + env names) | Storage independence; avoids C8 contract coupling |
| `services/rate_limit.py`, `keyed_lock.py` | Not imported; II has its own | Independence |
| Frontend design system (Tailwind tokens, `components/ui/*`) | **Reused for UI** | UI consistency is not backend coupling; "Premium MECE-quality interface" |
| `lib/supabase/client.ts` (browser) | **Not needed** — the II client gets its token from the MECE token route (cookie-authenticated, same origin) | Thinner coupling |

## 4. Constraints discovered that shape the design

1. **Render memory (RENDER_OOM_FIX.md)**: the existing backend sits at 350–450 MB of a
   512 MB box. Adding II to that process would raise OOM risk for the live product.
   → II runs as its **own service** (own process, own memory budget, own deploy).
2. **DB latency**: backend in Oregon, Supabase in Tokyo (~100 ms per round trip). II
   keeps a compact interview state row and avoids chatty per-turn queries; one
   transaction per turn.
3. **PostgREST exposure**: anything in the `public` schema with grants is reachable via
   PostgREST. II uses a **direct Postgres connection** to its own schema
   (`interview_intel`), which is **not** in Supabase's exposed schemas and has no
   `anon`/`authenticated` grants.
4. **Two Windows profiles, one shared folder, git as the sync line**: II is a **new repo
   folder** `D:\dev\mece\consilio-interview-intelligence` (nothing existing is edited
   except one additive admin-nav entry in the frontend).
5. **Brain protocol**: brains author, Antigravity lands. This session authored code +
   `ANTIGRAVITY_HANDOFF_interview-intelligence.md`; it does not edit STATE/CHANGELOG/
   CONTRACTS. No existing CONTRACTS surface (C1–C9) is touched. A new contract
   (proposed **C10 · II entitlement assertion**) is proposed in the handoff.
6. **Existing voice stack** (`/transcribe`, `/speak`, realtime) belongs to the case
   product. II gets its own flagged STT/TTS endpoints (no reuse).

## 5. Things noticed that are NOT part of this work (reported, not changed)

* `routes/copilot.py` hard-codes a default allowlist email in source — II deliberately
  does the opposite (no emails in code; admin-managed table + env placeholders).
* `services/access_guard.py` and `lib/access.ts` mirror tier logic by hand. II avoids a
  third copy: tier is computed once (MECE side, `effectiveTier`) and asserted.
