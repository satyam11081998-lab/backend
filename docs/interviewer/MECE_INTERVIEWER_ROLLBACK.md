# MECE Interviewer — Rollback

Date: 2026-09-30. Nothing from this work is merged into `main` or deployed. Rolling back is
therefore mostly "don't merge" — this file covers every stage in case it is merged later.

## 1. Git state before this work (recorded)

| Where | Repo | State |
|---|---|---|
| GitHub | `satyam11081998-lab/backend` | `main` = `12308ec` at the start (2026-09-29); **`9e009c5`** when this work was finished (four V11 edits pushed from the GitHub web editor on 2026-09-30 00:43–00:46 IST: `9cb462a`, `a7764a1`, `f4114ec`, `9e009c5`) |
| GitHub | `satyam11081998-lab/company` | `main` = `f9841b6` at the start; **`a73be90`** at the end (India landing redesign: `e554abd`, `ed5890e`, `a73be90`) |
| `D:\dev\mece\consilio-backend` | local | branch `main` at `12308ec` (local `origin/main` not yet fetched past it), **with uncommitted changes that are not part of this work**: modified `.github/workflows/daily-cases-us.yml`, `requirements.txt`, `routes/cron.py`, `routes/daily.py`, `routes/submit.py`, `services/access_guard.py`, `services/content_generator.py`, `services/daily_scheduler.py`, `services/markets.py`, `tests/test_markets.py`; untracked `migrations/2026-09-15_seo_pages.sql`, `tools/eval_result_live*.txt` (3). Other local branches: `backup/pre-v10-local-ecd9fec`, `feat/interviewer-behavior-modes`, `feat/us-launch`, `wip/local-interviewer-iterations-2026-09-20`. |
| `D:\dev\mece\consilio` | local | branch `main` at `a73be90` (= `origin/main`), untracked `push_out.txt`. Other local branches: `feat/india-landing-redesign`, `feat/us-growth-seo`, `feat/us-launch`, `feat/us-redesign`. |

This work never modified, stashed, reset or checked out anything in those working trees.

## 2. What this work added

| Repo | Branch | Commits on top of `main` |
|---|---|---|
| backend | `feat/unified-interviewer-brain` | on `9e009c5`: brain + routes + tests + tools + docs (see `git log 9e009c5..feat/unified-interviewer-brain`) |
| frontend | `feat/unified-interviewer-brain` | on `a73be90`: client side + QA + migration 0071 + handoff (see `git log a73be90..feat/unified-interviewer-brain`) |

Delivered into the local repos under `D:\dev\mece` with `git fetch <bundle>` as the local branch
`feat/unified-interviewer-brain` only (object database + one new branch ref; no working-tree,
index or `main` change). Not pushed to GitHub.

## 3. Rollback, by stage

### Stage A — branches delivered, nothing merged (current state)
Delete the branches if you do not want them:
```
cd D:\dev\mece\consilio-backend
git branch -D feat/unified-interviewer-brain
cd D:\dev\mece\consilio
git branch -D feat/unified-interviewer-brain
```
(The fetched objects become unreachable and are pruned by `git gc` later. The bundle files in
`D:\dev\mece\Claude outputs\unified-interviewer\` can be deleted by hand.)

### Stage B — merged and deployed, flag OFF (`INTERVIEWER_BRAIN` unset or `off`)
Behaviour is the baseline V11 on every route; nothing to roll back functionally. The additive
routes exist but are unused by a flag-OFF frontend flow except:
- `/realtime/session` returns two extra fields (ignored by old clients);
- the realtime client sends lines out-of-band by default → to restore the exact previous
  in-band behaviour without any deploy set `REALTIME_RESPONSE_CONVERSATION=auto` on the backend;
- the noise-guard fix (digits / yes / no are no longer dropped) applies in both modes — it is a
  bug fix; to undo it revert `lib/voice/noise-guard.ts` only.

### Stage C — flag ON / allowlist
Instant: set `INTERVIEWER_BRAIN=off` in the Render environment and restart. The next turn of
every attempt runs V11. Brain state lives under `attempts.session_state.brain` and is ignored by
V11 (V11's own keys were never written by the brain), so no data migration is needed.

### Stage D — revert the code
```
# backend (from main, after the merge commit M_b)
git revert -m 1 <M_b>          # or: git revert <each brain commit, newest first>
# frontend
git revert -m 1 <M_f>
git push                        # then: node .brain\sync.mjs in consilio
```
The V11 engine files are byte-identical to `main` in this branch, so reverting cannot touch them.

### Stage E — the migration (0071)
Safe to leave in place: a nullable column and a partial unique index that only applies to rows
that carry a key. If it must go:
```sql
drop index if exists public.attempt_messages_client_turn_uidx;
alter table public.attempt_messages drop column if exists client_turn_id;
notify pgrst, 'reload schema';
```
Run the code revert (Stage D) or keep the flag OFF first; the backend tolerates the column being
absent (plain-insert fallback), so the order is not critical.

### Stage F — only the contextual-presence change (2026-10-01)
It lives in one commit on the branch ("feat(interviewer): contextual presence…"). To go back to the
fixed presence lines while keeping the rest of the brain: `git revert <that commit>` (it touches
`services/interviewer/{types,policy,classify,presence,prompting,responder,validate,engine}.py`,
`tests/interviewer_fakes.py` and adds `tests/test_interviewer_brain_contextual.py`). No data or
migration is involved.

## 4. What is NOT rolled back by the flag
- `services/copilot/engine/prompts_interview.py`: the unreachable static "EXAMINE, DO NOT TEACH /
  NO HINTS" prompt was replaced by a fail-closed stub (invariant 9). Reachable only if
  `ADAPTIVE_INTERVIEWER` were turned off, which `main.py` prevents. Revert that file alone to
  restore the old text.
- `services/ai_usage.py`: one added price line for `gpt-live-transcribe` (no effect unless the
  live STT transport is used).
