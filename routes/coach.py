"""
Prep Copilot — the per-user agentic-AI coach.

  GET  /coach/info   -> eligibility (has the user completed enough scored
                        attempts?), a profile pre-fill for the intake form, the
                        specialist roster, and whether live (model) mode is
                        available. Pro-gated.
  POST /coach/run    -> run the orchestrator over THIS user's real data. Body:
                        {goal, target_company, domain, mode}. Pro-gated AND
                        eligibility-gated. mode="live" uses the model planner;
                        if the model is unavailable/over-budget it falls back to
                        the GUIDED (deterministic) planner OVER THE SAME REAL
                        DATA — never sample data — so a real user always gets a
                        grounded plan, and the endpoint never returns 5xx.
  POST /coach/demo   -> admin-only stage demo. Runs the guided planner against a
                        chosen SAMPLE candidate (no key, no budget, no DB). This
                        is what you show people; different sample candidates take
                        visibly different paths.

Gating reuses the exact contracts already in the codebase:
  * Pro   -> services.access_guard.assert_tier_at_least(supabase, uid, "pro")
  * Admin -> users.is_admin (same as routes/ai_providers.py & routes/agentic.py)
"""

import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.auth import get_verified_user, is_guest_user
from services.supabase_client import get_supabase_client
from services.access_guard import assert_tier_at_least
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget

from services.coach import orchestrate, catalog, make_sim_planner, SAMPLE_CANDIDATES
from services.coach.simulation import SampleUserData
# make_live_planner / SupabaseUserData are imported lazily inside the handlers so
# a missing OpenAI SDK/key can never break module import or the /info endpoint.

router = APIRouter(prefix="/coach", tags=["coach", "agentic"])

# How many completed (scored) attempts unlock the copilot. Product default 4;
# override with COACH_MIN_ATTEMPTS without a redeploy.
MIN_ATTEMPTS = int(os.getenv("COACH_MIN_ATTEMPTS", "4") or "4")


def _require_pro(authorization: Optional[str]):
    """Bearer token -> (supabase, uid). Raises 401 (unauth) / 403 (not Pro)."""
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Create a free account, then upgrade to Pro to use the Prep Copilot.")
    assert_tier_at_least(supabase, uid, "pro")  # raises 403 with an upgrade message
    return supabase, uid


def _require_admin(authorization: Optional[str]) -> str:
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Admins only")
    try:
        res = supabase.table("users").select("is_admin").eq("id", uid).single().execute()
        is_admin = bool((res.data or {}).get("is_admin"))
    except Exception:
        raise HTTPException(status_code=403, detail="Admins only")
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    return uid


def _completed_count(supabase, uid: str) -> int:
    try:
        r = supabase.table("submissions").select("id", count="exact").eq("user_id", uid).execute()
        if getattr(r, "count", None) is not None:
            return int(r.count)
        return len(r.data or [])
    except Exception:
        return 0


class RunRequest(BaseModel):
    goal: str = ""
    target_company: str = ""
    domain: str = ""
    mode: str = "live"  # "live" (model planner) | "guided" (deterministic)


class DemoRequest(BaseModel):
    candidate: str = "aarav"
    goal: str = ""


@router.get("/info")
async def coach_info(authorization: Optional[str] = Header(default=None)):
    supabase, uid = _require_pro(authorization)
    completed = _completed_count(supabase, uid)

    prof = {}
    try:
        p = (supabase.table("users")
             .select("name, goal_text, placement_focus, weekly_hours_target")
             .eq("id", uid).maybe_single().execute())
        prof = dict(p.data or {})
    except Exception:
        prof = {}

    try:
        from services.ai_providers import openai_client
        live_available = openai_client() is not None
    except Exception:
        live_available = False

    return {
        "eligible": completed >= MIN_ATTEMPTS,
        "completed": completed,
        "required": MIN_ATTEMPTS,
        "specialists": catalog(),
        "live_available": live_available,
        "prefill": {
            "goal": prof.get("goal_text") or "",
            "placement_focus": prof.get("placement_focus") or "",
            "weekly_hours": prof.get("weekly_hours_target") or 5,
        },
    }


@router.post("/run")
async def coach_run(body: RunRequest, authorization: Optional[str] = Header(default=None)):
    supabase, uid = _require_pro(authorization)
    check_rate_limit(f"coach:run:{uid}", max_calls=6, window_seconds=60)

    completed = _completed_count(supabase, uid)
    if completed < MIN_ATTEMPTS:
        raise HTTPException(
            status_code=403,
            detail=f"Complete {MIN_ATTEMPTS} scored cases or guesstimates to unlock your Prep Copilot "
                   f"(you've done {completed}).",
        )

    from services.coach.live import SupabaseUserData
    data = SupabaseUserData(supabase, uid)

    goal = (body.goal or "").strip() or "Improve my case-interview performance."
    target_company = (body.target_company or "").strip()
    domain = (body.domain or "").strip()
    weekly = 5
    try:
        p = data.profile()
        weekly = int(p.get("weekly_hours_target") or 5)
    except Exception:
        weekly = 5

    want_live = (body.mode or "live").lower() != "guided"
    note = None

    def _guided():
        planner = make_sim_planner(target_company=target_company, domain=domain, weekly_hours=weekly)
        return orchestrate(goal, planner, data, mode="guided", max_steps=8)

    if want_live:
        try:
            from services.coach.live import make_live_planner
            assert_daily_budget()  # raises 503 if the day's spend is over budget
            context = {"goal": goal, "target_company": target_company, "domain": domain, "weekly_hours": weekly}
            planner, model = make_live_planner(uid, data, context)
            result = orchestrate(goal, planner, data, mode="live", max_steps=8, model=model)
            # If the model produced nothing useful, fall back to the guided plan.
            if not result.agents_used and result.stopped_reason in ("synthesized", "planner_error"):
                note = "Model returned no plan; used the guided planner on your real data."
                result = _guided()
        except HTTPException as e:
            note = f"Live unavailable ({e.detail}); used the guided planner on your real data."
            result = _guided()
        except Exception as e:  # noqa: BLE001
            note = f"Live failed ({type(e).__name__}); used the guided planner on your real data."
            result = _guided()
    else:
        result = _guided()

    payload = result.to_dict()
    payload["target_company"] = target_company
    payload["domain"] = domain
    if note:
        payload["note"] = note

    # Best-effort audit trail; never fails the response.
    try:
        supabase.table("coach_runs").insert({
            "user_id": uid, "goal": goal, "target_company": target_company, "domain": domain,
            "mode": result.mode, "summary": result.summary, "agents_used": result.agents_used,
            "trace": [s.to_dict() for s in result.steps], "model": result.model,
        }).execute()
    except Exception:
        pass

    return payload


@router.post("/demo")
async def coach_demo(body: DemoRequest, authorization: Optional[str] = Header(default=None)):
    """Admin-only, zero-cost stage demo over a sample candidate. Always works."""
    _require_admin(authorization)
    cand = SAMPLE_CANDIDATES.get((body.candidate or "").lower()) or next(iter(SAMPLE_CANDIDATES.values()))
    data = SampleUserData(cand)
    goal = (body.goal or "").strip() or cand.goal
    planner = make_sim_planner(target_company=cand.target_company, domain=cand.domain, weekly_hours=cand.weekly_hours)
    result = orchestrate(goal, planner, data, mode="demo", max_steps=8)
    payload = result.to_dict()
    payload["target_company"] = cand.target_company
    payload["domain"] = cand.domain
    payload["candidate"] = {"key": cand.key, "name": cand.name}
    return payload


@router.get("/demo/candidates")
async def coach_demo_candidates(authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    return {"candidates": [{"key": c.key, "name": c.name, "goal": c.goal,
                            "target_company": c.target_company, "domain": c.domain}
                           for c in SAMPLE_CANDIDATES.values()]}
