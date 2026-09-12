"""
Admin route for MECE's agentic AI — the orchestrator demo + live runner.

  GET  /admin/agentic/info   -> the specialist roster + mission presets + whether
                                live mode is available. Drives the admin UI.
  POST /admin/agentic/run    -> run the orchestrator. Body: {mission, mode}.
                                mode="sim"  : deterministic, zero-cost, ALWAYS works
                                              (this is what you show people).
                                mode="live" : real, model-driven orchestration over
                                              live data — and if anything is missing
                                              (no key, budget hit, empty tables) it
                                              transparently FALLS BACK to sim so the
                                              demo never errors in front of an audience.

Admin-gated with the same contract as routes/ai_providers.py. Never returns 5xx
for a run — failures are reported inside a 200 body so the UI degrades gracefully.
"""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.auth import get_verified_user, is_guest_user
from services.supabase_client import get_supabase_client
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget

# make_live_planner / SupabaseData are imported lazily inside the run handler so a
# missing OpenAI SDK/key can never break module import or the /info endpoint.

router = APIRouter(prefix="/admin/agentic", tags=["admin", "agentic"])


# Mission presets surfaced in the demo UI. Each is a real, different orchestration.
MISSION_PRESETS = [
    {"id": "daily_brief", "label": "Run today's platform brief",
     "mission": "Run today's platform brief: pick the best GD topic, diagnose the cohort's weakest skill, and design today's targeted practice case."},
    {"id": "cut_cost", "label": "Cut our AI costs (without hurting quality)",
     "mission": "Cut our AI costs without hurting scoring quality: find where spend goes and recommend safe provider switches."},
    {"id": "week_focus", "label": "Design this week's practice focus",
     "mission": "Design this week's practice focus: find the cohort's weakest dimension, build a hard case for it, and surface exemplar winning decks."},
]


def _require_admin(authorization: Optional[str]) -> str:
    """Bearer token -> admin user id. Same contract as routes/ai_providers.py."""
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


class RunRequest(BaseModel):
    mission: str = ""
    mode: str = "sim"  # "sim" | "live"


@router.get("/info")
async def agentic_info(authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    # Live is "available" only if the OpenAI client exists; the UI uses this to
    # enable/disable the Live toggle. Import lazily so a missing key never breaks import.
    try:
        from services.ai_providers import openai_client
        live_available = openai_client() is not None
    except Exception:
        live_available = False
    from services.agentic import catalog
    return {"specialists": catalog(), "missions": MISSION_PRESETS, "live_available": live_available}


@router.post("/run")
async def agentic_run(body: RunRequest, authorization: Optional[str] = Header(default=None)):
    uid = _require_admin(authorization)
    check_rate_limit(f"agentic:run:{uid}", max_calls=10, window_seconds=60)

    mission = (body.mission or "").strip() or MISSION_PRESETS[0]["mission"]
    mode = "live" if (body.mode or "").lower() == "live" else "sim"
    supabase = get_supabase_client()
    from services.agentic import orchestrate, make_sim_planner, SampleData

    note = None
    if mode == "live":
        # Try the real thing; degrade to sim on ANY problem so the demo can't fail.
        try:
            from services.agentic.live import make_live_planner, SupabaseData
            assert_daily_budget()  # raises 503 if the day's spend is over budget
            data = SupabaseData(supabase)
            planner, model = make_live_planner(uid, data)
            result = orchestrate(mission, planner, data, mode="live", max_steps=6, model=model)
        except HTTPException as e:
            note = f"Live unavailable ({e.detail}); showed a simulation instead."
            result = orchestrate(mission, make_sim_planner(), SampleData(), mode="sim", max_steps=6)
        except Exception as e:  # noqa: BLE001
            note = f"Live failed ({type(e).__name__}); showed a simulation instead."
            result = orchestrate(mission, make_sim_planner(), SampleData(), mode="sim", max_steps=6)
    else:
        result = orchestrate(mission, make_sim_planner(), SampleData(), mode="sim", max_steps=6)

    payload = result.to_dict()
    if note:
        payload["note"] = note

    # Best-effort audit trail; never fails the response.
    try:
        supabase.table("agentic_runs").insert({
            "user_id": uid, "mission": mission, "mode": result.mode,
            "summary": result.summary, "agents_used": result.agents_used,
            "trace": [s.to_dict() for s in result.steps],
            "stopped_reason": result.stopped_reason, "model": result.model,
        }).execute()
    except Exception:
        pass

    return payload
