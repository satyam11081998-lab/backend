"""
Prep Copilot v2 - ISOLATED role/company-aware prep route (USER-facing, Pro-gated).

Nothing here touches the live cases/guesstimates pipeline. It uses:
  * services.copilot.corpus     -> get-or-build a grounded Pack (Gemini research, self-improving)
  * services.copilot.casting    -> cast a scenario for the interviewer
  * services.copilot.engine.*   -> an ISOLATED COPY of the interview engine (tweakable per role)
  * services.copilot.scoring    -> an ISOLATED GPT scorer against the Pack rubric

Endpoints (all Pro-gated, all behind COPILOT_V2_ENABLED):
  GET  /copilot/status            -> feature + research availability
  POST /copilot/pack              -> research/ground a Pack for {role, company}
  POST /copilot/practice/start    -> cast a scenario, open a run
  POST /copilot/practice/message  -> one interviewer turn (copied engine)
  POST /copilot/practice/submit   -> score the run with the isolated rubric scorer

Heavy copilot imports are LAZY inside handlers so a copilot issue can never break
app import or any live route. Tables are service-role only; this route mediates.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.auth import get_verified_user, is_guest_user
from services.supabase_client import get_supabase_client
from services.access_guard import assert_tier_at_least
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget

router = APIRouter(prefix="/copilot", tags=["copilot", "prep-copilot-v2"])

MAX_TURNS = int(os.getenv("COPILOT_MAX_TURNS", "40") or "40")


def _enabled() -> bool:
    return os.getenv("COPILOT_V2_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _require_enabled() -> None:
    if not _enabled():
        raise HTTPException(status_code=404, detail="Not found")


def _require_pro(authorization: Optional[str]):
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403,
                            detail="Create a free account, then upgrade to Pro to use the Prep Copilot.")
    assert_tier_at_least(supabase, uid, "pro")  # 403 with upgrade message if not Pro
    return supabase, uid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PackRequest(BaseModel):
    role: str = ""
    company: str = ""
    force_refresh: bool = False


class StartRequest(BaseModel):
    role: str = ""
    company: str = ""
    focus: str = ""
    difficulty: str = ""


class MessageRequest(BaseModel):
    run_id: str
    content: str = ""


class SubmitRequest(BaseModel):
    run_id: str


def _pack_public(pack) -> Dict[str, Any]:
    """Client view of a pack: hide each scenario's solution_outline (shown only
    after an attempt)."""
    d = pack.to_dict()
    for s in d.get("scenarios", []):
        s.pop("solution_outline", None)
    return d


@router.get("/status")
async def copilot_status(authorization: Optional[str] = Header(default=None)):
    enabled = _enabled()
    research = False
    if enabled:
        try:
            from services.copilot.research import research_available
            research = research_available()
        except Exception:
            research = False
    return {"enabled": enabled, "research_available": research}


@router.post("/pack")
async def copilot_pack(body: PackRequest, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    supabase, uid = _require_pro(authorization)
    role = (body.role or "").strip()
    company = (body.company or "").strip()
    if not role and not company:
        raise HTTPException(status_code=422, detail="Tell me the role or company you're targeting.")
    check_rate_limit(f"copilot:pack:{uid}", max_calls=8, window_seconds=120)
    assert_daily_budget()  # Gemini research spend guard (503 if the day is over budget)
    from services.copilot.corpus import get_or_build
    pack = get_or_build(supabase, role, company, force_refresh=bool(body.force_refresh), user_id=uid)
    return {"pack": _pack_public(pack)}


@router.post("/practice/start")
async def copilot_start(body: StartRequest, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    supabase, uid = _require_pro(authorization)
    role = (body.role or "").strip()
    company = (body.company or "").strip()
    if not role and not company:
        raise HTTPException(status_code=422, detail="Tell me the role or company you're targeting.")
    check_rate_limit(f"copilot:start:{uid}", max_calls=10, window_seconds=120)
    assert_daily_budget()

    from services.copilot.corpus import get_or_build
    from services.copilot.casting import pick_scenario, to_case_content, role_emphasis_block

    pack = get_or_build(supabase, role, company, user_id=uid)
    scen = pick_scenario(pack, focus=(body.focus or ""), difficulty=(body.difficulty or ""))
    if not scen:
        raise HTTPException(status_code=422, detail="Couldn't build a practice scenario yet - try a broader role.")

    ins = supabase.table("copilot_runs").insert({
        "user_id": uid, "role_key": pack.role_key, "company_key": pack.company_key or "",
        "display_role": pack.display_role, "display_company": pack.display_company,
        "scenario": scen.to_dict(), "status": "active",
    }).execute()
    run = (ins.data or [None])[0]
    if not run or not run.get("id"):
        raise HTTPException(status_code=500, detail="Could not open a practice run.")
    run_id = run["id"]

    # The cast case_content the interviewer works from - stored once as the seed.
    case_content = role_emphasis_block(pack, scen) + "\n\n" + to_case_content(pack, scen)
    try:
        supabase.table("copilot_messages").insert(
            {"run_id": run_id, "role": "system", "content": case_content, "kind": "seed"}).execute()
    except Exception:
        pass

    scen_pub = scen.to_dict()
    scen_pub.pop("solution_outline", None)
    return {"run_id": run_id, "scenario": scen_pub, "pack": _pack_public(pack)}


@router.post("/practice/message")
async def copilot_message(body: MessageRequest, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    supabase, uid = _require_pro(authorization)
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=422, detail="Say something to the interviewer first.")
    check_rate_limit(f"copilot:msg:{uid}", max_calls=40, window_seconds=60)
    assert_daily_budget()

    r = (supabase.table("copilot_runs").select("id, status")
         .eq("id", body.run_id).eq("user_id", uid).maybe_single().execute())
    run = r.data or None
    if not run:
        raise HTTPException(status_code=404, detail="Practice run not found.")
    if run.get("status") != "active":
        raise HTTPException(status_code=409, detail="This practice is already submitted.")

    msgs = (supabase.table("copilot_messages").select("role, content, kind, created_at")
            .eq("run_id", body.run_id).order("created_at", desc=False).execute()).data or []
    seed = next((m["content"] for m in msgs if m.get("kind") == "seed"), "")
    transcript: List[Dict[str, str]] = [
        {"role": m["role"], "kind": m.get("kind") or "text", "content": m.get("content") or ""}
        for m in msgs if m.get("role") in ("user", "interviewer") and m.get("content")
    ]
    if len([m for m in transcript if m["role"] == "user"]) >= MAX_TURNS:
        raise HTTPException(status_code=409, detail="You've gone deep - submit to get your scored debrief.")

    from services.copilot.engine.interview_engine import complete_interviewer_reply, InterviewEngineError
    try:
        reply = complete_interviewer_reply(
            case_content=seed or "You are running a role-specific practice interview.",
            case_type="case", transcript=transcript, new_user_message=content,
        )
    except InterviewEngineError:
        raise HTTPException(status_code=502, detail="Interviewer is unavailable right now - try again.")
    except Exception:
        raise HTTPException(status_code=502, detail="Interviewer is unavailable right now - try again.")

    try:
        supabase.table("copilot_messages").insert(
            {"run_id": body.run_id, "role": "user", "content": content, "kind": "text"}).execute()
        supabase.table("copilot_messages").insert(
            {"run_id": body.run_id, "role": "interviewer", "content": reply, "kind": "text"}).execute()
    except Exception:
        pass

    return {"reply": reply}


@router.post("/practice/submit")
async def copilot_submit(body: SubmitRequest, authorization: Optional[str] = Header(default=None)):
    _require_enabled()
    supabase, uid = _require_pro(authorization)
    check_rate_limit(f"copilot:submit:{uid}", max_calls=10, window_seconds=120)
    assert_daily_budget()

    r = (supabase.table("copilot_runs")
         .select("id, status, role_key, company_key, display_role, display_company, scenario")
         .eq("id", body.run_id).eq("user_id", uid).maybe_single().execute())
    run = r.data or None
    if not run:
        raise HTTPException(status_code=404, detail="Practice run not found.")

    msgs = (supabase.table("copilot_messages").select("role, content")
            .eq("run_id", body.run_id).order("created_at", desc=False).execute()).data or []
    lines: List[str] = []
    have_user = False
    for m in msgs:
        if m.get("role") in ("user", "interviewer") and m.get("content"):
            who = "CANDIDATE" if m["role"] == "user" else "INTERVIEWER"
            if m["role"] == "user":
                have_user = True
            lines.append(f"{who}: {m['content']}")
    if not have_user:
        raise HTTPException(status_code=422, detail="Answer the scenario before submitting.")
    transcript_text = "\n".join(lines)

    from services.copilot.corpus import get_pack, get_or_build
    from services.copilot.schemas import Scenario
    from services.copilot.scoring import score_role_answer

    pack = get_pack(supabase, run["role_key"], run.get("company_key") or None)
    if pack is None:
        pack = get_or_build(supabase, run.get("display_role") or run["role_key"],
                            run.get("display_company") or "", user_id=uid)
    scen = Scenario.from_dict(run.get("scenario") or {})

    try:
        feedback = score_role_answer(pack, scen, transcript_text, user_id=uid)
    except Exception:
        raise HTTPException(status_code=502, detail="Scoring is unavailable right now - try again.")

    try:
        supabase.table("copilot_scores").insert({
            "run_id": body.run_id, "user_id": uid, "score": feedback.get("score"),
            "breakdown": feedback.get("breakdown") or {}, "feedback": feedback, "model": "gpt-4o",
        }).execute()
        supabase.table("copilot_runs").update(
            {"status": "scored", "submitted_at": _now_iso()}).eq("id", body.run_id).execute()
    except Exception:
        pass

    return {"feedback": feedback, "solution_outline": scen.solution_outline}
