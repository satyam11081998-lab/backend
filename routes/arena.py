"""Admin interviewer arena: blind A/B of two prompt variants on a real case.

Gating mirrors the other admin surfaces (routes/coach.py, routes/ai_providers.py):
Supabase bearer token -> users.is_admin -> shared daily AI budget -> rate limit.

STATE MODEL
The client holds the two transcripts and the two prior_states, so the backend keeps
no per-session user data and a restart loses nothing but the blinding. The ONLY
server-side state is the side->variant mapping, held in process precisely so the
client CANNOT know which side is which until a vote is cast. The client literally
never receives a variant id before voting.

Blinding here defends against the admin's own expectation bias, which is the real
threat to an eval: it is not a security boundary against the admin, who can read
the repo anyway.
"""
from __future__ import annotations

import json
import os
import random
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from prompts.variants import registry
from services.ai_providers import openai_client
from services.ai_usage import assert_daily_budget, log_ai_usage
from services.arena import DEFAULT_MODEL_MODE, model_modes, run_turn
from services.auth import get_verified_user, is_guest_user
from services.clarification_counter import count_clarifications
from services.interviewer_decision import detect_violations
from services.markets import llm_case_content
from services.rate_limit import check_rate_limit
from services.supabase_client import get_supabase_client

router = APIRouter(prefix="/admin/arena", tags=["admin-arena"])

MAX_TRANSCRIPT_TURNS = 60
MAX_MESSAGE_CHARS = 4000
_SESSION_TTL_S = 6 * 3600
_SESSION_CAP = 200

# session_id -> {sides: {"left": variant_id, "right": variant_id}, ...}
_SESSIONS: Dict[str, Dict[str, Any]] = {}

# Both arms of a turn run at once: each is one or two model calls, and running them
# one after the other made every turn wait for both. Latency is still measured per
# arm, inside run_turn.
_ARM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="arena-arm")

# --- production parity ------------------------------------------------------------
# Two things every real attempt has that v1 of the arena left out. Both are copied
# from routes/attempts.py rather than imported (importing that route drags the whole
# attempts stack into this one), and tests/test_arena_parity.py reads attempts.py and
# fails the moment either copy drifts.
#
# 1. The clarifying-question quota by plan (contract C9). When it runs out, the
#    interviewer is told so and declines in character -- different behaviour, and
#    most users are on the free plan's 7.
CLARIFICATION_QUOTA = {"free": 7, "lite": 12, "pro": 20}
DEFAULT_TIER = "free"


# 2. The system note seeded as the first row of every attempt. The interviewer reads
#    it on every turn (and the variants differ in how they treat a system turn), so a
#    conversation without it is not the production conversation.
def opening_transcript(case: Dict[str, Any]) -> List[Dict[str, str]]:
    return [{
        "role": "system",
        "kind": "system_note",
        "content": f"Case ready: {case.get('title')}. Ask any clarifying questions before structuring.",
    }]


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


def _gc_sessions() -> None:
    now = time.time()
    for sid in [k for k, v in _SESSIONS.items() if now - v["created_at"] > _SESSION_TTL_S]:
        _SESSIONS.pop(sid, None)
    while len(_SESSIONS) > _SESSION_CAP:
        oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k]["created_at"])
        _SESSIONS.pop(oldest, None)


def _session(session_id: str) -> Dict[str, Any]:
    _gc_sessions()
    s = _SESSIONS.get(session_id)
    if not s:
        raise HTTPException(
            status_code=404,
            detail="Arena session not found or expired (the backend may have restarted). "
                   "Start a new session; the blinding cannot be recovered.",
        )
    return s


def _seed_scenarios() -> List[Dict[str, Any]]:
    """The 50 adversarial situations from tools/eval_interviewer_behavior.py.

    Seeding from these matters: left to type your own turns you will drift towards
    the happy path, and the happy path is where the fast lane answers and the prompt
    never runs. Degrades to [] rather than breaking the arena if the tool moves.
    """
    try:
        from tools.eval_interviewer_behavior import SCENARIOS
    except Exception:  # noqa: BLE001
        return []
    out = []
    for s in SCENARIOS:
        out.append({
            "id": s.get("id"),
            "case_type": s.get("case_type"),
            "turns": len(s.get("transcript") or []),
            "new_message": s.get("new") or "",
            "transcript": s.get("transcript") or [],
            # What a good reply must do -- the adversarial check shows it next to
            # each build's reply, and the judge grades against it.
            "required": s.get("required") or "",
        })
    return out


# --------------------------------------------------------------------------- config

@router.get("/config")
def arena_config(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_admin(authorization)
    persisted, reason = _persistence_status()
    return {
        "variants": registry.describe_variants(),
        # Which frozen snapshot the deployed prompt tree equals, if any. Stops the
        # quiet mistake of believing you are comparing against production when you
        # are not.
        "live_matches": registry.live_matches(),
        "default_arm_a": registry.DEFAULT_ARM_A,
        "default_arm_b": registry.DEFAULT_ARM_B,
        "model_modes": model_modes(),
        "default_model_mode": DEFAULT_MODEL_MODE,
        "teaching_policies": ["coached", "exam"],
        # The plan the simulated candidate is on: sets the clarifying-question quota.
        "tiers": [{"id": t, "quota": q} for t, q in CLARIFICATION_QUOTA.items()],
        "default_tier": DEFAULT_TIER,
        "seed_scenarios": _seed_scenarios(),
        "votes_persisted": persisted,
        "persistence_note": reason,
        "limits": {"max_transcript_turns": MAX_TRANSCRIPT_TURNS, "max_message_chars": MAX_MESSAGE_CHARS},
        # Which code this backend runs, and where the candidate build lives (if set
        # on this service as ARENA_CANDIDATE_URL; the page also lets you type it).
        "build": build_info(),
        "candidate_url": (os.getenv("ARENA_CANDIDATE_URL") or "").strip() or None,
    }


def _persistence_status() -> tuple[bool, str]:
    """The arena works fully without its table; it just cannot keep a tally. Said
    plainly, because a silently-not-saving eval is how you lose a week of judgements
    (cf. the exemplar bank, dormant since September for exactly this reason)."""
    try:
        get_supabase_client().table("arena_votes").select("id").limit(1).execute()
        return True, "Votes are being saved to arena_votes."
    except Exception:
        return False, ("Votes are NOT being saved: run migrations/2026-10-14_arena_votes.sql. "
                       "The arena still works; you just lose the running tally.")


# --------------------------------------------------------------------------- cases

# Light columns only: the picker never needs a brief, and pulling `content` for the
# whole library would ship megabytes to list titles. `market` (migration 0070) and
# `unlisted` (0065) are selected defensively -- a database without them still lists.
_CASE_COLUMNS = "id,title,type,difficulty,market,unlisted,is_active,created_at"
_CASE_COLUMNS_MIN = "id,title,type,difficulty,is_active,created_at"
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _like_term(term: str) -> str:
    """Escape LIKE wildcards so a search for "20%" means the text, not a pattern."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@router.get("/cases")
def arena_cases(
    q: Optional[str] = None,
    case_type: Optional[str] = None,
    market: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Search the WHOLE case library: every case and guesstimate, active or not,
    listed or unlisted, India or US. The filter runs in the database, newest first,
    a page at a time -- v1 fetched 200 arbitrary rows and filtered those, so most of
    the library could never be picked. A pasted case id matches that case exactly."""
    _require_admin(authorization)
    sb = get_supabase_client()
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))
    term = (q or "").strip()

    def _run(columns: str, with_market: bool):
        qry = sb.table("cases").select(columns, count="exact")
        if term:
            qry = qry.eq("id", term) if _UUID_RE.match(term) else qry.ilike("title", f"%{_like_term(term)}%")
        if case_type and case_type != "all":
            qry = qry.eq("type", case_type)
        if with_market and market and market != "all":
            qry = qry.eq("market", market)
        return qry.order("created_at", desc=True).range(offset, offset + limit - 1).execute()

    markets_supported = True
    try:
        res = _run(_CASE_COLUMNS, True)
    except Exception:  # noqa: BLE001 -- a column a migration adds is missing here
        markets_supported = False
        try:
            res = _run(_CASE_COLUMNS_MIN, False)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Could not read cases: {type(e).__name__}")

    rows = res.data or []
    total = getattr(res, "count", None)
    if total is None:
        total = offset + len(rows)
    out = [{
        "id": r.get("id"),
        "title": str(r.get("title") or ""),
        "type": r.get("type"),
        "difficulty": r.get("difficulty"),
        "market": r.get("market"),
        "unlisted": r.get("unlisted"),
        "is_active": r.get("is_active"),
        "created_at": r.get("created_at"),
    } for r in rows]
    return {
        "cases": out,
        "count": len(out),
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(out) < total,
        "markets_supported": markets_supported,
    }


# --------------------------------------------------------------------------- session

class StartArenaRequest(BaseModel):
    case_id: str
    variant_a: str
    variant_b: str
    model_mode: str = DEFAULT_MODEL_MODE
    teaching_policy: str = "coached"
    tier: str = DEFAULT_TIER


@router.post("/session")
def start_session(
    body: StartArenaRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    uid = _require_admin(authorization)
    check_rate_limit(f"arena:session:{uid}", max_calls=60, window_seconds=600)
    for vid in (body.variant_a, body.variant_b):
        if vid not in registry.VARIANTS:
            raise HTTPException(status_code=400, detail=f"Unknown variant: {vid}")
    if body.teaching_policy not in ("coached", "exam"):
        raise HTTPException(status_code=400, detail="teaching_policy must be coached or exam")
    if body.tier not in CLARIFICATION_QUOTA:
        raise HTTPException(status_code=400, detail="tier must be free, lite or pro")

    sb = get_supabase_client()
    try:
        case = (sb.table("cases").select("*").eq("id", body.case_id).single().execute().data) or None
    except Exception:
        case = None
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Randomise which variant renders on the left, ONCE per session: consistent
    # within a conversation (or the dialogue makes no sense), unpredictable across
    # sessions so a left/right habit cannot masquerade as a preference.
    pair = [body.variant_a, body.variant_b]
    random.shuffle(pair)
    session_id = secrets.token_urlsafe(16)
    _SESSIONS[session_id] = {
        "created_at": time.time(),
        "admin_id": uid,
        "case_id": body.case_id,
        "case_type": case.get("type"),
        "sides": {"left": pair[0], "right": pair[1]},
        "model_mode": body.model_mode,
        "teaching_policy": body.teaching_policy,
        "tier": body.tier,
        "clar_quota": CLARIFICATION_QUOTA[body.tier],
        "clar_used": 0,
        "turns": 0,
        "votes": 0,
    }
    _gc_sessions()
    return {
        "session_id": session_id,
        # The case text the CANDIDATE sees; the models get llm_case_content().
        "case": {
            "id": case.get("id"),
            "title": case.get("title"),
            "type": case.get("type"),
            "difficulty": case.get("difficulty"),
            "content": case.get("content"),
        },
        "teaching_policy": body.teaching_policy,
        "model_mode": body.model_mode,
        # What a real attempt starts with. The client seeds BOTH arms with it, so
        # every turn the interviewer reads the same conversation production would.
        "opening_transcript": opening_transcript(case),
        "clarification": {
            "tier": body.tier,
            "quota": CLARIFICATION_QUOTA[body.tier],
            "used": 0,
            "remaining": CLARIFICATION_QUOTA[body.tier],
        },
        # Deliberately absent: which variant is on which side.
        "blinded": True,
        "self_noise_run": body.variant_a == body.variant_b,
    }


# --------------------------------------------------------------------------- turn

class ArmStateIn(BaseModel):
    transcript: List[Dict[str, Any]] = Field(default_factory=list)
    prior_state: Optional[Dict[str, Any]] = None


class ArenaTurnRequest(BaseModel):
    session_id: str
    message: str
    left: ArmStateIn = Field(default_factory=ArmStateIn)
    right: ArmStateIn = Field(default_factory=ArmStateIn)
    clarifications_exhausted: bool = False


def _clean_transcript(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for t in turns[-MAX_TRANSCRIPT_TURNS:]:
        role = str(t.get("role") or "user")
        content = str(t.get("content") or "")
        if not content.strip():
            continue
        out.append({"role": role, "content": content, "kind": str(t.get("kind") or "text")})
    return out


@router.post("/turn")
def arena_turn(
    body: ArenaTurnRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    uid = _require_admin(authorization)
    sess = _session(body.session_id)
    if sess["admin_id"] != uid:
        raise HTTPException(status_code=403, detail="Not your arena session")
    msg = (body.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    if len(msg) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"message over {MAX_MESSAGE_CHARS} chars")
    # Two arms per turn, each possibly two model calls: check the cap BEFORE spending.
    assert_daily_budget()
    check_rate_limit(f"arena:turn:{uid}", max_calls=120, window_seconds=600)

    sb = get_supabase_client()
    try:
        case = (sb.table("cases").select("*").eq("id", sess["case_id"]).single().execute().data) or {}
    except Exception:
        raise HTTPException(status_code=404, detail="Case not found")
    case_content = llm_case_content(case)
    case_type = case.get("type") or sess.get("case_type") or "profitability"

    # Clarifying-question quota, computed the way post_message computes it: the
    # same counter, spent before the model call, and "exhausted" only on a turn that
    # asks for something once the quota is gone. Both arms receive the same
    # candidate turn, so one count serves both.
    quota = int(sess.get("clar_quota") or CLARIFICATION_QUOTA[DEFAULT_TIER])
    used = int(sess.get("clar_used") or 0)
    clar_count = count_clarifications(msg, "text")
    quota_exhausted = (quota - used) <= 0
    clarifications_spent = clar_count > 0 and quota_exhausted
    if clar_count > 0 and not quota_exhausted:
        used = min(quota, used + clar_count)
    sess["clar_used"] = used
    exhausted_for_turn = clarifications_spent or bool(body.clarifications_exhausted)

    arm_inputs = {"left": body.left, "right": body.right}
    futures = {
        side: _ARM_POOL.submit(
            run_turn,
            case_content=case_content,
            case_type=case_type,
            variant_id=sess["sides"][side],
            transcript=_clean_transcript(arm_inputs[side].transcript),
            new_user_message=msg,
            prior_state=arm_inputs[side].prior_state,
            teaching_policy=sess["teaching_policy"],
            model_mode=sess["model_mode"],
            clarifications_exhausted=exhausted_for_turn,
            user_id=uid,
        )
        for side in ("left", "right")
    }
    results: Dict[str, Any] = {}
    for side, fut in futures.items():
        try:
            out = fut.result()
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown variant on {side}")
        # Strip the one field that would break the blind.
        out.pop("variant_id", None)
        results[side] = out

    sess["turns"] += 1
    both_model = all(results[s].get("prompt_consulted") for s in ("left", "right"))
    identical = (results["left"]["reply"] or "") == (results["right"]["reply"] or "")
    return {
        "session_id": body.session_id,
        "turn_index": sess["turns"],
        "left": results["left"],
        "right": results["right"],
        # Interpretation guards, not decoration. A turn neither arm used its prompt
        # for carries no information about the prompts.
        "comparable": bool(both_model),
        "not_comparable_reason": None if both_model else (
            "V11 answered this turn in a deterministic lane (silence or a canned beat), so "
            "neither arm built a prompt. Any difference here is not the variants."
        ),
        "replies_identical": identical,
        "self_noise_run": sess["sides"]["left"] == sess["sides"]["right"],
        "clarification": {
            "counted": clar_count,
            "quota": quota,
            "used": used,
            "remaining": max(0, quota - used),
            # True when this turn ran with the "you are out of questions" directive.
            "exhausted_this_turn": exhausted_for_turn,
        },
    }


# --------------------------------------------------------------------------- vote

class ArenaVoteRequest(BaseModel):
    session_id: str
    turn_index: int
    choice: str                      # "left" | "right" | "tie" | "both_bad"
    note: Optional[str] = None
    candidate_message: Optional[str] = None


@router.post("/vote")
def arena_vote(
    body: ArenaVoteRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Record a preference, THEN reveal the pairing. Reveal-after-vote is the whole
    point: knowing which arm is 'the new one' is enough to decide the vote."""
    uid = _require_admin(authorization)
    sess = _session(body.session_id)
    if sess["admin_id"] != uid:
        raise HTTPException(status_code=403, detail="Not your arena session")
    if body.choice not in ("left", "right", "tie", "both_bad"):
        raise HTTPException(status_code=400, detail="choice must be left, right, tie or both_bad")

    sides = sess["sides"]
    chosen_variant = sides.get(body.choice) if body.choice in ("left", "right") else None
    sess["votes"] += 1

    row = {
        "admin_id": uid,
        "session_key": body.session_id,
        "case_id": sess["case_id"],
        "case_type": sess.get("case_type"),
        "turn_index": body.turn_index,
        "left_variant": sides["left"],
        "right_variant": sides["right"],
        "left_fingerprint": registry.fingerprint(sides["left"]),
        "right_fingerprint": registry.fingerprint(sides["right"]),
        "choice": body.choice,
        "chosen_variant": chosen_variant,
        "model_mode": sess["model_mode"],
        "teaching_policy": sess["teaching_policy"],
        "candidate_message": (body.candidate_message or "")[:2000],
        "note": (body.note or "")[:1000],
    }
    persisted, detail = True, "saved"
    try:
        get_supabase_client().table("arena_votes").insert(row).execute()
    except Exception as e:  # noqa: BLE001
        persisted, detail = False, (
            f"not saved ({type(e).__name__}): run migrations/2026-10-14_arena_votes.sql"
        )

    return {
        "persisted": persisted,
        "detail": detail,
        # The reveal.
        "reveal": {
            "left": {"variant_id": sides["left"], **_variant_label(sides["left"])},
            "right": {"variant_id": sides["right"], **_variant_label(sides["right"])},
            "chosen_variant": chosen_variant,
        },
        "session_votes": sess["votes"],
        "small_n_warning": (
            "One preference is noise. Judge a variant on a dozen-plus votes across "
            "different scenario shapes, not on this turn."
            if sess["votes"] < 12 else None
        ),
    }


def _variant_label(variant_id: str) -> Dict[str, Any]:
    spec = registry.VARIANTS[variant_id]
    return {"label": spec["label"], "origin": spec["origin"], "frozen": bool(spec["frozen"])}


# --------------------------------------------------------------------------- results

@router.get("/results")
def arena_results(
    limit: int = 1000,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Tally of recorded votes.

    Three things this deliberately does NOT do, because each would manufacture a
    conclusion that is not in the data:

      * it does not pool SELF-NOISE runs (both arms the same variant) into win
        rates. Those votes measure the judge, so they are reported separately as
        your own noise floor -- the number a real result has to beat.
      * it does not pool votes cast against DIFFERENT SNAPSHOTS of the same variant
        id. Pairs are keyed by content fingerprint, so a re-snapshot starts a fresh
        bucket instead of silently averaging two experiments.
      * it does not name a winner below a minimum n, and it reports the position
        split so a left/right habit cannot pass itself off as a preference.
    """
    _require_admin(authorization)
    try:
        rows = (get_supabase_client().table("arena_votes")
                .select("*").order("created_at", desc=True)
                .limit(max(1, min(limit, 5000))).execute().data) or []
    except Exception:
        return {
            "available": False,
            "note": ("No vote history: run migrations/2026-10-14_arena_votes.sql. "
                     "The arena works without it, but nothing is being tallied."),
            "total_votes": 0, "pairs": [], "self_noise": None, "position": None,
        }

    def _fp(row: Dict[str, Any], side: str) -> str:
        return (row.get(f"{side}_fingerprint") or "live")[:12]

    pairs: Dict[tuple, Dict[str, Any]] = {}
    self_noise = {"votes": 0, "left": 0, "right": 0, "tie": 0, "both_bad": 0}
    position = {"left": 0, "right": 0, "tie": 0, "both_bad": 0}

    for r in rows:
        lv, rv = r.get("left_variant"), r.get("right_variant")
        choice = r.get("choice")
        if choice in position:
            position[choice] += 1
        if lv == rv:
            self_noise["votes"] += 1
            if choice in self_noise:
                self_noise[choice] += 1
            continue
        # Order-independent key, so A-vs-B and B-vs-A accumulate together, but
        # fingerprint-sensitive, so a re-snapshot does not.
        key = tuple(sorted([(lv, _fp(r, "left")), (rv, _fp(r, "right"))]))
        bucket = pairs.setdefault(key, {
            "variants": [
                {"id": key[0][0], "fingerprint": key[0][1], "wins": 0},
                {"id": key[1][0], "fingerprint": key[1][1], "wins": 0},
            ],
            "votes": 0, "ties": 0, "both_bad": 0, "by_case_type": {},
        })
        bucket["votes"] += 1
        if choice == "tie":
            bucket["ties"] += 1
        elif choice == "both_bad":
            bucket["both_bad"] += 1
        else:
            winner = r.get("chosen_variant")
            for v in bucket["variants"]:
                if v["id"] == winner:
                    v["wins"] += 1
        ct = r.get("case_type") or "unknown"
        bucket["by_case_type"][ct] = bucket["by_case_type"].get(ct, 0) + 1

    MIN_N = 12
    out_pairs = []
    for bucket in pairs.values():
        decided = sum(v["wins"] for v in bucket["variants"])
        leader = max(bucket["variants"], key=lambda v: v["wins"]) if decided else None
        runner = min(bucket["variants"], key=lambda v: v["wins"]) if decided else None
        verdict = None
        if decided < MIN_N:
            verdict = (f"Only {decided} decided vote(s). Too few to call -- judge a variant on "
                       f"{MIN_N}+ across different scenario shapes.")
        elif leader and runner and leader["wins"] == runner["wins"]:
            verdict = "Dead level on decided votes."
        elif leader:
            share = leader["wins"] / decided
            verdict = (f"{leader['id']} leads {leader['wins']}-{runner['wins']} "
                       f"({share:.0%} of decided votes).")
            if share < 0.65:
                verdict += " Close enough that your own noise floor could explain it."
        out_pairs.append({**bucket, "decided_votes": decided, "verdict": verdict})

    decided_sides = position["left"] + position["right"]
    position_note = None
    if decided_sides >= 10:
        left_share = position["left"] / decided_sides
        if left_share >= 0.7 or left_share <= 0.3:
            position_note = (f"You picked the left pane {left_share:.0%} of the time across all "
                             "sessions. Sides are randomised per session, so a split this lopsided "
                             "suggests position bias rather than a prompt difference.")
    if self_noise["votes"] >= 4:
        picked = self_noise["left"] + self_noise["right"]
        if picked:
            self_noise["note"] = (
                f"On {self_noise['votes']} self-noise turns (identical arms) you still picked a "
                f"side {picked} time(s). That is your noise floor: a real result needs to beat it.")
        else:
            self_noise["note"] = (
                f"On {self_noise['votes']} self-noise turns you called every one a tie or both-bad. "
                "Good -- your judgements are not inventing differences.")

    return {
        "available": True,
        "note": None,
        "total_votes": len(rows),
        "min_n_for_a_verdict": MIN_N,
        "pairs": sorted(out_pairs, key=lambda p: -p["votes"]),
        "self_noise": self_noise,
        "position": {**position, "note": position_note},
    }


# =========================================================================== builds
# CURRENT vs NEW. The admin page runs one conversation against TWO deployments: the
# live backend (main) for the left pane and the candidate backend (the `candidate`
# branch, a second Render service) for the right. Each pane calls /arm-turn on its
# OWN backend, so each side runs its own code end to end -- prompts, signals,
# decision logic, voice path, everything -- which a prompt snapshot cannot do.

_STARTED_AT = time.time()


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _git_ref() -> Tuple[str, str]:
    """(commit, branch) this process is running. Render sets RENDER_GIT_COMMIT and
    RENDER_GIT_BRANCH; locally the .git files are read directly -- no subprocess, no
    git lock, works in a worktree."""
    commit = (os.getenv("RENDER_GIT_COMMIT") or "").strip()
    branch = (os.getenv("RENDER_GIT_BRANCH") or "").strip()
    if commit and branch:
        return commit, branch
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gitdir = os.path.join(root, ".git")
        if os.path.isfile(gitdir):  # a worktree: ".git" is a file pointing at the real dir
            gitdir = _read(gitdir).split(":", 1)[1].strip()
            if not os.path.isabs(gitdir):
                gitdir = os.path.normpath(os.path.join(root, gitdir))
        common = gitdir
        if os.path.exists(os.path.join(gitdir, "commondir")):
            common = os.path.normpath(os.path.join(gitdir, _read(os.path.join(gitdir, "commondir"))))
        head = _read(os.path.join(gitdir, "HEAD"))
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            branch = branch or ref.split("refs/heads/", 1)[-1]
            for base in (gitdir, common):
                if os.path.exists(os.path.join(base, ref)):
                    commit = commit or _read(os.path.join(base, ref))
                    break
            else:
                packed = os.path.join(common, "packed-refs")
                if os.path.exists(packed):
                    for line in _read(packed).splitlines():
                        if line.endswith(" " + ref):
                            commit = commit or line.split()[0]
        else:
            commit, branch = commit or head, branch or "(detached)"
    except Exception:  # noqa: BLE001
        pass
    return commit or "unknown", branch or "unknown"


def build_info() -> Dict[str, Any]:
    commit, branch = _git_ref()
    return {
        "commit": commit[:12],
        "branch": branch,
        "service": os.getenv("RENDER_SERVICE_NAME") or "local",
        # "current" on main, "candidate" anywhere else; ARENA_BUILD_ROLE overrides.
        "role": os.getenv("ARENA_BUILD_ROLE") or ("current" if branch == "main" else "candidate"),
        # Which frozen snapshot this deployment's prompt layer equals, if any.
        "prompt_layer": registry.live_matches(),
        "started_at": datetime.fromtimestamp(_STARTED_AT, tz=timezone.utc).isoformat(),
    }


@router.get("/build")
def arena_build(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    """Which code this backend is running. The page shows it on each pane, so you
    always know which commit answered."""
    _require_admin(authorization)
    return {"build": build_info()}


def _scenario(scenario_id: str) -> Dict[str, Any]:
    try:
        from tools.eval_interviewer_behavior import SCENARIOS
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="The adversarial scenario set is not available on this build.")
    for s in SCENARIOS:
        if s.get("id") == scenario_id:
            return s
    raise HTTPException(status_code=404, detail=f"Unknown scenario: {scenario_id}")


def scenario_case_content(case_type: str) -> str:
    """The case text the behavioural eval gives the interviewer for a scenario
    (tools/eval_interviewer_behavior.py run_live), so the arena's adversarial check
    asks exactly what the eval asks. Pinned to the eval by test."""
    return f"A {case_type} case for practice on MECE."


class ArmTurnRequest(BaseModel):
    message: str
    case_id: Optional[str] = None          # a real case or guesstimate ...
    scenario_id: Optional[str] = None      # ... or one of the 50 adversarial scenarios
    transcript: List[Dict[str, Any]] = Field(default_factory=list)
    prior_state: Optional[Dict[str, Any]] = None
    variant: str = "live"                  # this build's own prompt layer, by default
    model_mode: str = DEFAULT_MODEL_MODE
    teaching_policy: str = "coached"
    channel: str = "text"                  # "text" (chat) or "voice" (the voice route's path)
    tier: str = DEFAULT_TIER
    clar_used: int = 0                     # questions this pane has already spent
    clarifications_exhausted: bool = False


@router.post("/arm-turn")
def arena_arm_turn(
    body: ArmTurnRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """ONE interviewer turn on THIS deployment's engine -- the building block of the
    Current vs New panel. Stateless on purpose: the page holds the transcript, the
    learner state and the question count, so a restart, a sleeping free-tier service
    waking up, or a second worker loses nothing. Not blinded: the panel is labelled,
    because you want to know which side is the new build."""
    uid = _require_admin(authorization)
    msg = (body.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    if len(msg) > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=400, detail=f"message over {MAX_MESSAGE_CHARS} chars")
    if body.channel not in ("text", "voice"):
        raise HTTPException(status_code=400, detail="channel must be text or voice")
    if body.tier not in CLARIFICATION_QUOTA:
        raise HTTPException(status_code=400, detail="tier must be free, lite or pro")
    if body.teaching_policy not in ("coached", "exam"):
        raise HTTPException(status_code=400, detail="teaching_policy must be coached or exam")
    if body.variant not in registry.VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unknown variant: {body.variant}")
    if not body.case_id and not body.scenario_id:
        raise HTTPException(status_code=400, detail="case_id or scenario_id is required")
    # Spend checks before any model call.
    assert_daily_budget()
    check_rate_limit(f"arena:arm:{uid}", max_calls=300, window_seconds=600)

    if body.scenario_id:
        s = _scenario(body.scenario_id)
        case_type = s.get("case_type") or "profitability"
        case_content = scenario_case_content(case_type)
    else:
        try:
            case = (get_supabase_client().table("cases").select("*")
                    .eq("id", body.case_id).single().execute().data) or None
        except Exception:  # noqa: BLE001
            case = None
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        case_content = llm_case_content(case)
        case_type = case.get("type") or "profitability"

    # Same quota rule as post_message, on this pane's own count.
    quota = CLARIFICATION_QUOTA[body.tier]
    used = max(0, min(int(body.clar_used or 0), quota))
    clar_count = count_clarifications(msg, "text")
    quota_exhausted = (quota - used) <= 0
    clarifications_spent = clar_count > 0 and quota_exhausted
    if clar_count > 0 and not quota_exhausted:
        used = min(quota, used + clar_count)
    exhausted_for_turn = clarifications_spent or bool(body.clarifications_exhausted)

    out = run_turn(
        case_content=case_content,
        case_type=case_type,
        variant_id=body.variant,
        transcript=_clean_transcript(body.transcript),
        new_user_message=msg,
        prior_state=body.prior_state,
        teaching_policy=body.teaching_policy,
        model_mode=body.model_mode,
        clarifications_exhausted=exhausted_for_turn,
        user_id=uid,
        channel=body.channel,
    )
    return {
        "build": build_info(),
        "channel": body.channel,
        "arm": out,
        "clarification": {
            "counted": clar_count,
            "quota": quota,
            "used": used,
            "remaining": max(0, quota - used),
            "exhausted_this_turn": exhausted_for_turn,
        },
    }


class JudgeRequest(BaseModel):
    scenario_id: str
    reply: str = ""


@router.post("/judge")
def arena_judge(
    body: JudgeRequest,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """The behavioural eval's own LLM judge (same prompt, same model) for one reply.
    The page sends BOTH builds' replies to the SAME backend's judge, so the two are
    graded by one judge, not by two judges that might disagree."""
    uid = _require_admin(authorization)
    s = _scenario(body.scenario_id)
    assert_daily_budget()
    check_rate_limit(f"arena:judge:{uid}", max_calls=300, window_seconds=600)
    from tools.eval_interviewer_behavior import judge_prompt

    reply = body.reply or ""
    violations = detect_violations(reply) if reply.strip() else []
    model = os.getenv("JUDGE_MODEL", "gpt-4o")
    verdict: Dict[str, Any] = {"pass": None, "why": "judge unavailable"}
    cli = openai_client()
    if cli is not None:
        t0 = time.time()
        try:
            jr = cli.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "user", "content": judge_prompt(s, reply)}],
                response_format={"type": "json_object"})
            log_ai_usage(user_id=uid, endpoint="/admin/arena/judge", model=model, response=jr,
                         latency_ms=int((time.time() - t0) * 1000))
            verdict = json.loads(jr.choices[0].message.content or "{}")
        except Exception as e:  # noqa: BLE001
            verdict = {"pass": None, "why": f"judge error: {type(e).__name__}"}
    judge_pass = verdict.get("pass")
    return {
        "scenario_id": body.scenario_id,
        "required": s.get("required") or "",
        # The eval's rule: the judge passes it AND no deterministic guardrail fired.
        "pass": bool(judge_pass) and not violations,
        "judge_pass": judge_pass,
        "why": verdict.get("why"),
        "violations": violations,
        "model": model,
    }
