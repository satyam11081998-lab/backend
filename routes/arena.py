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

import os
import random
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from prompts.variants import registry
from services.ai_usage import assert_daily_budget
from services.arena import DEFAULT_MODEL_MODE, model_modes, run_turn
from services.auth import get_verified_user, is_guest_user
from services.clarification_counter import count_clarifications
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

@router.get("/cases")
def arena_cases(
    q: Optional[str] = None,
    case_type: Optional[str] = None,
    limit: int = 50,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Read-only case picker. Any case or guesstimate in the live library."""
    _require_admin(authorization)
    sb = get_supabase_client()
    try:
        rows = (sb.table("cases").select("*").limit(max(1, min(limit, 200))).execute().data) or []
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not read cases: {type(e).__name__}")
    out = []
    for r in rows:
        if case_type and str(r.get("type") or "").lower() != case_type.lower():
            continue
        title = str(r.get("title") or "")
        if q and q.lower() not in title.lower():
            continue
        out.append({
            "id": r.get("id"),
            "title": title,
            "type": r.get("type"),
            "difficulty": r.get("difficulty"),
            "market": r.get("market"),
            "content_chars": len(str(r.get("content") or "")),
        })
    return {"cases": out, "count": len(out)}


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
