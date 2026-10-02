"""Live interview state (docs/E_INTERVIEW_STATE.md §2) — a plain dict persisted as JSON.

Kept deliberately small: the live loop reads and writes one row per turn. Helpers here are
pure functions over (state, blueprint) so the decision policy is deterministic and testable.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

STATE_SCHEMA = 1
MEMORY_KEEP = 14


def new_state(blueprint: dict) -> dict:
    sections = blueprint.get("sections", [])
    claims = {c["claim_id"]: {"id": c["claim_id"], "src": "cv", "text": c.get("text", ""),
                              "slots": c.get("slots", {}) or {}, "status": "unprobed", "ladder_step": 0}
              for c in blueprint.get("claims_to_investigate", []) if c.get("claim_id")}
    coverage = {c["competency_id"]: {"asked": 0, "strong": 0, "adequate": 0, "weak": 0, "declined": 0,
                                     "min": c.get("min_evidence", 1), "importance": c.get("importance", "medium")}
                for c in blueprint.get("competencies", [])}
    dv = blueprint.get("difficulty_vector", {})
    return {
        "schema": STATE_SCHEMA,
        "section_idx": 0,
        "section_id": sections[0]["id"] if sections else "intro",
        "queues": {s["id"]: [it["qid"] for it in s["items"]] for s in sections},
        "reserve": {s["id"]: [r["qid"] for r in s.get("reserve", [])] for s in sections},
        "current": None,
        "asked": [], "skipped": [],
        "coverage": coverage,
        "claims": claims,
        "interview_claims": [],
        "contradictions": [],
        "memory": [],
        "recent_openers": [],
        "consecutive": {"strong": 0, "weak": 0, "non_answers": 0},
        "difficulty": {"level": int(dv.get("level", 3)), "base": int(dv.get("level", 3)),
                       "max_probes": int(dv.get("max_probes", 2)), "pushback_rate": float(dv.get("pushback_rate", 0))},
        "clock": {"active_s": 0, "section_started_s": 0, "section_spent": {}},
        "closing": {"stage": "none"},
        "counters": {"exchanges": 0, "memory": 0, "ic": 0, "k": 0, "turns": 0},
        "degraded": False,
    }


def section(blueprint: dict, sid: str) -> Optional[dict]:
    return next((s for s in blueprint.get("sections", []) if s["id"] == sid), None)


def section_ids(blueprint: dict) -> List[str]:
    return [s["id"] for s in blueprint.get("sections", [])]


def item(blueprint: dict, qid: str) -> Optional[dict]:
    for s in blueprint.get("sections", []):
        for it in s.get("items", []) + s.get("reserve", []):
            if it.get("qid") == qid:
                return it
    return None


def total_budget_s(blueprint: dict) -> int:
    return int(blueprint.get("config", {}).get("duration_minutes", 45)) * 60


def closing_budget_s(blueprint: dict) -> int:
    s = section(blueprint, "closing")
    return int(s["budget_s"]) if s else 120


def section_budget_s(blueprint: dict, sid: str) -> int:
    s = section(blueprint, sid)
    return int(s["budget_s"]) if s else 0


def section_spent_s(state: dict, sid: str) -> int:
    clock = state["clock"]
    spent = int(clock["section_spent"].get(sid, 0))
    if state["section_id"] == sid:
        spent += max(0, clock["active_s"] - clock["section_started_s"])
    return spent


def remaining_s(state: dict, blueprint: dict) -> int:
    return max(0, total_budget_s(blueprint) - int(state["clock"]["active_s"]))


def add_memory(state: dict, *, exchange_id: str, summary: str) -> str:
    state["counters"]["memory"] += 1
    ref = f"M{state['counters']['memory']}"
    state["memory"].append({"ref": ref, "exchange": exchange_id, "summary": summary[:240]})
    state["memory"] = state["memory"][-MEMORY_KEEP:]
    return ref


def cov(state: dict, cid: str) -> Dict[str, Any]:
    return state["coverage"].setdefault(cid, {"asked": 0, "strong": 0, "adequate": 0, "weak": 0, "declined": 0,
                                              "min": 1, "importance": "medium"})


def sufficient(state: dict, cid: str) -> bool:
    c = cov(state, cid)
    return c["strong"] >= c["min"] or (c["strong"] + c["adequate"] >= c["min"] + 1)


def snapshot(state: dict) -> dict:
    return copy.deepcopy(state)
