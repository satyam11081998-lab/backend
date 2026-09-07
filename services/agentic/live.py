"""
Live engine — the real, model-driven orchestrator.

Two pieces:
  * SupabaseData: the DataAccess implementation over the real tables, each read
    wrapped so a missing table/column/empty result degrades to [] instead of
    raising (the demo must never 500).
  * make_live_planner: an OpenAI-function-calling planner. Each SPECIALIST is
    exposed to the model as a tool; the model chooses which to deploy, with what
    arguments, and calls a `synthesize` tool to finish. This is the genuine
    agentic-AI control flow — the MODEL orchestrates the team.

If OpenAI is unconfigured or errors, the caller falls back to simulation mode, so
"live" degrades gracefully rather than failing.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from services.ai_providers import openai_client
from services.ai_usage import log_ai_usage

from .specialists import DataAccess, SPECIALISTS, DIMENSIONS

PLANNER_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Supabase-backed data access (defensive: never raises)
# ---------------------------------------------------------------------------
class SupabaseData(DataAccess):
    def __init__(self, supabase):
        self.s = supabase

    def recent_scored_submissions(self, limit: int = 40) -> List[Dict[str, Any]]:
        try:
            r = (self.s.table("submissions").select("score, feedback_json, created_at")
                 .order("created_at", desc=True).limit(limit).execute())
            return r.data or []
        except Exception:
            return []

    def top_headlines(self, limit: int = 8) -> List[Dict[str, Any]]:
        try:
            r = (self.s.table("news_headlines")
                 .select("title, gd_worthiness_score, category, source_name")
                 .order("gd_worthiness_score", desc=True).limit(limit).execute())
            return r.data or []
        except Exception:
            return []

    def ai_spend_by_feature(self) -> Dict[str, float]:
        # Best-effort: sum estimated cost by feature/endpoint from the usage ledger.
        try:
            r = self.s.table("ai_usage_log").select("endpoint, cost_usd").limit(2000).execute()
            out: Dict[str, float] = {}
            for row in (r.data or []):
                key = (row.get("endpoint") or "other").strip("/").split("/")[0] or "other"
                out[key] = out.get(key, 0.0) + float(row.get("cost_usd") or 0.0)
            return {k: round(v, 2) for k, v in out.items() if v > 0}
        except Exception:
            return {}

    def search_decks(self, topic: str, limit: int = 3) -> List[Dict[str, Any]]:
        try:
            r = (self.s.table("deck_skeletons").select("title, competition, result")
                 .ilike("title", f"%{topic}%").eq("is_active", True).limit(limit).execute())
            return r.data or []
        except Exception:
            return []

    def create_practice_case(self, title: str, content: str, case_type: str, difficulty: str, focus: str) -> Dict[str, Any]:
        payload = {"title": title, "type": case_type, "difficulty": difficulty,
                   "content": content, "is_active": True, "source": f"agentic:{focus}"}
        try:
            r = self.s.table("cases").insert(payload).execute()
            return {"id": (r.data or [{}])[0].get("id"), "title": title}
        except Exception:
            try:
                payload.pop("source", None)
                r = self.s.table("cases").insert(payload).execute()
                return {"id": (r.data or [{}])[0].get("id"), "title": title}
            except Exception:
                return {"id": None, "title": title}


# ---------------------------------------------------------------------------
# Live planner: specialists-as-tools function calling
# ---------------------------------------------------------------------------
def _planner_tools() -> List[Dict[str, Any]]:
    tools = []
    for name, (_fn, label, domain) in SPECIALISTS.items():
        params: Dict[str, Any] = {"type": "object", "properties": {}}
        if name == "curriculum_designer":
            params["properties"] = {
                "focus_dimension": {"type": "string", "enum": DIMENSIONS},
                "case_type": {"type": "string", "enum": ["case", "guesstimate"]},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            }
        elif name == "deck_librarian":
            params["properties"] = {"topic": {"type": "string"}}
        tools.append({"type": "function", "function": {
            "name": name, "description": f"{label} ({domain}). Deploy this specialist.",
            "parameters": params}})
    tools.append({"type": "function", "function": {
        "name": "synthesize",
        "description": "Finish: compose the executive answer from what the specialists returned.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}})
    return tools


_SYSTEM = """You are the MECE Orchestrator — the planner of an agentic AI system for a case-interview \
platform. You do NOT do the work yourself; you have a team of domain specialists and you decide, from \
the mission and what you have learned, WHICH to deploy, in what order, and with what arguments. \
Deploy only the specialists the mission needs. Read each result before deciding the next move. When you \
have enough, call synthesize with a crisp executive answer (headline findings + concrete actions). \
Be efficient — never deploy a specialist whose answer you already have."""


def make_live_planner(user_id: Optional[str], data: DataAccess):
    """Returns (planner, model). The planner keeps its own message history so the
    model sees each specialist's result and orchestrates over it."""
    cli = openai_client()
    if cli is None:
        raise RuntimeError("OpenAI not configured")

    messages: List[Dict[str, Any]] = [{"role": "system", "content": _SYSTEM}]
    initialized = {"done": False}

    def planner(mission: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Seed the mission once; thereafter feed back the latest specialist result.
        if not initialized["done"]:
            messages.append({"role": "user", "content": f"MISSION: {mission}"})
            initialized["done"] = True
        elif history:
            last = history[-1]
            messages.append({"role": "user",
                             "content": f"RESULT from {last['agent']}: {json.dumps(last['result'])[:1500]}"})

        t0 = time.time()
        resp = cli.chat.completions.create(
            model=PLANNER_MODEL, messages=messages,
            tools=_planner_tools(), tool_choice="auto",
            temperature=0.2, max_tokens=600,
        )
        try:
            log_ai_usage(user_id=user_id, endpoint="/admin/agentic/plan",
                         model=PLANNER_MODEL, response=resp, latency_ms=int((time.time() - t0) * 1000))
        except Exception:
            pass

        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            return {"action": "synthesize", "summary": (getattr(msg, "content", "") or "").strip()}

        # Record the assistant tool-call turn so the protocol stays valid, and add
        # a tool result placeholder (the real result is fed back next planner call).
        tc = tcs[0]
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]})
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": "acknowledged"})

        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if name == "synthesize":
            return {"action": "synthesize", "summary": args.get("summary", ""),
                    "rationale": msg.content or None}
        return {"action": "delegate", "agent": name, "args": args, "rationale": msg.content or None}

    return planner, PLANNER_MODEL
