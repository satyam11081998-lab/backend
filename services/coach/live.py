"""
Live engine — the real, model-driven Prep Copilot orchestrator.

Two pieces:
  * SupabaseUserData: the DataAccess implementation over the real tables, SCOPED
    TO ONE user, each read wrapped so a missing table/column/empty result
    degrades to a safe default instead of raising (a run must never 500).
  * make_live_planner: an OpenAI function-calling planner. Each SPECIALIST is a
    tool the model can call; the model decides which to deploy, with what
    arguments, reads each result, and calls `synthesize` to finish. This is the
    genuine agentic-AI control flow — the MODEL orchestrates the team; the
    specialists keep every fact grounded in real data.

If OpenAI is unconfigured or errors, the caller falls back to simulation mode,
so "live" degrades gracefully rather than failing.
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
# Supabase-backed, user-scoped data access (defensive: never raises)
# ---------------------------------------------------------------------------
class SupabaseUserData(DataAccess):
    def __init__(self, supabase, user_id: str):
        self.s = supabase
        self.uid = user_id

    def profile(self) -> Dict[str, Any]:
        try:
            r = (self.s.table("users")
                 .select("name, goal_text, placement_focus, weekly_hours_target")
                 .eq("id", self.uid).maybe_single().execute())
            return dict(r.data or {})
        except Exception:
            return {}

    def my_scored_submissions(self, limit: int = 40) -> List[Dict[str, Any]]:
        try:
            r = (self.s.table("submissions").select("score, feedback_json, case_id, created_at")
                 .eq("user_id", self.uid).order("created_at", desc=True).limit(limit).execute())
            return r.data or []
        except Exception:
            return []

    def attempted_case_ids(self) -> List[str]:
        ids: set = set()
        try:
            r = self.s.table("case_attempts").select("case_id").eq("user_id", self.uid).execute()
            for row in (r.data or []):
                if row.get("case_id"):
                    ids.add(row["case_id"])
        except Exception:
            pass
        return list(ids)

    def recommend_cases(self, *, case_type: str, difficulty: Optional[str], firm: Optional[str],
                        exclude_ids: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        # Fetch a candidate pool of active cases of the right type, then filter in
        # Python (jsonb firm match + exclude-set) — robust across PostgREST quirks.
        try:
            r = (self.s.table("cases")
                 .select("id, title, type, difficulty, interview_meta, skill_node, skill_cluster")
                 .eq("is_active", True).eq("type", case_type).limit(60).execute())
            pool = r.data or []
        except Exception:
            return []
        exclude = set(exclude_ids or [])
        pool = [c for c in pool if c.get("id") not in exclude]

        def firm_of(c: Dict[str, Any]) -> str:
            meta = c.get("interview_meta") or {}
            return str((meta.get("firm") if isinstance(meta, dict) else "") or "").lower()

        if firm:
            f = firm.lower()
            firm_hits = [c for c in pool if f and f in firm_of(c)]
            if firm_hits:
                pool = firm_hits
        if difficulty:
            diff_hits = [c for c in pool if (c.get("difficulty") or "").lower() == difficulty.lower()]
            if diff_hits:
                pool = diff_hits
        out = []
        for c in pool[:limit]:
            out.append({"id": c.get("id"), "title": c.get("title"), "type": c.get("type"),
                        "difficulty": c.get("difficulty")})
        return out

    def top_headlines(self, domain: Optional[str] = None, limit: int = 6) -> List[Dict[str, Any]]:
        try:
            r = (self.s.table("news_headlines")
                 .select("title, gd_worthiness_score, category, source_name")
                 .order("gd_worthiness_score", desc=True).limit(limit).execute())
            return r.data or []
        except Exception:
            return []

    def search_decks(self, topic: str, limit: int = 3) -> List[Dict[str, Any]]:
        try:
            r = (self.s.table("deck_skeletons").select("title, competition, result")
                 .ilike("title", f"%{topic}%").eq("is_active", True).limit(limit).execute())
            return r.data or []
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Live planner: specialists-as-tools function calling
# ---------------------------------------------------------------------------
def _planner_tools() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    schemas: Dict[str, Dict[str, Any]] = {
        "diagnostician": {"type": "object", "properties": {}},
        "target_strategist": {"type": "object", "properties": {
            "target_company": {"type": "string"},
            "domain": {"type": "string"},
            "weakest_dimensions": {"type": "array", "items": {"type": "string", "enum": DIMENSIONS}},
        }},
        "case_curator": {"type": "object", "properties": {
            "focus_dimension": {"type": "string", "enum": DIMENSIONS},
            "case_type": {"type": "string", "enum": ["case", "guesstimate"]},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "firm": {"type": "string"},
        }},
        "news_analyst": {"type": "object", "properties": {"domain": {"type": "string"}}},
        "exemplar_scout": {"type": "object", "properties": {
            "focus_dimension": {"type": "string", "enum": DIMENSIONS},
            "topic": {"type": "string"},
        }},
        "roadmap_architect": {"type": "object", "properties": {
            "weakest_dimensions": {"type": "array", "items": {"type": "string", "enum": DIMENSIONS}},
            "target": {"type": "string"},
            "weekly_hours": {"type": "integer"},
        }},
    }
    for name, (_fn, label, domain) in SPECIALISTS.items():
        tools.append({"type": "function", "function": {
            "name": name, "description": f"{label} ({domain}). Deploy this specialist.",
            "parameters": schemas.get(name, {"type": "object", "properties": {}})}})
    tools.append({"type": "function", "function": {
        "name": "synthesize",
        "description": ("Finish the run: compose the final personalised prep plan from what the specialists "
                        "returned. Call this as soon as you have enough — do not keep deploying specialists "
                        "whose answers would not change the plan."),
        "parameters": {"type": "object", "properties": {"summary": {"type": "string", "description": (
            "The full prep plan in the required structure: a one-line verdict; a two to three sentence "
            "diagnosis citing the candidate's real weakest skill, its score and trend, and the colliding "
            "target-firm skill; a Days 1-3 / 4-6 / 7 plan that names the specific cases, the bespoke stretch "
            "case, the news angle and the exemplar the specialists returned; and one 'what good looks like' "
            "line. Answer-first, first person, concise, grounded, no filler.")}}, "required": ["summary"]}}})
    return tools


_SYSTEM = """You are the MECE Prep Copilot — the planning brain of an agentic system that builds a \
personalised case-interview prep plan for ONE candidate and hands them something they can start today. \
You do NOT do the analysis yourself: you command a team of six domain specialists and decide, from the \
candidate's goal, profile and what each specialist returns, WHICH to deploy, in what order, with what \
arguments, and WHEN you have learned enough to stop.

HOW TO ORCHESTRATE
- ALWAYS deploy the diagnostician FIRST. Never advise before you have measured where the candidate stands.
- Read each result before the next move, and pass what you learned forward: the diagnostician's \
weakest_dimensions into target_strategist and roadmap_architect; the priority skill (the target stresses \
it AND they are weak on it) into case_curator and exemplar_scout as focus_dimension; the target company \
into case_curator (firm) and roadmap_architect (target).
- Be genuinely selective — most candidates need three to five specialists, not all six. Deploy one only when \
its answer would change the plan. Skip news_analyst unless the target rewards current-affairs / GD fluency \
(consulting, general management, policy). Skip exemplar_scout when the priority is presence or pure quant, \
where a timed rep matters more than a role model. Never deploy a specialist whose answer you already have.
- If the diagnostician reports no graded attempts, do NOT invent a diagnosis. Deploy target_strategist only, \
then synthesize: tell them warmly to complete three or four scored cases so you can plan from real data.

WHEN YOU SYNTHESISE (call `synthesize`)
Write the plan the way a sharp MBB coach actually talks — direct, warm, specific, first person, Indian-English \
register, money in Rs/crore. Lead with the answer. No hype, no restating this brief, never say 'as an AI'. \
Ground every line in what the specialists returned: name the real case titles, the real headline, the real \
exemplar the team surfaced; never invent a case, a score or a number. Structure it exactly as:
  1. Verdict — one line: the single thing to fix this week and why it matters for their target.
  2. Diagnosis — two or three sentences: their weakest skill with its score and the trend, and the one \
     target-firm skill it collides with. Point at a concrete pattern, not a platitude.
  3. This week — Days 1-3, Days 4-6, Day 7: each a short line naming the specific case / stretch case / \
     news angle / exemplar to use and the technique to drill.
  4. What good looks like — one line on how they will know the gap has closed.
Keep it tight: a plan read once and acted on, not an essay. Short lines beat paragraphs."""


def make_live_planner(user_id: Optional[str], data: DataAccess, context: Dict[str, Any]):
    """Returns (planner, model). The planner keeps its own message history so the
    model sees each specialist's result and orchestrates over it."""
    cli = openai_client()
    if cli is None:
        raise RuntimeError("OpenAI not configured")

    prof = {}
    try:
        prof = data.profile() or {}
    except Exception:
        prof = {}
    ctx_lines = [
        f"GOAL: {context.get('goal', '')}",
        f"TARGET COMPANY: {context.get('target_company', '') or '(unspecified)'}",
        f"DOMAIN/ROLE: {context.get('domain', '') or '(unspecified)'}",
        f"WEEKLY HOURS: {context.get('weekly_hours') or prof.get('weekly_hours_target') or 5}",
        f"PROFILE GOAL NOTE: {prof.get('goal_text') or '(none)'}",
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n".join(ctx_lines)},
    ]
    initialized = {"done": False}

    def planner(goal: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not initialized["done"]:
            initialized["done"] = True  # context already seeded above
        elif history:
            last = history[-1]
            messages.append({"role": "user",
                             "content": f"RESULT from {last['agent']}: {json.dumps(last['result'])[:1800]}"})

        t0 = time.time()
        resp = cli.chat.completions.create(
            model=PLANNER_MODEL, messages=messages,
            tools=_planner_tools(), tool_choice="auto",
            temperature=0.3, max_tokens=900,
        )
        try:
            log_ai_usage(user_id=user_id, endpoint="/coach/plan", model=PLANNER_MODEL,
                         response=resp, latency_ms=int((time.time() - t0) * 1000))
        except Exception:
            pass

        msg = resp.choices[0].message
        tcs = getattr(msg, "tool_calls", None)
        if not tcs:
            return {"action": "synthesize", "summary": (getattr(msg, "content", "") or "").strip()}

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
            return {"action": "synthesize", "summary": args.get("summary", ""), "rationale": msg.content or None}
        return {"action": "delegate", "agent": name, "args": args, "rationale": msg.content or None}

    return planner, PLANNER_MODEL
