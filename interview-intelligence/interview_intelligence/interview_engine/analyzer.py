"""Live turn analysis wrapper: the interviewer's ears (fast model), with a safe fallback."""

from __future__ import annotations

import json
from typing import Optional

from ..ai.guard import wrap_untrusted
from ..ai.prompts import TURN_ANALYZER
from ..ai.runner import RunContext, StructuredOutputError, run_structured
from ..textutil import truncate
from .schemas import TurnAnalysis


def analyze_turn(*, ctx: RunContext, item: dict, answer: str, exchange_so_far: str, memory: list,
                 slots_context: list, probes_used: int) -> Optional[TurnAnalysis]:
    q = {
        "question": item.get("text"), "intent": item.get("intent"),
        "expected_evidence": item.get("expected_evidence", [])[:6],
        "strong_signals": item.get("strong_signals", [])[:4], "weak_signals": item.get("weak_signals", [])[:4],
        "competencies": item.get("competency_ids", []), "probes_used": probes_used,
        "claim_ids": (item.get("selection_reason") or {}).get("claim_ids") or [],
    }
    body = (
        f"QUESTION CONTEXT:\n{json.dumps(q, ensure_ascii=False)}\n\n"
        f"MEMORY (earlier summaries, refer by ref):\n{json.dumps(memory[-8:], ensure_ascii=False)}\n\n"
        f"COMPARABLE CLAIM SLOTS (refer by id):\n{json.dumps(slots_context, ensure_ascii=False)}\n\n"
    )
    if exchange_so_far:
        body += "THIS EXCHANGE SO FAR:\n" + wrap_untrusted("exchange", "x", truncate(exchange_so_far, 3000)) + "\n\n"
    body += "CANDIDATE'S LATEST MESSAGE:\n" + wrap_untrusted("answer", "latest", truncate(answer, 6000))
    try:
        return run_structured(TURN_ANALYZER, body, TurnAnalysis, ctx,
                              sim_input={"answer": answer, "item": q, "slots": slots_context, "memory": memory[-8:]})
    except StructuredOutputError:
        return None
