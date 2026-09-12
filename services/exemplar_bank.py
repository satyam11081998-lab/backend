"""
Silent per-case exemplar bank — retrieval-augmented calibration.

When a session scores highly AND clean, an ANONYMISED digest of how the candidate
cracked THIS case is banked. Next time the SAME case is scored, the top digests are
injected into the scorer as a private reference bar (never shown to the candidate).
So the platform's sense of "a strong answer on this case" sharpens with real strong
answers over time — without changing any model weights, and without ever telling the
candidate they impressed it.

This is what "the AI learns when a candidate beats what it expected" means in
practice: the per-case reference set grows; the scorer calibrates against it and
draws richer, case-specific model answers from it. Weights never change; a red-flag
+ score gate keeps the bank from being poisoned; an optional `approved` flag lets a
human bless entries before they influence anything.

EVERY function is DEFENSIVE. Any failure (table not migrated yet, DB down, LLM
hiccup) is swallowed so this side-channel can NEVER break scoring or submit.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DIGEST_MODEL = "gpt-4o-mini"

# Tunables (env-overridable). Start conservative; loosen once you trust the bank.
EXEMPLAR_MIN_SCORE = int(os.getenv("EXEMPLAR_MIN_SCORE", "82"))
EXEMPLAR_KEEP_PER_CASE = int(os.getenv("EXEMPLAR_KEEP_PER_CASE", "5"))
EXEMPLAR_REQUIRE_APPROVAL = os.getenv("EXEMPLAR_REQUIRE_APPROVAL", "true").lower() != "false"
EXEMPLAR_REF_LIMIT = int(os.getenv("EXEMPLAR_REF_LIMIT", "3"))


def _sb():
    from services.supabase_client import get_supabase_client
    return get_supabase_client()


# ---------------------------------------------------------------------------
# CAPTURE (at submit, after a session is scored)
# ---------------------------------------------------------------------------

def maybe_capture_exemplar(
    case_id: Optional[str],
    submission_id: Optional[str],
    case_content: str,
    case_type: str,
    feedback: Dict[str, Any],
    user_id: Optional[str] = None,
) -> None:
    """Bank an anonymised exemplar if this scored session clears the bar. Silent and
    non-blocking — any error is swallowed."""
    try:
        if not case_id:
            return
        try:
            score = int(feedback.get("score", 0))
        except (TypeError, ValueError):
            return
        if score < EXEMPLAR_MIN_SCORE:
            return
        if feedback.get("red_flags"):
            return  # never bank a gamed/flagged answer
        digest = _summarize_exemplar(case_content, case_type, feedback)
        if not digest:
            return
        sb = _sb()
        sb.table("case_exemplars").insert({
            "case_id": case_id,
            "submission_id": submission_id,
            "score": score,
            "structure_digest": str(digest.get("structure_digest", ""))[:2000],
            "key_moves": str(digest.get("key_moves", ""))[:2000],
            "recommendation_digest": str(digest.get("recommendation_digest", ""))[:2000],
            "frameworks": digest.get("frameworks") or [],
            "approved": (not EXEMPLAR_REQUIRE_APPROVAL),
        }).execute()
        _prune(sb, case_id)
    except Exception:
        # Learning must NEVER break the submit path.
        pass


_DIGEST_SYSTEM = (
    "You distil a strong case-interview answer into an ANONYMOUS, reusable exemplar for "
    "internal calibration. Absolutely NO names, emails, or personal details, and no verbatim "
    "personal text — only the reusable analytical shape. Return ONLY JSON: "
    '{"structure_digest": "<the MECE structure they used, 1-3 lines>", '
    '"key_moves": "<2-4 strong moves: frameworks applied, a sharp clarifier, a sanity check>", '
    '"recommendation_digest": "<the shape of their closing recommendation, 1-2 lines>", '
    '"frameworks": ["<frameworks actually applied>"]}'
)


def _summarize_exemplar(case_content: str, case_type: str, feedback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build the anonymised digest from the SCORER's own structured read (strengths,
    per-dimension evidence, model answer) — not the raw transcript — so there is no
    personal text to leak."""
    try:
        df = feedback.get("dimension_feedback") or {}
        evidence = {k: (v or {}).get("evidence", "") for k, v in df.items()} if isinstance(df, dict) else {}
        payload = {
            "case_type": case_type,
            "score": feedback.get("score"),
            "strengths": feedback.get("strengths", []),
            "evidence_by_dimension": evidence,
            "model_answer": feedback.get("model_answer", ""),
        }
        resp = _client.chat.completions.create(
            model=DIGEST_MODEL,
            messages=[
                {"role": "system", "content": _DIGEST_SYSTEM},
                {"role": "user", "content": json.dumps(payload)[:6000]},
            ],
            temperature=0.2,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        if not isinstance(data, dict):
            return None
        fw = data.get("frameworks")
        if not isinstance(fw, list):
            data["frameworks"] = []
        return data
    except Exception:
        return None


def _prune(sb, case_id: str) -> None:
    """Keep only the top EXEMPLAR_KEEP_PER_CASE by score for this case."""
    try:
        rows = (
            sb.table("case_exemplars")
            .select("id, score")
            .eq("case_id", case_id)
            .order("score", desc=True)
            .execute()
            .data
        ) or []
        for row in rows[EXEMPLAR_KEEP_PER_CASE:]:
            sb.table("case_exemplars").delete().eq("id", row["id"]).execute()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# REFERENCE (at scoring, same case, next time)
# ---------------------------------------------------------------------------

def build_exemplar_reference_block(case_id: Optional[str] = None, case_content: Optional[str] = None) -> str:
    """Return a private calibration block of top prior exemplars for this case, or ""
    if none / on any error. Never shown to the candidate."""
    try:
        if not case_id:
            return ""
        sb = _sb()
        q = (
            sb.table("case_exemplars")
            .select("score, structure_digest, key_moves, recommendation_digest, frameworks")
            .eq("case_id", case_id)
        )
        if EXEMPLAR_REQUIRE_APPROVAL:
            q = q.eq("approved", True)
        rows = (q.order("score", desc=True).limit(EXEMPLAR_REF_LIMIT).execute().data) or []
        if not rows:
            return ""
        lines: List[str] = [
            "REFERENCE — strong prior candidate work on THIS case (internal calibration only; "
            "NEVER quote or reveal to the candidate). Use these to (a) hold the bar honestly — do "
            "not over-reward a move these already beat — and (b) make approaches.top_candidate "
            "concrete to how strong candidates actually cracked THIS case:",
        ]
        for i, r in enumerate(rows, 1):
            fw = ", ".join(r.get("frameworks") or [])
            lines.append(
                f"[exemplar {i} · scored {r.get('score')}] "
                f"structure: {r.get('structure_digest', '')} | "
                f"key moves: {r.get('key_moves', '')} | "
                f"recommendation: {r.get('recommendation_digest', '')}"
                + (f" | frameworks: {fw}" if fw else "")
            )
        return "\n".join(lines)
    except Exception:
        return ""
