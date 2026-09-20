"""Keep the PAYWALLED case figures out of the submission row.

`visuals` — the worked profit bridge / 2x2 / driver tree the scorer produces —
are a Pro feature (frontend lib/tier.ts `caseFigures`). They must never be
stored inside `submissions.feedback_json`, because that row belongs to the
user:

  * `submissions_select_own` (migration 0006) lets any user read their OWN
    submission straight from PostgREST with their browser JWT. RLS is
    row-level, so `feedback_json` comes with the row —
    `GET /rest/v1/submissions?select=feedback_json&user_id=eq.<me>` returns
    every figure for every case they attempted.
  * Three server components already pass whole `feedback_json` blobs into
    CLIENT components (dashboard, cases/[id], profile), where they land in the
    RSC payload and are readable in devtools whatever is painted.

So anything left in `feedback_json` is effectively public to its owner, and no
amount of care in the UI can change that. The figures go to `public.case_figures`
instead (migration 0069), which has RLS enabled and NO policy — service role
only.

This module exists so the split is written ONCE. There are two submit paths
(`routes/attempts.py`, the conversational flow, and `routes/submit.py`, the
legacy single-answer flow) and both insert `feedback_json`. The second one was
missed on the first pass, which is exactly the failure this file prevents: a
future third writer calls `pop_figures` or it does not compile past review.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List


def pop_figures(feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Remove `visuals` from a scorer result, in place, and return them.

    Call this BEFORE writing feedback_json. Always safe: a scorer result
    without `visuals` (a guesstimate on an old prompt, a rejection payload)
    returns [].
    """
    if not isinstance(feedback, dict):
        return []
    figures = feedback.pop("visuals", None)
    return figures if isinstance(figures, list) else []


def bank_figures(
    supabase,
    case_id: str,
    figures: List[Dict[str, Any]],
    submission_id: str | None = None,
    score: int | None = None,
) -> None:
    """Store a case's figures, keeping the best-scoring version.

    One row per CASE, not per submission: the figures describe the case's
    economics, not a candidate's performance. A later attempt replaces them
    only if it scored at least as well, so what is on file always comes from
    the strongest answer seen.

    Entirely best-effort and never raises. A pre-0069 database or any other
    error must not fail a submit the user has already earned a score for —
    the figures are a garnish on the results page, the score is not.
    """
    if not figures or not case_id:
        return
    try:
        existing = (
            supabase.table("case_figures")
            .select("source_score")
            .eq("case_id", case_id)
            .maybe_single()
            .execute()
        )
        prev = ((existing.data or {}) or {}).get("source_score")
        if prev is not None and score is not None and int(score) < int(prev):
            return  # a better answer's figures are already on file
        row: Dict[str, Any] = {
            "case_id": case_id,
            "figures": figures,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if submission_id:
            row["source_submission_id"] = submission_id
        if score is not None:
            row["source_score"] = score
        supabase.table("case_figures").upsert(row, on_conflict="case_id").execute()
    except Exception as e:  # noqa: BLE001
        print(f"WARN: case_figures not written (run migration 0069?): {e}")
