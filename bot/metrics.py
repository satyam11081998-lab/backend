"""C4 Step 2 deterministic metrics (no model calls)."""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services.case_models import stage_of

from bot.data import WindowDataset, normalize_session_state
from bot.state import parse_iso_utc
from bot.turns import AttemptEval, stage_phase_index

REGIONS = ("IN", "US")

_PHASE_LABELS = {
    0: "Clarify",
    1: "Classify",
    2: "Decompose",
    3: "Assume",
    4: "Calculate",
    5: "Validate",
    6: "Conclude",
}


def _in_range(ts_raw: Optional[str], start: datetime, end: datetime, *, inclusive_end: bool = True) -> bool:
    if not ts_raw:
        return False
    try:
        dt = parse_iso_utc(str(ts_raw))
    except Exception:  # noqa: BLE001
        return False
    return (start <= dt <= end) if inclusive_end else (start <= dt < end)


def _structure_node_count(attempt: Dict[str, Any]) -> int:
    ss = normalize_session_state(attempt.get("session_state"))
    nodes = ss.get("structure") or []
    return sum(1 for n in nodes if isinstance(n, dict) and (n.get("kind") or "stated") != "gap")


def _reached_validate(attempt: Dict[str, Any], case_type: str) -> bool:
    ss = normalize_session_state(attempt.get("session_state"))
    try:
        st = stage_of(ss, case_type)
    except Exception:  # noqa: BLE001
        return False
    return st in ("Validate", "Diagnose")


def _coerce_float(val: Any) -> Optional[float]:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


def _structure_subscore_pct(sub: Dict[str, Any]) -> Optional[float]:
    fj = sub.get("feedback_json") or {}
    if isinstance(fj, str):
        try:
            fj = json.loads(fj)
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(fj, dict):
        return None
    bd = fj.get("breakdown") or {}
    if not isinstance(bd, dict):
        return None
    raw_val = _coerce_float(bd.get("structure"))
    if raw_val is None:
        return None
    rubric = str(fj.get("rubric") or "").strip().lower()
    if rubric == "guesstimate":
        return raw_val
    if rubric == "case" or raw_val <= 25.0:
        return raw_val * 100.0 / 25.0
    return raw_val


def _compute_slice_metrics(
    *,
    attempts_eval: List[AttemptEval],
    case_attempts: List[Dict[str, Any]],
    submissions: List[Dict[str, Any]],
    rolling_7d_attempts: List[Dict[str, Any]],
) -> Dict[str, Tuple[float, Optional[str]]]:
    n_att = len(attempts_eval)
    completion = round(len(case_attempts) / n_att, 4) if n_att else 0.0

    depths = [ae.user_turn_count for ae in attempts_eval]
    median_depth = round(float(statistics.median(depths)), 4) if depths else 0.0

    struct_hits = sum(1 for ae in attempts_eval if _structure_node_count(ae.attempt) >= 3)
    structure_reached = round(struct_hits / n_att, 4) if n_att else 0.0

    validate_hits = sum(1 for ae in attempts_eval if _reached_validate(ae.attempt, ae.case_type))
    self_check_rate = round(validate_hits / n_att, 4) if n_att else 0.0

    unfinished = [ae for ae in attempts_eval if str(ae.attempt.get("status") or "").lower() != "submitted"]
    if unfinished:
        ordered = sorted(unfinished, key=lambda ae: ae.final_stage_phase)
        mid = ordered[len(ordered) // 2]
        abandon_idx = float(mid.final_stage_phase)
        abandon_stage = mid.final_stage or _PHASE_LABELS.get(mid.final_stage_phase, "Clarify")
    else:
        abandon_idx = 0.0
        abandon_stage = "None"

    by_user: Dict[str, int] = {}
    for a in rolling_7d_attempts:
        uid = str(a.get("user_id") or "")
        if uid:
            by_user[uid] = by_user.get(uid, 0) + 1
    return_rate = round(sum(1 for c in by_user.values() if c >= 2) / len(by_user), 4) if by_user else 0.0

    scores = [v for s in submissions for v in [_coerce_float(s.get("score"))] if v is not None]
    mean_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    struct_scores = [v for s in submissions for v in [_structure_subscore_pct(s)] if v is not None]
    mean_structure = round(sum(struct_scores) / len(struct_scores), 4) if struct_scores else 0.0

    total_asst = sum(ae.assistant_reply_count for ae in attempts_eval)
    total_praise = sum(ae.praise_only_count for ae in attempts_eval)
    praise_only_rate = round(total_praise / total_asst, 4) if total_asst else 0.0

    total_cand = sum(ae.user_turn_count for ae in attempts_eval)
    total_rep = sum(ae.repeat_signal_count for ae in attempts_eval)
    repeat_rate = round(total_rep / total_cand, 4) if total_cand else 0.0

    return {
        "completion_rate": (completion, None),
        "median_depth": (median_depth, None),
        "structure_reached": (structure_reached, None),
        "self_check_rate": (self_check_rate, None),
        "abandon_point": (abandon_idx, abandon_stage),
        "return_rate": (return_rate, None),
        "mean_score": (mean_score, None),
        "mean_structure_subscore": (mean_structure, None),
        "praise_only_rate": (praise_only_rate, None),
        "repeat_rate": (repeat_rate, None),
    }


def _filter_region(items: Iterable[Any], region: str, get_region) -> List[Any]:
    return [it for it in items if get_region(it) == region]


def compute_all_metrics(
    dataset: WindowDataset,
    attempt_evals: List[AttemptEval],
    windows: Dict[str, datetime],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Compute C4 Step 2 metrics overall + per region (IN/US) vs 7d comparison."""
    w_from, w_to = windows["window_from"], windows["window_to"]
    p_from, p_to = windows["prev_7d_from"], windows["prev_7d_to"]
    r_from, r_to = windows["rolling_7d_from"], windows["rolling_7d_to"]

    cur_evals = [ae for ae in attempt_evals if _in_range(ae.attempt.get("created_at"), w_from, w_to, inclusive_end=True)]
    prev_evals = [ae for ae in attempt_evals if _in_range(ae.attempt.get("created_at"), p_from, p_to, inclusive_end=False)]

    cur_ca = [r for r in dataset.case_attempts if _in_range(r.get("created_at"), w_from, w_to, inclusive_end=True)]
    prev_ca = [r for r in dataset.case_attempts if _in_range(r.get("created_at"), p_from, p_to, inclusive_end=False)]

    cur_sub = [r for r in dataset.submissions if _in_range(r.get("created_at"), w_from, w_to, inclusive_end=True)]
    prev_sub = [r for r in dataset.submissions if _in_range(r.get("created_at"), p_from, p_to, inclusive_end=False)]

    roll_cur_att = [a for a in dataset.attempts if _in_range(a.get("created_at"), r_from, r_to, inclusive_end=True)]
    roll_prev_att = [a for a in dataset.attempts if _in_range(a.get("created_at"), p_from, p_to, inclusive_end=False)]

    cur_all = _compute_slice_metrics(
        attempts_eval=cur_evals, case_attempts=cur_ca, submissions=cur_sub, rolling_7d_attempts=roll_cur_att
    )
    prev_all = _compute_slice_metrics(
        attempts_eval=prev_evals, case_attempts=prev_ca, submissions=prev_sub, rolling_7d_attempts=roll_prev_att
    )

    cur_by_reg: Dict[str, Dict[str, Tuple[float, Optional[str]]]] = {}
    prev_by_reg: Dict[str, Dict[str, Tuple[float, Optional[str]]]] = {}
    region_counts: Dict[str, int] = {}

    for reg in REGIONS:
        c_ev = _filter_region(cur_evals, reg, lambda x: x.region)
        p_ev = _filter_region(prev_evals, reg, lambda x: x.region)
        region_counts[reg] = len(c_ev)
        cur_by_reg[reg] = _compute_slice_metrics(
            attempts_eval=c_ev,
            case_attempts=_filter_region(cur_ca, reg, dataset.record_region),
            submissions=_filter_region(cur_sub, reg, dataset.record_region),
            rolling_7d_attempts=_filter_region(roll_cur_att, reg, dataset.attempt_region),
        )
        prev_by_reg[reg] = _compute_slice_metrics(
            attempts_eval=p_ev,
            case_attempts=_filter_region(prev_ca, reg, dataset.record_region),
            submissions=_filter_region(prev_sub, reg, dataset.record_region),
            rolling_7d_attempts=_filter_region(roll_prev_att, reg, dataset.attempt_region),
        )

    metrics_out: Dict[str, Any] = {}
    for name, (val, stage_label) in cur_all.items():
        pval, pstage = prev_all[name]
        entry: Dict[str, Any] = {
            "value": val,
            "prev_7d": pval,
            "delta": round(val - pval, 4),
            "regions": {},
        }
        if stage_label is not None:
            entry["stage"] = stage_label
            entry["prev_7d_stage"] = pstage
        for reg in REGIONS:
            rv, rs = cur_by_reg[reg][name]
            pv, ps = prev_by_reg[reg][name]
            reg_entry: Dict[str, Any] = {
                "value": rv,
                "prev_7d": pv,
                "delta": round(rv - pv, 4),
            }
            if rs is not None:
                reg_entry["stage"] = rs
                reg_entry["prev_7d_stage"] = ps
            entry["regions"][reg] = reg_entry
        metrics_out[name] = entry

    return metrics_out, region_counts
