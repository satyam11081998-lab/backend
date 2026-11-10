"""C5 report builders and writers for bot/reports/<date>.json and <date>.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from bot.config import scrub_secrets

_METRIC_LABELS = (
    ("completion_rate", "Completion rate"),
    ("median_depth", "Median depth"),
    ("structure_reached", "Structure reached (>=3 nodes)"),
    ("self_check_rate", "Self-check rate (Validate)"),
    ("abandon_point", "Abandon point (stage index)"),
    ("return_rate", "Return rate (7d >=2 attempts)"),
    ("mean_score", "Mean score"),
    ("mean_structure_subscore", "Mean structure sub-score (0-100)"),
    ("praise_only_rate", "Praise-only rate"),
    ("repeat_rate", "Repeat rate"),
)


def build_report_payload(
    *,
    report_date: str,
    window_from_iso: str,
    window_to_iso: str,
    sessions_total: int,
    regions: Dict[str, int],
    cost_usd: float,
    suite: Dict[str, Any],
    metrics: Dict[str, Any],
    failures: List[Dict[str, Any]],
    notes: List[str],
    report_only: bool = False,
    consecutive_failures: int = 0,
) -> Dict[str, Any]:
    """Assemble the C5 JSON contract dict."""
    payload: Dict[str, Any] = {
        "date": report_date,
        "window": {
            "from": window_from_iso,
            "to": window_to_iso,
            "sessions": int(sessions_total),
            "regions": {
                "IN": int(regions.get("IN", 0)),
                "US": int(regions.get("US", 0)),
            },
        },
        "cost_usd": round(float(cost_usd), 6),
        "suite": {
            "pytest_passed": int((suite or {}).get("pytest_passed", 0)),
            "pytest_failed": int((suite or {}).get("pytest_failed", 0)),
            "offline_eval_drift": list((suite or {}).get("offline_eval_drift") or []),
        },
        "metrics": metrics,
        "failures": failures,
        "notes": [scrub_secrets(str(n)) for n in notes if str(n).strip()],
    }
    if report_only:
        payload["report_only"] = True
        payload["consecutive_failures"] = int(consecutive_failures)
    return payload


def _fmt_metric_cell(entry: Dict[str, Any], is_stage: bool = False) -> str:
    if not entry:
        return "0.0"
    val = entry.get("value", 0.0)
    prev = entry.get("prev_7d", 0.0)
    delta = entry.get("delta", 0.0)
    sign = "+" if delta > 0 else ""
    if is_stage and entry.get("stage"):
        return f"{entry['stage']} ({val:.1f}, prev {entry.get('prev_7d_stage', '?')} {prev:.1f}, {sign}{delta:.1f})"
    return f"{val:.4g} (7d {prev:.4g}, {sign}{delta:.4g})"


def render_markdown_report(payload: Dict[str, Any]) -> str:
    """Render the human-readable bot/reports/<date>.md report."""
    lines: List[str] = []

    if payload.get("report_only"):
        cf = payload.get("consecutive_failures", 3)
        lines.extend([
            "> [!CAUTION]",
            f"> **REPORT-ONLY MODE ACTIVE (consecutive_failures = {cf})**",
            "> Three or more consecutive failures occurred. Gemini model calls are paused and this run is report-only until investigated and reset.",
            "",
        ])

    lines.append(f"# MECE Backend Daily Report — {payload['date']}")
    lines.append("")
    lines.append(f"**est_cost_usd:** `${payload['cost_usd']:.6f}`")
    win = payload["window"]
    reg = win.get("regions") or {}
    lines.append(
        f"**Window:** `{win['from']}` -> `{win['to']}` | "
        f"**Sessions:** {win['sessions']} (IN: {reg.get('IN', 0)}, US: {reg.get('US', 0)})"
    )
    suite = payload["suite"]
    drift = suite.get("offline_eval_drift") or []
    lines.append(
        f"**Suite:** pytest {suite.get('pytest_passed', 0)} passed / {suite.get('pytest_failed', 0)} failed | "
        f"offline eval drift lines: {len(drift)}"
    )
    lines.append("")

    # Metrics table
    lines.append("## Metrics (Window vs Prior 7d)")
    lines.append("")
    lines.append("| Metric | Overall | IN | US |")
    lines.append("|---|---|---|---|")
    metrics = payload.get("metrics") or {}
    for key, label in _METRIC_LABELS:
        m = metrics.get(key) or {}
        m_regs = m.get("regions") or {}
        is_stage = key == "abandon_point"
        lines.append(
            f"| {label} | {_fmt_metric_cell(m, is_stage)} | "
            f"{_fmt_metric_cell(m_regs.get('IN') or {}, is_stage)} | "
            f"{_fmt_metric_cell(m_regs.get('US') or {}, is_stage)} |"
        )
    lines.append("")

    # Failures section
    failures = payload.get("failures") or []
    lines.append(f"## Candidate Failures ({len(failures)})")
    lines.append("")
    if not failures:
        lines.append("No candidate failures flagged for this window.")
        lines.append("")
    else:
        for f in failures:
            lines.append(
                f"### #{f['rank']} `{f['signal']}` — 7d freq: {f['frequency_7d']} | "
                f"attempt `{f['attempt_hash']}` ({f['case_type']} / {f['stage']})"
            )
            lines.append(
                f"- **Response function:** `{f['response_function']}` | "
                f"**Abandoned <=2 turns:** `{f['abandoned_within_2_turns']}`"
            )
            lines.append(f"- **Candidate turn:** {f['candidate_turn']}")
            lines.append(f"- **Interviewer reply:** {f['interviewer_reply']}")
            lines.append(f"- **Should have:** {f['should_have']}")
            lines.append("")

    if drift:
        lines.append("## Offline Eval Drift")
        lines.append("")
        lines.append("```diff")
        lines.extend(drift[:120])
        if len(drift) > 120:
            lines.append(f"... ({len(drift) - 120} more diff lines truncated)")
        lines.append("```")
        lines.append("")

    notes = payload.get("notes") or []
    lines.append("## Notes")
    lines.append("")
    if not notes:
        lines.append("- None.")
    else:
        for n in notes:
            lines.append(f"- {n}")
    lines.append("")
    return scrub_secrets("\n".join(lines))


def write_reports(reports_dir: Path, payload: Dict[str, Any]) -> tuple[Path, Path]:
    """Write both bot/reports/<date>.json and bot/reports/<date>.md atomically."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = payload["date"]
    json_path = reports_dir / f"{date_str}.json"
    md_path = reports_dir / f"{date_str}.md"

    json_text = scrub_secrets(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    md_text = render_markdown_report(payload)

    json_tmp = reports_dir / f"{date_str}.json.tmp"
    md_tmp = reports_dir / f"{date_str}.md.tmp"
    json_tmp.write_text(json_text, encoding="utf-8")
    md_tmp.write_text(md_text, encoding="utf-8")
    json_tmp.replace(json_path)
    md_tmp.replace(md_path)
    return md_path, json_path
