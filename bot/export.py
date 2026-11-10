"""Read-only raw session exporter (bot/exports/sessions_<date>.md, zero Gemini calls)."""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from bot.config import (
    EXIT_ABORT,
    EXIT_OK,
    SERVICE_ROLE_WARN,
    UUID_RE,
    default_repo_root,
    get_bot_hash_salt,
    get_supabase_credentials,
    scrub_secrets,
    scrub_uuids,
)
from bot.data import (
    SupabaseUnavailableError,
    WindowDataset,
    build_supabase_client,
    fetch_all,
    fetch_in_chunks,
    normalize_session_state,
    probe_supabase,
    valid_submission_row,
    valid_ts_row,
)
from bot.git_guard import capture_outside_bot_baseline, compare_outside_bot_snapshot
from bot.metrics import coerce_float
from bot.redact import RedactionLeakError, assert_no_identifiers, build_college_terms, redact_text
from bot.state import hash_identifier, ist_date_str, parse_iso_utc
from bot.turns import (
    CANONICAL_STAGES,
    canonical_stage_name,
    infer_attempt_phase,
    is_attempt_abandoned,
    resolve_policy,
)

DEFAULT_EXPORT_DAYS = 14


@dataclass
class ExportData:
    """Dataset fetched strictly for a pure read-only session export window."""

    dataset: WindowDataset
    signed_up_user_ids: Set[str] = field(default_factory=set)
    submissions_by_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    submissions_by_attempt_id: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ExportAttemptRecord:
    """Processed and redacted attempt ready for Markdown rendering."""

    attempt_hash: str
    created_dt: datetime
    created_iso: str
    attempt_id_raw: str
    user_id_raw: str
    case_type: str
    region: str
    policy: str
    status: str
    phase_idx: int
    stage: str
    turns: int
    duration_mmss: str
    score_display: str
    transcript_lines: List[str]
    has_messages: bool
    last_assistant_message: str


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {message}\n")


def compute_export_window(
    now_utc: datetime,
    *,
    days: Optional[int] = None,
    since: Optional[str] = None,
) -> Tuple[datetime, datetime]:
    """Resolve [window_from, window_to] in UTC for --export-sessions."""
    w_to = now_utc if now_utc.tzinfo else now_utc.replace(tzinfo=timezone.utc)
    w_to = w_to.astimezone(timezone.utc)
    if since:
        w_from = parse_iso_utc(since)
    else:
        lookback_days = int(days) if days is not None else DEFAULT_EXPORT_DAYS
        w_from = w_to - timedelta(days=lookback_days)
    return w_from, w_to


def _parse_msg_dt(raw: Any, fallback: datetime) -> datetime:
    if not raw:
        return fallback
    try:
        return parse_iso_utc(str(raw))
    except Exception:  # noqa: BLE001
        return fallback


def fetch_export_data(
    supabase: Any,
    *,
    window_from: datetime,
    window_to: datetime,
) -> ExportData:
    """Fetch users, attempts, attempt_messages, submissions, cases, and colleges."""
    probe_supabase(supabase)
    ds = WindowDataset()
    w_from_iso = window_from.isoformat()
    w_to_iso = window_to.isoformat()

    signed_up_user_ids: Set[str] = set()
    users_by_id: Dict[str, Dict[str, Any]] = {}
    submissions_by_id: Dict[str, Dict[str, Any]] = {}
    submissions_by_attempt_id: Dict[str, Dict[str, Any]] = {}

    try:
        # 1. Users who signed up inside [window_from, window_to]
        raw_signups = fetch_all(
            lambda: (
                supabase.table("users")
                .select("*")
                .gte("created_at", w_from_iso)
                .lte("created_at", w_to_iso)
            )
        )
        for u in raw_signups:
            if not isinstance(u, dict) or not u.get("id"):
                ds.note_skip("users", "non-dict row or missing id")
                continue
            ts = u.get("created_at")
            if not ts:
                ds.note_skip("users", "missing created_at")
                continue
            try:
                dt = parse_iso_utc(str(ts))
            except Exception:  # noqa: BLE001
                ds.note_skip("users", "unparseable created_at")
                continue
            if window_from <= dt <= window_to:
                uid = str(u["id"])
                signed_up_user_ids.add(uid)
                users_by_id[uid] = u

        # 2. Attempts started inside [window_from, window_to]
        raw_attempts = fetch_all(
            lambda: (
                supabase.table("attempts")
                .select("*")
                .gte("created_at", w_from_iso)
                .lte("created_at", w_to_iso)
                .order("created_at", desc=True)
            )
        )
        attempts: List[Dict[str, Any]] = []
        attempt_ids: Set[str] = set()
        sub_ids_from_attempts: Set[str] = set()
        user_ids: Set[str] = set()
        case_ids: Set[str] = set()

        for a in raw_attempts:
            if not valid_ts_row(a, "attempts", ds):
                continue
            dt = parse_iso_utc(str(a["created_at"]))
            if not (window_from <= dt <= window_to):
                continue
            aid = str(a["id"])
            if aid in attempt_ids:
                continue
            clean_a = dict(a)
            clean_a["session_state"] = normalize_session_state(a.get("session_state"))
            attempt_ids.add(aid)
            attempts.append(clean_a)
            if clean_a.get("user_id"):
                user_ids.add(str(clean_a["user_id"]))
            if clean_a.get("case_id"):
                case_ids.add(str(clean_a["case_id"]))
            if clean_a.get("submission_id"):
                sub_ids_from_attempts.add(str(clean_a["submission_id"]))

        # 3. Messages for window attempts
        all_messages = fetch_in_chunks(
            supabase,
            "attempt_messages",
            "attempt_id",
            attempt_ids,
            select_cols="id, attempt_id, role, kind, content, is_clarification, created_at",
            order_col="created_at",
        )
        messages_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
        for m in all_messages:
            if not isinstance(m, dict) or not m.get("attempt_id"):
                ds.note_skip("attempt_messages", "missing attempt_id or non-dict")
                continue
            if m.get("created_at"):
                try:
                    parse_iso_utc(str(m["created_at"]))
                except Exception:  # noqa: BLE001
                    ds.note_skip("attempt_messages", "unparseable created_at")
                    continue
            aid = str(m["attempt_id"])
            messages_by_attempt.setdefault(aid, []).append(m)

        # 4. Submissions linked via submission_id or attempt_id
        raw_subs_by_id = fetch_in_chunks(supabase, "submissions", "id", sub_ids_from_attempts, select_cols="*")
        raw_subs_by_att = fetch_in_chunks(supabase, "submissions", "attempt_id", attempt_ids, select_cols="*")
        seen_sub_ids: Set[str] = set()
        for s in raw_subs_by_id + raw_subs_by_att:
            if isinstance(s, dict) and s.get("id") and str(s["id"]) in seen_sub_ids:
                continue
            if isinstance(s, dict) and s.get("id"):
                seen_sub_ids.add(str(s["id"]))
            if not valid_submission_row(s, ds):
                continue
            sid = str(s["id"])
            submissions_by_id[sid] = s
            if s.get("attempt_id"):
                submissions_by_attempt_id[str(s["attempt_id"])] = s

        # 5. Referenced users + cases
        missing_user_ids = {uid for uid in user_ids if uid not in users_by_id}
        if missing_user_ids:
            extra_users = fetch_in_chunks(supabase, "users", "id", missing_user_ids, select_cols="*")
            for u in extra_users:
                if not isinstance(u, dict) or not u.get("id"):
                    ds.note_skip("users", "non-dict row or missing id")
                    continue
                users_by_id[str(u["id"])] = u

        cases_rows = fetch_in_chunks(supabase, "cases", "id", case_ids, select_cols="*")
        cases_by_id: Dict[str, Dict[str, Any]] = {}
        for c in cases_rows:
            if not isinstance(c, dict) or not c.get("id"):
                ds.note_skip("cases", "non-dict row or missing id")
                continue
            cases_by_id[str(c["id"])] = c
    except SupabaseUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SupabaseUnavailableError(f"Supabase export query failed: {type(exc).__name__}: {exc}") from exc

    # Best-effort fetch of colleges purely to augment PII redaction terms.
    colleges: List[Dict[str, Any]] = []
    try:
        colleges = fetch_all(lambda: supabase.table("colleges").select("id, name, short_name, slug"))
    except Exception:  # noqa: BLE001
        colleges = []

    ds.attempts = attempts
    ds.messages_by_attempt = messages_by_attempt
    ds.submissions = list(submissions_by_id.values())
    ds.users_by_id = users_by_id
    ds.cases_by_id = cases_by_id
    ds.colleges = colleges

    return ExportData(
        dataset=ds,
        signed_up_user_ids=signed_up_user_ids,
        submissions_by_id=submissions_by_id,
        submissions_by_attempt_id=submissions_by_attempt_id,
    )


def _format_duration_mmss(seconds: int) -> str:
    secs = max(0, int(seconds))
    return f"{secs // 60:02d}:{secs % 60:02d}"


def _format_score(sub: Optional[Dict[str, Any]]) -> str:
    if not sub:
        return "-"
    val = coerce_float(sub.get("score"))
    if val is None:
        return "-"
    if val.is_integer():
        return str(int(val))
    return f"{val:.1f}"


def _build_redaction_context(export_data: ExportData) -> Dict[str, Any]:
    ds = export_data.dataset
    raw_ids: List[str] = []
    for uid in ds.users_by_id:
        raw_ids.append(uid)
    for a in ds.attempts:
        if a.get("id"):
            raw_ids.append(str(a["id"]))
        if a.get("submission_id"):
            raw_ids.append(str(a["submission_id"]))
    for cid in ds.cases_by_id:
        raw_ids.append(cid)
    for sid in export_data.submissions_by_id:
        raw_ids.append(sid)

    raw_emails = [
        str(u.get(k))
        for u in ds.users_by_id.values()
        for k in ("email", "college_email")
        if u.get(k)
    ]
    raw_phones = [str(u.get("phone")) for u in ds.users_by_id.values() if u.get("phone")]
    raw_names = [
        str(u.get(k))
        for u in ds.users_by_id.values()
        for k in ("full_name", "name")
        if u.get(k)
    ]
    raw_colleges = [
        str(u.get("college_other")).strip()
        for u in ds.users_by_id.values()
        if u.get("college_other")
        and str(u.get("college_other")).strip().lower() not in {"other", "none", "na", "n/a", "-"}
    ]
    college_terms = build_college_terms(ds.colleges, raw_colleges)

    return {
        "raw_ids": raw_ids,
        "raw_emails": raw_emails,
        "raw_phones": raw_phones,
        "raw_names": raw_names,
        "raw_colleges": raw_colleges,
        "college_terms": college_terms,
    }


def _clean_text(text: str, user_info: Optional[Dict[str, Any]], ctx: Dict[str, Any]) -> str:
    red = redact_text(
        text,
        user_info=user_info,
        known_ids=ctx["raw_ids"],
        known_emails=ctx["raw_emails"],
        known_phones=ctx["raw_phones"],
        college_terms=ctx["college_terms"],
        extra_names=ctx["raw_names"],
    )
    return scrub_uuids(scrub_secrets(red))


def build_session_records(
    export_data: ExportData,
    *,
    bot_hash_salt: str,
    reference_now: datetime,
    redaction_ctx: Dict[str, Any],
) -> List[ExportAttemptRecord]:
    """Convert raw rows into redacted ExportAttemptRecord list sorted newest first."""
    ds = export_data.dataset
    records: List[ExportAttemptRecord] = []

    for att in ds.attempts:
        try:
            aid = str(att["id"])
            uid = str(att.get("user_id") or "")
            created_dt = parse_iso_utc(str(att["created_at"]))
            case = ds.cases_by_id.get(str(att.get("case_id") or "")) or {}
            user_info = ds.users_by_id.get(uid) or ({"id": uid} if uid else {})

            case_type = _clean_text(str(case.get("type") or "case").strip() or "case", user_info, redaction_ctx)
            region = ds.attempt_region(att)
            policy = resolve_policy(att, case)

            raw_msgs = list(ds.messages_by_attempt.get(aid) or [])
            msgs_with_dt: List[Tuple[datetime, str, Dict[str, Any]]] = []
            for m in raw_msgs:
                mdt = _parse_msg_dt(m.get("created_at"), created_dt)
                msgs_with_dt.append((mdt, str(m.get("id") or ""), m))
            msgs_with_dt.sort(key=lambda item: (item[0], item[1]))
            sorted_msgs = [item[2] for item in msgs_with_dt]

            raw_status = str(att.get("status") or "active").strip().lower()
            if raw_status == "submitted" or att.get("submitted_at"):
                effective_status = "submitted"
            elif is_attempt_abandoned(att, sorted_msgs, reference_now):
                effective_status = "abandoned"
            else:
                effective_status = raw_status or "active"

            phase_idx = infer_attempt_phase(
                att,
                sorted_msgs,
                case_type,
                policy,
                submitted_reaches_conclude=True,
            )
            stage_label = canonical_stage_name(phase_idx)

            non_empty_msgs = [
                (mdt, m)
                for mdt, _mid, m in msgs_with_dt
                if str(m.get("content") or "").strip()
            ]
            candidate_turns = sum(
                1 for _mdt, m in non_empty_msgs if str(m.get("role") or "").strip() == "user"
            )

            t0 = non_empty_msgs[0][0] if non_empty_msgs else created_dt
            end_candidates = [t0] + [mdt for mdt, _m in non_empty_msgs]
            if att.get("submitted_at"):
                try:
                    end_candidates.append(parse_iso_utc(str(att["submitted_at"])))
                except Exception:  # noqa: BLE001
                    pass
            duration_secs = int((max(end_candidates) - t0).total_seconds())

            sub_id = str(att.get("submission_id") or "")
            sub = export_data.submissions_by_id.get(sub_id) or export_data.submissions_by_attempt_id.get(aid)
            score_display = _format_score(sub)

            transcript_lines: List[str] = []
            last_asst = ""
            last_system = ""
            for mdt, m in non_empty_msgs:
                role = str(m.get("role") or "unknown").strip() or "unknown"
                offset = max(0, int((mdt - t0).total_seconds()))
                cleaned = _clean_text(str(m.get("content") or "").strip(), user_info, redaction_ctx)
                transcript_lines.append(f"[{_format_duration_mmss(offset)}] {role}: {cleaned}")
                if role == "assistant":
                    last_asst = cleaned
                elif role == "system":
                    last_system = cleaned

            last_interviewer_facing = last_asst or (f"[opening system prompt] {last_system}" if last_system else "")

            records.append(
                ExportAttemptRecord(
                    attempt_hash=hash_identifier(aid, bot_hash_salt),
                    created_dt=created_dt,
                    created_iso=created_dt.isoformat(),
                    attempt_id_raw=aid,
                    user_id_raw=uid,
                    case_type=case_type,
                    region=region,
                    policy=policy,
                    status=effective_status,
                    phase_idx=phase_idx,
                    stage=stage_label,
                    turns=candidate_turns,
                    duration_mmss=_format_duration_mmss(duration_secs),
                    score_display=score_display,
                    transcript_lines=transcript_lines,
                    has_messages=bool(non_empty_msgs),
                    last_assistant_message=last_interviewer_facing,
                )
            )
        except Exception as exc:  # noqa: BLE001
            ds.note_skip("attempts_export", f"{type(exc).__name__}: {exc}")
            continue

    records.sort(key=lambda r: (r.created_dt, r.attempt_hash), reverse=True)
    return records


def render_sessions_markdown(
    export_data: ExportData,
    records: List[ExportAttemptRecord],
    *,
    export_date: str,
    window_from: datetime,
    window_to: datetime,
    used_service_role: bool = False,
) -> str:
    """Render raw three-section session export Markdown."""
    ds = export_data.dataset
    lines: List[str] = []

    if used_service_role:
        lines.extend(["> [!WARNING]", f"> {SERVICE_ROLE_WARN}", ""])

    skipped_total = ds.total_skipped()
    skipped_detail = ""
    if skipped_total > 0:
        reasons_txt = "; ".join(ds.skipped_reasons[:10])
        skipped_detail = f" ({dict(ds.skipped_counts)}: {reasons_txt})"

    lines.extend([
        f"# Session Export — {export_date}",
        "",
        f"Window: `{window_from.isoformat()}` -> `{window_to.isoformat()}`",
        f"Skipped unparseable rows: {skipped_total}{skipped_detail}",
        "",
    ])

    # Section 1 — Funnel counts
    user_turn_totals: Dict[str, int] = {}
    for rec in records:
        if rec.user_id_raw:
            user_turn_totals[rec.user_id_raw] = user_turn_totals.get(rec.user_id_raw, 0) + rec.turns
    zero_msg_users = sum(1 for total in user_turn_totals.values() if total == 0)

    attempts_started = len(records)
    attempts_with_cand_msg = sum(1 for r in records if r.turns >= 1)
    attempts_submitted = sum(1 for r in records if r.status == "submitted")

    region_counts: Counter[str] = Counter(r.region for r in records)
    policy_counts: Counter[str] = Counter(r.policy for r in records)

    lines.extend([
        "## Section 1 — Funnel Counts",
        "",
        f"- users who signed up: {len(export_data.signed_up_user_ids)}",
        f"- users who opened a case but sent zero messages: {zero_msg_users}",
        f"- attempts started: {attempts_started}",
        f"- attempts with >=1 candidate message: {attempts_with_cand_msg}",
        "- attempts reaching each stage:",
    ])
    for idx, st_name in enumerate(CANONICAL_STAGES):
        cnt = sum(1 for r in records if r.phase_idx >= idx)
        lines.append(f"  - {st_name}: {cnt}")
    lines.append(f"- attempts submitted: {attempts_submitted}")
    lines.append(
        f"- attempts by region: IN={region_counts.get('IN', 0)}, US={region_counts.get('US', 0)}"
    )
    policy_keys = ["coached", "exam"] + sorted(k for k in policy_counts if k not in ("coached", "exam"))
    policy_parts = [f"{k}={policy_counts.get(k, 0)}" for k in policy_keys]
    lines.append(f"- attempts by teaching_policy: {', '.join(policy_parts)}")
    lines.append("")

    # Section 2 — Every attempt in the window, newest first
    lines.extend(["## Section 2 — Attempts (Newest First)", ""])
    if not records:
        lines.extend(["(no attempts in window)", ""])
    for rec in records:
        lines.extend([
            f"### attempt {rec.attempt_hash} · {rec.case_type} · {rec.region} · {rec.policy} · {rec.status}",
            f"stage reached: {rec.stage}   turns: {rec.turns}   duration: {rec.duration_mmss}   score: {rec.score_display}",
            "--- transcript ---",
        ])
        if not rec.transcript_lines:
            lines.append("(empty transcript — 0 messages sent)")
        else:
            lines.extend(rec.transcript_lines)
            if rec.turns == 0:
                lines.append("(0 candidate messages sent)")
        lines.extend(["--- end ---", ""])

    # Section 3 — Abandoned attempts only: last assistant message grouped by stage
    lines.extend(["## Section 3 — Abandoned Attempts: Last Assistant Message by Stage", ""])
    abandoned = [r for r in records if r.status == "abandoned"]
    if not abandoned:
        lines.extend(["(no abandoned attempts in window)", ""])
    else:
        for idx, st_name in enumerate(CANONICAL_STAGES):
            group = [r for r in abandoned if r.phase_idx == idx]
            if not group:
                continue
            group.sort(key=lambda r: (r.last_assistant_message or "", r.attempt_hash))
            lines.extend([f"### {st_name} ({len(group)})", ""])
            for rec in group:
                lines.append(f"- attempt {rec.attempt_hash} (turns: {rec.turns}):")
                lines.append(rec.last_assistant_message or "(no assistant message)")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_session_export(
    *,
    repo_root: Optional[Path] = None,
    now_utc: Optional[datetime] = None,
    days: Optional[int] = None,
    since: Optional[str] = None,
    require_bot_key: Optional[bool] = None,
    supabase_client: Any = None,
    before_write_hook: Any = None,
) -> int:
    """Execute read-only DB session export to bot/exports/sessions_<date>.md (0 Gemini calls)."""
    root = (repo_root or default_repo_root()).resolve()
    bot_dir = root / "bot"
    logs_dir = bot_dir / "logs"
    exports_dir = bot_dir / "exports"

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    export_date = ist_date_str(now)
    log_path = logs_dir / f"{export_date}.log"

    def log(msg: str) -> None:
        safe = scrub_uuids(scrub_secrets(msg))
        _append_log(log_path, safe)
        print(safe)

    bot_hash_salt = get_bot_hash_salt()
    if not bot_hash_salt:
        log("[ABORT] BOT_HASH_SALT is unset; aborting export without writing unhashed identifiers.")
        return EXIT_ABORT

    git_ok, git_baseline, git_err = capture_outside_bot_baseline(root)
    if not git_ok:
        log(f"[ABORT] {git_err}")
        return EXIT_ABORT

    strict_bot_key = (
        require_bot_key
        if require_bot_key is not None
        else os.getenv("BOT_REQUIRE_BOT_KEY", "").strip() == "1"
    )
    sb_url, sb_key, used_service_role = get_supabase_credentials()
    if strict_bot_key and used_service_role:
        log("[ABORT] SUPABASE_BOT_KEY is unset and --require-bot-key is active.")
        return EXIT_ABORT
    if used_service_role:
        log(SERVICE_ROLE_WARN)

    try:
        window_from, window_to = compute_export_window(now, days=days, since=since)
    except Exception as exc:  # noqa: BLE001
        log(f"[ABORT] Invalid export window parameter: {type(exc).__name__}: {exc}")
        return EXIT_ABORT

    if supabase_client is None:
        try:
            supabase_client = build_supabase_client(sb_url, sb_key)
        except Exception as exc:  # noqa: BLE001
            log(f"[ABORT] Supabase client initialization failed: {type(exc).__name__}: {exc}")
            return EXIT_ABORT

    try:
        export_data = fetch_export_data(
            supabase_client,
            window_from=window_from,
            window_to=window_to,
        )
    except SupabaseUnavailableError as exc:
        log(f"[ABORT] Supabase unavailable during export: {exc}")
        return EXIT_ABORT

    redaction_ctx = _build_redaction_context(export_data)
    records = build_session_records(
        export_data,
        bot_hash_salt=bot_hash_salt,
        reference_now=now,
        redaction_ctx=redaction_ctx,
    )
    markdown = render_sessions_markdown(
        export_data,
        records,
        export_date=export_date,
        window_from=window_from,
        window_to=window_to,
        used_service_role=bool(used_service_role and not os.getenv("SUPABASE_BOT_KEY", "").strip()),
    )

    try:
        assert_no_identifiers(
            [markdown],
            raw_ids=redaction_ctx["raw_ids"],
            raw_emails=redaction_ctx["raw_emails"],
            raw_names=redaction_ctx["raw_names"],
            raw_colleges=redaction_ctx["raw_colleges"],
        )
        if UUID_RE.search(markdown):
            raise RedactionLeakError("Raw UUID survived redaction in session export")
    except RedactionLeakError as exc:
        log(f"[ABORT] Privacy verification failed before writing export: {exc}")
        return EXIT_ABORT

    if before_write_hook is not None:
        before_write_hook()

    git_ok_end, changed_end, git_err_end = compare_outside_bot_snapshot(root, git_baseline)
    if not git_ok_end:
        if git_err_end:
            log(f"[ABORT] Pre-write git check failed: {git_err_end}")
        else:
            log(
                "[ABORT] Refusing to write export — path(s) outside bot/ changed during run: "
                + ", ".join(changed_end)
            )
        return EXIT_ABORT

    exports_dir.mkdir(parents=True, exist_ok=True)
    out_path = exports_dir / f"sessions_{export_date}.md"
    out_path.write_text(markdown, encoding="utf-8")
    log(
        f"[DONE] Wrote {out_path.relative_to(root)} "
        f"(attempts={len(records)}, skipped={export_data.dataset.total_skipped()})"
    )
    return EXIT_OK
