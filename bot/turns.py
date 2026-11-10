"""Shared deterministic turn-level replay over attempt transcripts."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from services.case_models import stages_for, stage_of
from services.session_signals import (
    _PRESENCE_TEXT_MATCHES,
    _help_hit,
    _norm,
    _similar,
    _structured,
    compute_signals,
)

from bot.data import WindowDataset
from bot.state import parse_iso_utc

ABANDON_IDLE_MINUTES = 30

_GUIDANCE_RE = re.compile(
    r"(?i)\b(?:start|begin|anchor\w*|try|think\s+about|consider|break\w*|split|population|"
    r"top[- ]?down|bottom[- ]?up|first\s+step|look\s+at|focus\s+on|segment\w*|"
    r"revenue|cost|price|volume|demand|supply|fixed|variable|margin|per\b|divide|multiply|"
    r"calculate|estimate|household|denominator|numerator|sanity|compare|check)\b"
)
_GENERIC_UNACTIONABLE_RE = re.compile(
    r"(?i)^(?:what\s+do\s+you\s+think|how\s+would\s+you\s+(?:start|begin|approach\s+it|solve\s+it)|"
    r"where\s+would\s+you\s+start|keep\s+going|go\s+on|tell\s+me\s+more|what\s+else)\s*[?.!]*$"
)
_PRESENCE_LINE_NORMS = {re.sub(r"[^\w\s]", "", p.lower()).strip() for p in _PRESENCE_TEXT_MATCHES}


def _fast_line_norms() -> set[str]:
    from services.interviewer_decision import _FAST_PHRASES

    return _PRESENCE_LINE_NORMS | {
        re.sub(r"[^\w\s]", "", opt["text"].lower()).strip()
        for opts in _FAST_PHRASES.values()
        for opt in opts
    }

# Canonical phase index 0..5 matching services/case_models._PHASE_INDEX
_STAGE_TO_PHASE = {
    "clarify": 0,
    "classify": 1,
    "define success": 1,
    "hypothesise": 2,
    "decompose": 2,
    "structure": 2,
    "assume": 3,
    "prioritise": 3,
    "calculate": 4,
    "analyse": 4,
    "validate": 5,
    "diagnose": 5,
    "solve": 5,
    "evaluate": 5,
    "conclude": 6,
    "recommend": 6,
}

CANONICAL_STAGES = (
    "Clarify",
    "Classify",
    "Decompose",
    "Assume",
    "Calculate",
    "Validate",
    "Conclude",
)


def canonical_stage_name(phase_idx: int) -> str:
    """Return canonical 7-stage label (Clarify..Conclude) for phase index 0..6."""
    return CANONICAL_STAGES[max(0, min(len(CANONICAL_STAGES) - 1, int(phase_idx)))]


def infer_attempt_phase(
    attempt: Dict[str, Any],
    messages: List[Dict[str, Any]],
    case_type: str,
    policy: str,
    *,
    submitted_reaches_conclude: bool = False,
) -> int:
    """Deterministically infer phase index 0..6 without importing interviewer LLM clients."""
    from bot.data import normalize_session_state

    status = str(attempt.get("status") or "").strip().lower()
    if submitted_reaches_conclude and (
        status == "submitted"
        or bool(attempt.get("submitted_at"))
        or bool((attempt.get("final_recommendation") or "").strip())
    ):
        return 6

    ss = normalize_session_state(attempt.get("session_state"))
    if ss.get("structure"):
        try:
            return stage_phase_index(stage_of(ss, case_type))
        except Exception:  # noqa: BLE001
            return 0

    transcript_rows: List[Dict[str, str]] = []
    cur_phase = 0
    for msg in messages:
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user" and (msg.get("kind") or "text") != "recommendation":
            sig = compute_signals(transcript_rows, content, teaching_policy=policy)
            cur_phase = max(cur_phase, _inferred_turn_phase(sig, content))
        transcript_rows.append({"role": role or "system", "kind": msg.get("kind") or "text", "content": content})
    return cur_phase


def resolve_policy(attempt: Dict[str, Any], case: Optional[Dict[str, Any]]) -> str:
    """Resolve teaching_policy matching routes/attempts._resolve_policy."""
    c = case or {}
    for candidate in (attempt.get("teaching_policy"), c.get("teaching_policy")):
        p = (candidate or "").strip().lower()
        if p in ("exam", "coached"):
            return p
    return "coached"


def stage_phase_index(stage_name: str) -> int:
    """Map a guesstimate or case stage name to canonical phase index 0..6."""
    return _STAGE_TO_PHASE.get((stage_name or "").strip().lower(), 0)


def stage_name_for_phase(case_type: str, phase_idx: int) -> str:
    """Map canonical phase index 0..5 to the case_type's own stage label."""
    stages = stages_for(case_type)
    is_guess = (case_type or "").strip().lower() == "guesstimate"
    if is_guess:
        idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    else:
        idx_map = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 9}
    target = idx_map.get(max(0, min(6, phase_idx)), 0)
    return stages[min(target, len(stages) - 1)]


def is_attempt_abandoned(
    attempt: Dict[str, Any],
    messages: List[Dict[str, Any]],
    reference_now: datetime,
) -> bool:
    """True when the attempt did not submit and is marked abandoned or idle."""
    status = str(attempt.get("status") or "active").strip().lower()
    if status == "submitted":
        return False
    if status == "abandoned":
        return True
    timestamps = [m.get("created_at") for m in messages if m.get("created_at")]
    if attempt.get("created_at"):
        timestamps.append(attempt["created_at"])
    if not timestamps:
        return True
    try:
        latest = max(parse_iso_utc(str(ts)) for ts in timestamps)
        ref = reference_now if reference_now.tzinfo else reference_now.replace(tzinfo=timezone.utc)
        return (ref - latest) >= timedelta(minutes=ABANDON_IDLE_MINUTES)
    except Exception:  # noqa: BLE001
        return True


def _is_unactionable_reply(reply_text: str, policy: str) -> bool:
    from services.interviewer_decision import is_praise_only, looks_like_refusal

    t = (reply_text or "").strip()
    if not t:
        return True
    stripped_norm = re.sub(r"[^\w\s]", "", t.lower()).strip()
    if stripped_norm in _fast_line_norms():
        return True
    if is_praise_only(t, policy):
        return True
    if looks_like_refusal(t):
        return True
    if _GENERIC_UNACTIONABLE_RE.match(t):
        return True
    if "→" in t or "->" in t or re.search(r"\d", t):
        return False
    if _GUIDANCE_RE.search(t) and len(t.split()) >= 5:
        return False
    return len(t.split()) <= 10


def _question_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [_norm(p) for p in parts if "?" in p and len(_norm(p)) >= 6]


def _reply_repeats_prior_question(
    reply_text: str,
    prior_assistant_texts: List[str],
) -> bool:
    if not reply_text or not prior_assistant_texts:
        return False
    norm_reply = _norm(reply_text)
    if norm_reply in [_norm(a) for a in prior_assistant_texts]:
        return True
    cur_qs = _question_sentences(reply_text)
    if not cur_qs:
        return False
    prior_qs = [q for a in prior_assistant_texts for q in _question_sentences(a)]
    for cq in cur_qs:
        for pq in prior_qs:
            if _similar(cq, pq) or difflib.SequenceMatcher(None, cq, pq).ratio() >= 0.85:
                return True
    return False


def _inferred_turn_phase(signals: Dict[str, Any], cand_text: str) -> int:
    if signals.get("final_declared") or signals.get("states_final_estimate"):
        return 5
    tt = signals.get("turn_type") or ""
    norm = _norm(cand_text)
    if tt in ("calculation_result", "calculation"):
        return 4
    if tt in ("estimate", "strategic_insight", "hypothesis"):
        return 3
    if tt in ("structure", "plan") or _structured(norm):
        return 2
    if signals.get("has_work"):
        return 2
    return 0


@dataclass
class TurnEval:
    attempt_id: str
    user_id: str
    region: str
    case_type: str
    policy: str
    stage: str
    stage_phase: int
    turn_index: int
    candidate_turn: str
    interviewer_reply: str
    signals: Dict[str, Any]
    response_function: str
    abandoned_within_2_turns: bool
    flags: List[str] = field(default_factory=list)


@dataclass
class AttemptEval:
    attempt: Dict[str, Any]
    region: str
    case_type: str
    policy: str
    user_turn_count: int
    assistant_reply_count: int
    praise_only_count: int
    repeat_signal_count: int
    is_abandoned: bool
    final_stage: str
    final_stage_phase: int
    turns: List[TurnEval] = field(default_factory=list)


def analyze_attempt(
    attempt: Dict[str, Any],
    dataset: WindowDataset,
    *,
    reference_now: datetime,
) -> AttemptEval:
    """Deterministically replay one attempt and evaluate metrics/failure signals."""
    from services.interviewer_decision import decide_response, is_praise_only

    from bot.data import normalize_session_state

    aid = str(attempt.get("id") or "")
    uid = str(attempt.get("user_id") or "")
    case = dataset.cases_by_id.get(str(attempt.get("case_id") or "")) or {}
    case_type = str(case.get("type") or "case")
    policy = resolve_policy(attempt, case)
    region = dataset.attempt_region(attempt)
    messages = list(dataset.messages_by_attempt.get(aid) or [])

    session_state = normalize_session_state(attempt.get("session_state"))
    try:
        final_db_stage = stage_of(session_state, case_type)
    except Exception:  # noqa: BLE001
        final_db_stage = "Clarify"
    final_db_phase = stage_phase_index(final_db_stage)
    is_abandoned = is_attempt_abandoned(attempt, messages, reference_now)

    # Identify candidate turns (non-empty user messages excluding final recommendation)
    user_indices = [
        idx
        for idx, m in enumerate(messages)
        if (m.get("role") or "") == "user"
        and (m.get("content") or "").strip()
        and (m.get("kind") or "text") != "recommendation"
    ]
    total_user_turns = len(user_indices)

    sim_state: Dict[str, Any] = {}
    transcript_rows: List[Dict[str, str]] = []
    prior_assistant_raw: List[str] = []

    raw_turns: List[Dict[str, Any]] = []
    assistant_reply_count = 0
    praise_only_count = 0
    repeat_signal_count = 0

    for msg_idx, msg in enumerate(messages):
        role = (msg.get("role") or "").strip()
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            assistant_reply_count += 1
            if is_praise_only(content, policy):
                praise_only_count += 1
            prior_assistant_raw.append(content)
            transcript_rows.append({"role": "assistant", "kind": msg.get("kind") or "text", "content": content})
            continue
        if role != "user" or (msg.get("kind") or "text") == "recommendation":
            transcript_rows.append({"role": role or "system", "kind": msg.get("kind") or "text", "content": content})
            continue

        # Pair with the next assistant reply before the next user message
        reply_text = ""
        for j in range(msg_idx + 1, len(messages)):
            nxt = messages[j]
            nrole = (nxt.get("role") or "").strip()
            ncontent = (nxt.get("content") or "").strip()
            if nrole == "user":
                break
            if nrole == "assistant" and ncontent:
                reply_text = ncontent
                break

        sig = compute_signals(
            transcript_rows,
            content,
            teaching_policy=policy,
            prior_state=sim_state,
        )
        decision = decide_response(sig)
        fn = decision["function"]

        if sig.get("interviewer_repeating"):
            repeat_signal_count += 1

        u_pos = user_indices.index(msg_idx)
        remaining_after = (total_user_turns - 1) - u_pos
        abandoned_2 = bool(is_abandoned and reply_text and remaining_after <= 2)

        raw_turns.append({
            "u_pos": u_pos,
            "cand_text": content,
            "reply_text": reply_text,
            "sig": sig,
            "fn": fn,
            "mode": decision.get("mode"),
            "abandoned_2": abandoned_2,
            "remaining_after": remaining_after,
            "prior_asst": list(prior_assistant_raw),
        })

        # Update simulation memory for next turn's compute_signals
        recent_fns = list(sim_state.get("recent_functions") or [])
        sim_state["recent_functions"] = (recent_fns + [fn])[-4:]
        sim_state["last_function"] = fn
        sim_state["function_turn"] = sig.get("candidate_turn_count", 0)

        transcript_rows.append({"role": "user", "kind": msg.get("kind") or "text", "content": content})

    # Second pass: compute stage tracking and all 6 failure triggers
    has_persisted_tree = bool(session_state.get("structure"))
    cur_phase = 0
    turns_out: List[TurnEval] = []
    any_primary_flag = False
    last_replied_idx = max((i for i, t in enumerate(raw_turns) if t["reply_text"]), default=-1)

    for idx, t in enumerate(raw_turns):
        sig = t["sig"]
        cand_text = t["cand_text"]
        reply_text = t["reply_text"]
        fn = t["fn"]

        turn_phase = _inferred_turn_phase(sig, cand_text)
        stage_before = cur_phase
        cur_phase = max(cur_phase, turn_phase)
        stage_label = stage_name_for_phase(case_type, cur_phase)

        flags: List[str] = []

        # 1. is_praise_only() would have fired on the reply that was sent
        if reply_text and is_praise_only(reply_text, policy):
            flags.append("praise_only")

        # 2. Interviewer repeated a question it had already asked
        if reply_text and _reply_repeats_prior_question(reply_text, t["prior_asst"]):
            flags.append("repeated_question")

        # 3. Candidate hit a _HELP pattern and reply contained nothing actionable
        if (_help_hit(_norm(cand_text)) or sig.get("help_requested")) and reply_text:
            if _is_unactionable_reply(reply_text, policy):
                flags.append("help_not_actionable")

        # 4. Reply stacked two questions, or asked one where policy said HAND_BACK
        if reply_text:
            if reply_text.count("?") >= 2:
                flags.append("stacked_questions")
            elif (fn == "HAND_BACK" or t.get("mode") == "HAND_BACK") and "?" in reply_text:
                flags.append("question_on_hand_back")

        # 5. Candidate gave substantive reasoning and the stage did not advance
        if sig.get("is_substantive_reasoning") and reply_text:
            stalled_next = False
            if idx + 1 < len(raw_turns):
                nxt_sig = raw_turns[idx + 1]["sig"]
                nxt_phase = _inferred_turn_phase(nxt_sig, raw_turns[idx + 1]["cand_text"])
                if nxt_phase <= stage_before and (
                    nxt_sig.get("intent") in ("asking_for_help", "wants_to_stop")
                    or nxt_sig.get("frustration") != "none"
                    or nxt_sig.get("candidate_repeating")
                    or nxt_sig.get("turns_without_progress", 0) >= 1
                ):
                    stalled_next = True
            persisted_stuck = has_persisted_tree and final_db_phase <= stage_before and final_db_phase < 5
            unsubmitted_no_tree = (
                attempt.get("status") != "submitted"
                and not has_persisted_tree
                and idx == last_replied_idx
                and turn_phase >= 2
            )
            if stalled_next or persisted_stuck or unsubmitted_no_tree:
                flags.append("substantive_no_stage_advance")

        if flags:
            any_primary_flag = True

        # 6. Attempt was abandoned within 2 turns of that reply
        if t["abandoned_2"] and not flags and (idx == last_replied_idx) and not any_primary_flag:
            flags.append("abandoned_within_2_turns")

        turns_out.append(
            TurnEval(
                attempt_id=aid,
                user_id=uid,
                region=region,
                case_type=case_type,
                policy=policy,
                stage=stage_label,
                stage_phase=cur_phase,
                turn_index=t["u_pos"],
                candidate_turn=cand_text,
                interviewer_reply=reply_text,
                signals=sig,
                response_function=fn,
                abandoned_within_2_turns=t["abandoned_2"],
                flags=flags,
            )
        )

    # Effective stage for unfinished/finished summary
    effective_phase = final_db_phase if has_persisted_tree else cur_phase
    effective_stage = final_db_stage if has_persisted_tree else stage_name_for_phase(case_type, effective_phase)

    return AttemptEval(
        attempt=attempt,
        region=region,
        case_type=case_type,
        policy=policy,
        user_turn_count=total_user_turns,
        assistant_reply_count=assistant_reply_count,
        praise_only_count=praise_only_count,
        repeat_signal_count=repeat_signal_count,
        is_abandoned=is_abandoned,
        final_stage=effective_stage,
        final_stage_phase=effective_phase,
        turns=turns_out,
    )
