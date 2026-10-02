"""Live interview orchestration (docs/E_INTERVIEW_STATE.md §3).

One turn = one transaction under the session row lock:
  persist answer -> measure -> analyze (fast model) -> update memory/claims/contradictions
  -> decide (pure policy) -> speak (fast model, filtered, fallback) -> persist -> close the
  exchange if the policy moved on and enqueue its evidence extraction.
Nothing evaluative is ever returned to the client during the interview (spec §78).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from ..access import flags
from ..ai.guard import scan_injection
from ..ai.runner import RunContext, pending_cost
from ..db.models import (Claim, InterviewBlueprint, InterviewEvent, InterviewExchange, InterviewMessage,
                         InterviewSession, InterviewState, utcnow)
from ..errors import Conflict, NotFound, Unprocessable
from ..evidence_engine.metrics import answer_metrics
from ..interview_memory import claims as CL
from ..jobs.queue import enqueue
from ..textutil import truncate
from . import intents
from . import state as S
from .analyzer import analyze_turn
from .interviewer import opener_of, speak
from .lifecycle import find_owned, next_message_seq, transition
from .policy import Action, decide

MAX_ANSWER_CHARS = 12000
IDLE_CAP_S = 360  # thinking time beyond 6 minutes between messages is not counted as interview time
QUALITY_RANK = {"weak": 0, "adequate": 1, "strong": 2}


# ------------------------------------------------------------------------------------------
# loading / saving
# ------------------------------------------------------------------------------------------
def _load(db: Session, user_id: uuid.UUID, session_id: uuid.UUID) -> Tuple[InterviewSession, InterviewState, dict]:
    sess = find_owned(db, user_id, session_id, lock=True)
    if sess is None:
        raise NotFound("Interview not found.")
    st = db.execute(select(InterviewState).where(InterviewState.session_id == sess.id).with_for_update()).scalar_one_or_none()
    bp = db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == sess.id)).scalar_one_or_none()
    if st is None or bp is None:
        raise Conflict("This interview isn't ready yet.", code="not_ready")
    return sess, st, bp.blueprint


def _save_state(st: InterviewState, state: dict) -> None:
    # The JSON column does not track in-place mutation of nested dicts; always mark it dirty.
    st.state = state
    flag_modified(st, "state")
    st.version = (st.version or 0) + 1
    st.updated_at = utcnow()


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=utcnow().tzinfo)


def _tick(sess: InterviewSession, state: dict) -> None:
    now = utcnow()
    gap = (now - _aware(sess.last_activity_at)).total_seconds() if sess.last_activity_at else 0
    # Accumulate fractional seconds: truncating each turn's gap lost up to a second per turn.
    state["clock"]["active_s"] = round(float(state["clock"]["active_s"]) + max(0.0, min(gap, IDLE_CAP_S)), 3)
    sess.active_seconds = int(round(state["clock"]["active_s"]))
    sess.last_activity_at = now


def _switch_section(state: dict, bp: dict, new_sid: str) -> None:
    old = state["section_id"]
    if old == new_sid:
        return
    clock = state["clock"]
    clock["section_spent"][old] = clock["section_spent"].get(old, 0) + max(0, clock["active_s"] - clock["section_started_s"])
    clock["section_started_s"] = clock["active_s"]
    ids = S.section_ids(bp)
    # Items left in the sections we jump over are recorded as skipped (coverage map shows it).
    if new_sid in ids:
        start, end = ids.index(old) if old in ids else 0, ids.index(new_sid)
        for sid in ids[start:end]:
            state["skipped"].extend(state["queues"].get(sid, []))
            state["queues"][sid] = []
        state["section_idx"] = end
    state["section_id"] = new_sid


def _msg(db: Session, sess: InterviewSession, *, role: str, content: str, exchange_id=None, action: str = "",
         kind: str = "text", client_turn_id: Optional[str] = None, meta: Optional[dict] = None) -> InterviewMessage:
    m = InterviewMessage(session_id=sess.id, seq=next_message_seq(db, sess.id), role=role, kind=kind, content=content,
                         exchange_id=exchange_id, action=action, client_turn_id=client_turn_id, meta=meta or {})
    db.add(m)
    db.flush()
    return m


def _event(db: Session, sess: InterviewSession, type_: str, payload: dict) -> None:
    db.add(InterviewEvent(session_id=sess.id, type=type_, payload=payload))


def _open_exchange(db: Session, sess: InterviewSession, state: dict, item: dict) -> InterviewExchange:
    state["counters"]["exchanges"] += 1
    meta_keys = ["qid", "origin", "archetype_id", "curated_id", "section_kind", "question_type", "intent",
                 "expected_evidence", "strong_signals", "weak_signals", "red_flags", "must_not_infer", "probe_tree",
                 "selection_reason", "sub_competency", "difficulty"]
    ex = InterviewExchange(session_id=sess.id, seq=state["counters"]["exchanges"], planned_qid=item.get("qid", ""),
                           section_id=state["section_id"], question_text=item.get("text", ""),
                           question_meta={k: item.get(k) for k in meta_keys},
                           competency_ids=item.get("competency_ids", []), difficulty=int(item.get("difficulty", 3)))
    db.add(ex)
    db.flush()
    state["current"] = {"exchange_id": str(ex.id), "qid": item.get("qid"), "probes": 0, "challenged": False,
                        "qualities": [], "declined": False}
    if item.get("qid") and item["qid"] not in state["asked"]:
        state["asked"].append(item["qid"])
    for sid, q in state["queues"].items():
        if item.get("qid") in q:
            q.remove(item["qid"])
    return ex


def _close_exchange(db: Session, sess: InterviewSession, state: dict, *, declined: bool = False) -> None:
    cur = state.get("current")
    if not cur:
        return
    ex = db.get(InterviewExchange, uuid.UUID(cur["exchange_id"]))
    if ex is None or ex.status != "open":
        state["current"] = None
        return
    best = max(cur.get("qualities") or [], key=lambda q: QUALITY_RANK.get(q, 0), default=None)
    for cid in ex.competency_ids or []:
        c = S.cov(state, cid)
        c["asked"] += 1
        if declined or cur.get("declined"):
            c["declined"] += 1
        elif best:
            c[best] += 1
    ex.status = "skipped" if (declined or cur.get("declined")) else "closed"
    ex.closed_at = utcnow()
    ex.probes = int(cur.get("probes", 0))
    # claim ladder bookkeeping
    claim_ids = ((ex.question_meta or {}).get("selection_reason") or {}).get("claim_ids") or []
    for cid in claim_ids:
        if cid in state["claims"]:
            state["claims"][cid]["status"] = "probed"
    has_answer = db.execute(select(InterviewMessage.id).where(InterviewMessage.exchange_id == ex.id,
                                                              InterviewMessage.role == "candidate")).first()
    if has_answer and ex.section_id != "closing" and ex.status == "closed":
        enqueue(db, "extract_exchange_evidence", {"exchange_id": str(ex.id)}, dedupe_key=f"evidence:{ex.id}",
                max_attempts=3)
    else:
        ex.evidence_status = "skipped"
    state["current"] = None


def _ctx(sess: InterviewSession) -> RunContext:
    return RunContext(user_id=sess.user_id, session_id=sess.id)


def _degraded(db: Session, sess: InterviewSession, state: dict) -> bool:
    if state.get("degraded"):
        return True
    cap = float(flags.flag(db, "limits.session_cost_cap_usd") or 0)
    if cap and float(sess.cost_usd or 0) + pending_cost(sess.id) >= cap:
        state["degraded"] = True
        _event(db, sess, "degraded", {"reason": "session cost cap reached", "cap_usd": cap})
        return True
    return False


def _public_message(m: InterviewMessage) -> dict:
    return {"id": str(m.id), "seq": m.seq, "role": m.role, "content": m.content, "kind": m.kind,
            "created_at": m.created_at.isoformat()}


def session_progress(sess: InterviewSession, state: Optional[dict], bp: Optional[dict]) -> dict:
    out = {"status": sess.status, "ended_reason": sess.ended_reason or None}
    if state and bp:
        sec = S.section(bp, state["section_id"]) or {}
        planned = sum(len(s["items"]) for s in bp.get("sections", []) if s["id"] not in ("intro", "closing"))
        asked = len([q for q in state.get("asked", []) if q])
        out.update({
            "section": state["section_id"], "section_title": sec.get("title", ""),
            "elapsed_s": int(state["clock"]["active_s"]), "remaining_s": S.remaining_s(state, bp),
            "duration_s": S.total_budget_s(bp),
            "questions_asked": max(0, asked - 1), "questions_planned": planned,
        })
    return out


# ------------------------------------------------------------------------------------------
# public operations
# ------------------------------------------------------------------------------------------
def start(db: Session, user_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    sess, st, bp = _load(db, user_id, session_id)
    if sess.status == "active" and sess.started_at is not None:
        return resume_view(db, sess, st.state, bp)
    if sess.status != "ready":
        raise Conflict("This interview can't be started right now.", code="not_ready")
    transition(db, sess, "active", reason="started")
    state = st.state
    first = S.item(bp, state["queues"][S.section_ids(bp)[0]][0])
    ex = _open_exchange(db, sess, state, first)
    action = Action("OPEN", qid=first["qid"])
    text, used, guard = speak(action, ctx=_ctx(sess), bp=bp, question_text=first["text"], last_answer="",
                              memory_refs=[], recent_openers=[], grounding=[first["text"]])
    m = _msg(db, sess, role="interviewer", content=text, exchange_id=ex.id, action="OPEN",
             meta={"qid": first["qid"], "used_model": used, **({"guard": guard} if guard else {})})
    state["recent_openers"].append(opener_of(text))
    state["lines"] = [{"x": str(ex.id), "a": "OPEN", "t": truncate(text, 500)}]
    _event(db, sess, "decision", {"turn": 0, **action.to_event()})
    _save_state(st, state)
    return {"messages": [_public_message(m)], "session": session_progress(sess, state, bp)}


def resume_view(db: Session, sess: InterviewSession, state: dict, bp: dict) -> dict:
    msgs = db.execute(select(InterviewMessage).where(InterviewMessage.session_id == sess.id)
                      .order_by(InterviewMessage.seq)).scalars().all()
    return {"messages": [_public_message(m) for m in msgs if m.role != "system"],
            "session": session_progress(sess, state, bp)}


SAID_KEEP = 16000  # characters of the candidate's own words kept as grounding for the interviewer
LINES_KEEP = 60


def _lines(state: dict) -> list:
    return state.setdefault("lines", [])


_PREFACES = ("Welcome back. Let's pick up where we left off.", "That's fine, we can leave that one.", "Sure.",
             "Fair question. Any example that fits works — what I'm asking is:", "Let me put it another way.")


def _last_line(state: dict, exchange_id: Optional[str], bp: Optional[dict] = None) -> str:
    """What the interviewer last asked in this exchange, without greetings or transitions — the
    thing a candidate wants repeated or clarified."""
    from .interviewer import TRANSITIONS
    for ln in reversed(_lines(state)):
        if ln.get("x") != exchange_id or ln.get("a") in ("WAIT", "PAUSE"):
            continue
        if ln.get("a") in ("ASK", "OPEN") and bp is not None:
            cur = state.get("current") or {}
            planned = (S.item(bp, cur.get("qid")) or {}).get("text", "") if cur.get("qid") else ""
            if planned:
                return planned
        text = ln.get("t", "")
        changed = True
        while changed:
            changed = False
            for pre in (*_PREFACES, *TRANSITIONS.values()):
                if text.startswith(pre.strip()):
                    text, changed = text[len(pre.strip()):].strip(), True
        return text
    return ""


def turn(db: Session, user_id: uuid.UUID, session_id: uuid.UUID, *, client_turn_id: str, content: str,
         kind: str = "text", answer_ms: Optional[int] = None, skip: bool = False) -> dict:
    client_turn_id = (client_turn_id or "").strip()[:64]
    if not client_turn_id:
        raise Unprocessable("client_turn_id is required.", code="missing_turn_id")
    sess, st, bp = _load(db, user_id, session_id)

    # Idempotency: a retried request returns what the first one produced.
    prior = db.execute(select(InterviewMessage).where(InterviewMessage.session_id == sess.id,
                                                      InterviewMessage.client_turn_id == client_turn_id)).scalar_one_or_none()
    if prior is not None:
        after = db.execute(select(InterviewMessage).where(InterviewMessage.session_id == sess.id,
                                                          InterviewMessage.seq > prior.seq)
                           .order_by(InterviewMessage.seq).limit(2)).scalars().all()
        reply = [m for m in after if m.role == "interviewer"][:1]
        return {"messages": [_public_message(prior)] + [_public_message(m) for m in reply],
                "session": session_progress(sess, st.state, bp), "replayed": True}

    if sess.status == "paused":
        transition(db, sess, "active", reason="resumed by answering")
    if sess.status != "active":
        raise Conflict("This interview is not in progress.", code="not_active")
    text = (content or "").replace("\x00", "").strip()
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS]
    if not text and not skip:
        raise Unprocessable("Type or say your answer first.", code="empty_answer")

    state = st.state
    _tick(sess, state)
    state["counters"]["turns"] += 1
    turn_no = state["counters"]["turns"]
    cur = state.get("current") or {}
    item = S.item(bp, cur.get("qid")) if cur.get("qid") else None
    degraded = _degraded(db, sess, state)

    intent = "refusal" if skip else intents.detect(text)
    inj = scan_injection(text)
    if inj and intent == "answer" and len(text.split()) < 40:
        # A short message that is mostly an instruction to the system is not an answer.
        intent = "off_topic_question"
    metrics = answer_metrics(text, duration_ms=answer_ms)
    cand = _msg(db, sess, role="candidate", content=text or "[skipped]", exchange_id=uuid.UUID(cur["exchange_id"]) if cur.get("exchange_id") else None,
                kind="voice" if kind == "voice" else "text", client_turn_id=client_turn_id,
                meta={"intent": intent, "metrics": metrics, "injection_flags": inj, "answer_ms": answer_ms})
    if inj:
        _event(db, sess, "injection_attempt", {"turn": turn_no, "flags": inj})
    if text:
        state["said"] = (state.get("said", "") + "\n" + text)[-SAID_KEEP:]

    analysis = None
    in_closing = state["section_id"] == "closing" and state["closing"]["stage"] == "invited"
    if intent == "answer" and item is not None and not in_closing and not degraded:
        exchange_text = _exchange_transcript(db, cur.get("exchange_id"), exclude_id=cand.id)
        ta = analyze_turn(ctx=_ctx(sess), item=item, answer=text, exchange_so_far=exchange_text,
                          memory=state["memory"], slots_context=CL.slots_context(state), probes_used=int(cur.get("probes", 0)))
        if ta is not None:
            analysis = ta.model_dump()
            # The analyzer hears what the keyword rules miss — above all a clarifying question
            # ("Do you mean in my current role?"), which must be answered, not scored and skipped.
            if ta.intent in ("off_topic_question", "refusal", "meta_question", "end_request", "break_request",
                             "clarification_request", "repeat_request", "thinking_pause") and \
                    len(text.split()) < 40:
                # ...but a short ANSWER misheard as an unrelated question would get the same question
                # asked again: only a message that is actually a question can be one.
                statement_only = ta.intent in ("off_topic_question", "meta_question") and \
                    not intents.looks_like_question(text)
                if not statement_only:
                    intent = ta.intent
    if intent == "answer":
        _absorb_answer(db, sess, state, bp, cur, analysis, text)
        state["consecutive"]["non_answers"] = 0
    elif intent == "non_answer":
        state["consecutive"]["non_answers"] += 1

    # Contradiction raised on the previous turn is now answered.
    for k in state["contradictions"]:
        if k["status"] == "raised":
            k["status"] = "clarified"

    if degraded and state["section_id"] != "closing" and intent not in ("end_request", "break_request"):
        action = Action("CLOSE_INVITE", close_exchange=True, new_section="closing", reasons=["degraded mode"])
    else:
        action = decide(state, bp, intent=intent, analysis=analysis, session_key=str(sess.id))
    _event(db, sess, "decision", {"turn": turn_no, "intent": intent,
                                  "analysis": _redacted_analysis(analysis), **action.to_event()})
    reply_msgs = _apply(db, sess, st, state, bp, action, last_answer=text, degraded=degraded)
    return {"messages": [_public_message(cand)] + [_public_message(m) for m in reply_msgs],
            "session": session_progress(sess, state, bp)}


def _redacted_analysis(a: Optional[dict]) -> Optional[dict]:
    if not a:
        return None
    return {k: a.get(k) for k in ("intent", "addresses_question", "answer_quality", "probe_focus", "gaps", "summary")}


def _exchange_transcript(db: Session, exchange_id: Optional[str], *, exclude_id=None) -> str:
    if not exchange_id:
        return ""
    rows = db.execute(select(InterviewMessage).where(InterviewMessage.exchange_id == uuid.UUID(exchange_id))
                      .order_by(InterviewMessage.seq)).scalars().all()
    return "\n".join(f"{'INTERVIEWER' if m.role == 'interviewer' else 'CANDIDATE'}: {m.content}"
                     for m in rows if m.id != exclude_id)


def _absorb_answer(db: Session, sess: InterviewSession, state: dict, bp: dict, cur: dict, analysis: Optional[dict],
                   text: str) -> None:
    quality = (analysis or {}).get("answer_quality", "adequate")
    if cur:
        cur.setdefault("qualities", []).append(quality)
        ex = db.get(InterviewExchange, uuid.UUID(cur["exchange_id"])) if cur.get("exchange_id") else None
        if ex is not None and analysis:
            ex.live_signals = list(ex.live_signals or []) + [_redacted_analysis(analysis)]
    cons = state["consecutive"]
    if quality == "strong":
        cons["strong"] += 1
        cons["weak"] = 0
    elif quality == "weak":
        cons["weak"] += 1
        cons["strong"] = 0
    else:
        cons["strong"] = 0
        cons["weak"] = 0
    summary = (analysis or {}).get("summary") or truncate(text, 200)
    S.add_memory(state, exchange_id=cur.get("exchange_id", "") if cur else "", summary=summary)
    if analysis:
        item = S.item(bp, cur.get("qid")) if cur.get("qid") else None
        anchors = ((item or {}).get("selection_reason") or {}).get("claim_ids") or []
        added = CL.register(state, analysis.get("new_claims") or [], exchange_id=cur.get("exchange_id", ""),
                            answer_text=text, default_anchor=anchors[0] if anchors else "")
        for ic in added:
            db.add(Claim(session_id=sess.id, claim_key=ic["id"], source="interview", text=ic["text"] or "(claim)",
                         slots=ic.get("slots", {}), priority=1 if ic.get("high_impact") else 2,
                         exchange_id=uuid.UUID(cur["exchange_id"]) if cur.get("exchange_id") else None))
        CL.detect(state)
        CL.from_analyzer(state, analysis.get("conflicts") or [], CL.valid_refs(state))


def _apply(db: Session, sess: InterviewSession, st: InterviewState, state: dict, bp: dict, action: Action, *,
           last_answer: str, degraded: bool) -> List[InterviewMessage]:
    cur = state.get("current") or {}
    current_item = S.item(bp, cur.get("qid")) if cur.get("qid") else None
    question_text = (current_item or {}).get("text", "")
    contradiction_text = ""
    exchange_id = uuid.UUID(cur["exchange_id"]) if cur.get("exchange_id") else None

    if action.difficulty_delta:
        state["difficulty"]["level"] = max(1, min(5, state["difficulty"]["level"] + action.difficulty_delta))
        _event(db, sess, "difficulty", {"level": state["difficulty"]["level"], "delta": action.difficulty_delta})

    t = action.type
    if t in ("ASK", "CLOSE_INVITE", "CLOSE_FINAL", "END_EARLY") and action.close_exchange:
        _close_exchange(db, sess, state, declined=action.declined)
    if action.new_section:
        _switch_section(state, bp, action.new_section)

    if t == "ASK":
        nxt = S.item(bp, action.qid)
        ex = _open_exchange(db, sess, state, nxt)
        exchange_id, question_text = ex.id, nxt["text"]
        state["consecutive"]["non_answers"] = 0
    elif t == "CLOSE_INVITE":
        if state["section_id"] != "closing":
            _switch_section(state, bp, "closing")
        closing_item = S.item(bp, (S.section(bp, "closing") or {"items": [{"qid": ""}]})["items"][0]["qid"]) or {
            "qid": "", "text": "What questions do you have for me?", "competency_ids": []}
        ex = _open_exchange(db, sess, state, closing_item)
        exchange_id, question_text = ex.id, closing_item["text"]
        state["closing"]["stage"] = "invited"
    elif t == "PROBE":
        cur["probes"] = int(cur.get("probes", 0)) + 1
        if current_item and current_item.get("origin") == "cv_specific":
            for cid in ((current_item.get("selection_reason") or {}).get("claim_ids") or []):
                if cid in state["claims"]:
                    state["claims"][cid]["ladder_step"] = state["claims"][cid].get("ladder_step", 0) + 1
                    state["claims"][cid]["status"] = "probing"
    elif t == "CHALLENGE":
        cur["probes"] = int(cur.get("probes", 0)) + 1
        cur["challenged"] = True
    elif t == "CLARIFY_CONTRADICTION":
        cur["probes"] = int(cur.get("probes", 0)) + 1
        k = next((c for c in state["contradictions"] if c["id"] == action.contradiction_id), None)
        if k:
            k["status"] = "raised"
            contradiction_text = CL.describe(k, state["memory"])
    elif t == "NUDGE":
        cur["probes"] = int(cur.get("probes", 0)) + 1

    memory_refs = []
    if t in ("PROBE", "CHALLENGE", "ASK") and state["memory"] and len(state["memory"]) > 2:
        memory_refs = state["memory"][-4:-1]

    xid = str(exchange_id) if exchange_id else None
    # Repeating or clarifying is about the interviewer's LAST line (often a follow-up), not the main question.
    if t in ("REPEAT", "CLARIFY_QUESTION"):
        question_text = _last_line(state, xid, bp) or question_text
    target_item = S.item(bp, action.qid) if t == "ASK" and action.qid else current_item
    claim_texts = [state["claims"][c]["text"] for c in (((target_item or {}).get("selection_reason") or {})
                                                        .get("claim_ids") or []) if c in state["claims"]]
    exchange_lines = [ln["t"] for ln in _lines(state) if ln.get("x") == xid]
    grounding = [question_text, (target_item or {}).get("text", ""), *exchange_lines, *claim_texts,
                 state.get("said", ""), contradiction_text]
    previous_questions = [ln["t"] for ln in _lines(state) if ln.get("a") in ("ASK", "OPEN") and ln.get("x") != xid]
    previous_questions += [(S.item(bp, q) or {}).get("text", "") for q in state["asked"] if q != action.qid]
    exchange_text = _exchange_transcript(db, xid) if t in ("PROBE", "CHALLENGE", "CLARIFY_QUESTION") else ""

    text, used, guard = speak(action, ctx=_ctx(sess), bp=bp, question_text=question_text, last_answer=last_answer,
                              memory_refs=memory_refs, recent_openers=state["recent_openers"],
                              contradiction=contradiction_text, closing_reply=last_answer if t == "CLOSE_FINAL" else "",
                              degraded=degraded, grounding=grounding, exchange_text=exchange_text,
                              previous_questions=previous_questions, used_foci=cur.get("foci", []) if cur else [])
    if t == "PROBE" and cur:
        cur["foci"] = (cur.get("foci") or []) + [action.focus or "specificity"]
    state["recent_openers"] = (state["recent_openers"] + [opener_of(text)])[-6:]
    state["lines"] = (_lines(state) + [{"x": xid, "a": t, "t": truncate(text, 500)}])[-LINES_KEEP:]
    if guard:
        _event(db, sess, "line_guard", {"action": t, "note": guard})
    m = _msg(db, sess, role="interviewer", content=text, exchange_id=exchange_id, action=t,
             meta={"qid": action.qid or cur.get("qid"), "focus": action.focus, "used_model": used,
                   "reasons": action.reasons[:6], **({"guard": guard} if guard else {})})

    if t == "PAUSE":
        transition(db, sess, "paused", reason="candidate requested a break")
    elif t in ("CLOSE_FINAL", "END_EARLY"):
        if state.get("current"):
            _close_exchange(db, sess, state)
        sess.ended_reason = action.end_reason or "completed_naturally"
        _finish(db, sess, state)
    _save_state(st, state)
    return [m]


def _flush_section_clock(state: dict) -> None:
    clock = state["clock"]
    sid = state["section_id"]
    clock["section_spent"][sid] = clock["section_spent"].get(sid, 0) + max(0, clock["active_s"] - clock["section_started_s"])
    clock["section_started_s"] = clock["active_s"]


def _finish(db: Session, sess: InterviewSession, state: dict) -> None:
    _flush_section_clock(state)
    transition(db, sess, "completed", reason=sess.ended_reason)
    sess.assessment_status = "pending"
    enqueue(db, "assess_session", {"session_id": str(sess.id)}, dedupe_key=f"assess:{sess.id}", max_attempts=3,
            delay_s=2)


def pause(db: Session, user_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    sess, st, bp = _load(db, user_id, session_id)
    if sess.status != "active":
        raise Conflict("Only an interview in progress can be paused.", code="not_active")
    _tick(sess, st.state)
    transition(db, sess, "paused", reason="paused by candidate")
    _save_state(st, st.state)
    return {"session": session_progress(sess, st.state, bp)}


def resume(db: Session, user_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    sess, st, bp = _load(db, user_id, session_id)
    if sess.status != "paused":
        raise Conflict("This interview isn't paused.", code="not_paused")
    transition(db, sess, "active", reason="resumed")
    sess.last_activity_at = utcnow()
    state = st.state
    cur = state.get("current") or {}
    item = S.item(bp, cur.get("qid")) if cur.get("qid") else None
    # pick up exactly where it stopped: the last thing asked (often a follow-up), else the question
    q = _last_line(state, cur.get("exchange_id"), bp) or (item or {}).get("text", "")
    if q.lower().startswith("welcome back"):
        q = (item or {}).get("text", "")
    text = f"Welcome back. Let's pick up where we left off. {q}" if q else "Welcome back. Let's continue."
    m = _msg(db, sess, role="interviewer", content=text, exchange_id=uuid.UUID(cur["exchange_id"]) if cur.get("exchange_id") else None,
             action="RESUME", meta={"qid": cur.get("qid")})
    _save_state(st, state)
    return {"messages": [_public_message(m)], "session": session_progress(sess, state, bp)}


def end_early(db: Session, user_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    sess = find_owned(db, user_id, session_id, lock=True)
    if sess is None:
        raise NotFound("Interview not found.")
    if sess.status in ("completed", "abandoned", "expired", "failed"):
        return {"session": {"status": sess.status, "ended_reason": sess.ended_reason}}
    if sess.status not in ("active", "paused"):
        # Ending before it began = abandoning the slot; there is nothing to assess.
        transition(db, sess, "abandoned", reason="ended before starting")
        return {"session": {"status": sess.status}}
    st = db.execute(select(InterviewState).where(InterviewState.session_id == sess.id).with_for_update()).scalar_one()
    state = st.state
    if sess.status == "active":
        _tick(sess, state)
    if state.get("current"):
        _close_exchange(db, sess, state)
    sess.ended_reason = "ended_early_by_user"
    _event(db, sess, "ended_early", {"active_s": state["clock"]["active_s"]})
    _finish(db, sess, state)
    _save_state(st, state)
    return {"session": {"status": sess.status, "ended_reason": sess.ended_reason}}


def abandon(db: Session, user_id: uuid.UUID, session_id: uuid.UUID) -> dict:
    sess = find_owned(db, user_id, session_id, lock=True)
    if sess is None:
        raise NotFound("Interview not found.")
    if sess.status in ("completed", "abandoned", "expired", "failed"):
        return {"session": {"status": sess.status}}
    transition(db, sess, "abandoned", reason="abandoned by candidate")
    sess.ended_reason = "abandoned"
    return {"session": {"status": sess.status}}
