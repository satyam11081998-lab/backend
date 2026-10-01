"""
Control-tag parsing, streaming strip, Contextual Assessor, and Intervention Gate.
Implements V10.2: Safely routing imperatives to Deep Lane and cleanly falling back 
to contextual PROBEs or HAND_BACKs.
V12: the gates now feed a policy layer that picks a conversational RESPONSE
FUNCTION; the model words it from a JSON INTERVIEWER CONTROL PACKET (see
decide_response / build_interviewer_control_packet / validate_contextual_line).
"""
from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Dict, Generator, Tuple, Any, List, Optional

from services.learning_model import evaluate_intervention_outcome, update_learning_profile
from services.ai_providers import openai_client

_TAG_RE = re.compile(r"^\s*<<(.*?)>>\s*", re.DOTALL)
_DEC = "\u0001"  


def _sentences(text: str, keep_punct: bool = True):
    t = re.sub(r"(\d)\.(\d)", r"\1" + _DEC + r"\2", text or "")
    if keep_punct:
        parts = re.findall(r"[^.!?]*[.!?]|[^.!?]+$", t)
    else:
        parts = re.split(r"[.!?]+", t)
    return [p.replace(_DEC, ".").strip() for p in parts if p.strip()]


def parse_control_tag(text: str) -> Tuple[Dict[str, str], str]:
    if not text:
        return {}, ""
    m = _TAG_RE.match(text)
    if not m:
        return {}, text.strip()
    tag: Dict[str, str] = {}
    for part in re.split(r"[;,]", m.group(1)):
        if "=" in part:
            k, v = part.split("=", 1)
            tag[k.strip().lower()] = v.strip().lower()
    cleaned = text[m.end():].lstrip("\n").strip()
    return (tag, cleaned) if cleaned else (tag, text.strip())


class StreamTagStripper:
    def __init__(self, budget: int = 240):
        self.buf = ""
        self.resolved = False
        self.budget = budget
        self._strip_leading = False
        self.tag = {}

    def feed(self, token: str) -> Generator[str, None, None]:
        if self.resolved:
            if token:
                if self._strip_leading:
                    token = token.lstrip()
                    if not token:
                        return
                    self._strip_leading = False
                yield token
            return
        self.buf += token or ""
        stripped = self.buf.lstrip()
        if not stripped:
            return
        if stripped.startswith("<<"):
            end = self.buf.find(">>")
            if end != -1:
                self.tag = parse_control_tag(self.buf[:end + 2])[0]
                rest = self.buf[end + 2:].lstrip("\n").lstrip()
                self.resolved, self.buf = True, ""
                if rest:
                    yield rest
                else:
                    self._strip_leading = True
            elif len(self.buf) > self.budget:
                out, self.buf, self.resolved = self.buf, "", True
                if out:
                    yield out
            return
        if len(stripped) >= 2 or len(self.buf) > 8:
            out, self.buf, self.resolved = self.buf, "", True
            if out:
                yield out

    def flush(self) -> Generator[str, None, None]:
        if not self.buf:
            return
        out, self.buf, self.resolved = self.buf, "", True
        m = _TAG_RE.match(out)
        if m:
            out = out[m.end():].lstrip()
        if out:
            yield out


# --- STATE ESTIMATION (Contextual Assessor) ----------------------------------

_ASSESSOR_PROMPT = """You are a state assessor for a consulting case interview.
Evaluate the candidate's last message in the context of the transcript.
Output ONLY a JSON object with these exact keys and values from the enums below:
{
  "intent": "answering" | "clarification" | "asking_for_help" | "asking_for_solution" | "hypothesis" | "meta" | "wants_to_stop" | "working" | "unknown",
  "progress": "progressing" | "recovering" | "stuck" | "repeatedly_stuck" | "stalled" | "completed_step" | "unclear",
  "materiality": "none" | "minor" | "material" | "critical" | "unknown",
  "case_phase": "opening" | "clarification" | "structuring" | "analysis" | "quantification" | "synthesis" | "closing" | "unknown"
}
"""

def assess_context_with_llm(transcript: list, new_message: str, case_content: str) -> Dict[str, Any]:
    cli = openai_client()
    if not cli:
        return {}
        
    history_lines = []
    for t in transcript[-4:]:
        role = t.get('role', 'user')
        content = t.get('content', '')
        if content:
            history_lines.append(f"{role.upper()}: {content}")
    history = "\n".join(history_lines)
    
    case_ctx = (case_content or "")[:500]
    user_prompt = f"CASE CONTEXT: {case_ctx}...\n\nRECENT TRANSCRIPT:\n{history}\n\nCANDIDATE CURRENT TURN:\n{new_message}"
    
    try:
        resp = cli.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _ASSESSOR_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


# --- INTERVENTION GATE (V10 TWO-GATE SYSTEM) --------------------------------

def _evaluate_substantive_gate(signals: Dict[str, Any]) -> Tuple[bool, str, str]:
    if signals.get("is_session_open"): return True, "OPEN", "session_start"
    if signals.get("is_session_close"): return True, "CLOSE", "session_end"
    if signals.get("wants_to_advance") or signals.get("intent") == "wants_to_stop":
        return True, "TRANSITION", "candidate_transition"
        
    if signals.get("solution_requested") or signals.get("intent") == "asking_for_solution":
        return True, "DELIVER_SOLUTION", "solution_requested"
    if signals.get("help_requested") or signals.get("intent") == "asking_for_help":
        return True, "HINT", "help_requested"
    if signals.get("looks_garbage"): return True, "NOISE", "noise"
    if signals.get("is_meta") or signals.get("intent") == "meta": return True, "DEFLECT_META", "meta"

    if signals.get("error_materiality") == "material" or signals.get("materiality") in ["material", "critical"]:
        return True, "CORRECT_MATERIAL", "material_error"
    if signals.get("states_final_estimate") or signals.get("confident_unsupported_claim"):
        return True, "RETHINK_CUE", "check_suspicious_claim"
    if signals.get("frustration_explicit") or signals.get("progress") in ["stuck", "repeatedly_stuck"]:
        return True, "REPAIR", "frustration_or_stuck"

    if signals.get("why_this_question") or signals.get("product_ux_question") or signals.get("intent") == "clarification":
        return True, "ANSWER_DIRECT", "clarification"
    if signals.get("asks_owned_fact") or signals.get("is_scope_question"):
        return True, "DATA_REVEAL", "fact_requested"

    if signals.get("intent") == "hypothesis":
        if signals.get("progress") != "progressing":
            return True, "DATA_REVEAL", "hypothesis_needs_data"

    return False, "", ""

def _evaluate_presence_gate(signals: Dict[str, Any]) -> Tuple[bool, str, str]:
    if signals.get("last_action_was_presence"):
        return False, "", "recently_acknowledged"
        
    if signals.get("is_affirmation_only"):
        return True, "HAND_BACK", "affirmation_received"
        
    if signals.get("hedged_self_estimate") or "?" in signals.get("new_message_norm", ""):
        return True, "HAND_BACK", "return_floor_explicitly"
        
    if signals.get("candidate_working") or signals.get("wants_space"):
        return True, "HAND_BACK", "candidate_needs_space"
        
    if signals.get("just_recovered"):
        return True, "VALIDATE", "candidate_recovered"
        
    if signals.get("has_work"):
        return True, "ACKNOWLEDGE", "conversational_presence"
        
    return False, "", ""

def evaluate_intervention_gate(signals: Dict[str, Any]) -> Tuple[str, str, str]:
    if not signals:
        return "SUBSTANTIVE", "PROBE", "no_signals"

    channel = signals.get("channel", "text")
    is_voice_partial = signals.get("is_voice_partial", False)

    if is_voice_partial:
        return "SILENCE", "NO_OUTPUT", "voice_partial"

    is_substantive, sub_mode, sub_reason = _evaluate_substantive_gate(signals)
    if is_substantive:
        return "SUBSTANTIVE", sub_mode, sub_reason

    is_presence, pres_mode, pres_reason = _evaluate_presence_gate(signals)
    if is_presence:
        return "PRESENCE", pres_mode, pres_reason

    # Text Fallback: Prevents Blank Bubbles on un-recognized states
    if channel == "text":
        if signals.get("has_work") or signals.get("candidate_working"):
            return "PRESENCE", "ACKNOWLEDGE", "text_fallback_prevent_blank"
        else:
            # The candidate said something short that wasn't an affirmation or a known directive.
            # E.g. "I don't know." -> It's best to gently probe instead of saying "Got it".
            return "SUBSTANTIVE", "PROBE", "text_fallback_no_work"

    return "SILENCE", "NO_OUTPUT", "no_presence_needed_for_voice"


# --- V12 POLICY LAYER: STATE -> RESPONSE FUNCTION ---------------------------
#
# Function first, language second. The two gates above still decide WHETHER the
# interviewer intervenes and every business rule they encode (help is honoured,
# solutions are delivered, material errors are corrected, frustration is
# repaired, meta is deflected, voice partials are silent). This layer decides
# WHAT CONVERSATIONAL FUNCTION the turn needs. It never writes the sentence:
#   - NO_OUTPUT                      -> nothing (fast lane)
#   - SHORT_ACK / HAND_BACK          -> one of a few fixed words, ONLY when the
#                                      candidate's turn itself carried almost
#                                      nothing (fast lane, no model call)
#   - every other function           -> worded by the model from the JSON
#                                      INTERVIEWER CONTROL PACKET below.
# A substantive candidate turn never reaches the fixed-phrase lane.

RESPONSE_FUNCTIONS = (
    "NO_OUTPUT", "SHORT_ACK", "HAND_BACK", "ACKNOWLEDGE_AND_CONTINUE",
    "ACKNOWLEDGE_AND_ORIENT", "REFLECT_PROGRESS", "VALIDATE_AND_HAND_BACK",
    "TARGETED_QUESTION", "SANITY_CHECK", "MICRO_HINT", "CORRECT_AND_CONTINUE",
    "REPAIR_AND_RESET", "ANSWER_DIRECT", "DATA_REVEAL", "DELIVER_SOLUTION",
    "TRANSITION", "OPEN", "CLOSE", "DEFLECT_META", "NOISE", "CONTINUE_AS_AGREED",
)
FAST_FUNCTIONS = frozenset({"SHORT_ACK", "HAND_BACK"})
CONTEXTUAL_PRESENCE = frozenset({
    "ACKNOWLEDGE_AND_CONTINUE", "ACKNOWLEDGE_AND_ORIENT", "REFLECT_PROGRESS",
    "VALIDATE_AND_HAND_BACK",
})

# Gate mode (unchanged V10.2 vocabulary) -> the function it now names.
_SUBSTANTIVE_FUNCTION = {
    "OPEN": "OPEN",
    "CLOSE": "CLOSE",
    "TRANSITION": "TRANSITION",
    "DELIVER_SOLUTION": "DELIVER_SOLUTION",
    "HINT": "MICRO_HINT",
    "NOISE": "NOISE",
    "DEFLECT_META": "DEFLECT_META",
    "CORRECT_MATERIAL": "CORRECT_AND_CONTINUE",
    "RETHINK_CUE": "SANITY_CHECK",
    "REPAIR": "REPAIR_AND_RESET",
    "ANSWER_DIRECT": "ANSWER_DIRECT",
    "DATA_REVEAL": "DATA_REVEAL",
    "PROBE": "TARGETED_QUESTION",
}

_CONTEXTUAL_BY_TURN = {
    "strategic_insight": "REFLECT_PROGRESS",
    "structure": "REFLECT_PROGRESS",
    "hypothesis": "REFLECT_PROGRESS",
    "calculation_result": "ACKNOWLEDGE_AND_ORIENT",
    "plan": "ACKNOWLEDGE_AND_CONTINUE",
    "calculation": "ACKNOWLEDGE_AND_CONTINUE",
    "estimate": "ACKNOWLEDGE_AND_CONTINUE",
}
# When the same contextual function was used on the previous turn, rotate.
_ROTATE = {
    "REFLECT_PROGRESS": "ACKNOWLEDGE_AND_CONTINUE",
    "ACKNOWLEDGE_AND_CONTINUE": "REFLECT_PROGRESS",
    "ACKNOWLEDGE_AND_ORIENT": "ACKNOWLEDGE_AND_CONTINUE",
    "VALIDATE_AND_HAND_BACK": "ACKNOWLEDGE_AND_CONTINUE",
}


def _contextual_function(signals: Dict[str, Any], gate_mode: str) -> str:
    if gate_mode == "VALIDATE" or signals.get("just_recovered"):
        fn = "VALIDATE_AND_HAND_BACK"
    else:
        fn = _CONTEXTUAL_BY_TURN.get(signals.get("turn_type") or "", "ACKNOWLEDGE_AND_CONTINUE")
    if fn == signals.get("last_function"):
        fn = _ROTATE.get(fn, fn)
    return fn


_OWN_WORK_RE = re.compile(r"^(so |okay |ok |now )?(i'll|i will|i'd|i would|let me|i'm going to|im going to)\b")


def _announces_own_work(norm: str) -> bool:
    """A short "I'll split households first" announces work to come: hand the floor back."""
    return bool(_OWN_WORK_RE.search(norm))


def _decision(lane: str, mode: str, function: str, reason: str) -> Dict[str, Any]:
    render = ("silent" if function == "NO_OUTPUT"
              else "fast" if function in FAST_FUNCTIONS else "model")
    return {"lane": lane, "mode": mode, "function": function, "reason": reason, "render": render}


def contextual_presence_enabled() -> bool:
    """Kill switch: INTERVIEWER_CONTEXTUAL_PRESENCE=off sends the model-worded
    presence beats back to the fixed short lines (pre-V12 behaviour, 0 tokens).
    Read per turn, so flipping it on Render needs a restart, not a deploy."""
    return os.getenv("INTERVIEWER_CONTEXTUAL_PRESENCE", "on").strip().lower() not in ("off", "0", "false", "no")


def decide_response(signals: Dict[str, Any]) -> Dict[str, Any]:
    d = _decide_response(signals)
    if d["function"] in CONTEXTUAL_PRESENCE and not contextual_presence_enabled():
        fn = "HAND_BACK" if d["function"] == "VALIDATE_AND_HAND_BACK" else "SHORT_ACK"
        return _decision(d["lane"], d["mode"], fn, d["reason"] + "|contextual_off")
    return d


def _decide_response(signals: Dict[str, Any]) -> Dict[str, Any]:
    """The policy layer: candidate state -> {lane, mode, function, reason, render}.

    `mode` is the V10.2 gate mode (kept for telemetry, the learner-state fold and
    existing callers); `function` is what the interviewer must accomplish.
    """
    lane, mode, reason = evaluate_intervention_gate(signals or {})
    sig = signals or {}

    # "Shall we proceed with that?" -> "Yes, go ahead.": the interviewer carries on
    # with what it proposed (model-worded). Help, solution, meta, close etc. above
    # in the gate still win; a voice partial is still silent.
    if (sig.get("is_affirmation_only") and sig.get("last_assistant_asked")
            and not sig.get("is_voice_partial") and lane != "SUBSTANTIVE"):
        return _decision("SUBSTANTIVE", "TRANSITION", "CONTINUE_AS_AGREED", "go_ahead_after_question")

    if lane == "SILENCE":
        return _decision(lane, mode, "NO_OUTPUT", reason)

    if lane == "SUBSTANTIVE":
        # A candidate who lays out real work and then hands the floor back
        # ("... shall I proceed?") is not asking a clarification: they need the
        # work acknowledged and the floor returned. Anything that is a real
        # question (facts, scope, why, product) keeps its V10.2 route.
        if (reason == "clarification" and sig.get("asks_to_proceed")
                and sig.get("is_substantive_reasoning")
                and not (sig.get("asks_owned_fact") or sig.get("why_this_question")
                         or sig.get("product_ux_question"))):
            fn = _contextual_function(sig, "ACKNOWLEDGE")
            return _decision("PRESENCE", "ACKNOWLEDGE", fn, "floor_yield_after_work")
        return _decision(lane, mode, _SUBSTANTIVE_FUNCTION.get(mode, mode), reason)

    # PRESENCE: the gate decided a presence beat is due. Pick the function.
    if sig.get("candidate_mid_thought"):
        # Still calculating ("let me think", "hold on", trailing "..."): in voice
        # any line interrupts them; in text a short turn simply gets the floor back
        # (a substantive one still gets its content acknowledged, below).
        if sig.get("channel") == "voice":
            return _decision("SILENCE", "NO_OUTPUT", "NO_OUTPUT", "candidate_mid_thought")
        if not sig.get("is_substantive_reasoning"):
            return _decision(lane, mode, "HAND_BACK", "candidate_mid_thought")
    if sig.get("is_substantive_reasoning"):
        return _decision(lane, mode, _contextual_function(sig, mode), reason)
    if mode == "VALIDATE":
        # Recovery after a correction: acknowledge the specific fix (model-worded),
        # never a stock "That works." that certifies it.
        return _decision(lane, mode, "VALIDATE_AND_HAND_BACK", reason)
    if (mode == "HAND_BACK" or sig.get("turn_type") in ("plan", "affirmation")
            or _announces_own_work(sig.get("new_message_norm") or "")):
        return _decision(lane, mode, "HAND_BACK", reason)
    return _decision(lane, mode, "SHORT_ACK", reason)


def determine_response_function(signals: Dict[str, Any]) -> str:
    return decide_response(signals)["function"]


# --- V12 INTERVIEWER CONTROL PACKET -------------------------------------------

_OBJECTIVE = {
    "NO_OUTPUT": "stay silent; the candidate holds the floor",
    "SHORT_ACK": "register that you heard a minimal turn",
    "HAND_BACK": "return the floor so the candidate keeps working",
    "ACKNOWLEDGE_AND_CONTINUE": "recognise the specific progress the candidate just made and let them continue",
    "ACKNOWLEDGE_AND_ORIENT": "recognise the step the candidate just completed and point them to the next part of their own plan",
    "REFLECT_PROGRESS": "reflect back the distinction or progress the candidate just made, then let them carry it forward",
    "VALIDATE_AND_HAND_BACK": "acknowledge the adjustment the candidate just made and hand the floor back",
    "TARGETED_QUESTION": "move the case forward with one targeted question",
    "SANITY_CHECK": "prompt the candidate to sanity-check a number or claim themselves",
    "MICRO_HINT": "give the next rung of help tied to the candidate's own work",
    "CORRECT_AND_CONTINUE": "name the one material issue, then let them continue",
    "REPAIR_AND_RESET": "reset the frustrated or stuck candidate with a simpler framing",
    "ANSWER_DIRECT": "answer the candidate's question directly, then hand back",
    "DATA_REVEAL": "give only the case data requested",
    "DELIVER_SOLUTION": "give the approach the candidate asked for, concisely",
    "TRANSITION": "move cleanly to the next stage",
    "OPEN": "open the case",
    "CLOSE": "close the case",
    "DEFLECT_META": "stay in role and return to the case",
    "NOISE": "ask the candidate to restate a garbled turn",
    "CONTINUE_AS_AGREED": "the candidate agreed to what you just proposed or asked: carry on with exactly that",
}
_STRONG = {"MICRO_HINT", "CORRECT_AND_CONTINUE", "REPAIR_AND_RESET", "DELIVER_SOLUTION"}
_MODERATE = {"TARGETED_QUESTION", "SANITY_CHECK", "ANSWER_DIRECT", "DATA_REVEAL",
             "TRANSITION", "OPEN", "CLOSE", "DEFLECT_META", "NOISE"}
_NEW_FACTS_OK = {"ANSWER_DIRECT", "DATA_REVEAL", "OPEN", "DELIVER_SOLUTION", "MICRO_HINT",
                 "REPAIR_AND_RESET", "CORRECT_AND_CONTINUE"}


def _intervention_level(fn: str) -> str:
    if fn == "NO_OUTPUT":
        return "none"
    if fn in FAST_FUNCTIONS:
        return "minimal"
    if fn in CONTEXTUAL_PRESENCE:
        return "light"
    if fn in _STRONG:
        return "strong"
    return "moderate"


def _reasoning_stage(sig: Dict[str, Any]) -> str:
    phase = sig.get("case_phase")
    if phase and phase != "unknown":
        return phase
    tt = sig.get("turn_type")
    if sig.get("is_session_open"):
        return "opening"
    if sig.get("is_session_close"):
        return "closing"
    if tt in ("structure", "plan"):
        return "structuring"
    if tt in ("calculation", "calculation_result", "estimate"):
        return "quantification"
    if tt in ("strategic_insight", "hypothesis"):
        return "analysis"
    return "unknown"


def _progress_state(sig: Dict[str, Any]) -> str:
    if sig.get("progress") and sig.get("progress") != "unclear":
        return sig["progress"]
    if sig.get("just_recovered"):
        return "recovering"
    if sig.get("turns_without_progress", 0) >= 2:
        return "stuck"
    if sig.get("step_completed"):
        return "completed_step"
    if sig.get("has_work"):
        return "progressing"
    return "unclear"


def _gist(text: str, limit: int = 180) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    first = _sentences(t)[0] if _sentences(t) else t
    first = first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + "..."
    return first


def build_interviewer_control_packet(signals: Dict[str, Any], decision: Any,
                                     candidate_text: str = "",
                                     recent_lines: Optional[List[str]] = None,
                                     allow_questions: Optional[bool] = None) -> Dict[str, Any]:
    """The decision brief the model words a turn from. Plain data: no sentence to
    copy, no fallback line, no hidden case solution."""
    if isinstance(decision, str):
        decision = {"function": decision, "lane": "", "mode": decision, "reason": ""}
    sig = signals or {}
    fn = decision["function"]
    voice = sig.get("channel") == "voice"
    contextual = fn in CONTEXTUAL_PRESENCE
    q_ok = bool(allow_questions) if allow_questions is not None else False
    alerts = []
    if sig.get("repair_due"):
        alerts.append("your recent moves are not working: stop probing, change strategy")
    if sig.get("interviewer_repeating"):
        alerts.append("you have repeated yourself: do not reuse a previous line")
    if sig.get("turns_without_progress", 0) >= 2:
        alerts.append(f"no progress for {sig['turns_without_progress']} turns")
    return {"interviewer_control": {
        "response_function": fn,
        "conversational_objective": _OBJECTIVE.get(fn, "move the case forward"),
        "candidate_state": {
            "candidate_turn_type": sig.get("turn_type", "statement"),
            "candidate_turn_summary": _gist(candidate_text) if candidate_text else "",
            "numbers_stated": list(sig.get("numbers_stated") or []),
            "reasoning_stage": _reasoning_stage(sig),
            "progress_state": _progress_state(sig),
            "reasoning_quality": ("material_issue" if fn == "CORRECT_AND_CONTINUE"
                                  else "tentative" if sig.get("candidate_confidence") == "low"
                                  else "not_verified"),
            "candidate_confidence": sig.get("candidate_confidence", "moderate"),
            "candidate_is_working": bool(sig.get("has_work") or sig.get("candidate_working")),
            "candidate_needs_space": bool(sig.get("wants_space") or sig.get("candidate_mid_thought")),
            "explicit_help_request": bool(sig.get("help_requested")),
            "explicit_solution_request": bool(sig.get("solution_requested")),
            "material_error": fn == "CORRECT_AND_CONTINUE",
            "frustration_state": sig.get("frustration", "none"),
        },
        "interviewer_decision": {
            "lane": decision.get("lane", ""),
            "intervention_level": _intervention_level(fn),
            "question_allowed": q_ok,
            "hint_allowed": fn in ("MICRO_HINT", "REPAIR_AND_RESET", "DELIVER_SOLUTION"),
            "correction_allowed": fn == "CORRECT_AND_CONTINUE",
            "solution_allowed": fn == "DELIVER_SOLUTION",
            "silence_allowed": fn == "NO_OUTPUT",
            "may_confirm_correctness": False,
            "new_case_facts_allowed": fn in _NEW_FACTS_OK,
        },
        "response_generation": {
            "medium": "spoken" if voice else "chat",
            "tone": "calm_human_interviewer",
            "target_length": "brief" if (contextual or voice) else "short",
            "target_sentence_count": 1 if voice else (2 if contextual else 3),
            "content_specificity": "required" if contextual else "normal",
            "must_reference_candidate_content": contextual,
            "generic_acknowledgement_only": False,
            "do_not_overpraise": True,
            "do_not_take_over_candidate_reasoning": True,
        },
        "conversation_memory": {
            "recent_response_functions": list(sig.get("recent_functions") or []),
            "recent_interviewer_lines": [l for l in (recent_lines or [])][-3:],
            "repetition_avoidance": True,
            "candidate_plan": sig.get("candidate_plan") if fn == "ACKNOWLEDGE_AND_ORIENT" else None,
        },
        "alerts": alerts,
    }}


# --- V12 LAYER C: CONTEXTUAL LINE VALIDATION ----------------------------------

_CLAIM_RE = re.compile(
    r"(?i)\b(that'?s|that is|this is|you'?re|you are|it'?s|it is|looks|sounds|seems)\s+"
    r"(exactly\s+|absolutely\s+|completely\s+|totally\s+|quite\s+)?"
    r"(correct|right|accurate|spot[- ]on|on track|on the right track)\b"
    r"|\bcorrect(ly)?\b|\byou'?ve got it\b|\bthat checks out\b")
_LEAK_RE = re.compile(
    r"[{}]|<<|>>|response[_ ]function|control[_ ]packet|interviewer[_ ]control|"
    r"may[_ ]confirm|question[_ ]allowed|candidate[_ ]turn[_ ]type|"
    r"\b(ACKNOWLEDGE_AND_CONTINUE|ACKNOWLEDGE_AND_ORIENT|REFLECT_PROGRESS|VALIDATE_AND_HAND_BACK)\b")
_STOP = frozenset("""
a an the and or but so then that this these those there their they them you your yours we our
i i'm im me my it its it's is are was were be been being to of in on at for from with by as
into about over under just only also very really quite some any all each other than more most
first next now let lets let's will would can could should shall may might do does did done have
has had get got go going keep carry take make makes made think okay ok right alright sure yes
yeah well good great nice step steps part parts thing things way point
""".split())


def _content_keys(text: str) -> set:
    t = (text or "").lower().replace("’", "'")
    keys = set()
    for tok in re.findall(r"[a-z][a-z'\-]+|\d[\d,.]*", t):
        tok = tok.strip("'.,-")
        if not tok:
            continue
        if tok[0].isdigit():
            keys.add(tok.replace(",", "").rstrip("."))
        elif len(tok) >= 4 and tok not in _STOP:
            keys.add(tok[:5])
    return keys


def references_candidate(line: str, candidate_text: str) -> bool:
    """True when the line picks up something the candidate actually said (a number
    or a content word, compared on a 5-letter stem)."""
    return bool(_content_keys(line) & _content_keys(candidate_text))


def validate_contextual_line(text: str, candidate_text: str, channel: str = "text",
                             max_sentences: int = 2) -> Tuple[str, List[str]]:
    """Clean a model-worded presence line. Returns (line, problems).

    Problems: "leak" (control metadata in the text), "empty", "unreferenced"
    (nothing the candidate said is picked up). Questions, solicitations, gush and
    unverified correctness claims are removed, not just flagged: no check verified
    the candidate's work, so the line may not certify it.
    """
    t = (text or "").strip()
    if _LEAK_RE.search(t):
        return "", ["leak"]
    t = sanitize_reply(t)
    t = strip_solicitations(strip_all_questions(t)).strip()
    kept = [s for s in _sentences(t, keep_punct=True) if not _CLAIM_RE.search(s)]
    n = 1 if channel == "voice" else max(1, max_sentences)
    t = " ".join(kept[:n]).strip()
    if not t:
        return "", ["empty"]
    problems = [] if references_candidate(t, candidate_text) else ["unreferenced"]
    return t, problems


_JSON_TEXT_KEYS = ("reply", "text", "response", "line", "say", "message", "output", "interviewer")


def scrub_control_leak(text: str) -> str:
    """Deep-lane guard: the prompt now carries a JSON packet, so a model may answer
    in JSON or echo a field name. Recover the spoken line from a JSON reply, drop
    any sentence that carries control metadata; return "" if nothing is left."""
    t = (text or "").strip()
    if not t or not _LEAK_RE.search(t):
        return t
    body = re.sub(r"^```(?:json)?|```$", "", t).strip()
    if body.startswith("{"):
        try:
            obj = json.loads(body)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            for k in _JSON_TEXT_KEYS:
                v = obj.get(k)
                if isinstance(v, str) and v.strip() and not _LEAK_RE.search(v):
                    return v.strip()
        return ""
    kept = [x for x in _sentences(t, keep_punct=True) if not _LEAK_RE.search(x)]
    return " ".join(kept).strip()


class StreamLeakGuard:
    """Streams normally, unless the reply opens like JSON / a code fence: then it
    holds the whole reply and releases only the scrubbed line at the end."""

    def __init__(self):
        self.decided = False
        self.hold = False
        self.buf = ""

    def feed(self, piece: str):
        if self.decided and not self.hold:
            yield piece
            return
        self.buf += piece or ""
        head = self.buf.lstrip()
        if not head:
            return
        if not self.decided:
            if head.startswith("{") or head.startswith("`"):
                self.decided, self.hold = True, True
                return
            self.decided = True
            out, self.buf = self.buf, ""
            yield out

    def flush(self, fallback: str = "") -> str:
        if not self.hold:
            return ""
        return scrub_control_leak(self.buf) or fallback


# --- FAST LANE SEMANTIC EVENT GENERATOR --------------------------------------
# Only for turns that carried almost nothing (an affirmation, a bare number, a
# short "let me work it out"). Never used for substantive reasoning.

_FAST_PHRASES = {
    "SHORT_ACK": [
        {"code": "ACK_01", "text": "Got it."},
        {"code": "ACK_02", "text": "Alright."},
        {"code": "ACK_03", "text": "Okay."},
        {"code": "ACK_04", "text": "Right."}
    ],
    "HAND_BACK": [
        {"code": "HB_01", "text": "Go ahead."},
        {"code": "HB_02", "text": "Sure, continue."},
        {"code": "HB_03", "text": "Take it from there."},
        {"code": "HB_04", "text": "Alright, continue."},
        {"code": "HB_05", "text": "Makes sense. Go ahead."}
    ],
}
_FAST_ALIASES = {"ACKNOWLEDGE": "SHORT_ACK", "VALIDATE": "HAND_BACK"}

def get_fast_lane_event(mode: str, signals: Dict[str, Any]) -> Dict[str, Any]:
    key = _FAST_ALIASES.get(mode, mode)
    options = _FAST_PHRASES.get(key, _FAST_PHRASES["SHORT_ACK"])
    
    msg = signals.get("new_message_norm", "default")
    hash_val = int(hashlib.md5(msg.encode('utf-8')).hexdigest(), 16)
    
    recent_phrases = signals.get("recent_assistant_turns", [])
    recent_norm = [re.sub(r"[^\w\s]", "", t.lower().strip()) for t in recent_phrases]
    
    valid_options = []
    for opt in options:
        opt_norm = re.sub(r"[^\w\s]", "", opt["text"].lower().strip())
        if opt_norm not in recent_norm:
            valid_options.append(opt)
            
    if not valid_options:
        valid_options = options
        
    selection = valid_options[hash_val % len(valid_options)]
    
    return {
        "event_type": "interviewer_presence",
        "data": {
            "mode": mode,
            "code": selection["code"],
            "text": selection["text"]
        }
    }


# --- state update + guardrail telemetry --------------------------------------

_HINT_INTERVENTIONS = {"micro_hint", "reframe", "analogy", "decompose", "demonstrate", "reveal"}
_HINT_FLOOR = {"reframe": 1, "analogy": 1, "micro_hint": 2, "decompose": 3, "demonstrate": 4, "reveal": 5}

_BANNED_PHRASES = (
    "isn't specified", "isnt specified", "not specified", "isn't in the prompt",
    "not in the prompt", "isn't provided", "not provided", "isn't given", "not given",
    "i can't give you that", "i cannot give you that", "i won't provide", "i wont provide",
    "i cannot provide", "that detail isn't", "information isn't provided",
)
_RUBBER_STAMP = (
    "solid structure", "great job", "well done", "impressive", "well-structured",
    "well structured", "comprehensive", "excellent job",
)

def _infer_learner_level(prior_level, signals):
    tw = signals.get("turns_without_progress", 0)
    if tw >= 3:
        return "weak" if prior_level in ("developing", "weak") else "developing"
    if signals.get("has_work") and signals.get("frustration") == "none" and tw == 0:
        return "strong" if prior_level in ("unknown", "developing") else prior_level
    return prior_level if prior_level else "unknown"

def update_session_state(prior_state, tag, signals, function=None):
    ps = dict(prior_state or {})
    tag = tag or {}
    intervention = (tag.get("intervention") or "").strip().lower()
    prior_hint = int(ps.get("hint_level", 0) or 0)
    try:
        tag_hint = int(tag["hint"]) if tag.get("hint") not in (None, "") else None
    except (TypeError, ValueError, KeyError):
        tag_hint = None

    progressed = not (signals.get("help_requested") or signals.get("looks_garbage")
                      or signals.get("turns_without_progress", 0) >= 1)
    if intervention in _HINT_INTERVENTIONS:
        rung = tag_hint if tag_hint is not None else max(_HINT_FLOOR.get(intervention, 1), prior_hint + 1)
        hint_level = max(prior_hint, rung)
    elif progressed and prior_hint > 0:
        hint_level = prior_hint - 1
    else:
        hint_level = prior_hint
    hint_level = max(0, min(5, hint_level))

    ps.update({
        "mode": (tag.get("mode") or ps.get("mode") or "interviewer"),
        "hint_level": hint_level,
        "last_intervention": intervention or ps.get("last_intervention"),
        "repairs_done": int(ps.get("repairs_done", 0) or 0) + (1 if intervention == "repair" else 0),
        "frustration": signals.get("frustration", "none"),
        "consecutive_probes": signals.get("recent_probes", 0),
        "updated_turn": signals.get("candidate_turn_count", ps.get("updated_turn", 0)),
        "learner_level": _infer_learner_level(ps.get("learner_level", "unknown"), signals),
    })
    if function:
        # V12: the response function is the presence cool-down and the
        # repetition memory for the next turn (model-worded lines have no fixed text).
        recent = [f for f in (ps.get("recent_functions") or []) if isinstance(f, str)]
        ps["recent_functions"] = (recent + [function])[-4:]
        ps["last_function"] = function
        ps["function_turn"] = signals.get("candidate_turn_count", 0)
    outcome = evaluate_intervention_outcome(prior_state, signals)
    ps["last_intervention_effect"] = outcome
    ps["profile"] = update_learning_profile(
        ps.get("profile"), tag, outcome,
        hint_level=int((prior_state or {}).get("hint_level", 0) or 0),
        signals=signals, prior_modality=(prior_state or {}).get("last_intervention"),
    )
    return ps

def detect_violations(reply_text, policy="coached", tag=None):
    import re
    t = (reply_text or "").lower()
    out = []
    if any(p in t for p in _BANNED_PHRASES):
        out.append("banned_isnt_specified")
    if any(p in t for p in _RUBBER_STAMP):
        out.append("possible_rubber_stamp")
    sents = _sentences(reply_text, keep_punct=False)
    if len(sents) > 5:
        out.append("too_long")
    if (reply_text or "").count("?") >= 2:
        out.append("multiple_questions")
    interv = ((tag or {}).get("intervention") or "").strip().lower()
    looks_full_solution = (len(re.findall(r"\d", reply_text or "")) >= 4 and len(sents) >= 5
                           and ("=" in (reply_text or "") or "the answer is" in t or "the total is" in t))
    if interv in ("continue", "probe") and looks_full_solution:
        out.append("solved_when_declared_continue")
    if policy == "exam" and looks_full_solution:
        out.append("revealed_full_solution_in_exam")
    return out

_GUSH = (
    "great job", "great work", "great instinct", "great answer", "well done",
    "impressive", "brilliant", "awesome", "amazing", "excellent", "excellent job",
    "fantastic", "outstanding", "superb", "spot on", "nailed it", "perfect",
    "flawless", "masterful", "genius", "incredible", "love it", "really impressive",
    "solid structure", "well structured", "well-structured", "comprehensive",
    "you're on the right track", "youre on the right track", "phenomenal", "stellar",
    "you nailed", "beautifully done", "top notch", "top-notch",
)

def _is_gush(sentence: str) -> bool:
    s = sentence.lower().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    return any(g in s for g in _GUSH)

def sanitize_reply(text: str, policy: str = "coached") -> str:
    t = (text or "").strip()
    if not t:
        return t
    m = re.match(r"^\s*([^.!?\n]{0,90}?[.!?,])\s+(\S.*)$", t, re.DOTALL)
    if m and _is_gush(m.group(1)):
        t = m.group(2).strip()
    kept = [s for s in _sentences(t, keep_punct=True) if not _is_gush(s)]
    if kept:
        t = " ".join(kept).strip()
    if t.count("?") >= 2:
        t = t[: t.find("?") + 1].strip()
    return t

_SOLICIT_RE = re.compile(
    r"(?i)\b(please\s+)?(share|tell me|give me|walk me|provide|explain|describe|outline|"
    r"let me know|talk me through|elaborate)\b[^.!?]*\b"
    r"(thoughts?|final|recommendations?|insights?|feedback|answer|reasoning|next step)\b")

_MODE_FALLBACK = {
    "CLOSE": "Good — that's a reasonable place to close the case.",
    "DELIVER_SOLUTION": "In short: work the cost side first, then tie it back to the profit impact.",
    "DATA_REVEAL": "Carry on with your next step.",
    "CORRECT_MATERIAL": "Check the denominator.",
    "HINT": "Consider a different structural approach here.",
    "REPAIR": "Let's step back and look at the broader picture.",
    "ANSWER_DIRECT": "That is the standard assumption.",
    "TRANSITION": "Let's move on to the next phase."
}

def strip_all_questions(text: str) -> str:
    kept = [s for s in _sentences(text, keep_punct=True) if not s.rstrip().endswith("?")]
    return " ".join(kept).strip()

def strip_solicitations(text: str) -> str:
    kept = [s for s in _sentences(text, keep_punct=True) if not _SOLICIT_RE.search(s)]
    return " ".join(kept).strip()

def enforce_mode(text: str, mode: str, allow_questions: bool = True, policy: str = "coached") -> str:
    t = sanitize_reply(text, policy)
    if not allow_questions:
        t = strip_solicitations(strip_all_questions(t)).strip()
        if not t:
            t = _MODE_FALLBACK.get(mode, "Okay, continue.")
    return t.strip()
