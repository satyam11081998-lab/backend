# ============================================================================
# MECE PREP COPILOT - ISOLATED COPY. Do NOT sync with the original.
# Copied services/interviewer_decision.py on 2026-09-23 for the role/company-aware Prep Copilot (v2).
# Tweak freely here; the LIVE cases/guesstimates engine is the ORIGINAL and is
# never imported from this package. See .brain/handoffs/ANTIGRAVITY_HANDOFF_prep-copilot-v2.md
# ============================================================================
"""
Control-tag parsing, streaming strip, Contextual Assessor, and Intervention Gate.
Implements the v0.51 MECE architecture separating Substantive Gates from Conversational Presence.
HARD RULE APPLIED: Zero blank responses. All turns yield conversational presence.
"""
from __future__ import annotations

import re
import json
from typing import Dict, Generator, Tuple, Any

from services.copilot.engine.learning_model import evaluate_intervention_outcome, update_learning_profile
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


# --- INTERVENTION GATE -------------------------------------------------------

def evaluate_intervention_gate(signals: Dict[str, Any]) -> Tuple[bool, str, str]:
    """
    Returns (intervention_required, recommended_modality, reason).
    HARD RULE APPLIED: Zero blank responses allowed. NO_INTERVENTION is removed.
    All non-substantive turns route to a conversational presence mode.
    """
    if not signals:
        return True, "PROBE", "no_signals"

    # --- STAGE 1: SUBSTANTIVE INTERVENTION GATE ---
    
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
            
    # --- STAGE 2: CONVERSATIONAL PRESENCE GATE ---
    # Hard Rule Applied: Never any blank responses.
    # We must return LISTENING_BEAT or HAND_BACK for all remaining healthy cases.
    
    if signals.get("hedged_self_estimate") or "?" in signals.get("new_message_norm", ""):
        return True, "HAND_BACK", "return_floor_explicitly"
        
    if signals.get("candidate_working") or signals.get("wants_space"):
        return True, "HAND_BACK", "candidate_needs_space"
        
    # Default for progressing math, valid structure, and fragmented reasoning:
    return True, "LISTENING_BEAT", "conversational_presence"


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


def update_session_state(prior_state, tag, signals):
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
    "HAND_BACK": "Go ahead.",
    "LISTENING_BEAT": "Got it.",
    "DELIVER_SOLUTION": "In short: work the cost side first, then tie it back to the profit impact.",
    "DATA_REVEAL": "The data confirms that.",
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
