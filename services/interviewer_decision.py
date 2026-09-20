"""
Control-tag parsing + streaming strip for the adaptive interviewer (Phase 1).

The adaptive prompt asks the model to prepend a one-line control tag, e.g.
    <<mode=coach; materiality=material; intervention=micro_hint; hint=1>>
which drives state/telemetry and must NEVER reach the candidate. This module
strips it -- from a full string (non-streaming) and from a token STREAM -- and
degrades safely if the model ignores the contract (the reply is never lost).
"""
from __future__ import annotations

import re
from typing import Dict, Generator, Tuple

from services.learning_model import evaluate_intervention_outcome, update_learning_profile

_TAG_RE = re.compile(r"^\s*<<(.*?)>>\s*", re.DOTALL)

_DEC = "\u0001"  # placeholder protecting decimal points from sentence splitting


def _sentences(text: str, keep_punct: bool = True):
    """Split into sentences WITHOUT breaking on decimal points (0.95, 5.5 crore).
    keep_punct=True keeps each sentence's trailing .!? (for question/solicitation
    stripping); False drops it (for counting)."""
    t = re.sub(r"(\d)\.(\d)", r"\1" + _DEC + r"\2", text or "")
    if keep_punct:
        parts = re.findall(r"[^.!?]*[.!?]|[^.!?]+$", t)
    else:
        parts = re.split(r"[.!?]+", t)
    return [p.replace(_DEC, ".").strip() for p in parts if p.strip()]


def parse_control_tag(text: str) -> Tuple[Dict[str, str], str]:
    """Return (tag_dict, cleaned_reply). Strips a leading <<...>> tag if present."""
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
    # If the model put its whole reply inside the tag, keep the original text.
    return (tag, cleaned) if cleaned else (tag, text.strip())


class StreamTagStripper:
    """Strip a leading <<...>> control tag from a token STREAM.

    Buffers only until it can decide whether a tag is present, then passes
    tokens through. If the first real chars are not '<<', or no closing '>>'
    arrives within `budget`, it flushes and assumes no tag -- a model that
    ignores the contract never loses its reply.
    """

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
            elif len(self.buf) > self.budget:      # never closed -> flush as-is
                out, self.buf, self.resolved = self.buf, "", True
                if out:
                    yield out
            return
        # first real chars are not '<<' -> there is no tag
        if len(stripped) >= 2 or len(self.buf) > 8:
            out, self.buf, self.resolved = self.buf, "", True
            if out:
                yield out

    def flush(self) -> Generator[str, None, None]:
        if not self.buf:
            return
        out, self.buf, self.resolved = self.buf, "", True
        m = _TAG_RE.match(out)          # strip a dangling, unterminated tag
        if m:
            out = out[m.end():].lstrip()
        if out:
            yield out


# --- state update + guardrail telemetry (Phase 2) ----------------------------
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
    """Fold the model's control tag + this turn's signals into the persisted state.
    Keeps the hint ladder monotonic within a stuck streak and steps it DOWN when the
    learner regains momentum (preserves productive struggle). Pure; caller persists it."""
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
    # Phase 4: fold this turn into the learner model (skill/error/modality + outcome).
    outcome = evaluate_intervention_outcome(prior_state, signals)
    ps["last_intervention_effect"] = outcome
    ps["profile"] = update_learning_profile(
        ps.get("profile"), tag, outcome,
        hint_level=int((prior_state or {}).get("hint_level", 0) or 0),
        signals=signals, prior_modality=(prior_state or {}).get("last_intervention"),
    )
    return ps


def detect_violations(reply_text, policy="coached", tag=None):
    """Deterministic invariant checks on the model's reply, for telemetry + evals.
    Streaming means we don't rewrite the live reply; we RECORD violations so the eval
    harness and metrics catch a disobedient model (and flag the case for review)."""
    import re
    t = (reply_text or "").lower()
    out = []
    if any(p in t for p in _BANNED_PHRASES):
        out.append("banned_isnt_specified")
    if any(p in t for p in _RUBBER_STAMP):
        out.append("possible_rubber_stamp")
    sents = _sentences(reply_text, keep_punct=False)   # decimal-safe: 0.95 is not 2 sentences
    if len(sents) > 5:
        out.append("too_long")
    # Behavioral: DECLARED action (tag) vs ACTUAL reply (points 20, 21). Textual
    # bans catch phrasing; these catch behaviour the model narrated but didn't do.
    if (reply_text or "").count("?") >= 2:
        out.append("multiple_questions")   # a good interviewer asks ONE
    interv = ((tag or {}).get("intervention") or "").strip().lower()
    looks_full_solution = (len(re.findall(r"\d", reply_text or "")) >= 4 and len(sents) >= 5
                           and ("=" in (reply_text or "") or "the answer is" in t or "the total is" in t))
    if interv in ("continue", "probe") and looks_full_solution:
        out.append("solved_when_declared_continue")   # solved it while claiming to only nudge
    if policy == "exam" and looks_full_solution:
        out.append("revealed_full_solution_in_exam")
    return out


# GUSH = effusive / rubber-stamp praise ONLY. A brief earned "Good", "Fair point",
# "That's a good point" is natural and matches the real casebook transcripts -- we do NOT
# strip that. We strip only over-the-top praise, and praise of weak work (rubber-stamp).
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
    """Deterministic guardrail behind the prompt: strip GUSH (effusive / rubber-stamp
    praise) while keeping brief earned acknowledgement, and keep at most ONE question.
    Small models over-praise even when told not to, so we ENFORCE it here."""
    t = (text or "").strip()
    if not t:
        return t
    # 1) strip a leading gush clause ("Great job, ..." / "Well done. ...")
    m = re.match(r"^\s*([^.!?\n]{0,90}?[.!?,])\s+(\S.*)$", t, re.DOTALL)
    if m and _is_gush(m.group(1)):
        t = m.group(2).strip()
    # 2) drop any remaining standalone gush sentence ("Well done." / "Impressive!")
    kept = [s for s in _sentences(t, keep_punct=True) if not _is_gush(s)]
    if kept:
        t = " ".join(kept).strip()
    # 3) at most one question
    if t.count("?") >= 2:
        t = t[: t.find("?") + 1].strip()
    return t


# --- mode-aware enforcement (Phase: intervention necessity) ------------------
# Some interviewer moves must ask NOTHING: CLOSE (take the recommendation and end),
# HOLD_SPACE (let them think), RELEASE (they recovered -> step back), and an EARNED
# DELIVER_SOLUTION (give the answer, don't deflect). The prompt asks the model to do
# this; the gate GUARANTEES it, because small models still tack on a question. A
# question hidden as an imperative ("please share your final thoughts") slips past a
# '?'-only strip, so we also drop trailing solicitations.

_ZERO_Q_MODES = {"CLOSE", "HOLD_SPACE", "RELEASE", "DELIVER_SOLUTION"}

_SOLICIT_RE = re.compile(
    r"(?i)\b(please\s+)?(share|tell me|give me|walk me|provide|explain|describe|outline|"
    r"let me know|talk me through|elaborate)\b[^.!?]*\b"
    r"(thoughts?|final|recommendations?|insights?|feedback|answer|reasoning|next step)\b")

_MODE_FALLBACK = {
    "CLOSE": "Good — that's a reasonable place to close the case.",
    "HOLD_SPACE": "Take your time.",
    "RELEASE": "Exactly — run with that.",
    "DELIVER_SOLUTION": "In short: work the cost side first, then tie it back to the profit impact.",
}


def strip_all_questions(text: str) -> str:
    """Drop every sentence that ends in '?'."""
    kept = [s for s in _sentences(text, keep_punct=True) if not s.rstrip().endswith("?")]
    return " ".join(kept).strip()


def strip_solicitations(text: str) -> str:
    """Drop imperative asks ('please share your final thoughts') that dodge the '?' strip."""
    kept = [s for s in _sentences(text, keep_punct=True) if not _SOLICIT_RE.search(s)]
    return " ".join(kept).strip()


def enforce_mode(text: str, mode: str, q_budget=None, policy: str = "coached") -> str:
    """Apply the base guardrail (praise strip + 1-question cap) and, for a zero-budget
    mode, strip ALL questions and solicitations. Falls back to a short mode-appropriate
    line if nothing survives (never returns empty)."""
    t = sanitize_reply(text, policy)
    budget = q_budget if q_budget is not None else (0 if mode in _ZERO_Q_MODES else 1)
    if budget == 0:
        t = strip_solicitations(strip_all_questions(t)).strip()
        if not t:
            t = _MODE_FALLBACK.get(mode, "Let's move on.")
    return t.strip()
