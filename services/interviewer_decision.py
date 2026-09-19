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
    sents = [x for x in re.split(r"[.!?]+", reply_text or "") if x.strip()]
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


_PRAISE_OPENERS = (
    "good start", "great", "good question", "great question", "nice", "well done",
    "solid approach", "solid figure", "solid reasoning", "good plan", "perfect",
    "excellent", "impressive", "brilliant", "awesome", "that sounds",
    "that's a solid", "thats a solid", "that's a good", "thats a good",
    "that's a great", "thats a great", "that's a strong", "thats a strong",
    "that's a bold", "thats a bold", "that's a thoughtful", "thats a thoughtful",
    "that's an interesting", "thats an interesting", "that's interesting", "thats interesting",
    "you're on the right track", "youre on the right track", "you've got", "youve got",
    "that's a good start", "thats a good start", "solid start", "solid starting",
    "solid rationale", "sound direction", "sound approach", "good rationale",
)


def sanitize_reply(text: str, policy: str = "coached") -> str:
    """Deterministic guardrail behind the prompt: strip a leading praise opener and
    keep at most ONE question. Small models do not reliably obey these format rules
    even when told, so we ENFORCE them here (the prompt still asks first)."""
    t = (text or "").strip()
    if not t:
        return t
    m = re.match(r"^\s*([^.!?\n]{0,80}[.!?])\s+(\S.*)$", t, re.DOTALL)
    if m:
        first = m.group(1).lower().replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
        if any(pp in first for pp in _PRAISE_OPENERS):
            t = m.group(2).strip()
    if t.count("?") >= 2:
        t = t[: t.find("?") + 1].strip()
    return t
