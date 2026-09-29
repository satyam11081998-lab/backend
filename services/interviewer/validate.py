"""
Output validation for model-worded interviewer lines.

Enforced on every generated line (non-stream and stream):
  * no markdown / list formatting,
  * no praise or rubber-stamping ("Great", "Excellent", "Solid structure", ...),
  * no refusals or interrogation loops ("That's the exercise", "What's your next
    step?", "No hints", "Think harder", "isn't specified", ...),
  * no internal labels or instruction text (MOVE:, TASK:, hint level, ...),
  * question budget (0 or 1), no generic questions, no repeated questions,
  * sentence cap per move and channel.

The validator REMOVES sentences; it never invents content. If nothing survives,
the caller gets EmptyModelOutput - never a canned stand-in answer.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

_ABBREV = {"e.g", "i.e", "rs", "approx", "vs", "etc", "mr", "mrs", "dr", "no", "fig", "est", "inc", "ltd", "co", "cf", "u.s"}

_PRAISE_SENT = re.compile(
    r"^\s*(great|excellent|brilliant|awesome|amazing|fantastic|perfect|wonderful|superb|outstanding|impressive|"
    r"nice|good job|great job|well done|nicely done|love it|spot on|nailed it|good start|good question|"
    r"great question|good thinking|great thinking|exactly right|that'?s exactly right|solid (structure|start|approach|work)|"
    r"that'?s (a )?(great|good|solid|excellent|brilliant|nice|perfect) (structure|start|approach|point|thought|question|answer|framework)|"
    r"you'?re on the right track|good point|great point)\b[\s!.,:;-]*",
    re.IGNORECASE)
_PRAISE_ANY = re.compile(
    r"\b(excellent|brilliant|awesome|amazing|fantastic|wonderful|superb|outstanding|impressive|great job|well done|"
    r"nailed it|spot on|solid structure|great structure|perfect structure|great thinking|excellent thinking|"
    r"great question|good question)\b", re.IGNORECASE)

_BANNED = [
    r"that'?s the exercise", r"that is the exercise", r"what'?s your next step", r"what is your next step",
    r"(i )?(can'?t|cannot|won'?t|will not|am not able to|'m not able to) (provide|give|offer|share)( you)?( any| a)? (hints?|the answer|the solution|solutions|help)",
    r"\bno hints\b", r"\bno solutions\b", r"think harder", r"try harder", r"figure (it|that) out (yourself|on your own)",
    r"that'?s for you to (figure|work) out", r"that'?s what i'?m here to (see|watch|test)",
    r"i'?m not (able|allowed) to help",
    r"\b(isn'?t|is not|wasn'?t|was not|not) (specified|provided|given|mentioned|stated) (in|by) (the )?(case|prompt|problem|question|brief)\b",
    r"^\s*(that|this|it)('s| is| isn'?t)? ?(not )?(specified|provided|given)\s*[.!]?$",
    r"\b(isn'?t|is not|not) specified\b", r"\bnot in the (prompt|case text)\b",
    r"i don'?t have (that|the|this) (information|data|number)", r"as an ai( language model)?",
    r"i'?m (a|an) (human|real person)",
]
_BANNED_RE = re.compile("|".join(f"(?:{b})" for b in _BANNED), re.IGNORECASE)

_LEAK_RE = re.compile(
    r"\b(MOVE|TASK|RULES|CONTEXT)\s*:|\b(MICRO_HINT|TARGETED_HINT|STRUCTURAL_HINT|DELIVER_SOLUTION|DATA_REVEAL|"
    r"DIRECT_CORRECTION|ANSWER_DIRECT|TARGETED_PROBE|RETHINK_CUE|HAND_BACK|NO_OUTPUT|ACKNOWLEDGE|REPAIR)\b|"
    r"assistance level|hint[_ ]level|system prompt|my instructions|these instructions|the application (has|decided)|"
    r"\bcandidate=|\bstage=", re.IGNORECASE)

_GENERIC_Q = re.compile(
    r"^\s*(and )?(what else|anything else|why|any other (factors|thoughts|ideas)|what do you think|thoughts|"
    r"what would you do|what'?s next|what now|how would you proceed|how do you want to proceed)\s*\?\s*$",
    re.IGNORECASE)

_MD = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"), (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"(?m)^\s*#{1,6}\s*"), ""), (re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+"), ""),
    (re.compile(r"`([^`]*)`"), r"\1"), (re.compile(r"\*(\S[^*]*?)\*"), r"\1"),
]


def strip_markdown(text: str) -> str:
    t = text or ""
    for rx, rep in _MD:
        t = rx.sub(rep, t)
    t = t.replace("*", "")
    return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", " ", t)).strip()


def split_sentences(text: str) -> List[str]:
    out: List[str] = []
    buf = ""
    t = text or ""
    i = 0
    while i < len(t):
        ch = t[i]
        buf += ch
        if ch in ".!?":
            nxt = t[i + 1] if i + 1 < len(t) else ""
            if nxt == "" or nxt.isspace() or nxt in "\"')":
                word = re.findall(r"([A-Za-z.]+)\.$", buf)
                if ch == "." and word and word[0].lower().rstrip(".") in _ABBREV:
                    i += 1
                    continue
                while i + 1 < len(t) and t[i + 1] in "\"')":
                    i += 1
                    buf += t[i]
                out.append(buf.strip())
                buf = ""
        i += 1
    if buf.strip():
        out.append(buf.strip())
    return [s for s in out if s]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower()).strip()


@dataclass
class ValidationResult:
    text: str
    violations: List[str] = field(default_factory=list)
    dropped: int = 0
    questions: int = 0

    @property
    def empty(self) -> bool:
        return not self.text.strip()


class SentenceFilter:
    """Stateful per-sentence filter shared by the stream gate and the batch validator."""

    def __init__(self, max_questions: int, max_sentences: int, recent_lines: Iterable[str] = (),
                 recent_questions: Iterable[str] = ()):
        self.max_questions = max_questions
        self.max_sentences = max_sentences
        self.recent_lines = [_norm(x) for x in recent_lines if x]
        self.recent_questions = [_norm(x) for x in recent_questions if x]
        self.kept: List[str] = []
        self.questions = 0
        self.violations: List[str] = []
        self.dropped = 0
        self.closed = False  # a question was asked; nothing may follow it

    def _drop(self, why: str):
        self.dropped += 1
        if why not in self.violations:
            self.violations.append(why)
        return None

    def accept(self, sentence: str) -> Optional[str]:
        if self.closed or len(self.kept) >= self.max_sentences:
            return self._drop("over_length" if not self.closed else "after_question")
        s = strip_markdown(sentence)
        if not s:
            return None
        if _LEAK_RE.search(s):
            return self._drop("leak")
        if _BANNED_RE.search(s):
            return self._drop("banned_phrase")
        m = _PRAISE_SENT.match(s)
        if m:
            rest = s[m.end():].strip()
            if not rest or len(re.findall(r"\w+", rest)) < 2:
                return self._drop("praise")
            s = rest[0].upper() + rest[1:]
            if "praise_opener" not in self.violations:
                self.violations.append("praise_opener")
        if _PRAISE_ANY.search(s):
            return self._drop("praise")
        n_q = s.count("?")
        is_q = n_q > 0
        if is_q:
            if self.questions + n_q > self.max_questions:
                return self._drop("question_budget")
            if _GENERIC_Q.match(s):
                return self._drop("generic_question")
            n = _norm(s)
            if any(difflib.SequenceMatcher(None, n, q).ratio() >= 0.85 for q in self.recent_questions):
                return self._drop("repeated_question")
        n = _norm(s)
        if n and n in self.recent_lines and "repeated_line" not in self.violations:
            # Soft: a verbatim repeat of a recent STATEMENT is recorded (telemetry) but kept -
            # repeating a hint is better than failing the turn. Repeated QUESTIONS are dropped above.
            self.violations.append("repeated_line")
        self.kept.append(s)
        if is_q:
            self.questions += n_q
            self.closed = True
        return s

    def result(self) -> ValidationResult:
        return ValidationResult(text=" ".join(self.kept).strip(), violations=list(self.violations),
                                dropped=self.dropped, questions=self.questions)


def validate_text(text: str, *, max_questions: int, max_sentences: int, recent_lines: Iterable[str] = (),
                  recent_questions: Iterable[str] = ()) -> ValidationResult:
    f = SentenceFilter(max_questions, max_sentences, recent_lines, recent_questions)
    for s in split_sentences(strip_markdown(text)):
        f.accept(s)
    return f.result()


class SentenceGate:
    """Streaming: buffer tokens, emit each sentence only after it passes the filter."""

    def __init__(self, sentence_filter: SentenceFilter):
        self.f = sentence_filter
        self.buf = ""

    def feed(self, token: str) -> List[str]:
        self.buf += token or ""
        out: List[str] = []
        while True:
            m = self._boundary()
            if m is None:
                break
            sent, self.buf = self.buf[:m], self.buf[m:]
            kept = self.f.accept(sent)
            if kept:
                out.append(kept)
        return out

    def _boundary(self) -> Optional[int]:
        for i, ch in enumerate(self.buf):
            if ch in ".!?" and i + 1 < len(self.buf) and self.buf[i + 1].isspace():
                word = re.findall(r"([A-Za-z.]+)\.$", self.buf[: i + 1])
                if ch == "." and word and word[0].lower().rstrip(".") in _ABBREV:
                    continue
                return i + 1
        return None

    def flush(self) -> List[str]:
        rest, self.buf = self.buf, ""
        out = []
        for s in split_sentences(rest):
            kept = self.f.accept(s)
            if kept:
                out.append(kept)
        return out
