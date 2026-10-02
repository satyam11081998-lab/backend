"""Keeping the interviewer honest about the conversation.

Two failure modes testers hit with real models:
  * a follow-up that refers to something the candidate never said ("What was the segment about?"
    when no segment was mentioned) — usually a follow-up written into the plan BEFORE the interview,
    or a detail picked up from the CV or from a model summary;
  * the same question asked twice in different words.

Both are checked deterministically on every line before it is spoken. A line that fails is not
spoken; the interviewer falls back to the planned question or a neutral follow-up instead.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set

_WORD = re.compile(r"[a-z0-9][a-z0-9%'&.-]*")

STOP = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with", "at", "by", "from", "as", "is",
    "are", "was", "were", "be", "been", "being", "it", "its", "that", "this", "these", "those", "your", "you",
    "i", "we", "our", "my", "me", "us", "they", "them", "their", "he", "she", "his", "her", "how", "what", "would",
    "do", "did", "does", "done", "can", "could", "should", "will", "about", "when", "which", "why", "who", "whom",
    "where", "there", "here", "if", "so", "than", "then", "into", "onto", "over", "under", "up", "down", "out",
    "off", "just", "also", "very", "really", "more", "much", "many", "some", "any", "all", "each", "every", "not",
    "no", "yes", "have", "has", "had", "get", "got", "make", "made", "take", "took", "go", "went", "come", "came",
    "say", "said", "see", "saw", "know", "knew", "think", "thought", "want", "like", "need", "use", "used", "let",
    "lets", "let's", "now", "again", "still", "even", "only", "too", "such", "own", "while", "during", "after",
    "before", "since", "until", "because", "though", "although", "whether", "within", "without", "between",
    "across", "around", "behind", "against", "toward", "towards", "upon", "per", "via", "amid", "once",
}

# Words an interviewer may use for whatever the candidate described, without presupposing a fact.
GENERIC = {
    "question", "questions", "answer", "answers", "example", "examples", "instance", "situation", "scenario",
    "case", "story", "project", "projects", "role", "roles", "team", "teams", "company", "organisation",
    "organization", "business", "decision", "decisions", "result", "results", "outcome", "outcomes", "impact",
    "effect", "approach", "process", "problem", "problems", "challenge", "challenges", "goal", "goals", "target",
    "targets", "metric", "metrics", "number", "numbers", "data", "evidence", "baseline", "timeline", "deadline",
    "budget", "customer", "customers", "client", "clients", "user", "users", "stakeholder", "stakeholders",
    "manager", "leadership", "leader", "experience", "cv", "resume", "background", "career", "context", "plan",
    "plans", "strategy", "idea", "ideas", "change", "changes", "issue", "issues", "risk", "risks", "trade-off",
    "tradeoff", "trade-offs", "tradeoffs", "alternative", "alternatives", "option", "options", "reason", "reasons",
    "reasoning", "part", "contribution", "work", "task", "tasks", "time", "end", "start", "beginning", "first",
    "last", "next", "main", "biggest", "hardest", "key", "most", "same", "other", "others", "whole", "overall",
    "specific", "specifics", "exact", "right", "best", "worst", "interview", "rest", "point", "step", "steps", "way",
    "ways", "thing", "things", "constraint", "constraints", "resource", "resources", "cost", "costs", "feedback",
    "conflict", "disagreement", "market", "product", "service", "industry", "sector", "function", "department",
    "people", "person", "individual", "group", "colleague", "colleagues", "peer", "peers", "senior", "junior",
    "panel", "previous", "current", "earlier", "later", "initial", "final", "original", "detail", "details",
    "level", "deeper", "one", "two", "three", "few", "difference", "situation's", "assumption", "assumptions",
    "lesson", "lessons", "learning", "learnings", "mistake", "mistakes", "success", "failure", "measure",
    "measures", "measurement", "priority", "priorities", "pressure", "role's", "team's", "company's", "position",
    "job", "opportunity", "skills", "skill", "strengths", "strength", "weakness", "weaknesses", "motivation",
    "area", "areas", "topic", "topics", "scope", "size", "scale", "period", "phase", "stage", "year", "years",
    "month", "months", "week", "weeks", "day", "days", "quarter", "quarters", "percent", "percentage", "share",
    "growth", "revenue", "sales", "volume", "profit", "margin", "money", "spend", "investment", "return",
    "framework", "model", "analysis", "approach's", "thinking", "logic", "structure", "method", "methods",
    "tools", "tool", "system", "systems", "solution", "solutions", "recommendation", "recommendations",
    "personal", "personally", "new", "old", "big", "small", "large", "good", "bad", "high", "low", "major",
    "minor", "early", "late", "different", "similar", "additional", "extra", "full", "actual", "real", "broader",
    "wider", "long", "short", "entire", "important", "difficult", "hard", "easy", "simple", "complex", "core",
}

# Interview filler that says nothing about the subject of a question.
QUESTION_WORDS = {
    "tell", "describe", "walk", "through", "explain", "give", "share", "talk", "example", "examples", "instance",
    "time", "times", "situation", "specific", "one", "please", "briefly", "little", "bit", "able", "kind",
    "sort", "something", "anything", "someone", "anyone", "particular", "recent", "recently", "ever", "had",
    "handled", "handle", "faced", "face", "dealt", "deal",
}

# "you mentioned the pricing change", "earlier you said ..." — presuppose the candidate said it.
_SAID = re.compile(r"\byou(?:'ve| have)? (?:mentioned|said|talked about|described|spoke about|brought up|noted|"
                   r"referred to|told me about)\s+(?:that\s+|how\s+|the\s+|your\s+|a\s+|an\s+)?((?:[\w%'&.-]+\s*){1,5})",
                   re.IGNORECASE)
# "the" / "your" + at most two words: an adjective and the noun it presents as already known.
_DEF = re.compile(r"\b(?:the|your)\s+((?:[\w%'&.-]+\s+)?[\w%'&.-]+)", re.IGNORECASE)
_NUM = re.compile(r"\b\d[\d,.]*\s*(?:%|x|k|m|cr|crore|lakh|lakhs|million|billion|percent)?", re.IGNORECASE)


def _stem(w: str) -> str:
    w = w.lower().strip("'.-&")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w[:6]


def _words(text: str) -> List[str]:
    out: List[str] = []
    for tok in _WORD.findall((text or "").lower()):
        out.extend(w for w in (p.strip(".'&-") for p in re.split(r"[-/]", tok)) if w)
    return out


def _stems(text: str) -> Set[str]:
    return {_stem(w) for w in _words(text) if w not in STOP}


def _content(phrase: str) -> List[str]:
    out = []
    for w in _words(phrase):
        if w in STOP or w in GENERIC or len(w) < 4 or w.isdigit():
            continue
        out.append(w)
    return out


def _nums(text: str) -> Set[str]:
    return {re.sub(r"[^\d.]", "", m.group(0)).rstrip(".") for m in _NUM.finditer(text or "")} - {""}


def ungrounded(line: str, sources: Iterable[str]) -> List[str]:
    """Phrases in `line` that present something as already said ("the segment", "you mentioned the
    pilot", a number) when nothing in `sources` (the question asked, the candidate's own words, the
    CV claim under discussion) contains it. Empty list = the line is grounded."""
    src_text = " \n ".join(s for s in sources if s)
    src = _stems(src_text)
    src_nums_all = _nums(src_text)
    problems: List[str] = []
    seen: Set[str] = set()
    for rx in (_SAID, _DEF):
        for m in rx.finditer(line or ""):
            phrase = m.group(1).strip()
            words = _content(phrase.split(" and ")[0])
            if not words:
                continue
            if any(_stem(w) in src for w in words) or (_nums(phrase) & src_nums_all):
                continue
            key = " ".join(words)
            if key not in seen:
                seen.add(key)
                problems.append(phrase.strip(" ,.?!"))
    for n in _nums(line):
        if n and n not in src_nums_all and n not in seen:
            seen.add(n)
            problems.append(n)
    return problems


def _subject(text: str) -> Set[str]:
    return {_stem(w) for w in _words(text) if w not in STOP and w not in QUESTION_WORDS and len(w) > 2}


def overlap(a: str, b: str) -> float:
    """How much two questions ask about the same thing (share of the shorter one's subject words)."""
    sa, sb = _subject(a), _subject(b)
    if len(sa) < 3 or len(sb) < 3:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def same_question(a: str, b: str, *, threshold: float = 0.6) -> bool:
    sa, sb = _subject(a), _subject(b)
    return len(sa & sb) >= 3 and overlap(a, b) >= threshold
