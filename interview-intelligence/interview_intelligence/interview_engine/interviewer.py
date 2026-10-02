"""The interviewer's voice. Turns a decided Action into one natural spoken line.

The model only phrases; WHAT happens was decided by the policy. Every output passes an
evaluation-leak filter, and every action has a deterministic fallback line so the
interview never stalls on a model failure (spec §30, §57, §78)."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import List, Optional, Sequence

from ..ai.guard import wrap_untrusted
from ..ai.prompts import INTERVIEWER
from ..ai.runner import RunContext, run_text
from ..textutil import truncate
from .grounding import same_question, ungrounded
from .policy import Action

FOCUS_FALLBACK = {
    "specificity": "Can you make that concrete — what exactly happened, with a specific example?",
    "ownership": "What was your personal part in that, as distinct from the team's?",
    "reasoning": "Why did you choose that approach over the alternatives?",
    "outcome": "What was the result, and how did you measure it?",
    "quantification": "Can you put a number on that — what was the baseline, and what changed?",
    "reflection": "Looking back, what would you do differently?",
    "tradeoff": "What did you give up by doing it that way?",
    "depth": "Take me one level deeper — how does that actually work?",
    "example": "Can you walk me through one specific instance?",
}
TRANSITIONS = {
    "cv": "Let's talk about your experience. ",
    "personal": "Let me ask a little about you beyond the CV. ",
    "functional": "Let me ask about the functional side of the role. ",
    "technical": "Let's get into some technical questions. ",
    "behavioral": "Let's switch to some behavioural questions. ",
    "situational": "Let me give you a scenario. ",
    "case": "Let's work through a short problem. ",
    "company": "Let's talk about the company and the role. ",
    "awareness": "Let's talk about what's happening in business right now. ",
    "motivation": "A couple of questions about what you're looking for. ",
}

# How the interviewer announces each part of the plan at the start ({m} = " for about N minutes").
AGENDA_PHRASES = {
    "cv": "go through your CV{m}",
    "personal": "talk a little about you beyond the CV",
    "functional": "get into the functional side of the role{m}",
    "technical": "get into some technical questions{m}",
    "case": "work through a short problem together{m}",
    "company": "talk about the company and the role{m}",
    "awareness": "discuss what's happening in business right now",
    "behavioral": "switch to behavioural questions{m}",
    "situational": "go through a few real-world scenarios{m}",
    "motivation": "talk about what you're looking for",
}
MODE_NAMES = {
    "grill": "grill mode", "mixed": "a mixed interview", "final_round": "a final-round interview",
    "cv_attack": "a CV attack-and-defence interview", "hr_behavioral": "an HR and behavioural interview",
    "case": "a case interview", "cv_jd": "an interview on your CV against this role",
    "cv_deep_dive": "a CV deep dive", "stress": "a pressure interview", "hiring_manager": "a hiring-manager interview",
    "company_simulation": "a company and role simulation", "weakness_targeting": "a focused re-attempt on the areas "
    "your last interview flagged", "technical_deep_dive": "a technical deep dive",
}
MODE_TONE = {
    "grill": "Expect me to push for specifics, numbers and exactly what you did.",
    "stress": "I'll add some pressure along the way; that's part of the exercise.",
    "cv_attack": "I'll test the main claims on your CV one by one.",
}


def agenda_sentence(agenda: Sequence[dict]) -> str:
    """'We'll start with a quick introduction, then go through your CV for about 10 minutes, ...'"""
    parts = []
    for a in agenda:
        phrase = AGENDA_PHRASES.get(a.get("kind", ""))
        if not phrase:
            continue
        mins = int(a.get("minutes") or 0)
        parts.append(phrase.format(m=f" for about {mins} minutes" if mins >= 4 else ""))
    if not parts:
        return ""
    if len(parts) == 1:
        body = parts[0]
    else:
        linked = [parts[0]]
        for i, p in enumerate(parts[1:-1], start=1):
            linked.append(("then " if i % 2 == 0 else "") + p)
        body = ", ".join(linked) + ", and " + parts[-1]
    return f"We'll start with a quick introduction, then {body}. I'll keep a few minutes at the end for your questions."


def opening_line(bp: dict, question_text: str) -> str:
    """The first thing the candidate hears: who is speaking, what kind of interview this is, how
    long, the running order, then the first question. Deterministic — built from the real plan,
    so it can never announce a part that is not there."""
    cfg = bp.get("config") or {}
    role_title = (bp.get("role") or {}).get("title", "")
    minutes = int(cfg.get("duration_minutes", 45))
    mode = cfg.get("mode", "mixed")
    kind = MODE_NAMES.get(mode, "a practice interview")
    role = f"the {role_title} role" if role_title else "this role"
    lines = [f"Hi, I'm your MECE interviewer. This is {kind}, about {minutes} minutes, for {role}."]
    tone = MODE_TONE.get(mode) or ("Take your time; there are no trick questions." if cfg.get("difficulty") == "easy"
                                    else "")
    if tone:
        lines.append(tone)
    agenda = agenda_sentence(bp.get("agenda") or [
        {"kind": x["kind"], "minutes": round(int(x.get("budget_s", 0)) / 60)} for x in bp.get("sections", [])
        if x.get("kind") not in ("intro", "closing") and x.get("items")])
    if agenda:
        lines.append(agenda)
    lines.append(question_text)
    return " ".join(lines)
CHALLENGE_FALLBACK = [
    "I'm not convinced yet. What evidence do you have that this was the right call?",
    "Suppose you only had two weeks instead. What changes in your approach?",
    "If you could only do one of those things, which would it be, and why?",
]
OPENER_RX = re.compile(r"^\s*(great|good|excellent|perfect|nice|awesome|wonderful|fantastic|brilliant|impressive|"
                       r"that'?s (a )?(great|good|excellent|perfect|nice|right|correct|interesting)|well done|"
                       r"exactly|correct|absolutely|love (that|it)|good (answer|point|question))\b[^.?!]*[.!,]?\s*",
                       re.IGNORECASE)
EVAL_RX = re.compile(r"\b(great answer|good answer|excellent answer|well answered|you('| a)re (doing )?(well|great)|"
                     r"that('?s| is) (correct|right|wrong|incorrect)|your score|you scored|strong answer|weak answer)\b",
                     re.IGNORECASE)
MD_RX = re.compile(r"[*_#`>]|^\s*[-•]\s+", re.MULTILINE)

NEEDS_QUESTION = {"OPEN", "ASK", "PROBE", "CLARIFY_CONTRADICTION", "CHALLENGE", "REPEAT", "NUDGE", "CLOSE_INVITE",
                  "REDIRECT", "CLARIFY_QUESTION"}
# Lines that must stay grounded in what was actually said (closing / control lines carry no claims).
GROUNDED = {"OPEN", "ASK", "PROBE", "CHALLENGE", "CLARIFY_QUESTION", "REPEAT", "NUDGE", "REDIRECT"}


def _clean(text: str) -> str:
    t = (text or "").strip().strip('"').strip()
    t = MD_RX.sub("", t)
    for _ in range(2):
        t = OPENER_RX.sub("", t).lstrip()
    t = re.sub(r"\s+", " ", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return truncate(t, 900)


def acceptable(action: Action, text: str) -> bool:
    if not text or len(text) < 8:
        return False
    if EVAL_RX.search(text):
        return False
    if action.type in NEEDS_QUESTION and "?" not in text:
        return False
    return True


def fallback(action: Action, *, question_text: str, role_title: str, minutes: int, contradiction: str = "",
             closing_reply: str = "") -> str:
    t = action.type
    if t == "OPEN":
        role = f"the {role_title} role" if role_title else "this role"
        return (f"Hi, I'm your MECE interviewer. This will be about {minutes} minutes for {role}, and I'll ask "
                f"follow-ups as we go. {question_text}")
    if t == "ASK":
        pre = {"ack_refusal": "That's fine, we can leave that one. ", None: ""}.get(action.preface, "")
        if action.new_section:
            pre += TRANSITIONS.get(action.new_section, "Let's move to a different area. ")
        return pre + question_text
    if t == "PROBE":
        return action.probe_text if action.probe_text and "?" in action.probe_text else FOCUS_FALLBACK.get(
            action.focus or "specificity", FOCUS_FALLBACK["specificity"])
    if t == "CLARIFY_CONTRADICTION":
        return f"I want to clarify something. {contradiction} Can you help me reconcile those?"
    if t == "CHALLENGE":
        return CHALLENGE_FALLBACK[len(question_text or "") % len(CHALLENGE_FALLBACK)]
    if t == "REPEAT":
        return f"Sure. {question_text}"
    if t == "CLARIFY_QUESTION":
        return f"Fair question. Any example that fits works — what I'm asking is: {question_text}"
    if t == "WAIT":
        return "Of course, take your time."
    if t == "NUDGE":
        return "Could you say a bit more? A specific example would help."
    if t == "REDIRECT":
        if action.preface == "meta":
            return ("I won't share how it's going during the interview — you'll get a detailed report at the end. "
                    f"Coming back to the question: {question_text}")
        # No "Good ..." opener: the evaluation-leak filter strips praise-like sentences.
        return f"Let's keep our focus on the interview for now. {question_text}"
    if t == "PAUSE":
        return "Sure, let's take a break. Resume whenever you're ready and we'll pick up where we left off."
    if t == "CLOSE_INVITE":
        return "We're coming to the end of our time. Before we wrap up, what questions do you have for me?"
    if t == "CLOSE_FINAL":
        return ("Thanks — those are fair questions; in a real process the panel would be the best people to answer "
                "the company-specific ones. That's all from my side. Thank you for your time today.")
    if t == "END_EARLY":
        if action.preface == "time":
            return "We're out of time, so let's stop here. Thank you — your report will be ready shortly."
        return "Understood, we'll stop here. Thank you for your time — your report will be ready shortly."
    return question_text


def _action_brief(action: Action, *, question_text: str, item: Optional[dict], contradiction: str,
                  closing_reply: str) -> str:
    t = action.type
    if t == "OPEN":
        return ("ACTION: OPEN — greet in one short sentence, say this is a practice interview for the role in one "
                "short clause, then ask the TARGET QUESTION.")
    if t == "ASK":
        extra = ""
        if action.preface == "ack_refusal":
            extra = " First acknowledge briefly and neutrally that you'll leave the previous question."
        if action.new_section:
            extra += " Make a short natural transition to a new area."
        if action.difficulty_delta > 0:
            extra += " The candidate is handling this well; you may make the question a touch more demanding."
        return f"ACTION: ASK the TARGET QUESTION.{extra}"
    if t == "PROBE":
        hint = (f" A follow-up idea from the plan (use it only if it fits what they actually said): "
                f"\"{action.probe_text}\".") if action.probe_text else ""
        return (f"ACTION: PROBE — one follow-up focused on {action.focus}, built on THEIR words in this exchange. "
                f"Do not mention anything they did not say.{hint} Do not repeat or rephrase the original question.")
    if t == "CLARIFY_CONTRADICTION":
        return ("ACTION: CLARIFY — neutrally point out the two statements below and ask them to reconcile. "
                f"No accusation.\nSTATEMENTS: {contradiction}")
    if t == "CHALLENGE":
        return (f"ACTION: CHALLENGE — professional pushback on their last answer (focus: {action.focus}). Ask for "
                "evidence, add a realistic constraint, or force a choice. Never rude.")
    if t == "REPEAT":
        return "ACTION: REPEAT the TARGET QUESTION (your last line), rephrased slightly, same substance."
    if t == "CLARIFY_QUESTION":
        return ("ACTION: CLARIFY — instead of answering, the candidate asked about your last line (their message "
                "below). Answer THEIR question directly in one short sentence: what you mean, which role, period or "
                "example is fine, how much detail. If they ask whether they may answer a certain way, say yes or "
                "point to the closest fit. Do not hint at what a good answer contains. Then invite them to answer "
                "the TARGET QUESTION (your last line) — same question, not a new one.")
    if t == "WAIT":
        return "ACTION: WAIT — tell them to take their time, in a few words. No question needed but a question mark is fine."
    if t == "NUDGE":
        return "ACTION: NUDGE — their answer was very brief; invite them to elaborate with a specific example."
    if t == "REDIRECT":
        if action.preface == "meta":
            return ("ACTION: REDIRECT — they asked how they are doing. Say you won't discuss performance during the "
                    "interview and they'll get a detailed report at the end, then re-ask the TARGET QUESTION.")
        return ("ACTION: REDIRECT — they asked something unrelated. Answer in at most one short clause only if it is "
                "about the interview process; otherwise say you'll keep focus, then re-ask the TARGET QUESTION.")
    if t == "CLOSE_INVITE":
        return "ACTION: CLOSING — say you're near the end and invite their questions for you."
    if t == "CLOSE_FINAL":
        return ("ACTION: FINAL — respond to their questions in one or two sentences WITHOUT inventing company facts "
                "(say the panel would know specifics), then thank them and end. No question.\n"
                "THEIR QUESTIONS (untrusted): " + wrap_untrusted("answer", "closing", truncate(closing_reply, 800)))
    if t == "END_EARLY":
        if action.preface == "time":
            return "ACTION: END — the time is up: say so briefly, thank them, say the report will follow. No question."
        return "ACTION: END — acknowledge they want to stop, thank them, say the report will follow. No question."
    if t == "PAUSE":
        return "ACTION: PAUSE — agree to a break, say they can resume when ready. No question."
    return f"ACTION: {t}"


def _focus_fallback(focus: Optional[str], used: Sequence[str]) -> str:
    """A neutral follow-up for `focus`; one not already used in this exchange if possible."""
    order = [focus or "specificity"] + [f for f in FOCUS_FALLBACK if f != focus]
    for f in order:
        if f in FOCUS_FALLBACK and f not in used:
            return FOCUS_FALLBACK[f]
    return FOCUS_FALLBACK.get(focus or "specificity", FOCUS_FALLBACK["specificity"])


def speak(action: Action, *, ctx: RunContext, bp: dict, question_text: str, last_answer: str,
          memory_refs: List[dict], recent_openers: List[str], contradiction: str = "",
          closing_reply: str = "", degraded: bool = False, grounding: Optional[Sequence[str]] = None,
          exchange_text: str = "", previous_questions: Sequence[str] = (), used_foci: Sequence[str] = ()
          ) -> tuple[str, bool, str]:
    """Returns (utterance, used_model, guard_note). Falls back deterministically on any failure, and
    whenever the line would refer to something never said, restate the question, or repeat an
    earlier question (guard_note says which)."""
    role_title = (bp.get("role") or {}).get("title", "")
    minutes = int((bp.get("config") or {}).get("duration_minutes", 45))
    sources = list(grounding) if grounding is not None else None
    if sources is not None:
        # what the interviewer itself may always talk about: the role, the company, the sections
        role = bp.get("role") or {}
        sources += [role_title, role.get("family_name", ""), role.get("industry", ""),
                    str((bp.get("config") or {}).get("company_name") or ""), *TRANSITIONS.values()]
    guard = ""
    if action.type == "PROBE":
        # A follow-up written into the plan before the interview may presuppose things they never said.
        hint = action.probe_text
        if hint and ((sources is not None and ungrounded(hint, sources)) or same_question(hint, question_text)):
            guard = f"planned follow-up not used: {hint[:80]}"
            hint = None
        action = replace(action, probe_text=hint)
    fb = fallback(action, question_text=question_text, role_title=role_title, minutes=minutes,
                  contradiction=contradiction, closing_reply=closing_reply)
    if action.type == "PROBE" and not action.probe_text:
        fb = _focus_fallback(action.focus, used_foci)
    if action.type == "OPEN" and bp.get("sections"):
        return opening_line(bp, question_text), False, guard
    if degraded or action.type in ("WAIT", "PAUSE"):
        return fb, False, guard
    persona = bp.get("persona") or {}
    cfg = bp.get("config") or {}
    parts = [
        f"INTERVIEW: {role_title or 'role'} | MODE: {cfg.get('mode')} | DIFFICULTY: {cfg.get('difficulty')}",
        f"STYLE: {persona.get('style', '')} Acknowledgements: {persona.get('acknowledgement', 'occasional')}.",
        _action_brief(action, question_text=question_text, item=None, contradiction=contradiction,
                      closing_reply=closing_reply),
    ]
    if question_text and action.type not in ("CLOSE_INVITE", "CLOSE_FINAL", "END_EARLY", "PAUSE", "WAIT"):
        parts.append(f"TARGET QUESTION: {question_text}")
    if exchange_text and action.type in ("PROBE", "CHALLENGE", "CLARIFY_QUESTION"):
        parts.append("THIS EXCHANGE SO FAR (the only things they have said about this question):\n" +
                     wrap_untrusted("exchange", "x", truncate(exchange_text, 3000)))
    if last_answer and action.type not in ("OPEN",):
        parts.append("CANDIDATE'S LAST MESSAGE:\n" + wrap_untrusted("answer", "last", truncate(last_answer, 1800)))
    if memory_refs:
        parts.append("EARLIER STATEMENTS YOU MAY REFER TO NATURALLY:\n" +
                     "\n".join(f"- {m['summary']}" for m in memory_refs[:4]))
    if recent_openers:
        parts.append("AVOID STARTING WITH: " + " | ".join(f'"{o}"' for o in recent_openers[-5:]))
    parts.append("Reply with the spoken line only.")
    text = run_text(INTERVIEWER, "\n\n".join(parts), ctx,
                    sim_input={"action": action.to_event(), "question": question_text, "fallback": fb})
    cleaned = _clean(text)
    if not acceptable(action, cleaned):
        return fb, False, guard
    if sources is not None and action.type in GROUNDED:
        bad = ungrounded(cleaned, sources)
        if bad:
            return fb, False, "referred to something never said: " + ", ".join(bad[:3])
    if action.type == "PROBE" and same_question(cleaned, question_text):
        return fb, False, "the follow-up restated the question"
    if action.type in ("ASK", "OPEN") and any(same_question(cleaned, p) for p in previous_questions if p):
        return fb, False, "repeated an earlier question"
    return cleaned, True, guard


def opener_of(text: str) -> str:
    first = re.split(r"[.?!,—-]", (text or "").strip(), maxsplit=1)[0]
    return " ".join(first.split()[:3])
