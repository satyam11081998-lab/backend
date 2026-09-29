"""
Deterministic signal extraction for one candidate turn.

Design rules (from the product brief and the V11 audit):
  * Request intents are matched as REQUEST FORMS, not substrings: "this will help
    margins" is not a help request; "leave 20% for rural" is not "leave it".
  * Length is never a proxy for state. "3", "50%", "0.46B" are valid turns.
  * Causal words alone do not make a hypothesis: "I'll start with revenue because
    it is easier" is the candidate explaining their own process.
  * Nothing here calls a model. Ambiguity is resolved in policy.py (and, for a
    narrow class of step-completing analytic turns, by the optional assessor).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from services.interviewer import numbers as num
from services.interviewer.normalize import (
    normalize, dedupe_repeated_words, ends_with_question, is_asr_filler_only, base_clean,
)


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# Discourse markers that may precede an anchored request ("ok help", "umm, I don't know").
_LEAD = r"^\s*(?:(?:ok|okay|so|umm+|uh+|hmm+|well|actually|i think|please|pls|sir|ma'?am|and|but|no|yes|right)[,.!]?\s+)*"

# --- explicit requests -------------------------------------------------------
_SOLUTION = _rx(
    r"\b(give|tell|show|share|reveal|provide|send)( me| us)? (the |your )?(correct |right |full |final |complete |model |ideal )?(answer|solution|approach|method|working|answer key)\b",
    r"\b(what('s| is) the (correct |right |final |actual )?(answer|solution|approach))\b",
    r"\bhow would you (solve|approach|do|tackle|crack|structure) (this|it|the case|the problem)\b",
    r"\b(can|could|would|will) you (just )?(solve|do) (this|it)( for me)?\b",
    r"\bsolve (it|this) for me\b",
    r"\bwalk me through (the |your )?(solution|answer|approach|correct approach)\b",
    r"\bjust tell me\b",
    r"\b(show|tell) me how (you('d| would)|to) solve\b",
    r"\bi (give up|surrender)\b",
    r"\bwhat (should|would) the (answer|final number|estimate) be\b",
)
# Level of a solution request: the full worked answer, or the approach spine.
_SOLUTION_FULL = _rx(r"\banswer\b", r"\bsolution\b", r"\bfinal (number|answer|estimate)\b",
                     r"\bgive up\b", r"\bsurrender\b", r"\bjust tell me\b", r"\bsolve (it|this) for me\b",
                     r"\b(can|could|will) you (just )?(solve|do) (this|it)\b")

_HELP = _rx(
    _LEAD + r"help\b",
    r"\bhelp me\b", r"\b(please|pls|plz) help\b", r"\bhelp (please|pls|plz)\b",
    r"\b(can|could|would|will) you (please |just |kindly )?(help|guide|assist)\b",
    r"\bneed (some |a little |a bit of |your )?(help|hint|clue|nudge|pointer|guidance)\b",
    r"\b(give|want|need|get|have|provide|offer|share|drop|throw) (me |us )?(a |some |one |another |any |a small |a little |a bigger |more of a )?(hint|clue|nudge|pointer)s?\b",
    r"\b(hint|clue|nudge)\s*(please|pls|plz|\?|$)",
    _LEAD + r"(hint|clue|nudge)\b",
    r"\bany (hint|hints|clue|clues|pointers?)\b",
    r"\bi'?m (so |really |completely |totally |a bit |kind of |kinda )?(stuck|lost)\b",
    r"\bi am (so |really |completely |totally |a bit |kind of )?(stuck|lost)\b",
    r"\b(stuck|lost) (here|now|on this|at this)\b",
    _LEAD + r"(stuck|lost)\s*[.!?]*\s*$",
    r"\bno idea\b",
    r"\bi (really |honestly |just )?(don't|do not) know (how|what|where|which)\b",
    _LEAD + r"(i )?(really |honestly )?(don't|do not) know\s*[.!?]*\s*$",
    r"\bnot sure how to (start|begin|proceed|approach|go about|move forward|continue)\b",
    r"\bguide me\b",
    r"\bhow (do|should|can|would) i (even )?(start|begin|proceed|go about|approach|move forward|tackle)\b",
    r"\bwhere (do|should|can) i (even )?(start|begin)\b",
    r"\bwhat (should|do) i do (now|next|here)?\b",
    r"\bshow me how to (start|begin|proceed|approach)\b",
    r"\bpoint me (in the right direction|somewhere)\b",
    r"\bi don't understand\b", r"\bi do not understand\b",
    r"\bcan'?t (figure|work) (it|this|out)\b",
)
_INSIST = _rx(
    r"^\s*no[,.!]? (i want|give me|just give|i need)",
    r"\b(still|really) (stuck|lost|confused|don't get)\b",
    r"\b(that|this) (doesn't|does not|didn't|did not) help\b",
    r"\b(another|bigger|more|better|clearer|stronger) (hint|clue|nudge)\b",
    r"\bmore (help|direct)\b",
    r"\bjust (give|tell) me\b",
)
_FRUSTRATION = _rx(
    r"\birritat(ing|ed)\b", r"\bannoy(ing|ed)\b", r"\bfrustrat(ing|ed)\b", r"\bgetting annoying\b",
    r"\bgoing (round )?in circles\b", r"\bround and round\b", r"\bsame question\b",
    r"\bnot helping\b", r"\bisn'?t helping\b", r"\bnot helpful\b", r"\buseless\b", r"\bwaste of (my )?time\b",
    r"\bno help at all\b", r"\bnot getting anywhere\b",
    r"\bstop asking( me)?( so many)? questions\b", r"\btoo many questions\b",
    r"\bbeating around the bush\b", r"\bpointless\b", r"\bridiculous\b",
    r"\bwhat the (hell|heck)\b", r"\bwtf\b", r"\bugh+\b", r"\bseriously\?",
    r"\bjust answer (me|the question)\b", r"\bare you even listening\b",
)
_RECOVERY = _rx(
    r"^\s*(oh+|ah+|aha+)[,!. ]*(right|ok|i see|got it|yes|yeah|makes sense|so)\b",
    r"^\s*(oh+|ah+|aha+)\s*[.!]*\s*$",
    r"^\s*(got it|makes sense|that makes sense|understood|i see|right, so|ok so i|ok, so i|ok got it|clear now|now i get it|i get it now)\b",
    r"\bso i (just )?(need|have) to\b",
    r"\bthat helps\b",
)
_THINKING = _rx(
    r"^\s*(let me|lemme) (think|see|calculate|compute|work (it|this) out|redo|recompute|recalculate|check|jot)\b",
    r"^\s*(give me|one|just a|wait a) (sec|second|minute|moment|min)\b",
    r"^\s*(hold on|hang on|one moment|wait|wait wait|no wait|hmm+|umm+|uh+|erm+|so+|okay so|ok so|well)\s*[.,!…]*\s*$",
    r"^\s*(hmm+|umm+|uh+)[, ]+(let me|so|ok)\b",
    r"(\.\.\.|…)\s*$",
    r"^\s*(actually|wait),? (let me|i'll|i will)\b",
)
_SELF_CORRECTION = _rx(
    r"^\s*(no wait|wait|sorry|oops|actually|correction|scratch that|i mean|my bad)[,.! ]",
    r"\bi meant\b", r"\bnot \S+,? (but|it's|it is)\b",
)
_TRANSITION = _rx(
    r"\b(can|could|shall|should) we (move on|move ahead|go to|proceed to|move to|look at the next|get to)\b",
    r"\blet'?s (move on|move ahead|go to the next|do the (numbers|math|calculation)|get to|look at the next)\b",
    r"\bmove on\b", r"\bnext (part|section|step|question|stage|phase)\b",
    r"^\s*(what'?s|what is) (the )?next( step| part| stage| bit)?\s*\??\s*$",
    r"\bi'?m done with (this|the structure|structuring|this part)\b",
    r"\b(skip|skipping) (this|that|ahead)\b",
)
_FINAL = _rx(
    r"\b(my|the|our) (final )?recommendation\b", r"\bi('d| would)? recommend\b", r"\bi recommend\b",
    r"\bfinal (answer|estimate|number|figure)\b", r"\bmy (final )?estimate is\b",
    r"\bto (sum up|summari[sz]e|conclude)\b", r"\bin (summary|conclusion)\b",
    r"\b(so|therefore),? (overall|the (final )?answer is)\b",
    r"\bthat'?s my (final )?(answer|number|estimate)\b",
)
_OPEN = _rx(
    r"^\s*(hi|hii+|hello|hey|hey there|good (morning|afternoon|evening))\b[\s!.,?]*$",
    r"^\s*(hi|hello|hey)[,!. ]+(let'?s (start|begin|go)|i'?m ready|ready)\b",
    r"^\s*(let'?s (start|begin|go|get started|do this)|i'?m ready|ready|shall we (start|begin)|start|begin)\s*[.!?]*\s*$",
)
_META = _rx(
    r"\bignore (all |your |the |previous |prior |above )*(instructions|rules|prompt|guidelines)\b",
    r"\b(system|hidden|secret|developer) (prompt|instructions|message)\b",
    r"\byour (instructions|prompt|rules|guidelines|system prompt)\b",
    r"\byou are now\b", r"\bpretend (to be|you('re| are))\b", r"\bact as (the |an? )?(admin|scorer|grader|evaluator|developer|system)\b",
    r"\b(reveal|show|give|tell|leak|print)( me)? (the |your )?(answer key|rubric|hidden (case )?(data|facts|solution)|internal (score|notes)|scoring (rubric|criteria)|evaluator instructions)\b",
    r"\b(i am|i'?m) (the |an )?(admin|developer|owner|founder)\b",
    r"\bdeveloper mode\b", r"\bjailbreak\b", r"\bdan mode\b",
    r"\b(which|what) (model|llm|ai) (are you|is this|do you use)\b",
    r"\bare you (an? )?(ai|bot|robot|chatgpt|gpt|llm|human|real person|real)\b",
    r"\bare you (chat ?gpt|gpt-?\d|claude|gemini)\b",
    r"\b(your|the) api[_ ]?key\b",
)
_IDENTITY = _rx(r"\bare you (an? )?(ai|bot|robot|chatgpt|gpt|llm|human|real person|real)\b",
                r"\b(which|what) (model|llm|ai) (are you|is this|do you use)\b")
_UX = _rx(
    r"\bwhere (do|will|can) i (see|find|get) (my )?(results?|score|feedback)\b",
    r"\bhow (do|will) i (submit|finish|end)\b", r"\bsee my score\b", r"\bresults page\b",
    r"\bwhat happens (after|when i finish|next after)\b", r"\bhow (long|much time) (do i have|is this)\b",
    r"\bwhat('s| is) my (current |internal )?score\b", r"\bhow am i (doing|scoring)\b", r"\bhow did i do\b",
)
_REPEAT = _rx(r"\b(don't|do not) understand (the|your|what you) (question|asked|mean)\b", r"\bwhat do you mean\b",
              r"\b(can|could) you (repeat|rephrase|say that again|clarify)( (that|the question|it))?\b",
              r"^\s*(sorry|pardon|come again)\s*\??\s*$", r"\bsay (that|it) again\b")
# Information the interviewer owns: scope, data, figures, timeframe, objective.
_CLAR_TOPIC = _rx(
    r"\bpopulation\b", r"\bmarket( size)?\b", r"\b(time ?frame|timeline|period|horizon|annual|annually|yearly|per year|monthly)\b",
    r"\bgeograph(y|ic)\b", r"\b(india|urban|rural|city|country|region|global|domestic)\b", r"\bcustomers?\b",
    r"\bprice|pricing\b", r"\bcosts?\b", r"\brevenues?\b", r"\bprofits?\b", r"\bmargins?\b", r"\bcompetit(ors?|ion)\b",
    r"\bshare\b", r"\bgrowth\b", r"\bdata\b", r"\bnumbers?\b", r"\bunits?\b", r"\bcurrency\b", r"\bobjective\b",
    r"\bgoal\b", r"\bclient\b", r"\bproducts?\b", r"\bsegments?\b", r"\bchannel\b", r"\bvolume\b",
    r"\b(b2b|b2c)\b", r"\bscope\b", r"\bdefin(e|ition)\b", r"\bmean by\b", r"\bincluding|include\b",
    r"\bassume\b", r"\bstores?\b|\boutlets?\b|\bplants?\b|\bcapacity\b", r"\bsales\b", r"\bdemand\b",
    r"\bbudget\b", r"\binvestment\b", r"\bbreak.?even\b", r"\bindustry\b", r"\bbusiness model\b",
    r"\bare we (talking|looking|considering|sizing|estimating|focusing)\b", r"\bdo we (know|have)\b",
    r"\bis there (any )?(data|information|info)\b", r"\bwhat (is|was|are) the\b",
)
_ASSUME_Q = _rx(r"\b(can|could|shall|should|may) (i|we) (assume|take|use|consider|go with)\b",
                r"\bis it (ok|okay|fine|fair|reasonable) (to|if) (i |we )?(assume|take|use)\b",
                r"\bassume\b.*\?\s*$")
_WHY_THIS = _rx(r"\bwhy (are you asking|do you ask|this question|that question|does (that|this) matter|is (that|this) (important|relevant))\b")
_DATA_REQ = _rx(
    r"\b(can|could|may) (i|we) (get|have|see|look at) (the |some |any )?(data|numbers|figures|breakdown|split|trend|details|information|info)\b",
    r"\b(do|does) (we|the client|you) have (any |the )?(data|numbers|figures|information|info|breakdown)\b",
    r"\b(i'?d|i would|i want to|i'?d like to|let'?s|we should|can we|could we) (look at|check|see|analy[sz]e|examine|get|review)\b[^.?!]{0,32}\b(data|numbers|figures|trend|trends|breakdown|split|details)\b",
    r"\bany (data|numbers|figures|information|info) on\b",
    r"\bwhat (does|do) the (data|numbers) (say|show)\b",
    r"\b(share|give me) (the |some )?(data|numbers|figures)\b",
)
# Floor yields: the candidate hands the turn back and expects a (short) reply.
_FLOOR_PROCESS = _rx(
    r"\b(shall|should|can|may) i (proceed|continue|go ahead|move ahead|carry on|start|begin|go on)\b",
    r"\b(is|does) (that|this|it) (ok|okay|fine|alright|make sense|sound (ok|okay|good|right|reasonable)|work|seem (ok|right|reasonable))\b",
    r"\b(sound|sounds|look|looks|seem|seems) (ok|okay|good|fine|right|reasonable)\s*\?",
    r"\b(ok|okay|fine|right|correct|alright|yes|yeah|agree|fair)\s*\?\s*$",
    r"\bmakes sense\s*\?\s*$",
    r"\bam i (on the right track|going in the right direction|thinking (about )?(this|it) right)\b",
    r"\b(any|your) (thoughts|feedback|comments)\b",
    r"\bwhat do you think\b", r"\bhow('s| is| does) (that|this|my structure|it) (look|sound)\b",
    r"\bis my (structure|approach|framework|logic|math|calculation|assumption) (ok|okay|fine|right|correct|reasonable|good)\b",
)
_VALIDATION_OF_WORK = _rx(
    r"\bam i on the right track\b", r"\b(any|your) (thoughts|feedback)\b", r"\bwhat do you think\b",
    r"\bhow('s| is| does) (that|this|my structure|my approach) (look|sound)\b",
    r"\bis my (structure|approach|framework|logic|math|calculation) (ok|okay|fine|right|correct|reasonable|good)\b",
    r"\bis (this|that) (structure|approach|framework) (ok|okay|fine|right|correct|reasonable|good)\b",
    r"\bdid i miss (anything|something)\b",
)
_COMPLETION = _rx(
    r"\bthat'?s (my|the) (structure|approach|framework|plan|breakdown|answer|number|logic)\b",
    r"\bthat'?s (it|all)\b\s*[.!]*\s*$", r"^\s*(done|that'?s it|finished)\s*[.!]*\s*$",
    r"\bso (those|these) are (my|the) (buckets|branches|drivers|segments|factors)\b",
)
_STRUCTURE = _rx(
    r"\b(i'?d|i would|i will|i'?ll|let me|let'?s|i want to|i'?m going to) (break|split|divide|segment|structure|bucket|decompose|look at|approach|categori[sz]e)\b",
    r"\b(framework|buckets?|branches|issue tree|mece|driver tree|profit tree)\b",
    r"\b(first|firstly)\b.*\b(second|secondly|then)\b.*\b(third|thirdly|finally|lastly)\b",
    r"\b(revenue|internal) (side|factors)\b.*\b(cost|external) (side|factors)\b",
    r"\bon (the )?one (side|hand)\b.*\bon the other\b",
)
_PROCESS_BECAUSE = _rx(
    r"\b(i'?ll|i will|i'?d|i would|let me|let'?s|i want to|i'?m going to|i prefer to|i chose to|i'?m starting|i start|i'?ll start|starting)\b[^.?!]*\b(because|since|as it|as that)\b",
)
_CAUSAL = _rx(r"\bbecause\b", r"\bdue to\b", r"\bdriven by\b", r"\bwhich suggests\b", r"\btherefore\b",
              r"\bmy hypothesis\b", r"\bi hypothesi[sz]e\b", r"\bleads? to\b", r"\bcaused by\b", r"\bresult of\b",
              r"\bexplains?\b", r"\bimplies\b")
_BUSINESS_SUBJECT = _rx(r"\b(revenues?|sales|profits?|margins?|costs?|prices?|pricing|volumes?|demand|market share|share|customers?|"
                        r"footfall|traffic|conversion|churn|retention|competit\w*|market|growth|occupancy|utili[sz]ation|"
                        r"capacity|units|orders|basket|aov|ticket size|throughput|yield|ebitda)\b")
_HEDGE = _rx(r"\bmaybe\b", r"\bi think\b", r"\bnot (fully |totally |very )?sure\b", r"\bi guess\b", r"\bprobably\b",
             r"\bi'?d guess\b", r"\bkind of\b", r"\bsort of\b", r"\broughly\b", r"\baround\b", r"\bish\b")
_STUCK_SOFT = _rx(r"\bi'?m not sure\b", r"\bnot sure (what|about)\b", r"\bi'?m confused\b", r"\bconfused\b",
                  r"\bhmm+,? (i'?m )?not sure\b", r"\bblank\b", r"\bdrawing a blank\b")


@dataclass
class Signals:
    raw: str
    norm: str
    is_partial: bool = False
    numeric_only: bool = False
    has_number: bool = False
    word_count: int = 0
    question: bool = False                 # ends with '?' or interrogative opener
    # explicit intents
    solution: bool = False
    solution_level: str = "approach"       # approach | full
    help: bool = False
    insist: bool = False
    frustration: bool = False
    meta: bool = False
    identity_question: bool = False
    ux_question: bool = False
    clarification: bool = False            # information request the interviewer owns
    assumption_check: bool = False
    data_request: bool = False
    why_question: bool = False
    repeat_request: bool = False           # "what do you mean?", "can you repeat that?"
    transition: bool = False
    final: bool = False
    opening: bool = False
    # conversational
    recovery_language: bool = False
    thinking: bool = False
    self_correction: bool = False
    floor_yield: bool = False
    validation_request: bool = False
    completion: bool = False
    structure: bool = False
    hypothesis: bool = False
    hedged: bool = False
    stuck_soft: bool = False
    unintelligible: bool = False
    asr_noise: bool = False
    # numeric findings
    arithmetic: List[num.ArithmeticFinding] = field(default_factory=list)
    anchors: List[num.AnchorFinding] = field(default_factory=list)

    @property
    def material_arithmetic(self) -> Optional[num.ArithmeticFinding]:
        return next((f for f in self.arithmetic if f.severity == "material"), None)

    @property
    def minor_arithmetic(self) -> Optional[num.ArithmeticFinding]:
        return next((f for f in self.arithmetic if f.severity == "minor"), None)

    @property
    def material_anchor(self) -> Optional[num.AnchorFinding]:
        return next((f for f in self.anchors if f.severity == "material"), None)

    def flags(self) -> Dict[str, bool]:
        return {k: v for k, v in self.__dict__.items() if isinstance(v, bool) and v}


def _looks_garbage(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if num.has_number(t):
        return False
    if re.search(r"(.)\1{5,}", t):
        return True
    letters = re.sub(r"[^a-zA-Z]", "", t)
    if len(letters) >= 6:
        vowels = len(re.findall(r"[aeiouy]", letters.lower()))
        if vowels / max(1, len(letters)) < 0.15:
            return True
    if " " not in t and len(letters) >= 9 and re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", letters.lower()):
        return True
    return False


_INTERROGATIVE = re.compile(
    r"^\s*(so,? |and,? |ok,? |okay,? |also,? )?(what|which|how|why|when|where|who|is|are|do|does|did|can|could|should|shall|may|will|would)\b",
    re.IGNORECASE)


def _is_question_to_interviewer(raw: str, norm: str) -> bool:
    """True when the turn ENDS on a question. 'How many buy a car? Maybe 10%.' is the
    candidate answering their own rhetorical question - not a question to us."""
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", raw.strip()) if x.strip()]
    if not sentences:
        return False
    last = sentences[-1]
    if ends_with_question(last):
        return True
    if len(sentences) == 1 and not re.search(r"[.!]\s*$", last) and _INTERROGATIVE.match(normalize(last)):
        # Unpunctuated speech ("can I assume india only") still counts.
        return True
    return False


def extract(text: str, *, is_partial: bool = False) -> Signals:
    raw = base_clean(text or "").strip()
    norm = dedupe_repeated_words(normalize(raw))
    s = Signals(raw=raw, norm=norm, is_partial=is_partial)
    s.word_count = len(re.findall(r"\w+", norm))
    s.has_number = num.has_number(raw)
    s.numeric_only = num.is_numeric_only(raw)
    s.question = _is_question_to_interviewer(raw, norm)
    s.asr_noise = is_asr_filler_only(raw)

    s.meta = bool(_META.search(norm))
    s.identity_question = bool(_IDENTITY.search(norm))
    s.frustration = bool(_FRUSTRATION.search(norm))
    s.solution = bool(_SOLUTION.search(norm))
    if s.solution:
        s.solution_level = "full" if _SOLUTION_FULL.search(norm) else "approach"
    s.help = bool(_HELP.search(norm))
    s.insist = bool(_INSIST.search(norm))
    s.ux_question = bool(_UX.search(norm))
    s.why_question = bool(_WHY_THIS.search(norm))
    s.repeat_request = bool(_REPEAT.search(norm))
    if s.repeat_request:
        s.help = False  # "I don't understand the question" asks for the question, not a hint
    s.data_request = bool(_DATA_REQ.search(norm))
    # A PROPOSED assumption ("can I assume 1.4 billion?", "should we take India only?"), not a request
    # for the interviewer to supply one ("what population should I use?" is a clarification).
    s.assumption_check = (bool(_ASSUME_Q.search(norm)) and (s.question or "?" in raw)
                          and not re.match(r"^\s*(so,? |and,? |ok,? )?(what|which|how|where|who)\b", norm))
    s.transition = bool(_TRANSITION.search(norm))
    s.final = bool(_FINAL.search(norm))
    s.opening = bool(_OPEN.search(norm))
    s.recovery_language = bool(_RECOVERY.search(norm))
    s.thinking = bool(_THINKING.search(norm))
    s.self_correction = bool(_SELF_CORRECTION.search(norm))
    s.floor_yield = bool(_FLOOR_PROCESS.search(norm))
    s.validation_request = bool(_VALIDATION_OF_WORK.search(norm))
    s.completion = bool(_COMPLETION.search(norm))
    s.structure = bool(_STRUCTURE.search(norm))
    s.hedged = bool(_HEDGE.search(norm))
    s.stuck_soft = bool(_STUCK_SOFT.search(norm))

    # Hypothesis: causal language about the BUSINESS, not about the candidate's process.
    causal = bool(_CAUSAL.search(norm))
    process_only = bool(_PROCESS_BECAUSE.search(norm)) and not re.search(r"\bmy hypothesis\b", norm)
    s.hypothesis = causal and not process_only and bool(_BUSINESS_SUBJECT.search(norm))

    # Clarification: a question about information the interviewer owns. Excludes process
    # floor-yields ("shall I proceed?"), help and solution requests, meta and UX.
    if (s.question and not s.floor_yield and not s.help and not s.solution and not s.meta
            and not s.ux_question and not s.transition and not s.validation_request):
        s.clarification = (bool(_CLAR_TOPIC.search(norm)) or s.why_question or s.assumption_check
                           or s.data_request or s.repeat_request)
    if s.data_request or s.repeat_request:
        s.clarification = True

    recognised = any((s.meta, s.frustration, s.solution, s.help, s.ux_question, s.why_question, s.repeat_request,
                      s.data_request, s.transition, s.final, s.opening, s.recovery_language, s.thinking,
                      s.floor_yield, s.structure, s.clarification))
    s.unintelligible = (not s.asr_noise) and (not recognised) and _looks_garbage(raw) and _looks_garbage(norm)
    # "Does that make sense?" after real work asks us to check the work, not to hand back.
    if (not s.validation_request and s.word_count >= 12
            and re.search(r"\b(does|do) (that|this|it) (make sense|sound (right|ok|okay|reasonable))\b", norm)):
        s.validation_request = True

    s.arithmetic = num.check_arithmetic(raw)
    s.anchors = num.check_anchors(raw)

    # Guard against over-triggering: help language inside a longer analytic statement that is
    # clearly not asking for help ("the loyalty scheme would help margins").
    if s.help and s.word_count > 25 and not s.question and not re.search(
            r"\b(help me|i'?m stuck|i am stuck|hint|guide me|no idea|don't know how|lost)\b", norm):
        s.help = False
    # A bare "I don't know" answering a factual question is uncertainty; treat as a help signal
    # (policy decides the rung) rather than as content.
    return s
