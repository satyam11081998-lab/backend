"""
Deterministic session signals for the adaptive interviewer.
V10.2: Fixed 'has_work' sticky bug, enhanced interrogative parsing (strips fillers),
and broadened imperative/solution exact-matching.
V12: adds a read of WHAT KIND of turn this is (turn_type, is_substantive_reasoning,
step_completed, asks_to_proceed) so the policy layer can pick a conversational
function, and reads the presence cool-down from the persisted response function
(a model-worded acknowledgement can no longer be recognised by its exact text).
"""
from __future__ import annotations

import re
import difflib
from typing import Any, Dict, Iterable, List

_HELP = (
    "help", "hint", "i am stuck", "i'm stuck", "im stuck", " stuck",
    "not getting", "not able to", "unable to", "don't get", "dont get",
    "how do i", "how do we", "how do you", "how to start", "how should i", 
    "where do i start", "where to start", "what should i do", "what do i do", 
    "confused", "i am lost", "i'm lost", "no idea", "not sure how", "guide me",
    "give me direction", "point me", "don't understand", "dont understand",
    "help karo", "help kar do", "help kardo", "madad karo", "madad kar do", "guide karo", "samajh nahi",
)
_SOLUTION = (
    "give me the answer", "what is the answer", "what's the answer",
    "tell me the answer", "just tell me", "show me the answer",
    "show me the approach", "show me the solution", "give me the approach",
    "give me the solution", "provide me with the approach", "provide the approach",
    "correct approach", "the correct approach", "solve it for me",
    "solve this for me", "what is the correct", "show the correct",
    "show me how", "what is the answer you", "you tell me first",
    "give me the final", "tell me the final", "just the final answer",
    "the final answer now", "what's the final answer", "whats the final answer",
    "give me the answer", "just give me the", "how would you solve",
    "share the approach", "share your approach", "tell me the approach", "what's the approach", "whats the approach",
    "what is the approach", "explain the approach",
    "approach bata do", "approach batao", "approach bta do", "solution bata do", "solution batao",
    "answer bata do", "answer batao", "bata do approach",
)
_EXACT_SOLUTION = {"answer", "solution", "solve it", "tell me"}

_SKIP_STOP = (
    "leave it", "skip", "move on", "next question", "forget it",
    "i don't want to", "i dont want to", "i don't want", "dont want",
    "not interested", "let's move", "lets move", "i'm done", "im done",
    "give up", "leave", "quit",
)
_FRUSTRATION = (
    "irritating", "irritated", "frustrat", "annoying", "annoyed",
    "beating around the bush", "going too into deep", "too deep",
    "too many questions", "erratic", "waste of time", "useless",
    "guide or leave", "not even saying", "you are not agreeing",
    "you're not helping", "not helping", "going in circles",
    "round and round", "same question",
)
_META = (
    "are you a bot", "are you an ai", "are you ai", "are you human",
    "which model", "what model", "chatgpt", "chat gpt", "claude", "gemini",
    "gemina", "chargipity", "i am the admin", "i am admin", "ignore the",
    "ignore your", "ignore all", "reveal your", "system prompt",
    "your instructions",
)
_INTERROGATIVE = (
    "what", "why", "how", "when", "where", "who", "which", "is", "are",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "will",
)
_DIRECTIVES = (
    "tell me", "give me", "explain to me", "share the", "share with", "please share", 
    "provide the", "provide me", "please provide", "clarify",
    "i need to know", "let me know"
)
_GREETING = ("hi", "hii", "hello", "hey", "ok", "okay", "start", "let's start",
             "lets start", "so", "good morning", "good evening", "yo")

_MID_THOUGHT = (
    "let me think", "let me redo", "let me recompute", "let me recalculate",
    "no wait", "wait no", "hold on", "give me a sec", "give me a second",
    "one sec", "let me re-do", "scratch that", "let me redo the",
    # adversarial review (Oct 21)
    "give me a minute", "give me a min", "one minute", "1 minute", "ek minute", "ek min", "ek second",
    "soch raha", "soch rahi", "sochne do", "sochta hu", "sochti hu",
)
_WANTS_SPACE = (
    "let me solve", "let me work", "let me try", "just let me", "let me do it",
    "stop asking", "too many questions", "let me finish", "give me a moment",
)
_WANTS_ADVANCE = (
    "move on", "next part", "next question", "let's move", "lets move",
    "moving on", "shift gears", "next topic", "next section",
)
_PRODUCT_UX = (
    "where do i see", "where will i see", "my results", "results page",
    "results after", "where do i find", "how do i submit", "where is the",
    "what happens after", "after this", "after we finish", "see my score",
)
_WHY_THIS = (
    "why are you asking", "why do you ask", "why this question",
    "why that question", "what's the point of", "whats the point of",
    "why do you want to know", "why are we", "why does that matter",
)
_OWNED_FACT_WIDE = (
    "population", "growth", "competitor", "how many", "how much", "market size", "market share",
    "per capita", "price", "cost", "revenue", "salary", "wage", "margin", "share of", "number of",
    "average", "data", "exhibit", "split", "breakdown", "budget", "trend", "spend", "volume", "rate",
    "households", "income", "profit", "sales", "customers", "members", "stores", "outlets", "fleet",
)
_OWNED_FACT = (
    "population", "growth rate", "market growth", "competitors", "how many players",
    "market size", "gdp", "per capita", "how big is the market", "what's the price",
    "whats the price", "how many people", "what is the market",
)
_CONFIDENT = (
    "obviously", "definitely", "everyone in", "everyone buys", "always",
    "100%", "hundred percent", "surely", "no doubt", "clearly", "must be 100",
    "almost everyone", "nearly everyone", "basically everyone", "everyone uses",
    "everyone drives", "everyone has", "literally everyone",
)
_RECOVERY = (
    "oh right", "ohh", "oh i see", "i see so", "i see, so", "got it, so",
    "okay so i just", "so i just", "makes sense so", "ah so", "oh so",
)
_END_SESSION = (
    "i'm done", "im done", "i am done", "leave it", "i don't want to continue",
    "dont want to continue", "don't want to continue", "i give up", "give up",
    "quit", "let's wrap", "lets wrap", "we can wrap", "i'm finished", "im finished",
)
_RECOMMEND = (
    "my recommendation", "i'd recommend", "id recommend", "i would recommend",
    "i recommend", "overall i'd", "overall i would", "so overall i", "in summary",
    "to summarize", "to sum up", "my final recommendation", "overall my",
)
_END_RE = re.compile(
    r"^(?:ok(?:ay)?[\s,]+|so[\s,]+|yeah[\s,]+)*(?:i'?m|i am|im) (?:done|finished)(?:\s+(?:now|here|for today|sir))?(?=\s*(?:[.!,;]|$|let'?s|lets|my|and i))"
    r"|\blet'?s wrap(?: (?:it )?up)?\b|\blets wrap\b|\bwe can wrap\b|\bi (?:don'?t|do not|dont) want to continue\b"
    r"|\bi give up(?: (?:now|here|on this(?: case)?|for (?:today|now)))?(?=\s*(?:[.!?,]|$))|\bi quit\b"
    r"|^leave it(?=\s*(?:[.!?,]|$|na|yaar))")
_POST_CLOSE = (
    "how did i do", "how'd i do", "any tips", "any feedback", "how was i",
    "what's my score", "whats my score", "how did that go", "any advice",
    "any pointers",
)
_FINAL_ESTIMATE = (
    "final answer", "final number", "so my answer", "my answer is",
    "thats my number", "that's my number", "final figure",
)
_PLANNING = (
    "i need to think about", "let me first", "i'll start with", "step one",
    "i need to figure out how many",
)
_WORK_MARKERS = (
    "i'll ", "i'd ", "i will ", "let me ", "i want to", "i'm going to", "im going to",
    "instead of", "rather than", "i'll go", "i'll use", "i'll take", "i'll split",
    "i'll size", "i would", "i'd go", "i'd start",
)
_HEDGE = (
    "maybe", "i think", "not sure", "not fully sure", "not totally", "feels about",
    "feels right", "seems okay", "seems right", "i'd guess", "around", "roughly",
    "probably", "kind of", "sort of", "somewhere near", "ish",
)
_KICKOFF = (
    "start", "begin", "let's go", "lets go", "let's do this", "lets do this",
    "ready", "shall we", "kick off", "kickoff", "go ahead",
    "give me the case", "tell me the problem", "what is the problem", "what's the problem"
)

_HYPOTHESIS_MARKERS = (
    "because", "due to", "driven by", "explains", "therefore", 
    "which suggests", "my hypothesis", "if we", "if they", "implies"
)

_PRESENCE_TEXT_MATCHES = [
    "got it.", "alright.", "okay.", "right.", "understood.",
    "go ahead.", "makes sense. go ahead.", "sure, continue.", 
    "take it from there.", "alright, continue.", "that works.", "that makes sense."
]

_AFFIRMATION_WORDS = {
    "ok", "okay", "got", "right", "understood", "yes", "yeah", "yep",
    "sure", "makes", "agreed", "correct", "exactly", "proceed", "continue",
    # V12.1: Hinglish yes and plain thanks are affirmations too ("hn", "haan ji", "thanks")
    "hn", "haan", "han", "ha", "ji", "acha", "accha", "achha", "theek", "thik", "thanks", "thank",
    "thankyou", "ty",
}

# --- V12: what kind of turn is this? ------------------------------------------
# These read the SHAPE of the candidate's turn. They never decide wording; the
# policy layer (interviewer_decision.decide_response) turns them into a function.
_STRUCTURE_WORDS = (
    "split", "segment", "bucket", "break it into", "break this into", "break it down",
    "break this down", "breakdown", "categor", "framework", "my structure", "the structure",
    "drivers", "branches", "mece", "two parts", "three parts", "two sides", "both sides",
    "revenue side", "cost side", "demand side", "supply side", "top-down", "bottom-up",
)
_SEQUENCE_WORDS = (
    "first", "then ", "after that", "next,", "next i", "finally", "step one", "step two",
    "followed by", "once i have", "once we have", "from there", "from the updated",
)
_STRATEGIC_WORDS = (
    "rather than", "instead of", "the real issue", "the key driver", "key question",
    "trade-off", "tradeoff", "trade off", "on the other hand", "bigger lever",
    "biggest lever", "root cause", "the underlying", "more important", "what matters",
    "premium", "realization", "realisation", "differentiat", "the risk is", "the catch",
    "not just", "not only", "the flip side", "second-order", "second order",
)
_RESULT_WORDS = (
    "that gives", "which gives", "this gives", "gives us", "gives me", "so we get",
    "we get ", "comes to", "comes out", "works out", "that's about", "thats about",
    "that is about", "which is about", "so the total", "so total", "total is",
    "total cost is", "in total", "so roughly", "so about", " = ", "equals", "so overall",
    "that means", "new profit margin is", "the new margin is", "the margin is",
)
_CALC_WORDS = (
    "multiply", "multiplied", "divide", "divided", "times", " x ", "*", "/", " per ",
    "%", "percent", "plus", "minus", "subtract", "add ",
)
# Wanting to skip a step or stop - said about the session, not the client ("the client
# wants to QUIT the market", "customers GIVE UP on the brand", "LEAVE the market").
_SKIP_STOP_RE = re.compile(
    r"^(?:(?:ok|okay|so|arre|yaar|please)[\s,]+)*(?:leave it|skip(?: it| this)?|forget it|move on|next question)\b"
    r"|\b(?:let'?s|lets|can we|could we|shall we|we can)(?: just| please)? (?:move on|skip)\b|\bnext question\b"
    r"|\bskip (?:this|it|to the)\b|^(?:(?:eh|ok|okay|so)[\s,]+)*(?:just )?move on\b"
    r"|\bi (?:don'?t|dont|do not) want to (?:continue|do (?:this|it)|answer|go on|proceed|carry on)\b"
    r"|\bi (?:don'?t|dont|do not) want to[\s.!]*$|\bi (?:don'?t|dont|do not) want (?:this|to do this)\b"
    r"|\bnot interested\b|\b(?:i'?m|i am|im) done\b(?! with)|\bi give up\b(?! on (?:the|this) (?!case))|\bi quit\b"
    r"|\bleave it\b(?=\s*(?:[.!?,]|$|na|yaar|i))")
_PROCEED_ASKS = (
    "shall i proceed", "should i proceed", "can i proceed", "may i proceed",
    "shall i continue", "should i continue", "can i continue", "shall i go ahead",
    "should i go ahead", "can i go ahead", "can i move on", "shall i move on",
    "should i move on", "shall we proceed", "should we proceed", "is that okay",
    "is that ok", "is this okay", "is this ok", "is that fine", "is this fine",
    "does that make sense", "does this make sense", "sound good", "sounds good?",
    "okay to proceed", "ok to proceed", "good to go",
)
_CONFIDENCE_HIGH = ("definitely", "clearly", "obviously", "certainly", "for sure", "surely")

# --- V12.1: what the candidate is ASKING FOR -----------------------------------
# From the 15-day review (Oct 2026): questions and requests that got "Okay." /
# "Sure, continue." / "Let's move on to the next phase." -- "Show me the case study",
# "Are we done?", "i want to exist the interview i am not prepared", "How it work",
# "how i adjust my assumtion", "could you answer the questions?", "i estimate it to
# around 20k". Each of these now has a named signal and a real response function.

# A question that only tags work ("... three parts. Upper, middle and lower class. Right?")
_TAG_Q_RE = re.compile(
    r"(?:^|[\s,.;-])(right|ok|okay|correct|yes|na|naa|no|isn'?t it|is it|innit|yeah|fine|agreed|"
    r"sound good|sounds good|makes sense|good)\s*\?+\s*$")
_REQUEST_RE = re.compile(
    r"\b(show me|tell me|give me|can you|could you|would you|will you|can u|could u|u tell|you tell|"
    r"please (?:tell|give|share|explain|show|provide|help)|explain (?:to )?me\b|"
    r"share (?:the |your |some |my )?(?:feedback|data|numbers|answer|case)|let me know|"
    r"i need (?:the|some|to know)|brief me)\b")
# A question asked mid-sentence without a question mark ("Budget was fixed all over
# the past 5 years or what was the budget trends").
_EMBEDDED_Q_RE = re.compile(
    r"\bor what\b|\b(?:do|did) (?:we|you) (?:have|know)\b|\b(?:is|are) there (?:any|a)\b|"
    r"\bwhat (?:was|were|is|are) the (?:\w+ )?(?:trend|trends|number|numbers|data|split|share|budget|price|cost|"
    r"revenue|size|growth|rate|figure|figures|breakdown|target|timeline|objective)\b|"
    r"\bcan (?:i|we) (?:get|have|know)\b|\bi(?:'d| would| will|'ll)? (?:like to )?ask\b|\bclarifying question\b|"
    r"\b(?:i(?:'d| would)|would) (?:like|love) to (?:know|understand|check)\b|\bi (?:want|need|wanted) to know\b|"
    r"\b(?:was |am )?wondering (?:if|whether|about)\b|"
    r"\b(?:kitna|kitne|kitni|kya|kaise|kyun|kyu|kaun|kab|kahan|kidhar)\b")
_DATA_NOUN_RE = re.compile(
    r"\b(?:price|prices|pricing|cost|costs|breakdown|margins?|share|revenue|data|split|size|number of|trend|trends|"
    r"growth|budget|sales|volume|competitors?|population|salary|salaries|rate|average|avg|ebitda|profit|"
    r"capacity|utili[sz]ation|churn|footfall|aov|skus?|stores|outlets|exhibit)\b")
_DATA_ASK_LEAD_RE = re.compile(r"^(?:and|ok|okay|so|also|what about|how about|any|need|the|plus|then|now)\b")
# "What I would do is first check...", "How I see it is..." open a statement.
_STATEMENT_START_RE = re.compile(r"^(?:what|how|why|where) i(?:'d| would| will|'ll| did| do| see| think| mean| am| was)\b"
                                 r"[^?]*\b(?:is|was|are)\b")
_NOT_A_REQUEST_RE = re.compile(r"\bgive me (?:a |one )?(?:sec|second|moment|minute|min)\b|\bgive me time\b")
_Q_STARTERS = {"what", "whats", "what's", "why", "how", "hows", "how's", "when", "where", "who", "which",
               "is", "are", "do", "does", "did", "can", "could", "should", "would", "shall", "am", "any"}
_Q_FILLERS = re.compile(r"^(?:(?:so|and|but|then|well|now|actually|basically|okay|ok|hmm+|umm+|sir|ma'?am|bro|"
                        r"yes|yeah|no|also|right|alright|hello|hi)[\s,.!]+)+")

# Asking for the case again. Directed at the interviewer: "can you show me the case",
# "brief me the case", "show me the case study ...", a whole message "what is the
# question?", or "I can't see it on my screen". NOT "To summarize the case: ..." or
# "So the problem statement is ..." or "What is the problem with the cost side".
_CASE_NOUN = r"(?:case(?:\s+study)?|question|problem(?:\s+statement)?|brief|prompt|scenario)"
_RESTATE_VERB = r"(?:show|give|send|share|repeat|read|tell|brief|recap|summari[sz]e|remind|explain)"
_RESTATE_RE = re.compile(
    rf"\b(?:can|could|would|will) (?:you|u) (?:please |pls |plz |kindly )?(?:again )?{_RESTATE_VERB}"
    rf"(?: me| us)?(?: again)? (?:the |this |that |our )?(?:full |whole |complete )?{_CASE_NOUN}\b"
    rf"|^(?:(?:please|pls|plz|kindly|ok|okay|so|sir|hi|hello|first)[\s,]+)*{_RESTATE_VERB} (?:me|us)\b[^.?!]{{0,25}}?{_CASE_NOUN}\b"
    rf"|^(?:(?:please|pls|plz|kindly|ok|okay|so|sir)[\s,]+)*(?:show|repeat|read|recap|summari[sz]e|explain) "
    rf"(?:the|this|that) {_CASE_NOUN}(?: again| once more| please)?[\s?.!]*$"
    r"|\b(?:can'?t|cannot|can not|unable to|not able to|don'?t|dont|could not|couldn'?t)\s+(?:see|find|view|read|open)\s+"
    r"(?:the\s+|this\s+)?(?:case|question|brief|prompt|problem|it on my (?:screen|phone|mobile)|anything on (?:my )?(?:screen|phone))\b"
    rf"|^(?:(?:so|ok|okay|sorry|sir|hi|hello|wait)[\s,]+)*what(?:'s| is| was) (?:the|this|our) {_CASE_NOUN}"
    r"(?: again| here| exactly| about| we are solving)?[\s?.!]*$"
    r"|\bwhere(?:'s| is)\s+the\s+(?:case|question|brief|prompt)\b|^brief me\b|\b(?:the )?case again\b")

_HOW_IT_WORKS_RE = re.compile(
    r"\bhow (?:does |do )?(?:it|this|the (?:interview|case|session|platform|app)|this (?:interview|case|session|platform|app)) works?\b"
    r"|\bhow (?:do|does|should|can) (?:i|we) (?:use|start|begin|do|play) (?:this|it)\b"
    r"|\bwhat am i supposed to\b|\bhow to (?:use|start|begin) (?:this|it)\b"
    r"|\b(?:first time|new to (?:this|cases?|case interviews?|guesstimates?|consulting))\b"
    r"|\bwhat (?:type|kind|sort|types|kinds) of (?:\w+ )?(?:questions?|clarifications?) (?:should |can |do |shall )?(?:i|we)\b"
    r"|\b(?:which|what) (?:\w+ )?questions? (?:(?:should|can|do|shall|to) (?:i )?|i (?:should |can )?)ask\b"
    r"|\bwhat (?:do|should) i (?:do|ask) (?:here|now|first)\b")

_CONCEPT_T = (r"(sanity[- ]?check(?:ing)?|reasonableness|mece|top[- ]?down|bottom[- ]?up|issue[- ]tree|"
              r"profit[- ]tree|driver[- ]tree|hypothes[ie]s|break[- ]?even|cagr|ebitda|"
              r"(?:profit |contribution |gross |net |operating )?margin|market share|penetration|"
              r"synthesis|value chain|unit economics|payback(?: period)?|npv|irr|elasticity|tam|sam|som|"
              r"market sizing|demand[- ]side|supply[- ]side|run[- ]rate|weighted average|framework)")
# "what is market share of client?" asks for DATA, not a definition.
_CONCEPT_RE = re.compile(
    rf"\bwhat (?:does|do) (?:a |an |the |this |you mean by )?(?:\w+ ){{0,2}}?{_CONCEPT_T} means?\b"
    rf"|\bwhat (?:is|are|'s) (?:a |an )?{_CONCEPT_T}\b(?!\s+(?:of|for|in|at|with|here|now)\b)(?!\s+\w+\s+(?:is|are|was)\b)"
    rf"|\b(?:meaning of|what do you mean by|define|definition of) (?:a |an |the )?{_CONCEPT_T}"
    rf"|\bhow (?:to|do i|do you|should i|can i|would i|do we|does one) (?:do |make |build |run |perform |"
    rf"calculate |use |apply |draw |write |create |check )?(?:a |an |the |my |this )?{_CONCEPT_T}"
    rf"|\bexplain (?:me )?(?:a |an |the |what )?{_CONCEPT_T}")

_UNANSWERED_RE = re.compile(
    r"\b(?:you )?(?:didn'?t|did not|haven'?t|have not|never|not) answer(?:ed)?\b"
    r"|\banswer (?:my|the|these|those|all) (?:questions?|query|queries)\b"
    r"|\b(?:could|can|will|would) (?:you|u) (?:please )?answer\b|\byou (?:ignored|skipped|missed) my\b"
    r"|\bmy question was\b")

# Wanting to leave the SESSION - anchored to the speaker or the session. Not "the rural
# market is not ready", "consumers are not prepared to pay", "customers give up on the
# brand", "stop the practice of 60-day credit", "Bye-laws aside".
_EXIT_RE = re.compile(
    r"\b(?:exit|exist|end|stop|quit|leave|close) (?:the |this |my )?(?:interview|session|chat|practice session)\b"
    r"|\bi (?:want|wanna|would like|need|have) to (?:exit|exist|quit)\b"
    r"|\bi (?:want|wanna|would like|need|have) to (?:stop|leave)(?: now| here| this| for (?:today|now)| the (?:interview|session|case))?(?=\s*(?:[.!?,]|$|i am|i'm))"
    r"|\bi(?:'m| am)(?: really| just)? not (?:prepared|ready)\b|\bi give up(?: (?:now|here|on this(?: case)?))?(?=\s*(?:[.!?,]|$))"
    r"|\bi quit\b|\bi(?:'m| am) done for (?:today|now)\b"
    r"|^(?:ok(?:ay)?[\s,]+)?bye(?:[\s.!]|$)|^(?:ok(?:ay)?[\s,]+)?good ?bye\b")

# A final answer: the phrase AND a figure right after it ("my final estimate is about 9
# lakh"), or a short "this is my final answer" when the figure came in earlier turns.
_FINAL_RE = re.compile(
    r"\bfinal (?:answer|estimate|number|figure|value|output|result|count|total)\b(?! of (?:households|people|population|users|the population))"
    r"|^\s*(?:so\s+|ok\s+|okay\s+)?final\s*[-:=]\s*(?=(?:about |around |~|approx\w* )?(?:\d|rs|₹|\$))"
    r"|\bmy (?:answer|estimate) (?:is|would be|comes(?: to)?)(?: therefore| then| now)? (?:about |around |roughly |approx\w* |~|nearly )?(?=\d|rs|₹|\$)"
    r"|\bi (?:would )?estimate (?:it|this|that) (?:to be|to|at|as|is|around|about|roughly|approx\w*)\b"
    r"|\bi (?:would )?estimate the (?:total|answer|number|market|count)\b"
    r"|\bthat'?s my (?:answer|number|estimate|final)\b|\bthis is my (?:answer|final)\b"
    r"|\bi(?:'ve| have)? arrived at\b"
    r"|\b(?:so )?the (?:answer|estimate) (?:is|would be|comes to) (?:about |around |roughly |approx\w* |~|nearly )?(?=\d|rs|₹|\$)")
_FINAL_SHORT_RE = re.compile(
    r"\b(?:this|that)(?:'s| is) my (?:final(?: answer| estimate)?|answer)\b|^final answer\b|\bthat'?s my final\b|"
    r"\bi'?m done with (?:my )?(?:estimate|answer)\b")
_FINAL_ASK_RE = re.compile(r"\bfinal (?:answer|estimate|number|figure|value)\b|\bconfirm your final\b|\block (?:it|that) in\b")

# About the SESSION, not the case: "are we done?", "did I pass?", "how close am I?",
# "share the feedback". Not "any feedback from customers?", "the actual number of
# outlets", "is it over 50% of the market?", "we are done with households".
_STATUS_RE = re.compile(
    r"\b(?:are we|am i|is it|is this|is the (?:case|interview|session)|are we all)(?: all)? "
    r"(?:done|finished|over|complete|completed)(?: here| now| yet| for today| with (?:the|this) (?:case|interview|session))?(?=\s*(?:[?.!,]|$|did|right|na))"
    r"|\bdid i (?:pass|solve(?: (?:it|the (?:case|problem)))?|crack(?: it| the case)?|do (?:it|well|good|ok|okay))(?=\s*(?:[?.!,]|$))"
    r"|\bhow (?:did i do|close (?:am i|i am|was i|is (?:my|it))|was (?:i|my (?:answer|estimate|performance)))\b"
    r"|\bhow (?:accurate|good) (?:am i|is (?:my answer|my estimate|this answer))\b"
    r"|\b(?:share|give|any|your|some) (?:the |some |me |overall |my )?feedback(?: (?:please|pls|for me|on (?:my|this)))?(?=\s*(?:[?.!,]|$|and|for (?:overall|my|the overall)))"
    r"|\bfeedback (?:for|on) (?:overall|my|the overall)\b"
    r"|\bactual (?:value|answer)\b|\bwhat(?:'s| is) the (?:actual|real|right|correct) (?:answer|value)\b"
    r"|\bwhat(?:'s| is) my score\b|\bhow (?:much|many) (?:marks|points) (?:do|did|will) i\b")
_CORRECT_Q_RE = re.compile(
    r"\bam i (?:right|correct|close)\b|\bis (?:that|this|it|my (?:answer|calculation|math)) (?:right|correct)\b"
    r"|\bhope (?:that|it|this) is (?:right|correct)\b|\bdid i get (?:it|that) right\b")

# Asking to be TAUGHT - the candidate, about how to proceed. Not "we don't know what
# competitors charge, so I'll assume Rs 40k".
_TEACH_RE = re.compile(
    r"\bteach me\b|\bwalk me through\b|\bshow me how\b|\bexplain (?:to me )?how\b|\bhelp me understand\b"
    r"|\bhow (?:do|should|can|would) i (?:approach|structure|solve|start|begin|tackle|break|frame)\b"
    r"|\bhow to (?:approach|structure|solve|start|begin|tackle|break|frame)\b(?! (?:the|this) (?:market|customer|client))"
    r"|\bi (?:really |still |honestly |just )?(?:do ?n[o']?t|don'?t|dont) (?:know|understand) "
    r"(?:how|what (?:to|i|and|should)|where (?:to|i)|what is going on)\b|\bno idea how\b"
    r"|\bi'?m new\b|\bi am new\b|\bnever done (?:this|a case|one)\b"
    r"|\bi (?:still )?(?:do ?n[o']?t|don'?t|dont) get (?:it|this)\b|\bstill (?:confused|lost)\b"
    r"|\b(?:u|you) tell me (?:the questions?|how|what (?:to do|should i|do i|i should))\b"
    r"|\bactually (?:u|you) tell me\b")
_STUCK_BLANK_RE = re.compile(
    r"^(?:(?:so|but|and|hmm+|umm+|um|uh|sir|bro|actually)[\s,]+)*(?:i\s+)?(?:really\s+|honestly\s+)?"
    r"(?:do ?n[o']?t|don'?t|dont) (?:know|get it|understand)(?: (?:actually|anything|at all|sir|yet|really|"
    r"what to do|how))?[\s.!]*$"
    r"|^(?:no idea|idk|clueless(?: here)?|nothing(?: i (?:do ?n[o']?t|don'?t|dont) know)?|no clue|"
    r"i'?m (?:blank|lost|stuck)|i am (?:blank|lost|stuck)|blank|not sure|nope|i have no idea|"
    r"(?:mujhe |muje )?(?:pata nahi|nahi pata|pta nahi|nhi pata|kuch nahi pata|kuch samajh nahi(?: aa raha)?|"
    r"samajh nahi aa raha))(?: (?:bhai|yaar|sir|honestly))?[\s.!]*$")
# Arena adversarial run (Oct 21) + review: only a complaint about being questioned, or an
# explicit "just let me solve". A plain "ok let me solve it" announces work (an ordinary
# hand-back), and "too many questions in the survey" is case talk.
_SPACE_ASK_RE = re.compile(
    r"\b(?:you(?:'re| are)?|u r|ur|youre)\s+(?:asking\s+)?(?:too|so) many questions\b|"
    r"\btoo many questions\s*(?:man|bro|yaar|na|please|pls|sir|now|yr)?\s*(?:[,.!]|$|\bjust\b|\blet\b|\bstop\b)|"
    r"\bstop asking(?: me)?(?: (?:so many )?questions)?(?=\s*(?:[,.!?]|$|\b(?:and|let|please|pls|man|bro|yaar|na|so|just|i)\b))|"
    r"\b(?:stop|quit) (?:interrupting|probing|quizzing)(?: me)?(?=\s*(?:[,.!?]|$|\b(?:and|let|please|pls|man|bro|yaar|na|so|just|i)\b))|"
    r"\bdon'?t interrupt(?: me)?(?=\s*(?:[,.!?]|$|\b(?:and|let|please|pls|man|bro|yaar|na|so|just|i)\b))|"
    r"\bjust let me (?:solve|do it|do this|work|finish)\b|\blet me do it myself\b|"
    r"\bi'?ll figure (?:it|this) out(?: (?:myself|on my own))?\s*[.!]*$")
# About the app, never the case: "MY results", the results page, Submit, ending the
# session. Not "the results of the survey", "the solution, top down or bottom up",
# "the feedback report from customers", "finish this calculation".
_PRODUCT_RE = re.compile(
    r"\b(?:where|how) (?:do|can|will|would|should) i (?:see|find|get|check|view|access) my "
    r"(?:results?|score|scores|feedback|report|marks|performance)\b|\bresults? page\b|"
    r"\bsee my (?:score|results?|feedback|report|marks|performance)\b|\bhow (?:do|can|should) i submit\b|"
    r"\bwhere(?:'s| is) the submit\b|\bsubmit button\b|"
    r"\bhow (?:do|can|should) i (?:end|finish|exit|close|stop) (?:this |the )?(?:case|session|interview|attempt|chat)\b|"
    r"\bwill i (?:get|see|receive) (?:a |my |any )?(?:score|feedback|results?|report|marks)\b(?! (?:from|of|on) (?!me\b))|"
    r"\bcan i see the (?:solution|model answer|worked answer|full answer) (?:later|after|at the end)\b")
# About the interviewer's own question. The strict forms stand alone; the topic forms
# ("why is the value chain relevant here?", "why value chain?") must name a topic from
# the interviewer's last line, so "why is it important for the client to grow in north
# india" stays a case question.
_WHY_ASKED_RE = re.compile(
    r"\b(?:why|y) (?:are|r) (?:you|u) (?:asking|asks)\b|\bwhy (?:do|did|would) (?:you|u) ask\b|\bwhy (?:this|that) question\b|"
    r"\bwhy (?:do|would) (?:you|u) (?:want|need) to know\b|"
    r"\bwhat'?s the (?:point|use) of (?:the |this |that |your )?(?:[a-z-]+ ){0,4}question\b|"
    r"\bwhat'?s the point of asking\b|\bhow is (?:that|this) relevant\s*(?:here)?\s*\??\s*$|"
    r"\bwhy (?:does|would) (?:that|this) matter\s*(?:here)?\s*\??\s*$|"
    r"\bwhy (?:is|would) (?:that|this) (?:relevant|important|needed)\s*(?:here)?\s*\??\s*$")
_WHY_TOPIC_RE = re.compile(
    r"^\s*(?:but |and |so |ok |okay )?(?:why (?:is |are |does |do |would )?|how (?:is|are) )(?:the |a |an |this |that )?"
    r"(?P<topic>[a-z][a-z\s-]{1,40}?)(?: (?:relevant|important|needed|matter|matters))?"
    r"(?: here| for this case| now| to (?:the )?[a-z]+(?: [a-z]+)?)?\s*\??\s*$")
_WHY_STOP = {"is", "are", "the", "a", "an", "this", "that", "here", "now", "for", "case", "relevant", "important",
             "needed", "matter", "matters", "do", "does", "would", "we", "you", "it", "of", "to", "and", "so"}


def _asks_why_asked(norm: str, last_asst: str) -> bool:
    """'why are you asking about X' (or 'why X?' naming the topic of your last line)."""
    if re.search(r"\bbecause\b|\bcoz\b|\bsince\b", norm):
        return False  # "why does this matter? because cost is the driver..." answers itself
    la = _norm(last_asst)
    if len(re.findall(r"[a-z]+", la)) < 3:
        return False  # after "Go on." there is no question of yours to explain
    if _WHY_ASKED_RE.search(norm):
        return True
    m = _WHY_TOPIC_RE.search(norm)
    if not m:
        return False
    topic = {w for w in re.findall(r"[a-z]+", m.group("topic")) if w not in _WHY_STOP and len(w) >= 4}
    return bool(topic) and topic <= set(re.findall(r"[a-z]+", la))


_COURTESY_WORDS = {"thank", "thanks", "thankyou", "you", "so", "much", "ty", "ok", "okay", "great", "cool",
                   "sure", "alright", "bye", "sir", "maam", "nice", "very", "a", "lot", "again", "for", "help"}
_CLOSING_LINE_RE = re.compile(
    r"\bthank you for (?:sharing|walking|working|your|the)\b|\bthat'?s the end of the case\b|\bresults page\b|"
    r"\btap submit\b|\ball the best\b|\bgood luck\b|\btake care\b|\bwe'?re done\b|\bend of the case\b")
_NEXT_STEP_RE = re.compile(
    r"^(?:ok(?:ay)?[\s,.]+)?(?:so[\s,.]+)?(?:next|what next|what'?s next|whats next|next step|next what|then what|"
    r"now what|what now|what do (?:i|we) do next|what (?:should|do) i do next|"
    r"what do you want(?: me to do)? next|next what do you want)[\s?.!]*$")


def _strip_q_fillers(norm: str) -> str:
    return _Q_FILLERS.sub("", norm or "")


def asks_interviewer(text: str) -> Dict[str, bool]:
    """Is this turn a question or request aimed at the interviewer?

    tag_question: the only '?' tags the candidate's own work ("... Right?").
    A question to the interviewer must never get a canned "Okay." (V12.1)."""
    norm = _norm(text)
    wc = len(norm.split())
    if not norm or is_go_ahead(text):
        return {"asks": False, "tag": False}
    q = norm.count("?")
    if q:
        if q == 1 and wc >= 6 and _TAG_Q_RE.search(norm):
            return {"asks": False, "tag": True}
        if q == 1 and wc >= 8 and _has_any(norm, _PROCEED_ASKS) and not _REQUEST_RE.search(norm):
            return {"asks": False, "tag": False}
        return {"asks": True, "tag": False}
    stripped = _strip_q_fillers(norm)
    fw = stripped.split(" ", 1)[0] if stripped else ""
    if fw in _Q_STARTERS and wc <= 25 and not _STATEMENT_START_RE.match(stripped) and ":" not in stripped[:45]:
        return {"asks": True, "tag": False}
    if _REQUEST_RE.search(norm) and not _NOT_A_REQUEST_RE.search(norm):
        return {"asks": True, "tag": False}
    if wc <= 30 and _EMBEDDED_Q_RE.search(norm):
        return {"asks": True, "tag": False}
    if wc <= 6 and not re.search(r"\d", norm) and _DATA_NOUN_RE.search(norm) and (
            _DATA_ASK_LEAD_RE.match(norm) or re.search(r"\b(?:please|pls|plz)[.!]*$", norm) or wc <= 3):
        return {"asks": True, "tag": False}  # "And the margins", "number of SKUs", "cost breakdown please"
    return {"asks": False, "tag": False}


def _is_idea_turn(norm: str) -> bool:
    """A free-standing idea, not a structure, a calculation or a question."""
    if not norm or "?" in norm or re.search(r"\d", norm):
        return False
    if len(norm.split()) < 10:
        return False
    if _has_any(norm, _STRUCTURE_WORDS) or sum(1 for w in _SEQUENCE_WORDS if w in norm) >= 2:
        return False
    if re.search(r"\b(ask|question|clarif\w*|english|hindi|language|repeat)\b", norm):
        return False
    fw = _strip_q_fillers(norm).split(" ", 1)[0]
    return fw not in _Q_STARTERS


def _structured(norm: str) -> bool:
    return (_has_any(norm, _STRUCTURE_WORDS) or sum(1 for w in _SEQUENCE_WORDS if w in norm) >= 2
            or bool(re.search(r"(?:^|\n)\s*(?:1[.)]|-|•)", norm)) or "=" in norm)


_NUM_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|percent|crore|lakh|million|billion|k\b)?", re.I)


def numbers_stated(text: str, limit: int = 8) -> List[str]:
    """Numbers as the candidate wrote them (with a unit word when one is attached)."""
    out: List[str] = []
    for m in _NUM_TOKEN_RE.finditer(text or ""):
        tok = re.sub(r"\s+", " ", m.group(0)).strip().rstrip(",")
        if tok and tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return out


def classify_turn(new_norm: str, intent: Dict[str, Any], is_affirmation_only: bool) -> Dict[str, Any]:
    """Deterministic shape of one candidate turn.

    turn_type: affirmation | help_request | solution_request | question |
               strategic_insight | structure | calculation_result | calculation |
               plan | hypothesis | estimate | statement
    """
    words = new_norm.split()
    wc = len(words)
    has_number = bool(re.search(r"\d", new_norm))
    strategic = _has_any(new_norm, _STRATEGIC_WORDS)
    structure = _has_any(new_norm, _STRUCTURE_WORDS)
    result = has_number and _has_any(new_norm, _RESULT_WORDS)
    calc = has_number and _has_any(new_norm, _CALC_WORDS)
    sequence_hits = sum(1 for w in _SEQUENCE_WORDS if w in new_norm)
    plan = sequence_hits >= 2 or _has_any(new_norm, _PLANNING) or (
        _has_any(new_norm, _WORK_MARKERS) and wc >= 6)

    if is_affirmation_only:
        turn_type = "affirmation"
    elif intent.get("solution_requested"):
        turn_type = "solution_request"
    elif intent.get("help_requested"):
        turn_type = "help_request"
    elif intent.get("is_question") and wc < 12:
        turn_type = "question"
    elif strategic and wc >= 10:
        turn_type = "strategic_insight"
    elif result:
        turn_type = "calculation_result"
    elif structure and not calc:
        turn_type = "structure"
    elif plan and sequence_hits >= 2:
        turn_type = "plan"
    elif calc:
        turn_type = "calculation"
    elif structure:
        turn_type = "structure"
    elif plan:
        turn_type = "plan"
    elif intent.get("is_hypothesis"):
        turn_type = "hypothesis"
    elif has_number:
        turn_type = "estimate"
    else:
        turn_type = "statement"

    reasoning_shape = turn_type in ("strategic_insight", "structure", "calculation_result",
                                    "calculation", "plan", "hypothesis", "estimate")
    is_substantive = reasoning_shape and (
        wc >= 20 or (wc >= 12 and turn_type in ("calculation_result", "strategic_insight", "structure")))
    return {
        "turn_type": turn_type,
        "word_count": wc,
        "is_substantive_reasoning": bool(is_substantive),
        "step_completed": bool(result),
        "asks_to_proceed": _has_any(new_norm, _PROCEED_ASKS),
    }

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _has_any(text: str, needles) -> bool:
    return any(n in text for n in needles)

def _looks_garbage(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if re.search(r"(.)\1{4,}", t):
        return True
    letters = re.sub(r"[^a-zA-Z]", "", t)
    if len(letters) >= 6:
        vowels = len(re.findall(r"[aeiou]", letters.lower()))
        if vowels / max(1, len(letters)) < 0.12:
            return True
    if len(letters) >= 10 and len(set(letters.lower())) <= 5:
        return True
    if " " not in t and len(letters) >= 8 and re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", letters.lower()):
        return True
    return False

def _similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() > 0.9

# A short go-ahead / agreement ("Yes.", "Go ahead", "Sure, carry on", and the
# common ASR slip "Do ahead"). Never a question, never a request for data.
_GO_AHEAD_RE = re.compile(
    r"^(?:(?:yes|yeah|yep|yup|ya|sure|ok|okay|alright|all right|right|fine|cool|great|perfect|correct|"
    r"exactly|agreed|got it|understood|makes sense|sounds good|that works|let'?s do it)[\s,.!]*)*"
    r"(?:(?:please\s+)?(?:go|do|carry|move)\s+(?:ahead|on)|proceed|continue|let'?s\s+(?:go|proceed|continue|start|begin)"
    r"|go for it|please do)?[\s,.!]*$")


def is_go_ahead(text: str) -> bool:
    t = _norm(text)
    return bool(t) and "?" not in t and len(t.split()) <= 6 and bool(_GO_AHEAD_RE.match(t))


# "help" only as a REQUEST: "Can you help me here?", "help na ...", "need some help".
# Not "it would be really helpful" or "both metrics will help me make a decision".
_HELP_WORD_RE = re.compile(
    r"^(?:pls |please |plz |sir |bro )?help\b|\b(?:can|could|would|will) (?:you|u) (?:please |pls )?help\b"
    r"|\b(?:pls|please|plz) help\b"
    r"|\bneed (?:some |a little |a bit of |your |more )?help\b|\b(?:some|any|little) help\b|\bhelp (?:na|pls|please|plz)\b"
    r"|\bhelp me(?: out| here| with this| please)?[\s.!?]*$|\bhelp me (?:with|out|here|please|pls|start|understand|structure|solve)\b")


# "not helping" has to be aimed at the INTERVIEWER, not used as ordinary case analysis. As a bare substring it
# fired on "the current pricing strategy is not helping the client retain customers" - a completely normal
# sentence in a profitability case - and the interviewer would abandon the case to hand out a hint. It counts
# only with a second-person subject ("you're not helping"), an explicit object ("not helping me"), or when the
# utterance trails off on it ("...round and round, not helping").
_NOT_HELPING_RE = re.compile(
    r"(?:\b(?:you(?:'?re| are)?|u|ur|this|that|it)\s+(?:is\s+|are\s+)?not\s+helping\b)"
    r"|(?:\bnot\s+helping\s+(?:me|us|at all)\b)"
    r"|(?:\bnot\s+helping\s*[.!]*\s*$)")


def _help_hit(t: str) -> bool:
    """_HELP phrases, with 'help' itself counted only as a request (V12.1)."""
    if _HELP_WORD_RE.search(t):
        return True
    if len(t.split()) <= 4 and re.search(r"\bhelp\b", t):
        return True
    if _NOT_HELPING_RE.search(t):
        return True
    return any(n in t for n in _HELP if n != "help")


def detect_intent(text: str) -> Dict[str, Any]:
    t = _norm(text)
    
    # Strip conversational fillers from the start to accurately catch interrogatives (e.g. "so how do we...")
    stripped_t = re.sub(r"^(so|and|but|then|well|now|actually|basically)\s+", "", t)
    first_word = stripped_t.split(" ", 1)[0] if stripped_t else ""
    
    is_q = (("?" in t) or (first_word in _INTERROGATIVE)) and not is_go_ahead(text)
    is_sol = _has_any(t, _SOLUTION) or t in _EXACT_SOLUTION
    
    flags = {
        "help_requested": _help_hit(t),
        "solution_requested": is_sol,
        "skip_or_stop": bool(_SKIP_STOP_RE.search(t)),
        "frustration": _has_any(t, _FRUSTRATION),
        "is_meta": _has_any(t, _META),
        "looks_garbage": _looks_garbage(text),
        "is_question": is_q,
        "is_directive": _has_any(t, _DIRECTIVES),
        "is_greeting_only": t in _GREETING,
        "is_hypothesis": _has_any(t, _HYPOTHESIS_MARKERS)
    }
    if flags["is_meta"]:
        primary = "meta"
    elif flags["solution_requested"]:
        primary = "asking_for_solution"
    elif flags["skip_or_stop"]:
        primary = "wants_to_stop"
    elif flags["help_requested"]:
        primary = "asking_for_help"
    elif flags["is_greeting_only"]:
        primary = "greeting"
    elif flags["is_question"] or flags["is_directive"]:
        primary = "clarification"
    elif flags["is_hypothesis"]:
        primary = "hypothesis"
    else:
        primary = "answering"
    flags["intent"] = primary
    return flags

def _candidate_turns(transcript: Iterable[Dict[str, str]]) -> List[str]:
    return [_norm(t.get("content")) for t in transcript
            if (t.get("role") or "user") == "user" and (t.get("content") or "").strip()]

def _assistant_turns(transcript: Iterable[Dict[str, str]]) -> List[str]:
    return [_norm(t.get("content")) for t in transcript
            if t.get("role") == "assistant" and (t.get("content") or "").strip()]

def _stuckish(turn_norm: str, intent_flags: Dict[str, Any]) -> bool:
    if intent_flags.get("help_requested") or intent_flags.get("solution_requested") or intent_flags.get("looks_garbage") or intent_flags.get("frustration"):
        return True
    return False

def _error_materiality(t: str) -> str:
    unit_scale = ("ml" in t) and ("litre" in t or "liter" in t)
    double_count = (("add" in t and "whole" in t and "population" in t)
                    or "add all of them" in t or "add them all" in t)
    wrong_denom = ("market share" in t) and ("own sales" in t or "divided by our own" in t)
    mece_overlap = (("split them into" in t or "segment" in t or "categorize" in t
                     or "break them into" in t or "buckets" in t)
                    and ("buy a lot" in t or "heavy" in t or "people who" in t))
    if unit_scale or double_count or wrong_denom or mece_overlap:
        return "material"
    hedged = _has_any(t, ("roughly", "about", "call it", "approx", "i'll round",
                          "ill round", "round to", "ballpark", "give or take"))
    if hedged:
        return "trivial"
    return "none"

def compute_signals(transcript: Iterable[Dict[str, str]], new_user_message: str,
                    teaching_policy: str = "coached", prior_state=None, channel: str = "text", is_voice_partial: bool = False) -> Dict[str, Any]:
    transcript = list(transcript or [])
    # a pasted essay is read on its first few thousand characters (a 20k-char paste took 4s)
    new_user_message = (new_user_message or "")[:4000]
    cand = _candidate_turns(transcript)
    asst = _assistant_turns(transcript)
    new_norm = _norm(new_user_message)
    intent = detect_intent(new_user_message)

    # V10.2: Fixed historic memory bug. has_work is evaluated ONLY on the current turn to prevent infinite acks.
    has_work = (len(new_norm) > 60 or bool(re.search(r"\d", new_norm)) or _has_any(new_norm, _WORK_MARKERS))

    # V12.1 (2026-10-28): has_work being current-turn-only is correct for deciding whether to acknowledge, and
    # wrong for deciding how to HELP. "help na i'm not getting this at all" and "i've got the demand but i'm
    # stuck on what to do next" both read has_work=False, so a candidate who had already solved an equation or
    # computed a demand figure was offered a from-scratch restart instead of a hint tied to their own numbers.
    # Both failed the behavioural eval on BOTH models, which is what gives it away as a signal bug rather than a
    # model one. prior_work is session-level and purely additive: nothing that reads has_work changes.
    def _is_worky(t: str) -> bool:
        # Deliberately stricter than has_work: a figure or an explicit work marker, never mere length.
        return bool(re.search(r"\d", t)) or _has_any(t, _WORK_MARKERS)

    prior_work = any(_is_worky(t) for t in cand)
    
    recent_probes = sum(1 for a in asst[-3:] if a.endswith("?"))
    interviewer_repeating = len(asst) >= 2 and (_similar(asst[-1], asst[-2]) or asst[-1] in asst[:-1])
    candidate_repeating = any(_similar(new_norm, c) for c in cand[-4:])
    
    # V12: punctuation-insensitive ("Okay." / "Yes," from typing or ASR are affirmations too).
    words = re.findall(r"[a-z0-9']+", new_norm)
    is_affirmation_only = ((len(words) > 0 and len(words) <= 3 and not re.search(r"\d", new_norm)
                            and any(w in _AFFIRMATION_WORDS for w in words)) or is_go_ahead(new_user_message))
    # The interviewer's last line asked or proposed something ("Shall we proceed
    # with that?"): a go-ahead now means "carry on with it", not "I'm working".
    last_assistant_asked = bool(asst) and asst[-1].rstrip().endswith("?")

    recent_assistant_turns = asst[-4:] if len(asst) > 0 else []
    
    last_action_was_presence = False

    if len(asst) > 0:
        last_turn_content = asst[-1]
        content_norm = _norm(last_turn_content)
        if content_norm in _PRESENCE_TEXT_MATCHES or "event_type" in content_norm:
            last_action_was_presence = True

    # V12: a model-worded acknowledgement has no fixed text, so the cool-down is
    # read from the response function persisted for the previous candidate turn.
    ps = prior_state or {}
    recent_functions = [f for f in (ps.get("recent_functions") or []) if isinstance(f, str)][-4:]
    last_function = ps.get("last_function") if isinstance(ps.get("last_function"), str) else None
    try:
        function_turn = int(ps.get("function_turn") or 0)
    except (TypeError, ValueError):
        function_turn = 0
    if (last_function in PRESENCE_FUNCTIONS and function_turn and function_turn >= len(cand)
            and len(asst) > 0):
        last_action_was_presence = True

    shape = classify_turn(new_norm, intent, is_affirmation_only)

    tw = 0
    for c in reversed(cand + [new_norm]):
        if _stuckish(c, intent):
            tw += 1
        else:
            break

    if intent["frustration"]:
        frustration = "high"
    elif (intent["help_requested"] or intent["solution_requested"]) and tw >= 2:
        frustration = "high"
    elif candidate_repeating and intent["help_requested"]:
        frustration = "high"
    elif intent["help_requested"] or tw >= 1:
        frustration = "mild"
    else:
        frustration = "none"

    repair_due = (recent_probes >= 2 and (intent["help_requested"] or tw >= 2
                  or candidate_repeating or interviewer_repeating)) or frustration == "high"

    ends_ellipsis = new_norm.endswith("...") or "..." in new_norm
    candidate_mid_thought = (_has_any(new_norm, _MID_THOUGHT) or ends_ellipsis
                             or (len(new_norm.split()) <= 4 and bool(re.search(
                                 r"\bthinking\b|^\s*(?:\d+|one|two|a)\s*mins?\b|\bone min\b", new_norm))))
    wants_space = _has_any(new_norm, _WANTS_SPACE)
    wants_to_advance = _has_any(new_norm, _WANTS_ADVANCE)
    candidate_working = (_has_any(new_norm, _PLANNING)
                         and not intent["help_requested"]
                         and not intent["solution_requested"])
    product_ux_question = _has_any(new_norm, _PRODUCT_UX)
    why_this_question = _has_any(new_norm, _WHY_THIS)
    
    asks_owned_fact = (intent["intent"] == "clarification") and _has_any(new_norm, _OWNED_FACT) and not intent["help_requested"]

    # V12.1: a first message that is already work ("total population: 1.2 crore ...
    # total: 7 lakh building") is answered as work, not with a case kick-off.
    _figs = re.findall(r"\d[\d,.]*\s*(?:%|percent|lakh|lac|crore|cr\b|million|billion|mn\b|bn\b|k\b|m\b|b\b)", new_norm)
    first_turn_is_work = len(cand) == 0 and (
        bool(_figs) or len(re.findall(r"\d+", new_norm)) >= 3 or "=" in new_norm
        or shape["is_substantive_reasoning"])
    is_session_open = (
        # V12: a bare "okay"/"so" mid-case is not a greeting that re-opens the case.
        (intent["intent"] == "greeting" and len(cand) == 0)
        or (len(asst) == 0 and len(cand) == 0 and not first_turn_is_work)
        or (len(cand) == 0 and len(new_norm) <= 24 and _has_any(new_norm, _KICKOFF)
            and not intent["solution_requested"]
            and not intent["help_requested"]))
    is_final_recommendation = _has_any(new_norm, _RECOMMEND)
    # V12.1: "Do we have any feedback from customers?", "the client wants to quit the
    # market", "customers give up on the brand", "i am done with structuring" are case
    # talk, not the candidate ending the session.
    is_post_close = (_has_any(new_norm, _POST_CLOSE)
                     and (len(new_norm.split()) <= 8 or bool(_STATUS_RE.search(new_norm)))
                     and not re.search(r"\bfeedback (?:from|of|by) (?!you\b)", new_norm))
    wants_to_end = bool(_END_RE.search(new_norm)) and not wants_to_advance
    is_session_close = ((is_final_recommendation or is_post_close or wants_to_end)
                        and not is_session_open)

    has_number = bool(re.search(r"\d", new_norm))
    defended = (" because" in (" " + new_norm)) or (" since " in new_norm) or ("reason" in new_norm)
    hedged_100 = bool(re.search(r"\bnot (?:100|hundred) ?(?:%|percent|per cent) (?:sure|certain|confident)\b", new_norm))
    confident_unsupported_claim = (_has_any(new_norm, _CONFIDENT) and has_number and not defended
                                   and not hedged_100)
    
    # Must explicitly state finality to trigger sanity check
    states_final_estimate = _has_any(new_norm, _FINAL_ESTIMATE) and has_number
    
    hedged_self_estimate = has_number and _has_any(new_norm, _HEDGE)
    just_recovered = _has_any(new_norm, _RECOVERY) and len(asst) >= 1
    error_materiality = _error_materiality(new_norm)

    # --- V12.1: what the candidate is asking for (see the detectors above) ---
    ask = asks_interviewer(new_user_message)
    wc_now = len(new_norm.split())
    prior_user_text = [t for t in cand]
    work_so_far = any(re.search(r"\d", c) or _structured(c) for c in prior_user_text)
    stuck_blank = bool(_STUCK_BLANK_RE.search(new_norm)) and wc_now <= 10
    wants_teaching = bool(_TEACH_RE.search(new_norm))
    if stuck_blank:
        intent["help_requested"] = True
    last_asst = asst[-1] if asst else ""
    _fm = _FINAL_RE.search(new_norm)
    _asks_for_it = bool(intent["solution_requested"]
                        or re.search(r"\b(?:tell|give|show|share) me\b|\bwhat(?:'s| is) the final\b", new_norm))
    final_declared = bool(
        not _asks_for_it and (
            (_fm and re.search(r"\d", new_norm[_fm.end():_fm.end() + 60]))
            or (_fm and has_number and len(new_norm.rstrip(" .!")) - _fm.end() <= 2)  # "... 552 cr, final answer"
            or (_FINAL_SHORT_RE.search(new_norm) and wc_now <= 14 and any(re.search(r"\d", c) for c in cand[-3:]))))
    if (not final_declared and _FINAL_ASK_RE.search(last_asst) and wc_now <= 6
            and (has_number or re.match(r"^(yes|yeah|yep|yup|haan|han|ji|correct|confirm|confirmed|sure|right|"
                                        r"ok|okay)\b", new_norm))):
        final_declared = any(re.search(r"\d", c) for c in cand[-3:]) or has_number
    rf = [f for f in recent_functions if f]
    final_given_earlier = (any(_FINAL_RE.search(c) and re.search(r"\d", c) for c in cand)
                           or bool(_CLOSING_LINE_RE.search(last_asst))
                           or any(f in ("FINAL_REVIEW", "CLOSE", "SESSION_STATUS") for f in rf)
                           or any(_has_any(c, _RECOMMEND) for c in cand))
    words_now = re.findall(r"[a-z']+", new_norm)
    courtesy_only = bool(words_now) and all(w in _COURTESY_WORDS for w in words_now) and any(
        w in ("thank", "thanks", "thankyou", "ty", "bye") for w in words_now)
    asks_owned_fact_any = ask["asks"] and _has_any(new_norm, _OWNED_FACT_WIDE)
    # free ideas with no structure yet (passive "interesting perspective" sessions)
    streak = 0
    for c in reversed(cand + [new_norm]):
        if _is_idea_turn(c):
            streak += 1
        elif "?" in c or len(c.split()) < 6:
            continue
        else:
            break
    # consecutive help / stuck turns, each read on ITS OWN text (turns_without_progress
    # reads every earlier turn with the current turn's intent, so it is not usable here)
    help_streak = 0
    for c in reversed(cand + [new_norm]):
        ci = detect_intent(c)
        if ci["help_requested"] or ci["solution_requested"] or (_STUCK_BLANK_RE.search(c) and len(c.split()) <= 10):
            help_streak += 1
        else:
            break
    asks_for_space = bool(_SPACE_ASK_RE.search(new_norm)) and not has_number and wc_now <= 18
    space_requested_recently = any(_SPACE_ASK_RE.search(_norm(c)) and not re.search(r"\d", c)
                                   for c in cand[-2:])
    unanswered = [re.sub(r"\s+", " ", (t.get("content") or "")).strip()[:160]
                  for t in transcript[-8:]
                  if (t.get("role") or "user") == "user" and asks_interviewer(t.get("content") or "")["asks"]][-2:]

    return {
        "intent": intent["intent"],
        "teaching_policy": teaching_policy,
        "channel": channel,
        "is_voice_partial": is_voice_partial,
        "hint_level": int((prior_state or {}).get("hint_level", 0) or 0),
        "learner_level": (prior_state or {}).get("learner_level", "unknown"),
        "repairs_done": int((prior_state or {}).get("repairs_done", 0) or 0),
        "help_requested": intent["help_requested"],
        "solution_requested": intent["solution_requested"],
        "skip_or_stop": intent["skip_or_stop"],
        "is_meta": intent["is_meta"],
        "is_scope_question": intent["intent"] == "clarification",
        "looks_garbage": intent["looks_garbage"],
        "has_work": has_work,
        "prior_work": prior_work,
        "is_affirmation_only": is_affirmation_only,
        "recent_probes": recent_probes,
        "interviewer_repeating": interviewer_repeating,
        "candidate_repeating": candidate_repeating,
        "recent_assistant_turns": recent_assistant_turns,
        "last_action_was_presence": last_action_was_presence,
        "turns_without_progress": tw,
        "frustration": frustration,
        "frustration_explicit": intent["frustration"],
        "repair_due": repair_due,
        "candidate_turn_count": len(cand) + 1,
        "is_session_open": is_session_open,
        "is_session_close": is_session_close,
        "candidate_mid_thought": candidate_mid_thought,
        "wants_space": wants_space,
        "wants_to_advance": wants_to_advance,
        "candidate_working": candidate_working,
        "product_ux_question": product_ux_question,
        "why_this_question": why_this_question,
        # V12.1 (arena adversarial run): strict versions that route ahead of help / repair
        "asks_for_space": asks_for_space,
        "space_requested_recently": space_requested_recently,
        "asks_product_help": bool(_PRODUCT_RE.search(new_norm)) and not intent["solution_requested"],
        "asks_why_asked": _asks_why_asked(new_norm, last_asst),
        "asks_owned_fact": asks_owned_fact,
        "confident_unsupported_claim": confident_unsupported_claim,
        "states_final_estimate": final_declared,  # V12.1: one definition of a final answer
        "hedged_self_estimate": hedged_self_estimate,
        "just_recovered": just_recovered,
        "error_materiality": error_materiality,
        "new_message_norm": new_norm,
        # V12.1: requests that must get a real answer, never a canned beat
        "asks_interviewer": ask["asks"],
        "tag_question_only": ask["tag"],
        "wants_case_restate": bool(_RESTATE_RE.search(new_norm)),
        "asks_how_it_works": bool(_HOW_IT_WORKS_RE.search(new_norm)),
        "early_in_case": len(cand) <= 2 or not work_so_far,
        "concept_asked": (_CONCEPT_RE.search(new_norm).group(0) if _CONCEPT_RE.search(new_norm) else None),
        "complains_unanswered": bool(_UNANSWERED_RE.search(new_norm)),
        "unanswered_questions": unanswered,
        "wants_to_exit": bool(_EXIT_RE.search(new_norm)),
        "final_declared": final_declared,
        "final_given_earlier": final_given_earlier,
        "case_finished": final_given_earlier,
        "answers_interviewer_number": bool(last_assistant_asked and has_number and wc_now <= 6),
        "asks_session_status": bool(_STATUS_RE.search(new_norm)),
        "asks_if_correct": bool(_CORRECT_Q_RE.search(new_norm)),
        "wants_teaching": wants_teaching,
        "stuck_blank": stuck_blank,
        "courtesy_only": courtesy_only,
        "wants_next_step": bool(_NEXT_STEP_RE.search(new_norm)),
        "asks_owned_fact_any": bool(asks_owned_fact_any and not intent["solution_requested"]),
        "is_final_recommendation": is_final_recommendation,
        "idea_streak": streak,
        "help_streak": help_streak,
        "structured_ever": any(_structured(c) for c in cand + [new_norm]),
        "work_so_far": work_so_far,
        # V12 turn shape + conversation memory (read by interviewer_decision)
        "turn_type": shape["turn_type"],
        "word_count": shape["word_count"],
        "is_substantive_reasoning": shape["is_substantive_reasoning"],
        "step_completed": shape["step_completed"],
        "asks_to_proceed": shape["asks_to_proceed"],
        "numbers_stated": numbers_stated(new_user_message),
        "candidate_confidence": ("low" if hedged_self_estimate or _has_any(new_norm, _HEDGE)
                                 else "high" if _has_any(new_norm, _CONFIDENCE_HIGH) else "moderate"),
        "recent_functions": recent_functions,
        "last_function": last_function,
        "candidate_plan": _latest_plan(transcript),
        "last_assistant_asked": last_assistant_asked,
    }


# Functions that are a presence beat (acknowledge / hand back), fast-lane or
# model-worded. Kept here (not in interviewer_decision) to avoid an import cycle.
PRESENCE_FUNCTIONS = frozenset({
    "SHORT_ACK", "HAND_BACK", "ACKNOWLEDGE_AND_CONTINUE", "ACKNOWLEDGE_AND_ORIENT",
    "REFLECT_PROGRESS", "VALIDATE_AND_HAND_BACK",
})


def _latest_plan(transcript: Iterable[Dict[str, str]], max_chars: int = 240):
    """The candidate's most recent earlier turn that laid out a plan or structure
    (so ACKNOWLEDGE_AND_ORIENT can point to THEIR next part, never a new one)."""
    for t in reversed(list(transcript or [])):
        if (t.get("role") or "user") != "user":
            continue
        n = _norm(t.get("content"))
        if not n:
            continue
        sequence_hits = sum(1 for w in _SEQUENCE_WORDS if w in n)
        if (sequence_hits >= 2 or _has_any(n, _STRUCTURE_WORDS)) and len(n.split()) >= 12:
            raw = re.sub(r"\s+", " ", (t.get("content") or "").strip())
            return raw[:max_chars] + ("..." if len(raw) > max_chars else "")
    return None

def needs_contextual_assessment(sig: Dict[str, Any]) -> bool:
    if sig.get("help_requested") or sig.get("solution_requested"): return False
    if sig.get("is_meta") or sig.get("looks_garbage") or sig.get("is_session_open") or sig.get("is_session_close"): return False
    if sig.get("error_materiality") == "material": return False
    if sig.get("wants_to_advance") or sig.get("asks_owned_fact"): return False
        
    if sig.get("intent") in ["hypothesis", "clarification"]:
        return True
        
    return False

def build_signal_block(signals: Dict[str, Any]) -> str:
    lines = [f"- intent this turn: {signals.get('intent', 'unknown')}"]
    lines.append(f"- case phase: {signals.get('case_phase', 'unknown')}")
    lines.append(f"- progress status: {signals.get('progress', 'unknown')}")
    
    if signals.get("has_work"):
        lines.append("- learner has real work on the table already")
    elif signals.get("prior_work"):
        # The distinction that matters when they ask for help: they are not starting from nothing.
        lines.append("- learner put real work down in EARLIER turns (nothing concrete this turn)"
                     " -> build any hint on THEIR numbers; do NOT restart from scratch")
    else:
        lines.append("- learner has put little/nothing concrete down yet")
    if signals.get("repair_due"):
        lines.append("- YOUR RECENT MOVES ARE NOT WORKING -> repair is due: stop probing, change strategy.")
    if signals.get("turns_without_progress", 0) >= 2:
        lines.append(f"- stuck for {signals['turns_without_progress']} turns with no progress.")
    if signals.get("frustration", "none") != "none":
        lines.append(f"- frustration: {signals['frustration']} -> change what you DO.")
    if signals.get("interviewer_repeating"):
        lines.append("- you have repeated yourself -> do NOT reuse a previous line; find a fresh move.")
    if signals.get("looks_garbage"):
        lines.append("- last message looks like noise/typo -> ask them to restate briefly.")
    
    block = ("SESSION SIGNALS (contextual read of the learner right now; use them, never quote them):\n"
             + "\n".join(lines))
    block += f"\nTEACHING POLICY: {signals.get('teaching_policy', 'coached')}"
    return block
