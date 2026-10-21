"""
Reasoning slips that are not arithmetic (text interviewer, V12.1).

WHY (admin arena adversarial run, Oct 21 2026). Three material slips went through on
both builds because nothing in code saw them, and the model, left to notice them on
its own, either missed them or named them vaguely:
    "and then i'll also add the whole city population eating out, add all of them up"
        -> the families, singles and office orders are already inside the city
           population: a double count.
    "obviously its 100%, everyone in india buys this daily"
        -> 100% of people buying something every single day.
    "so the revenue drop must be because we're selling fewer units"
        -> the interviewer had just confirmed "volumes are flat".
This module names each one precisely (what they said, and what it collides with) so
the interviewer can point at it with one targeted question.

Precision over recall, as in numbers_check: each check needs an explicit signal in
the candidate's own words, skips negations, conditionals and questions, and never
raises.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_NEG = re.compile(r"(?i)\b(?:not|n't|no|never|cannot|can't|cant|isn'?t|aren'?t|wasn'?t|weren'?t|unlikely|"
                  r"rule out|ruled out|instead of|rather than|without|if|whether|unless|nobody|none)\b")

# --------------------------------------------------------------------- double count

_WHOLE_RE = re.compile(
    r"(?i)\b(?:whole|entire|total|full|overall|all(?:\s+of)?(?:\s+the)?)\s+(?:(?:city|town|country|state|area|"
    r"india)(?:'s)?\s+)?(?:population|people|residents|public)\b"
    r"|\b(?:whole|entire)\s+(?:city|town|country|state|area)\b")
# The whole place must be what is ADDED ("also add the whole city population", "plus
# the entire population"), not the base something is checked against or added to
# ("cross check against the entire city population", "tourists on top of the whole city").
_ADD_WHOLE_RE = re.compile(
    r"(?i)\b(?:add|adding|plus|sum|summing|include|including|count)\s+(?:in\s+|up\s+)?(?:also\s+)?"
    r"(?:the\s+|all\s+(?:of\s+)?the\s+|this\s+)?(?:whole|entire|total|full|overall)\b|\+\s*(?:the\s+)?(?:whole|entire|total)\b"
    r"|\bon\s+top\s+of\s+(?:that|this|these|those|it|them|all\s+this)\s*,?\s*(?:the\s+)?(?:whole|entire|total|full)\b"
    r"|\balso\s+(?:the\s+)?(?:whole|entire|full)\b")
_SEGMENTS = (
    "families", "family", "singles", "single people", "bachelors", "students", "office", "offices",
    "office goers", "office-goers", "corporates", "working professionals", "professionals", "couples",
    "kids", "children", "seniors", "elderly", "households", "hostels", "young people", "youth",
    "teenagers", "adults", "men", "women", "working women", "homemakers", "retirees",
)
_SEG_RE = re.compile(r"(?i)\b(" + "|".join(re.escape(s) for s in sorted(_SEGMENTS, key=len, reverse=True))
                     + r")\b")


_SEG_OF_RE = re.compile(r"(?i)\s+(?:of|in|among|from)\s+(?:the\s+|all\s+|these\s+|those\s+)?(?:"
                        + "|".join(re.escape(s) for s in sorted(_SEGMENTS, key=len, reverse=True)) + r")\b")


def _segments_named(texts: List[str]) -> List[str]:
    seen: List[str] = []
    for t in texts:
        for m in _SEG_RE.finditer(t):
            w = m.group(1).lower()
            w = {"family": "families", "offices": "office", "office goers": "office", "office-goers": "office",
                 "single people": "singles", "children": "kids"}.get(w, w)
            if w not in seen:
                seen.append(w)
    return seen


def _check_double_count(msg: str, recent_user: List[str]) -> Optional[Dict[str, Any]]:
    for sent in re.split(r"(?<=[.!?;])\s+|\n", msg):
        w = _WHOLE_RE.search(sent)
        if not w or "?" in sent:
            continue
        a = _ADD_WHOLE_RE.search(sent)
        if not a or not (a.start() <= w.start() <= a.end() + 2):
            continue
        if _NEG.search(sent[:w.start()]):
            continue  # "instead of adding the whole population", "if we add the whole city"
        if _SEG_OF_RE.match(sent[w.end():]):
            continue  # "the total population of singles" is a segment, not the whole place
        segs = _segments_named([sent[:w.start()]] + recent_user[-2:])
        if len(segs) < 2:
            continue
        named = [("office-goers" if s == "office" else s) for s in segs[:3]]
        names = named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]
        what = re.sub(r"(?i)^(?:all\s+of\s+the|all\s+the|all|overall)\s+", "whole ", w.group(0)).strip()
        what = what if what.lower().startswith("the ") else "the " + what
        return {"kind": "double_count", "said": w.group(0),
                "message": (f"{what} already includes the {names} counted above, so adding it on top "
                            f"counts the same people twice")}
    return None


# --------------------------------------------------------------------- extreme assumption

# Everyone in general (people, Indians, the population) - not "100% of restaurants",
# "100% of chai stalls" or "100% of metro riders", which are businesses or a segment that
# does the thing by definition.
_EVERYONE_RE = re.compile(
    r"(?i)\b(?:100\s?%|hundred\s+per\s?cent)\s+(?:of\s+)?(?:the\s+)?(?:people|population|indians|individuals|"
    r"persons|country|citizens|everyone)\b|\b(?:100\s?%|hundred\s+per\s?cent)(?=[^.!?;]{0,30}\b(?:everyone|everybody|"
    r"every\s+(?:single\s+)?(?:person|indian|individual))\b)|\beveryone\b|\beverybody\b|\bevery\s+(?:single\s+)?"
    r"(?:person|indian|individual|citizen)\b|\ball\s+(?:of\s+)?(?:the\s+)?(?:people|indians|individuals)\b"
    r"|\b(?:100\s?%|hundred\s+per\s?cent)\s+penetration\b|\bsab\s+log\b|\bsabhi\s+log\b|\bhar\s+(?:koi|aadmi|insaan)\b")
_DAILY_RE = re.compile(r"(?i)\b(?:daily|every\s?day|each\s+day|per\s+day|a\s+day|roz|har\s+din|rozana)\b")
# A subgroup, a business, a goal, someone else's claim, another time, or a share of a whole:
# "in a hostel everyone orders from the mess", "everyone of the 5000 shops", "the target is
# 100% coverage", "everyone says people buy it daily but i think 20%", "every person who
# commutes", "in my grandfather's generation", "out of 100% of the population, 40% buy".
_EXTREME_NOT = re.compile(
    r"(?i)\b(?:hostel|mess|campus|office|family|house|home|my|our|grand\w*|generation|used\s+to|earlier|back\s+then|"
    r"before|target|goal|aim|ideal|coverage|segment|group|who|that|which|commuters?|riders?|users?|members?|"
    r"customers?|subscribers?|students?|shops?|stores?|outlets?|restaurants?|businesses|companies|retailers?|"
    r"stalls?|vendors?|hotels?|says|say|thinks?|believes?|knows?|claims?|out\s+of|but|however|though|if|would)\b")
# Buying, not using: "everyone uses salt every day" is a fair household assumption;
# "everyone in india buys this daily" is not.
_USE_RE = re.compile(r"(?i)\b(?:buys?|buying|bought|purchases?|purchasing|orders?|ordering|shops?|shopping|"
                     r"kharid\w*|lete|leta|leti)\b")


def _check_extreme(msg: str) -> Optional[Dict[str, Any]]:
    for sent in re.split(r"(?<=[.!?;])\s+|\n", msg):
        e = _EVERYONE_RE.search(sent)
        if not e or "?" in sent:
            continue
        if re.search(r"(?i)\b(?:not|no|hardly|almost|nearly)\s+(?:quite\s+)?$", sent[:e.start()]):
            continue  # "not everyone", "almost everyone" (a strong claim, but not 100%)
        if re.search(r"(?i)\bevery\s*(?:one|body)\s+(?:in\s+(?:my|our|the)\s+(?:family|house|home|office)|"
                     r"who\b|that\b)", sent[e.start():]):
            continue  # "everyone in my family", "everyone who drives"
        if _EXTREME_NOT.search(sent):
            continue
        if any(float(x) < 100 for x in re.findall(r"(\d+(?:\.\d+)?)\s?%", sent)):
            continue  # "out of everyone, 40% buy it daily" - a share, not everyone
        d = _DAILY_RE.search(sent)
        if d and _USE_RE.search(sent) and not re.search(r"(?i)\b(?:not|never|n't)\s+(?:\w+\s+)?$",
                                                         sent[:d.start()]):
            return {"kind": "extreme_assumption", "said": e.group(0),
                    "message": ("this assumes every single person buys it every day - infants, the very old "
                                "and people who never buy it included")}
    return None


# --------------------------------------------------------------------- established facts

_METRICS: List[Tuple[str, str]] = [
    # cost lines are matched only so "raw material prices" is never read as OUR price;
    # costs have too many sub-lines (packaging vs raw material) to contradict safely
    ("cost", r"raw[\s-]?materials?(?:\s+(?:costs?|prices?))?|input\s+costs?|(?:\w+\s+)?costs?|expenses?|cogs"),
    ("volume", r"volumes?|units?\s+sold|unit\s+sales|sales\s+volumes?|quantity\s+sold|number\s+of\s+units|units?"),
    ("price", r"average\s+selling\s+prices?|selling\s+prices?|prices?|pricing|asp|ticket\s+size"),
    ("customers", r"customers?|customer\s+count|footfalls?|users?|subscribers?"),
    ("revenue", r"revenues?|top[\s-]?line|turnover"),
]
_FLAT = (r"flat|unchanged|constant|stable|steady|the\s+same|same|not\s+changed|hasn'?t\s+changed|haven'?t\s+changed|"
         r"didn'?t\s+change|no\s+change")
_UP = r"up|increased|increasing|rose|risen|rising|grew|grown|growing|higher|spiked|jumped|gone\s+up|went\s+up|hiked"
_DOWN = (r"down|decreased|decreasing|fell|fallen|falling|dropped|dropping|declined|declining|lower|lowered|less|"
         r"fewer|shrunk|shrinking|reduced|cut|gone\s+down|went\s+down")
_DIR = {"flat": _FLAT, "up": _UP, "down": _DOWN}
_ADJ = {"flat": r"flat|stable|constant|unchanged|steady",
        "up": r"higher|rising|increasing|growing",
        "down": r"lower|fewer|less|falling|declining|decreasing|shrinking"}
# A plan or recommendation, not a claim about what happened: "we should cut prices",
# "we need more customers", "to lower costs".
_PLAN_BEFORE = re.compile(
    r"(?i)\b(?:should|could|can|will|would|shall|need(?:s)?(?:\s+to)?|want(?:s)?(?:\s+to)?|let'?s|lets|"
    r"recommend|suggest|propose|plan|try|get|attract|bring|acquire|gain|drive|boost|aim|target|hope)\s+"
    r"(?:\w+\s+){0,2}$")
# Not a claim that contradicts OUR established figure: someone else (competitors, the
# industry), a part (a region, a segment), another period, a goal or lever, a contrast or
# self-correction ("my hypothesis was lower volumes, but you said flat"), a comparison.
_NOT_OUR_CLAIM = re.compile(
    r"(?i)\b(?:competitors?|competition|rivals?|industry|market(?:'s)?|peers?|others?|other\s+players|"
    r"region|regions|regional|north|south|east|west|zone|segment|segments|city|cities|state|states|store|stores|"
    r"outlet|outlets|channel|channels|online|offline|e-?commerce|modern\s+trade|general\s+trade|brand|brands|"
    r"some|certain|few|category|categories|product\s+line|sku|products|tier|range|"
    r"last\s+year|previous|prior|earlier|before|next\s+(?:year|quarter|month)|future|historically|used\s+to|"
    r"goal|target|aim|lever|levers|would|could|should|will|help|helps|if|unless|"
    r"but|however|though|although|instead|rather|hypothesis|thought|assumed|you\s+said|you\s+told|as\s+you|"
    r"than|relative|compared|versus|vs)\b")
_LINK = r"(?:\s+(?:have|has|had|been|is|are|was|were|stayed|remained|remain|stay|kept|keep|staying|being|really|must|" \
        r"pretty|quite|largely|mostly|basically|all|this|whole|period|the|over|time|year|years)){0,5}\s+"


def _metric_name(word: str) -> Optional[str]:
    for name, pat in _METRICS:
        if re.fullmatch(rf"(?:{pat})", word.strip(), re.I):
            return name
    return None


def _directions(text: str) -> List[Tuple[str, str, int, int]]:
    """(metric, direction, start, end) statements: 'volumes have stayed flat', 'flat volumes',
    'selling fewer units', 'prices were cut'."""
    found: List[Tuple[str, str, int, int]] = []
    metric_alt = "|".join(p for _n, p in _METRICS)
    for d, dpat in _DIR.items():
        # metric ... direction
        for m in re.finditer(rf"(?i)\b({metric_alt})\b{_LINK}(?:{dpat})\b", text):
            name = _metric_name(m.group(1))
            if name:
                found.append((name, d, m.start(), m.end()))
        # adjective metric ("flat volumes", "fewer units", "higher costs"); never a verb
        # ("cut prices" is a recommendation, not a statement about what happened)
        for m in re.finditer(rf"(?i)\b(?:{_ADJ[d]})\s+(?:\w+\s+)?({metric_alt})\b", text):
            name = _metric_name(m.group(1))
            if name:
                found.append((name, d, m.start(), m.end()))
    # "selling fewer", "sold less", "selling less units"
    for m in re.finditer(r"(?i)\b(?:selling|sell|sold|sells)\s+(?:a\s+lot\s+)?(fewer|less|lower|more)\b", text):
        found.append(("volume", "down" if m.group(1).lower() != "more" else "up", m.start(), m.end()))
    return found


_CONFIRM_RE = re.compile(r"(?i)^\s*(?:yes|yeah|yep|correct|right|that'?s right|exactly|true|indeed)\b")
_USER_QUOTES_RE = re.compile(r"(?i)\b(?:you (?:told|said|mentioned)|as you said|given that|we know)\b")


def established_facts(transcript: List[Dict[str, str]], case_content: str) -> Dict[str, Tuple[str, str]]:
    """metric -> (direction, where it was said). From the brief, the interviewer's own lines
    and a candidate line the interviewer confirmed ('correct, volumes are flat')."""
    facts: Dict[str, Tuple[str, str]] = {}
    sources: List[str] = [(case_content or "")[:3000]]
    for i, t in enumerate(transcript):
        c = t.get("content") or ""
        if t.get("role") == "assistant":
            sources.append(c)
        elif (t.get("role") or "user") == "user" and _USER_QUOTES_RE.search(c):
            nxt = transcript[i + 1] if i + 1 < len(transcript) else None
            if nxt and nxt.get("role") == "assistant" and _CONFIRM_RE.search(nxt.get("content") or ""):
                sources.append(c)
    for src in sources:
        for sent in re.split(r"(?<=[.!?;])\s+|\n", src):
            if "?" in sent or re.search(r"(?i)\b(?:industry|market|competitors?|rivals?|peers?)\b", sent):
                continue  # "Industry volumes are down 8%" is not a fact about ours
            for name, d, st, _en in _directions(sent):
                if _NEG.search(sent[max(0, st - 25):st]):
                    continue
                facts[name] = (d, re.sub(r"\s+", " ", sent).strip()[:120])
    return facts


def _check_contradiction(msg: str, facts: Dict[str, Tuple[str, str]]) -> Optional[Dict[str, Any]]:
    if not facts:
        return None
    for sent in re.split(r"(?<=[.!?;])\s+|\n", msg):
        if "?" in sent:
            continue
        if _NOT_OUR_CLAIM.search(sent):
            continue
        stated = _directions(sent)
        same_as_fact = {n for n, d, _s, _e in stated if n in facts and facts[n][0] == d}
        for name, d, st, en in stated:
            if name not in facts or name in same_as_fact or name == "cost":
                continue
            if re.search(r"(?i)\b(?:not|n't|no|never)\b", sent[en:]):
                continue  # "selling fewer units is not the reason"
            fact_dir, where = facts[name]
            if fact_dir == d or (fact_dir != "flat" and d == "flat"):
                continue
            if _NEG.search(sent[max(0, st - 30):st]) or _NEG.search(sent[st:en]):
                continue
            if _PLAN_BEFORE.search(sent[max(0, st - 40):st]):
                continue
            if re.match(r"(?i)\s*(?:\w+\s+){0,2}than\b", sent[en:en + 30]):
                continue  # "prices are lower than competitors" compares, it does not say they moved
            label = {"volume": "volumes", "price": "prices", "cost": "costs", "customers": "customer numbers",
                     "revenue": "revenue"}[name]
            word = {"flat": "flat", "up": "up", "down": "down"}[fact_dir]
            moved = {"up": "went up", "down": "went down", "flat": "stayed flat"}[d]
            claim = re.sub(r"\s+", " ", sent).strip()
            claim = claim if len(claim) <= 90 else claim[:90].rsplit(" ", 1)[0] + "..."
            return {"kind": "contradiction", "said": sent[st:en].strip(),
                    "message": (f"earlier it was established that {label} are {word} (\"{where}\"), but "
                                f"\"{claim}\" assumes {label} {moved}")}
    return None


# --------------------------------------------------------------------- entry point

def check_logic(transcript: Iterable[Dict[str, str]], msg: str, case_content: str = "") -> Dict[str, Any]:
    """{'logic_issue': {...}} for the first slip found, else {}. Never raises."""
    try:
        tl = [dict(t, content=(t.get("content") or "")[:2000]) for t in list(transcript or [])[-12:]]
        m = (msg or "")[:4000]
        recent_user = [(t.get("content") or "") for t in tl if (t.get("role") or "user") == "user"]
        issue = (_check_double_count(m, recent_user) or _check_extreme(m)
                 or _check_contradiction(m, established_facts(tl, case_content)))
        return {"logic_issue": issue} if issue else {}
    except Exception as e:  # noqa: BLE001 - a checker must never break a turn
        print(f"[logic_check] skipped: {type(e).__name__}: {e}")
        return {}
