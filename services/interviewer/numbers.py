"""
Deterministic numeric reasoning checks.

Two jobs, both conservative on purpose ("do not claim a number is wrong merely
because it is rough"):

1. parse_numbers()      - Indian + western scales, %, x-multipliers, currency,
                          scientific notation, Indian digit grouping (1,00,000).
2. check_arithmetic()   - explicit claims of the form "A op B [op C...] = R"
                          ("1.4 billion / 3 is about 0.46B", "30 lakh into 12
                          comes to 3.6 crore"). Only claims with an explicit
                          operator AND an explicit result marker are judged.
3. check_anchors()      - a handful of common guesstimate anchors (population,
                          household size, households) stated as fact. Flags only
                          gross (>~2x outside a generous range) errors.

A claim within +/-25% is fine; 1.25x-1.5x off is MINOR (let it stand);
beyond 1.5x is MATERIAL (it will visibly derail the downstream estimate).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

SCALES = {
    "thousand": 1e3, "k": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "lakh.": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "mil": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12,
}

_NUM = r"(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)"
_SCALE = r"(?:thousand|lakhs?|lacs?|crores?|cr|millions?|mn|mil|billions?|bn|trillion|tn|k|m|b|L)"
_NUMBER_RE = re.compile(
    r"(?P<cur>₹|rs\.?\s?|inr\s?|\$|usd\s?)?"
    r"(?P<num>" + _NUM + r")"
    r"(?:\s?(?P<scale>" + _SCALE + r")\b|(?P<pct>\s?%|\s?percent\b)|(?P<x>x)\b)?",
    re.IGNORECASE,
)


@dataclass
class Num:
    value: float          # absolute value (percent -> fraction)
    raw: str
    start: int
    end: int
    kind: str = "plain"   # plain | pct | mult | money
    scale: float = 1.0    # the scale word's factor, 1 if none
    has_scale: bool = False


def _to_float(num: str) -> Optional[float]:
    try:
        return float(num.replace(",", ""))
    except ValueError:
        return None


def parse_numbers(text: str) -> List[Num]:
    out: List[Num] = []
    for m in _NUMBER_RE.finditer(text or ""):
        base = _to_float(m.group("num"))
        if base is None:
            continue
        scale_word = m.group("scale")
        kind = "plain"
        scale = 1.0
        has_scale = False
        if scale_word:
            sw = scale_word
            # A bare "L" is lakh only when written as a capital L directly after digits ("5L").
            if sw == "L" or sw == "l":
                if sw == "l" or " " in m.group(0)[len(m.group('cur') or ''):]:
                    scale_word = None
                else:
                    scale, has_scale = 1e5, True
            if scale_word and not has_scale:
                key = sw.lower()
                # Single-letter m/b/k are only scales when glued or followed by a boundary we trust.
                scale = SCALES.get(key, 1.0)
                has_scale = scale != 1.0
        if m.group("pct"):
            kind = "pct"
            value = base / 100.0
        elif m.group("x"):
            kind = "mult"
            value = base
        else:
            value = base * scale
            if m.group("cur"):
                kind = "money"
        out.append(Num(value=value, raw=m.group(0).strip(), start=m.start(), end=m.end(),
                       kind=kind, scale=scale, has_scale=has_scale))
    return out


def has_number(text: str) -> bool:
    return bool(parse_numbers(text))


def is_numeric_only(text: str) -> bool:
    """'3', '50%', '1 crore', '460 million', '0.46B', '~ 1.2 lakh?' -> True."""
    t = (text or "").strip()
    if not t:
        return False
    nums = parse_numbers(t)
    if not nums:
        return False
    rest = t
    for n in sorted(nums, key=lambda n: -n.start):
        rest = rest[:n.start] + " " + rest[n.end:]
    rest = re.sub(r"[\s~≈=.,!?:;()\-+*/x×÷]", " ", rest.lower())
    leftover = [w for w in rest.split() if w not in {"about", "around", "approx", "approximately", "roughly",
                                                     "maybe", "say", "so", "like", "per", "year", "annum",
                                                     "month", "day", "rs", "inr", "usd", "people", "units",
                                                     "ish", "or", "is", "it's", "its", "i", "think", "guess"}]
    return len(leftover) <= 1


# ---------------------------------------------------------------------------
# Arithmetic claims
# ---------------------------------------------------------------------------
_OPS = [
    (r"multiplied by|multiply by|times|into|x|×|\*", "*"),
    (r"divided by|divide by|over|/|÷", "/"),
    (r"plus|\+", "+"),
    (r"minus|less|−|-", "-"),
]
_RESULT = r"(?:=|≈|~|->|→|equals|equal to|is equal to|gives(?: us| me)?|comes? (?:out )?to|is about|is around|is roughly|is approximately|is approx|is|which is|that's|thats|that is|so|makes|gets us|gets|would be|will be|becomes)"


@dataclass
class ArithmeticFinding:
    expression: str
    claimed: float
    actual: float
    ratio: float
    severity: str            # ok | minor | material
    op_word: str = ""
    operands: List[str] = field(default_factory=list)


def _tokenize(text: str) -> List[Tuple[str, object, int, int]]:
    """Tokens: ('num', Num), ('op', '*'|'/'|'+'|'-'), ('res', str), ('pctof', None), ('w', word)."""
    nums = parse_numbers(text)
    toks: List[Tuple[str, object, int, int]] = []
    pos = 0
    op_re = re.compile(r"\s*(?:" + "|".join(f"(?P<o{i}>{p})" for i, (p, _) in enumerate(_OPS)) + r")(?=\s|\d|₹|\$|\.|$)",
                       re.IGNORECASE)
    res_re = re.compile(r"\s*" + _RESULT + r"(?=\s|\d|₹|\$|\.|$)", re.IGNORECASE)
    of_re = re.compile(r"\s*of\b", re.IGNORECASE)

    def scan_gap(gap: str, offset: int):
        # "20-30%" is a range, not a subtraction: a bare hyphen glued to both numbers.
        if gap == "-" or gap == "\u2212":
            toks.append(("w", "-", offset, offset + 1))
            return
        i = 0
        while i < len(gap):
            if gap[i].isspace():
                i += 1
                continue
            m = res_re.match(gap, i)
            if m and m.group(0).strip():
                toks.append(("res", m.group(0).strip().lower(), offset + i, offset + m.end()))
                i = m.end()
                continue
            m = op_re.match(gap, i)
            if m and m.group(0).strip():
                sym = next(_OPS[int(k[1:])][1] for k, v in m.groupdict().items() if v)
                toks.append(("op", sym, offset + i, offset + m.end()))
                i = m.end()
                continue
            m = of_re.match(gap, i)
            if m:
                toks.append(("of", None, offset + i, offset + m.end()))
                i = m.end()
                continue
            m = re.match(r"[A-Za-z']+|[^\sA-Za-z']", gap[i:])
            w = m.group(0) if m else gap[i]
            toks.append(("w", w.lower(), offset + i, offset + i + len(w)))
            i += len(w)

    for n in nums:
        scan_gap(text[pos:n.start], pos)
        toks.append(("num", n, n.start, n.end))
        pos = n.end
    scan_gap(text[pos:], pos)
    return toks


def _eval(values: List[float], ops: List[str]) -> Optional[float]:
    """Standard precedence: * and / before + and -."""
    try:
        vals = [values[0]]
        pend: List[str] = []
        for op, v in zip(ops, values[1:]):
            if op == "*":
                vals[-1] = vals[-1] * v
            elif op == "/":
                if v == 0:
                    return None
                vals[-1] = vals[-1] / v
            else:
                pend.append(op)
                vals.append(v)
        total = vals[0]
        for op, v in zip(pend, vals[1:]):
            total = total + v if op == "+" else total - v
        return total
    except (OverflowError, ZeroDivisionError):
        return None


_FILLER = {"about", "around", "roughly", "approximately", "approx", "like", "almost", "nearly", "just",
           "we", "get", "i", "so", "then", "which", "that", "gives", "us", "=", "a", "total", "of", "the",
           "comes", "to", "out", "it's", "its", "is", "call", "it", "say"}


def _severity(ratio: float) -> str:
    if ratio <= 0:
        return "material"
    d = abs(math.log(ratio))
    if d <= math.log(1.25):
        return "ok"
    if d <= math.log(1.5):
        return "minor"
    return "material"


def check_arithmetic(text: str) -> List[ArithmeticFinding]:
    toks = _tokenize(text or "")
    findings: List[ArithmeticFinding] = []
    i = 0
    n = len(toks)
    while i < n:
        if toks[i][0] != "num":
            i += 1
            continue
        # Pattern A: P% of X <res> R
        if (toks[i][1].kind == "pct" and i + 2 < n and toks[i + 1][0] == "of" and toks[i + 2][0] == "num"):
            operands = [toks[i][1], toks[i + 2][1]]
            values = [toks[i][1].value, toks[i + 2][1].value]
            ops = ["*"]
            j = i + 3
        else:
            operands = [toks[i][1]]
            values = [toks[i][1].value]
            ops = []
            j = i + 1
            while j + 1 < n and toks[j][0] == "op" and toks[j + 1][0] == "num":
                ops.append(toks[j][1])
                operands.append(toks[j + 1][1])
                values.append(toks[j + 1][1].value)
                j += 2
            if not ops:
                i += 1
                continue
        # Optional filler words, then a result marker, then optional filler, then the result.
        k = j
        skipped = 0
        while k < n and toks[k][0] == "w" and toks[k][1] in _FILLER and skipped < 3:
            k += 1
            skipped += 1
        if k >= n or toks[k][0] != "res":
            i = j
            continue
        k += 1
        skipped = 0
        while k < n and toks[k][0] in ("w", "res") and (toks[k][0] == "res" or toks[k][1] in _FILLER) and skipped < 4:
            k += 1
            skipped += 1
        if k >= n or toks[k][0] != "num":
            i = j
            continue
        result: Num = toks[k][1]
        actual = _eval(values, ops)
        if actual is None or actual == 0 or not math.isfinite(actual):
            i = k + 1
            continue
        claimed_candidates = [result.value]
        # Dropped scale on the result ("1.4 billion / 3 = 0.46"): try the first operand's scale.
        first_scaled = next((o for o in operands if o.has_scale), None)
        if not result.has_scale and result.kind == "plain" and first_scaled is not None:
            claimed_candidates.append(result.value * first_scaled.scale)
        if result.has_scale and not any(o.has_scale for o in operands):
            claimed_candidates.append(result.value / result.scale)
        # "30 / 120 = 25" (a percentage written without the % sign).
        if result.kind == "plain" and not result.has_scale and 0 < abs(actual) < 1:
            claimed_candidates.append(result.value / 100.0)
        best = min(claimed_candidates, key=lambda c: abs(math.log(abs(c) / abs(actual))) if c else float("inf"))
        ratio = (best / actual) if actual else 0.0
        sev = _severity(abs(ratio)) if (best > 0) == (actual > 0) else "material"
        expr = text[toks[i][2]:toks[k][3]]
        findings.append(ArithmeticFinding(expression=expr.strip(), claimed=best, actual=actual, ratio=ratio,
                                          severity=sev, op_word=ops[0] if ops else "",
                                          operands=[o.raw for o in operands]))
        i = k + 1
    return findings


def describe_op(op: str) -> str:
    return {"*": "multiplication", "/": "division", "+": "addition", "-": "subtraction"}.get(op, "calculation")


def spoken_number(n: Num) -> str:
    return n.raw


def correction_line(f: ArithmeticFinding) -> str:
    """Deterministic, specific, non-solving correction for one arithmetic slip."""
    kind = describe_op(f.op_word)
    if f.operands and len(f.operands) >= 2 and f.op_word in ("*", "/"):
        joiner = " divided by " if f.op_word == "/" else " times "
        return f"Check that {kind}: {joiner.join(f.operands[:3])}."
    return f"Check that {kind} again before you build on it."


# ---------------------------------------------------------------------------
# Anchors: generous ranges; flag only gross errors
# ---------------------------------------------------------------------------
@dataclass
class Anchor:
    key: str
    subject: str            # regex that must match near the number
    context: Optional[str]  # regex that must appear in the same sentence (e.g. 'india')
    lo: float
    hi: float
    spoken: str             # what the interviewer says the anchor is
    exclude: Optional[str] = None  # sub-population qualifiers that make the anchor not apply


_SUBSET = (r"\b(urban|rural|working|adult|adults|male|female|women|men|target|addressable|middle[- ]class|"
           r"smartphone|internet|online|metro|tier|city|cities|state|states|young|youth|elderly|senior|kids|"
           r"children|student|students|employed|literate|affluent|rich|poor|income|segment|bpl|apl)\b")


ANCHORS: List[Anchor] = [
    Anchor("india_population", r"population", r"\bindia(n|'s)?\b", 1.2e9, 1.6e9, "about 1.4 billion", _SUBSET),
    Anchor("us_population", r"population", r"\b(us|u\.s\.|usa|united states|america(n|'s)?)\b", 3.0e8, 3.5e8,
           "about 335 million", _SUBSET),
    Anchor("world_population", r"population", r"\b(world|global|earth)\b", 7.5e9, 8.5e9, "about 8 billion", _SUBSET),
    Anchor("india_households", r"households", r"\bindia(n|'s)?\b", 2.5e8, 3.3e8, "roughly 300 million", _SUBSET),
    Anchor("household_size", r"(household size|people per household|members per household|persons per household|"
                              r"people in a household|family size|per family)", None, 2.0, 6.0,
           "typically 4 to 5 people in India"),
]
_GROSS = 2.0  # outside [lo/2, hi*2] only


@dataclass
class AnchorFinding:
    anchor: Anchor
    value: float
    raw: str
    severity: str  # ok | material


def check_anchors(text: str) -> List[AnchorFinding]:
    out: List[AnchorFinding] = []
    t = text or ""
    for sent in re.split(r"(?<=[.!?;])\s+|\n+", t):
        low = sent.lower()
        for a in ANCHORS:
            sm = re.search(a.subject, low)
            if not sm:
                continue
            if a.context and not re.search(a.context, low):
                continue
            if a.exclude and re.search(a.exclude, low):
                continue
            # The number must come within ~8 words after the subject (or right before it).
            nums = parse_numbers(sent)
            best = None
            for n in nums:
                gap = low[sm.end():n.start] if n.start >= sm.end() else low[n.end:sm.start()]
                if len(gap.split()) <= 8 and n.kind in ("plain", "money"):
                    best = n
                    break
            if best is None:
                continue
            val = best.value
            sev = "ok" if (a.lo / _GROSS) <= val <= (a.hi * _GROSS) else "material"
            out.append(AnchorFinding(anchor=a, value=val, raw=best.raw, severity=sev))
    return out


def anchor_correction_line(f: AnchorFinding) -> str:
    label = f.anchor.key.replace("_", " ")
    if f.anchor.key == "household_size":
        return f"Household size is {f.anchor.spoken}, not {f.raw} — rework it from there."
    if f.anchor.key.endswith("population"):
        region = {"india_population": "India's", "us_population": "The US", "world_population": "The world's"}[f.anchor.key]
        return f"{region} population is {f.anchor.spoken}, not {f.raw} — rework it from there."
    return f"Check the {label}: it's {f.anchor.spoken}, not {f.raw}."
