"""
Live number checks for a candidate's turn (text interviewer, V12.1).

WHY THIS EXISTS (15-day review of real sessions, Oct 2026)
The interviewer answered "Right." or "That's a solid estimate" to:
    "Assuming average household size of 4 there are 35 million households"  (1.4B / 4 = 350M)
    "Urban 300M fans, Rural 400M fans / total 600M fans"                     (= 700M)
    "35 crore is 35 million"                                                  (= 350M)
    "3% of 600 = 12M"                                                         (= 18M)
    "total: 7 lakh building" after 6 + 2 + 1 lakh                             (= 9 lakh)
    a weekly office commute built as daily x 7
and told a candidate whose 24.5x3 + 14.7x2 + 4.9 = 107.8 was exactly right that
"there might be some confusion". gpt-4o-mini cannot be trusted to recompute. Python
can, so this module does the arithmetic and the model only words the result.

TWO LAYERS
  check_turn()  deterministic, no network: explicit equations, unit conversions,
                listed totals, households from a population, a city or country
                population against a reference range, an office week built as x 7,
                a dangling expression answered by a bare number, and what a final
                answer implies per person. It records what CHECKED OUT as well, so
                the interviewer may say the arithmetic is right (only the arithmetic).
  audit_turn()  optional, one small JSON model call for arithmetic that spans turns
                ("1.3%" from numbers given three turns earlier). The model only
                WRITES the expression; Python evaluates it and decides, and the
                stated number must appear verbatim in the candidate's message.

Precision over recall: telling a learner that correct work is wrong is worse than
missing a slip, so a check reports an issue only when EVERY reading of a bare number
(as written, or in the scale its neighbours use) fails.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------- number parsing

_WORD_SCALE = {
    "k": 1e3, "thousand": 1e3, "thousands": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "lacks": 1e5,
    "cr": 1e7, "crs": 1e7, "crore": 1e7, "crores": 1e7,
    "mn": 1e6, "mil": 1e6, "million": 1e6, "millions": 1e6,
    "bn": 1e9, "billion": 1e9, "billions": 1e9,
}
# Single letters count as a scale only when written against the digits ("20M", "5L").
_ATTACHED_ONLY = {"m": 1e6, "b": 1e9, "l": 1e5}
_INDIAN_UNITS = {"lakh", "lakhs", "lac", "lacs", "lacks", "cr", "crs", "crore", "crores", "l"}

_QTY_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<cur>(?:rs\.?|inr|₹|\$|usd)\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<sp>\s?)"
    r"(?P<unit>%|percent\b|per\s?cent\b|"
    r"thousands?\b|lakhs?\b|lacs?\b|lacks\b|crores?\b|crs?\b|mn\b|mil\b|millions?\b|bn\b|billions?\b|"
    r"k\b|m\b|b\b|l\b)?",
    re.I)


def _valid_commas(num: str) -> bool:
    if "," not in num:
        return True
    whole = num.split(".")[0]
    return bool(re.fullmatch(r"\d{1,3}(,\d{3})+", whole) or re.fullmatch(r"\d{1,2}(,\d{2})*,\d{3}", whole))


class Qty:
    """One number as the candidate wrote it."""
    __slots__ = ("value", "raw", "scale", "pct", "cur", "start", "end", "indian", "alts")

    def __init__(self, value, raw, scale, pct, cur, start, end, indian, alts=None):
        self.value, self.raw, self.scale, self.pct = value, raw, scale, pct
        self.cur, self.start, self.end, self.indian = cur, start, end, indian
        self.alts = alts or ()  # other readings (a range "2-3" reads as 2, 2.5 or 3)

    @property
    def explicit(self) -> bool:
        return self.scale is not None

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Qty({self.raw!r}={self.value})"


def _normalize(text: str) -> str:
    t = (text or "").replace("−", "-").replace("–", "-").replace("×", "*")
    t = t.replace("÷", "/").replace("≈", "~")
    t = re.sub(r"[ \t]{2,}", " ", t)  # "1.43  billion" is one quantity
    # "420 000000000" (digit groups typed with a space) is one number.
    t = re.sub(r"(?<![\d.,])(\d{1,3}) (0\d{2}(?:\d{3})*)(?![\d.,])", r"\1\2", t)
    # Labels, not arithmetic: fiscal years ("2023-24", "FY24-25"), "24/7", and a year
    # after a preposition ("in 2024") are never operands.
    t = re.sub(r"(?i)\b(?:fy\s?)?(?:19|20)?\d\d\s?[-/]\s?(?:19|20)?\d\d\b(?=\s*(?:revenue|sales|profit|"
               r"budget|data|numbers?|figures?|is|was|=|:|,|\.|\)|$))", "fyyear", t)
    t = re.sub(r"(?i)\bfy\s?\d\d\b", "fyyear", t)
    t = re.sub(r"\b24\s?[/x*]\s?7\b", "allweek", t)
    t = re.sub(r"(?i)\b(in|since|by|from|till|until|of|during|for|year|before|after)\s+(?:19|20)\d\d\b",
               r"\1 yearlabel", t)
    return t


def parse_quantities(text: str) -> List[Qty]:
    t = _normalize(text)
    litres = bool(re.search(r"\b(litres?|liters?|ml)\b", t, re.I))
    out: List[Qty] = []
    for m in _QTY_RE.finditer(t):
        num = m.group("num")
        unit = (m.group("unit") or "").lower().replace(" ", "")
        end = m.end()
        if unit in _ATTACHED_ONLY and (m.group("sp") or (unit == "l" and litres)):
            unit, end = "", m.end("num")
        if not _valid_commas(num):
            continue
        try:
            base = float(num.replace(",", ""))
        except (ValueError, OverflowError):
            continue
        if not math.isfinite(base) or len(num) > 18:
            continue
        pct = unit == "%" or (unit.startswith("per") and unit.endswith("cent"))
        scale = None if pct or not unit else (_WORD_SCALE.get(unit) or _ATTACHED_ONLY.get(unit))
        indian = unit in _INDIAN_UNITS
        if scale and scale in (1e3, 1e5):
            # "2 lakh crore" / "50 thousand crore" is ONE scale
            cm = re.match(r"\s?(crores?|cr)\b", t[end:], re.I)
            if cm:
                scale, end, indian = scale * 1e7, end + cm.end(), True
        value = base * scale if scale else base
        cur = (m.group("cur") or "").strip().lower().rstrip(".")
        cur = "usd" if cur in ("$", "usd") else "inr" if cur else ""
        out.append(Qty(value, t[m.start():end].strip(), scale, pct, cur,
                       m.start(), end, indian))
    return out


def first_quantity(text: str) -> Optional[Qty]:
    qs = parse_quantities(text)
    return qs[0] if qs else None


# ----------------------------------------------------------------- formatting

def _sig(v: float, digits: int = 3) -> str:
    if v is None or not math.isfinite(v):
        return "a very large number"
    if v == 0:
        return "0"
    r = round(v, -int(math.floor(math.log10(abs(v)))) + (digits - 1))
    if abs(r) >= 1000:
        return f"{r:,.0f}"
    s = f"{r:.6f}".rstrip("0").rstrip(".")
    return s


def fmt_number(v: float, indian_first: bool = True) -> str:
    """350000000 -> '35 crore (350 million)'; 900000 -> '9 lakh (900,000)'."""
    a = abs(v)
    if a >= 1e7:
        ind = f"{_sig(v / 1e7)} crore"
    elif a >= 1e5:
        ind = f"{_sig(v / 1e5)} lakh"
    else:
        ind = None
    if a >= 1e9:
        intl = f"{_sig(v / 1e9)} billion"
    elif a >= 1e6:
        intl = f"{_sig(v / 1e6)} million"
    else:
        intl = _sig(v)
    if not ind:
        return intl
    return f"{ind} ({intl})" if indian_first else f"{intl} ({ind})"


def fmt_in(v: float, indian: bool) -> str:
    """One system only: '350 million' or '35 crore'."""
    a = abs(v)
    if indian and a >= 1e7:
        return f"{_sig(v / 1e7)} crore"
    if indian and a >= 1e5:
        return f"{_sig(v / 1e5)} lakh"
    if a >= 1e9:
        return f"{_sig(v / 1e9)} billion"
    if a >= 1e6:
        return f"{_sig(v / 1e6)} million"
    return _sig(v)


def fmt_pct(v: float) -> str:
    return f"{_sig(v * 100)}%"


# ----------------------------------------------------------------- safe arithmetic

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.USub, ast.UAdd)


def safe_eval(expr: str) -> Optional[float]:
    """Evaluate plain arithmetic (digits, + - * / and brackets). None if not that."""
    e = (expr or "").replace("×", "*").replace("x", "*").replace("[", "(").replace("]", ")")
    e = e.replace(",", "")
    if not re.fullmatch(r"[\d\s.+\-*/()eE]+", e) or len(e) > 300:
        return None
    try:
        tree = ast.parse(e, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None
    try:
        val = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, {})  # noqa: S307 - AST-whitelisted
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _rel(a: float, b: float) -> float:
    d = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / d


# ----------------------------------------------------------------- equation chains

_APPROX_WORDS = {"approx", "approximately", "approx.", "about", "around", "roughly", "nearly",
                 "almost", "circa", "ca", "~", "close", "only", "just", "abt", "aprox"}
_STOP_WORDS = {
    "and", "or", "but", "then", "so", "if", "which", "that", "this", "these", "those", "is",
    "are", "was", "were", "of", "for", "from", "to", "in", "on", "at", "with", "assuming",
    "assume", "take", "taking", "considering", "because", "since", "where", "when", "while",
    "each", "we", "i", "you", "they", "it", "he", "she", "will", "would", "can", "could",
    "should", "not", "no", "also", "be", "been", "have", "has", "had", "do", "does", "did",
    "remaining", "rest", "other", "others", "plus", "minus", "by", "as", "the", "a", "an",
}
_TOKEN_RE = re.compile(
    r"(?P<op_w>multiplied\s+by|multiply\s+by|divided\s+by|times|into|plus|minus)\b"
    r"|(?P<eq_w>is\s+equal\s+to|equal\s+to|equals?|comes?\s+out\s+to|comes?\s+to|which\s+gives|"
    r"that\s+gives|gives\s+us|gives\s+me|gives|results?\s+in|which\s+is|that\s+is|that'?s|thats|is|makes)\b"
    r"|(?P<eq_s>==|=>|=|~)"
    r"|(?P<op_s>[+\-*/xX])"
    r"|(?P<lp>[(\[])|(?P<rp>[)\]])"
    r"|(?P<word>approx\.|[A-Za-z][A-Za-z']*)"
    r"|(?P<sep>[,;:\n!?]|\.(?!\d))",
    re.I)


def _tokens(text: str) -> List[Tuple[str, Any, int, int]]:
    """(kind, value, start, end) over the normalized text. Quantities first."""
    t = _normalize(text)
    qs = parse_quantities(text)
    toks: List[Tuple[str, Any, int, int]] = []
    pos = 0
    qi = 0
    while pos < len(t):
        if qi < len(qs) and qs[qi].start <= pos:
            q = qs[qi]
            if q.start == pos or q.start < pos:
                toks.append(("Q", q, q.start, q.end))
                pos = max(pos, q.end)
                qi += 1
                continue
        nxt_q = qs[qi].start if qi < len(qs) else len(t)
        m = _TOKEN_RE.search(t, pos, nxt_q)
        if not m:
            pos = nxt_q
            continue
        if m.group("op_w"):
            w = re.sub(r"\s+", " ", m.group("op_w").lower())
            op = "+" if w == "plus" else "-" if w == "minus" else "/" if w.startswith("divided") else "*"
            toks.append(("OP", op, m.start(), m.end()))
        elif m.group("eq_w") or m.group("eq_s"):
            toks.append(("EQ", (m.group(0) or "").lower(), m.start(), m.end()))
        elif m.group("op_s"):
            s = m.group("op_s")
            toks.append(("OPS", "*" if s in "xX" else s, m.start(), m.end()))
        elif m.group("lp"):
            toks.append(("LP", "(", m.start(), m.end()))
        elif m.group("rp"):
            toks.append(("RP", ")", m.start(), m.end()))
        elif m.group("word"):
            toks.append(("W", m.group("word").lower().rstrip("."), m.start(), m.end()))
        else:
            toks.append(("SEP", m.group(0), m.start(), m.end()))
        pos = m.end()
    return toks


class _Chain:
    def __init__(self):
        self.items: List[Any] = []   # Qty | "+" "-" "*" "/" "(" ")"
        self.ops = 0
        self.result: Optional[Qty] = None
        self.more_results: List[Qty] = []  # "= 2,50,00,00,000 = 2500 crore"
        self.approx = False
        self.start = None
        self.eq_start = None


def _is_op_tok(tok) -> bool:
    return tok[0] in ("OP", "OPS")


def _chains(text: str) -> List[_Chain]:
    """Every 'operand (op operand)* = result' run in the text. Linear: a failed parse
    resumes where it stopped, not one token later (a 20,000-char "1-1-1-..." must not
    cost seconds)."""
    toks = _tokens(text)
    out: List[_Chain] = []
    i, n = 0, len(toks)
    while i < n:
        if toks[i][0] not in ("Q", "LP"):
            i += 1
            continue
        ch, j = _parse_chain(toks, i)
        if ch is not None and ch.result is not None:
            out.append(ch)
        i = max(i + 1, j)
    return out


_TIME_WORDS = {"day", "days", "week", "weeks", "month", "months", "year", "years", "annum", "annually",
               "daily", "weekly", "monthly", "yearly", "hour", "hours"}


def _skip_words(toks, j, allow_units: bool, limit: int = 2):
    """Skip up to `limit` filler words: unit nouns ('households'), a rate ('a month',
    'per day') or approx words."""
    k, skipped, approx = j, 0, False
    while k < len(toks) and toks[k][0] == "W" and skipped < limit:
        w = toks[k][1]
        if (allow_units and w in ("a", "per", "every", "each", "an") and k + 1 < len(toks)
                and toks[k + 1][0] == "W" and toks[k + 1][1] in _TIME_WORDS):
            k += 2
            skipped += 1
            continue
        if w in _APPROX_WORDS or w in ("to",) and skipped and toks[k - 1][1] == "close":
            approx = True
        elif not allow_units or w in _STOP_WORDS:
            break
        k += 1
        skipped += 1
    return k, approx


def _pct_of(toks, j, ch: _Chain):
    """'5% of 20% of 100 crore' -> ( 5% * ( 20% * 100 crore ) ). j points at a % Qty
    followed by 'of'. Returns the index after the chain, or None."""
    q = toks[j][1]
    k = j + 2
    if k < len(toks) and toks[k][0] == "W" and toks[k][1] == "the":
        k += 1
    if k >= len(toks) or toks[k][0] != "Q":
        return None
    nxt = toks[k][1]
    ch.items += ["(", q, "*"]
    ch.ops += 1
    if nxt.pct and k + 1 < len(toks) and toks[k + 1][0] == "W" and toks[k + 1][1] == "of":
        r = _pct_of(toks, k, ch)
        if r is None:
            return None
        ch.items.append(")")
        return r
    ch.items += [nxt, ")"]   # "50% of 40%" ends on a share: 0.5 * 0.4
    return k + 1


def _parse_operand(toks, j, ch: _Chain, depth: int = 0):
    """Q | Q% of Q | ( expr ). Returns new index or None."""
    if depth > 12:
        return None  # nobody nests brackets this deep in a calculation
    j, _ = _skip_words(toks, j, allow_units=False)
    if j >= len(toks):
        return None
    kind = toks[j][0]
    if kind == "Q":
        q = toks[j][1]
        if q.pct and j + 2 < len(toks) and toks[j + 1][0] == "W" and toks[j + 1][1] == "of":
            r = _pct_of(toks, j, ch)
            if r is not None:
                return r
            return None
        ch.items.append(q)
        return j + 1
    if kind == "LP":
        ch.items.append("(")
        k = _parse_operand(toks, j + 1, ch, depth + 1)
        if k is None:
            return None
        while True:
            k, _ = _skip_words(toks, k, allow_units=True)
            if k < len(toks) and _is_op_tok(toks[k]):
                ch.items.append(toks[k][1])
                ch.ops += 1
                k = _parse_operand(toks, k + 1, ch, depth + 1)
                if k is None:
                    return None
                continue
            break
        if k < len(toks) and toks[k][0] == "RP":
            ch.items.append(")")
            return k + 1
        return None
    return None


def _is_range(toks, j, ch: _Chain) -> bool:
    """'10-15%', '0-25 age group', '25-40': a hyphen written tight between an
    ascending pair is a range."""
    if toks[j][1] != "-" or not ch.items or not isinstance(ch.items[-1], Qty):
        return False
    left = ch.items[-1]
    if j + 1 >= len(toks) or toks[j + 1][0] != "Q":
        return False
    right = toks[j + 1][1]
    tight = toks[j][2] == left.end and toks[j + 1][2] == toks[j][3]
    return tight and left.value < right.value


def _parse_chain(toks, i):
    """(chain, next_index). On failure the index is where parsing stopped."""
    ch = _Chain()
    ch.start = toks[i][2]
    j = _parse_operand(toks, i, ch)
    if j is None:
        return None, i + 1
    while j < len(toks):
        j, _ = _skip_words(toks, j, allow_units=True)
        if j >= len(toks):
            break
        tok = toks[j]
        if _is_op_tok(tok) and tok[1] == "/" and j + 1 < len(toks) and toks[j + 1][0] == "W" \
                and toks[j + 1][1] in _TIME_WORDS | {"unit", "units", "person", "head", "household", "km", "litre"}:
            j += 2  # "1.5 lakh/day" is a rate written as a unit, not a division
            continue
        if _is_op_tok(tok):
            if _is_range(toks, j, ch):
                # "2-3 cups a day x 30": the range is one operand, read at either end or the middle
                left, right = ch.items[-1], toks[j + 1][1]
                ch.items[-1] = Qty((left.value + right.value) / 2, f"{left.raw}-{right.raw}", right.scale,
                                   right.pct, right.cur, left.start, right.end, right.indian,
                                   alts=(left.value, right.value))
                j += 2
                continue
            ch.items.append(tok[1])
            ch.ops += 1
            k = _parse_operand(toks, j + 1, ch)
            if k is None:
                return None, j + 1
            j = k
            continue
        if tok[0] == "EQ":
            ch.eq_start = tok[2]
            k = j + 1
            while k < len(toks) and toks[k][0] == "EQ":   # "= =" / "is ="
                k += 1
            k, approx = _skip_words(toks, k, allow_units=False, limit=2)
            if tok[1] == "~":
                approx = True
            if k < len(toks) and toks[k][0] == "Q":
                ch.result = toks[k][1]
                ch.approx = approx
                k += 1
                while True:  # restated results: "= 2,50,00,00,000 = 2500 crore"
                    k2, _ = _skip_words(toks, k, allow_units=True)
                    if k2 < len(toks) and toks[k2][0] == "EQ":
                        k3 = k2 + 1
                        while k3 < len(toks) and toks[k3][0] == "EQ":
                            k3 += 1
                        k3, _ = _skip_words(toks, k3, allow_units=False)
                        if k3 < len(toks) and toks[k3][0] == "Q":
                            ch.more_results.append(toks[k3][1])
                            k = k3 + 1
                            continue
                    break
                return ch, k
            return None, k
        break
    return None, max(i + 1, j)


def _interps(q: Qty, sib: List[float], is_result: bool, has_div: bool) -> List[float]:
    if q.alts:
        base = [q.value] + list(q.alts)
        if q.pct:
            return [v / 100.0 for v in base]
        if q.explicit:
            return base
        return base + [v * s for v in base for s in sib if v < 1000]
    if q.pct:
        return [q.value / 100.0, q.value] if is_result else [q.value / 100.0]
    if q.explicit:
        return [q.value]
    vals = [q.value]
    if q.value < 1000:
        vals += [q.value * s for s in sib]
    if is_result and has_div and q.value <= 100:
        vals.append(q.value / 100.0)
    return list(dict.fromkeys(vals))


def _eval_items(items: List[Any], vals: List[float], left_to_right: bool) -> Optional[float]:
    it = iter(vals)
    seq = [next(it) if isinstance(x, Qty) else x for x in items]
    if not left_to_right:
        expr = "".join(repr(x) if isinstance(x, float) else x for x in seq)
        return safe_eval(expr)

    def fold(k, depth=0):
        if depth > 50:
            return None, k
        acc, op = None, None
        while k < len(seq):
            x = seq[k]
            if x == "(":
                v, k = fold(k + 1, depth + 1)
                if v is None:
                    return None, k
            elif x == ")":
                return acc, k + 1
            elif x in ("+", "-", "*", "/"):
                op = x
                k += 1
                continue
            else:
                v = x
                k += 1
            if acc is None:
                acc = v
            elif op == "+":
                acc += v
            elif op == "-":
                acc -= v
            elif op == "*":
                acc *= v
            elif op == "/":
                if v == 0:
                    return None, k
                acc /= v
        return acc, k

    try:
        v, _ = fold(0)
    except (OverflowError, RecursionError, TypeError):
        return None
    return v if (v is None or math.isfinite(v)) else None


_COUNT_NOUNS = {"bottles", "bottle", "packets", "packet", "cups", "cup", "units", "unit", "people", "persons",
                "households", "homes", "cars", "fans", "laptops", "phones", "smartphones", "sarees", "orders",
                "customers", "users", "riders", "passengers", "trips", "stores", "shops", "outlets", "pcs",
                "vendors", "members", "students", "rickshaws", "pieces", "items", "tickets"}
# Physical and time units: a chain that moves between two of these converts units
# on the way (ml -> litres, minutes -> hours), which this checker does not model.
_MEASURE_NOUNS = {"litres", "liters", "litre", "liter", "ml", "kg", "kgs", "kilograms", "kilogram", "grams",
                  "gram", "g", "gm", "gms", "tonnes", "tonne", "tons", "ton", "kwh", "km", "kms", "kilometres",
                  "kilometers", "metres", "meters", "minutes", "minute", "mins", "min", "hours", "hour", "hrs",
                  "hr", "seconds", "secs", "sec", "days", "day", "weeks", "week", "months", "month", "years",
                  "year", "mw", "gw", "sq", "acres", "hectares"}


def _noun_after(text_norm: str, q: Qty) -> Optional[str]:
    m = re.match(r"\s*([a-z]+)", text_norm[q.end:q.end + 20].lower())
    return m.group(1) if m else None


def _chain_nouns(ch: _Chain, tn: str):
    qs = [x for x in ch.items if isinstance(x, Qty)]
    return [_noun_after(tn, q) for q in qs], (_noun_after(tn, ch.result) if ch.result else None)


def _unit_switch(ch: _Chain, text: str) -> Optional[Dict[str, Any]]:
    """'46 cr bottles a month into 12 thats 552 cr litres a year': the count became a
    volume with no conversion in between."""
    tn = _normalize(text)
    qs = [x for x in ch.items if isinstance(x, Qty)]
    nouns, r = _chain_nouns(ch, tn)
    a = nouns[0] if nouns else None
    if not a or not r or a == r:
        return None
    if a in _COUNT_NOUNS and r in _MEASURE_NOUNS and not any(n in _MEASURE_NOUNS for n in nouns[1:]):
        # "each bottle is 500 ml so 2 crore bottles is 1 crore litres": a size per item in
        # the result's kind of unit is stated, so the conversion happened in their head.
        fam = _UNIT_TABLE.get(r, (r,))[0]
        for q in parse_quantities(text):
            if q is ch.result or q.start == ch.result.start or q.pct:
                continue
            u = _noun_after(tn, q)
            if u and (u == r or (u in _UNIT_TABLE and _UNIT_TABLE[u][0] == fam)):
                return None
        return {"kind": "units", "verified": False, "expr": f"{qs[0].raw} {a}", "stated": f"{ch.result.raw} {r}",
                "message": (f"the {a} turned into {r} without a conversion: {qs[0].raw} {a} is a count of {a}, "
                            f"so the result is still {a} until you multiply by how many {r} each one holds")}
    return None


def _converts_units(ch: _Chain, tn: str) -> bool:
    """True when the chain changes physical/time unit on the way (not modelled)."""
    nouns, r = _chain_nouns(ch, tn)
    measures = {n for n in nouns if n in _MEASURE_NOUNS}
    if r in _MEASURE_NOUNS and measures and r not in measures:
        return True
    return len(measures) >= 2 and r not in measures


_RATE_AFTER = re.compile(r"\s*(?:/|per\b|a\s+(?:day|week|month|year)|an?\s+hour|every\b|each\b|daily|weekly|"
                         r"monthly|yearly|annually|p\.?a\.?|dollars?|usd|rupees?|inr|rs)", re.I)


def _with_relative_pct(items: List[Any]) -> Optional[List[Any]]:
    """'2 cr - 20% = 1.6 cr', '2 cr + 10% growth = 2.2 cr': Q +/- p% means Q*(1 +/- p).
    Returns the re-read items, or None when the chain has no such step."""
    out, changed, k = [], False, 0
    while k < len(items):
        x = items[k]
        if (x in ("+", "-") and k + 1 < len(items) and isinstance(items[k + 1], Qty) and items[k + 1].pct
                and not (k + 2 < len(items) and items[k + 2] == "*")):
            p = items[k + 1]
            f = 1 + p.value / 100.0 if x == "+" else 1 - p.value / 100.0
            out += ["*", Qty(f, p.raw, 1.0, False, "", p.start, p.end, False)]
            changed, k = True, k + 2
            continue
        out.append(x)
        k += 1
    return out if changed else None


def _check_conversion(ch: _Chain, a: Qty, res: Qty, tn: str) -> Optional[Dict[str, Any]]:
    """'35 crore is 35 million': only a bare restatement - nothing between the two
    figures but the 'is', no rate, noun or currency change on either side."""
    if not (a.explicit and res.explicit and a.indian != res.indian):
        return None
    if a.cur != res.cur:
        return None
    between = tn[a.end:ch.eq_start or a.end]
    if re.search(r"[a-z]", between, re.I):
        return None  # "1.4 billion PEOPLE which is 35 crore HOUSEHOLDS"
    if _RATE_AFTER.match(tn[a.end:]) or _RATE_AFTER.match(tn[res.end:]):
        return None  # "1 lakh per month is 1.2 million per year"
    after = re.match(r"\s*([a-z]+)", tn[res.end:res.end + 20].lower())
    if after and after.group(1) not in _STOP_WORDS and after.group(1) not in ("so", "and", "which", "i", "we"):
        return None  # a noun after the figure: a different quantity
    tol = 0.2 if ch.approx else 0.05
    if _rel(a.value, res.value) <= tol:
        return {"kind": "conversion", "verified": True, "expr": a.raw, "stated": res.raw}
    return {"kind": "conversion", "verified": False, "expr": a.raw, "stated": res.raw,
            "expected": a.value, "stated_value": res.value,
            "message": f"{a.raw} is {fmt_in(a.value, res.indian)}, not {res.raw}"}


def _check_chain(ch: _Chain, text: str) -> Optional[Dict[str, Any]]:
    qs = [x for x in ch.items if isinstance(x, Qty)]
    res = ch.result
    if not qs or res is None:
        return None
    tn = _normalize(text)
    switch = _unit_switch(ch, text)
    if switch:
        return switch
    if _converts_units(ch, tn):
        return None
    results = [res] + list(ch.more_results)
    if len(results) > 1:
        rnouns = {_noun_after(tn, r) for r in results}
        if len({n for n in rnouns if n in _MEASURE_NOUNS}) > 1:
            return None  # "= 200 crore ml = 2 crore litres": converts units between results
    ops = [x for x in ch.items if x in ("+", "-", "*", "/")]
    raw_expr = tn[ch.start:(ch.eq_start or ch.start)].strip(" =~:")
    if not ops:
        return _check_conversion(ch, qs[0], res, tn) if len(qs) == 1 else None
    if any(q.value is None for q in qs):
        return None
    if re.search(r"\b(?:tier|tiers|class|grade|phase|type|category|level|step|q|fy|round|ratio|rank|"
                 r"chapter|section|version|v)\s*$", tn[max(0, (ch.start or 0) - 12):ch.start or 0], re.I):
        return None  # "tier 2/3 = 225 cr" is a label, not a division
    if res.explicit and not any(q.explicit or q.pct or q.value >= 100 for q in qs):
        return None  # small bare numbers can't produce a scaled figure: "1/2 of them"
    variants = [ch.items]
    rel = _with_relative_pct(ch.items)
    if rel is not None:
        variants.append(rel)
    sib = sorted({q.scale for q in qs + [res] if q.explicit and q.scale != 1.0})
    has_div = "/" in ops
    has_pct = any(q.pct for q in qs)
    only_add = all(o in ("+", "-") for o in ops)
    tol = (0.06 if only_add else 0.15) + (0.10 if ch.approx else 0.0)
    best = None  # (distance, value)
    complement = False
    for items in variants:
        vqs = [x for x in items if isinstance(x, Qty)]
        cand_lists = [_interps(q, sib, False, has_div) for q in vqs]
        res_list = [x for r_ in results for x in _interps(r_, sib, True, has_div or has_pct)]
        combos = 1
        for cl in cand_lists:
            combos *= len(cl)
        if combos > 512:
            cand_lists = [cl[:1] for cl in cand_lists]
        for vals in product(*cand_lists):
            for ltr in (False, True):
                v = _eval_items(items, list(vals), ltr)
                if v is None:
                    continue
                for r in res_list:
                    if _rel(v, r) <= tol:
                        return {"kind": "arithmetic", "verified": True, "expr": raw_expr, "stated": res.raw}
                if res.pct and has_div and abs((1 - v) - res.value / 100.0) <= 0.02:
                    complement = True
                if v > 0 and res.value > 0:
                    d = abs(math.log10(v) - math.log10(res_list[0] if res.pct else res.value))
                    if best is None or d < best[0]:
                        best = (d, v)
    if best is None or rel is not None:
        return None  # ambiguous "+ 10%" reading that fits neither way: stay quiet
    expected = best[1]
    indian = res.indian or any(q.indian for q in qs)
    if res.pct:
        exp_text = fmt_pct(expected)
        if complement:
            msg = (f"{raw_expr} works out to {exp_text}, which is the cost share, not the margin: "
                   f"the margin is 100% minus that, i.e. profit / revenue")
            return {"kind": "complement", "verified": False, "expr": raw_expr, "stated": res.raw,
                    "expected": expected, "message": msg}
    else:
        exp_text = fmt_number(expected, indian_first=indian)
    stated_v = res.value / 100.0 if res.pct else res.value
    factor = expected / stated_v if stated_v else None
    msg = f"{raw_expr} = {exp_text}, not {res.raw}"
    if factor and (factor >= 3 or factor <= 1 / 3):
        msg += f" (about {_sig(factor if factor >= 1 else 1 / factor, 2)}x {'too small' if factor > 1 else 'too big'})"
    return {"kind": "arithmetic", "verified": False, "expr": raw_expr, "stated": res.raw,
            "expected": expected, "stated_value": res.value, "message": msg}


# ----------------------------------------------------------------- totals

_TOTAL_RE = re.compile(
    r"\b(in total|grand total|total(?:ly)?|altogether|combined|summing(?: up)?|sum(?: of)?|"
    r"adding (?:them|all|these|both|up)|add(?:s|ed)? up(?: to)?)\b", re.I)
_MULT_CUE = re.compile(r"[*x×]\s*\d|\btimes\b|\binto\b|\bmultipl|\bper\b|\beach\b|%|\bpercent", re.I)
_SEG_SPLIT = re.compile(r"\n+|;|(?<!\d),(?!\d)|\.(?!\d)\s*")
_PRICE_CUE = re.compile(r"\b(?:at|@)\s*(?:rs\.?|inr|\$|₹)|\bavg\b|\baverage (?:price|ticket|selling)|\bprice\b|\bper unit\b|\bARPU\b", re.I)
_TOTAL_FILLER = {"is", "are", "of", "the", "comes", "to", "come", "number", "count", "around", "about", "approx",
                 "approximately", "roughly", "nearly", "be", "would", "will", "so", "we", "get", "i", "it", "this",
                 "that", "a", "an", "rs", "inr", "usd", "my", "our", "estimate", "value", "figure", "becomes",
                 "works", "out", "at", "which", "gives", "us", "equals", "then"}


def _segments(text: str) -> List[Tuple[int, int, str]]:
    t = _normalize(text)
    segs, pos = [], 0
    for m in _SEG_SPLIT.finditer(t):
        if m.start() > pos:
            segs.append((pos, m.start(), t[pos:m.start()]))
        pos = m.end()
    if pos < len(t):
        segs.append((pos, len(t), t[pos:]))
    return [s for s in segs if s[2].strip()]


def _segment_value(seg_text: str, need_large: bool) -> Optional[Qty]:
    qs = [q for q in parse_quantities(seg_text) if not q.pct]
    qs = [q for q in qs if q.explicit or (need_large and q.value >= 1000)]
    return qs[-1] if qs else None


def _check_totals(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    segs = _segments(text)
    for si, (s0, s1, st) in enumerate(segs):
        m = _TOTAL_RE.search(st)
        if not m:
            continue
        after = st[m.end():]
        tq = first_quantity(after)
        if tq is None or tq.pct or not (tq.explicit or tq.value >= 1000):
            continue
        if re.search(r"[=+]", st):
            continue  # an explicit chain in the same segment is checked as a chain
        if si == 0:
            continue
        if _MULT_CUE.search(segs[si - 1][2]) and not re.search(r"\b(plus|rest|remaining|other|and)\b|\+", segs[si - 1][2], re.I):
            continue  # "36 lakh x 2 cups. Total 72 lakh" is a product, not a sum
        if _PRICE_CUE.search(segs[si - 1][2]) or _PRICE_CUE.search(st[:m.start()]):
            continue  # "At Rs 40k avg, total market Rs 3,200 cr" is a product
        between = re.sub(r"[^a-z ]", " ", after[:tq.start].lower())
        named = [w for w in between.split() if w not in _TOTAL_FILLER]
        window: List[Qty] = []
        for pj in range(si - 1, max(-1, si - 5), -1):
            pt = segs[pj][2]
            if _TOTAL_RE.search(pt):
                break
            v = _segment_value(pt, need_large=not tq.explicit)
            if v is None or v.value > 2 * tq.value:
                break
            window.insert(0, v)
        if len(window) < 2:
            continue
        if tq.cur and not all(w.cur == tq.cur for w in window):
            continue  # a money total over counted parts: a product, not a sum
        if named:
            parts_text = " ".join(segs[pj][2].lower() for pj in range(max(0, si - len(window)), si))
            if any(w not in parts_text for w in named):
                continue  # "total METRO USERS 12 lakh" after office goers and students: another quantity
        T = tq.value
        ok = _rel(window[-1].value, T) <= 0.08 or any(_rel(w.value, T) <= 0.05 for w in window)
        for k in range(len(window) - 1):
            if _rel(sum(w.value for w in window[k:]), T) <= 0.08:
                ok = True
        if ok:
            out.append({"kind": "total", "verified": True, "expr": " + ".join(w.raw for w in window),
                        "stated": tq.raw, "pos": s0})
            continue
        total = sum(w.value for w in window)
        indian = tq.indian or any(w.indian for w in window)
        out.append({"kind": "total", "verified": False, "pos": s0,
                    "expr": " + ".join(w.raw for w in window), "stated": tq.raw,
                    "expected": total, "stated_value": T,
                    "message": (f"the parts listed before the total ({' + '.join(w.raw for w in window)}) "
                                f"add up to {fmt_number(total, indian_first=indian)}, not {tq.raw}")})
    return out


# ----------------------------------------------------------------- population facts

# Plausible population ranges (people), wide on purpose: a guesstimate anchor only has
# to be defensible. City proper .. metro area, roughly 2025.
_PLACES: List[Tuple[str, Tuple[float, float], str]] = [
    (r"india|bharat", (1.2e9, 1.6e9), "India: about 140 crore (1.4 billion) people"),
    (r"mumbai|bombay", (1.0e7, 2.8e7), "Mumbai: about 1.25 crore people in the city, about 2.1 crore (21 million) in the metro region"),
    (r"delhi|new delhi|ncr", (1.3e7, 3.5e7), "Delhi: about 2.1 crore (21 million) in the NCT, about 3.3 crore with the NCR"),
    (r"bengaluru|bangalore|banglore|bangaluru|bengalore", (0.8e7, 1.6e7), "Bengaluru: about 1.4 crore (14 million) people"),
    (r"hyderabad|hyd", (0.65e7, 1.3e7), "Hyderabad: about 1.1 crore (11 million) people"),
    (r"chennai|madras", (0.45e7, 1.3e7), "Chennai: about 46 lakh in the city, about 1.2 crore (12 million) in the metro area"),
    (r"kolkata|calcutta", (0.45e7, 1.6e7), "Kolkata: about 45 lakh in the city, about 1.5 crore in the metro area"),
    (r"pune", (0.3e7, 0.8e7), "Pune: about 31 lakh in the city, about 70 lakh (7 million) in the metro area"),
    (r"ahmedabad", (0.55e7, 0.95e7), "Ahmedabad: about 85 lakh (8.5 million) people"),
    (r"jaipur", (0.3e7, 0.45e7), "Jaipur: about 40 lakh (4 million) people"),
    (r"lucknow", (0.28e7, 0.45e7), "Lucknow: about 40 lakh (4 million) people"),
    (r"surat", (0.45e7, 0.9e7), "Surat: about 80 lakh (8 million) people"),
    (r"united states|usa|u\.s\.a?\.?", (3.2e8, 3.45e8), "United States: about 335 million people"),
    (r"california", (3.7e7, 4.0e7), "California: about 39 million people"),
    (r"texas", (2.8e7, 3.2e7), "Texas: about 31 million people"),
    (r"new york city|nyc", (8.0e6, 2.0e7), "New York City: about 8.3 million people (about 20 million in the metro area)"),
]
# population and its common typos (poulation, popluation, poplation, pouplation,
# papulation) - but not penetration, participation, pollution.
_POP_WORD = r"\bp[aeiou]{1,2}p?[aeiou]?l[aeiou]{0,2}t?i?on\b|\bpopulace\b|\bpop\b"
_FILL = r"(?:\s+(?:of|is|as|to be|=|:|around|about|approx\.?|approximately|roughly|nearly|~|close to|total|city|is around|be))*\s*"


def _place_patterns():
    pats = []
    for alias, rng, note in _PLACES:
        p1 = re.compile(rf"(?:{_POP_WORD})\s+(?:of|in)\s+(?:the\s+)?(?:{alias})(?:\s+city)?{_FILL}(?=[\d₹$.])", re.I)
        p2 = re.compile(rf"\b(?:{alias})(?:n|'s|s)?(?:\s+city)?(?:\s+total)?\s+(?:{_POP_WORD}){_FILL}(?=[\d.])", re.I)
        p3 = re.compile(rf"\b(?:{alias})\s+has\s+(?:a\s+population\s+of\s+|around\s+|about\s+|approx\.?\s+|roughly\s+)*(?=\d)", re.I)
        pats.append((alias, rng, note, (p1, p2, p3)))
    return pats


_PLACE_PATS = _place_patterns()


_QUALIFIED = re.compile(
    r"\b(working|work|age|aged|office|urban|rural|female|male|women|men|adult|adults|target|student|students|"
    r"college|school|young|youth|old|senior|elderly|child|children|kids|travel\w*|commut\w*|relevant|"
    r"addressable|slum|middle|upper|lower|income|employed|coffee|tea|drinking|smartphone|internet|online|"
    r"digital|literate|female|segment|customer|consumer|buying|user|users|eligible)[\s-]+(?:age[\s-]+)?"
    r"(?:group\s+)?(?:\w+\s+)?p[aeiou]{1,2}p?[aeiou]?l[aeiou]{0,2}t?i?on\b", re.I)


def _place_population_claims(text: str) -> List[Tuple[str, Qty, Tuple[float, float], str]]:
    t = _normalize(text)
    out = []
    for alias, rng, note, pats in _PLACE_PATS:
        for p in pats:
            for m in p.finditer(t):
                lead = t[max(0, m.start() - 30):m.start()]
                if re.search(r"(?:%|percent|per cent)\s*(?:\w+\s+){0,2}(?:of\s+)?(?:the\s+)?$", lead, re.I):
                    continue  # "5 percent of mumbai's population is 35 lakh" is a share, not the population
                if _QUALIFIED.search(lead + " " + t[m.start():m.start() + 40].split(" of ")[0]):
                    continue  # "working age population in lucknow" is a segment, not the city
                q = first_quantity(t[m.end():m.end() + 30])
                if q is None or q.pct or not (q.explicit or q.value >= 1e5):
                    continue
                if p is pats[2] and not re.match(r"\s*(?:people|population|residents|inhabitants|persons|pop)\b",
                                                 t[m.end() + q.end:]):
                    continue  # "Mumbai has 50 lakh cars" is not a population
                q.start += m.end()
                out.append((alias, q, rng, note))
    return out


def _check_populations(text: str) -> List[Dict[str, Any]]:
    out = []
    for _alias, q, (lo, hi), note in _place_population_claims(text):
        if lo / 1.5 <= q.value <= hi * 1.5:
            continue
        out.append({"kind": "fact", "verified": False, "expr": "", "stated": q.raw, "pos": q.start,
                    "expected": (lo + hi) / 2, "stated_value": q.value,
                    "message": f"{q.raw} is far off for this place - reference: {note}"})
    return out


def reference_facts(case_content: str, limit: int = 3) -> List[str]:
    """Population anchors for the places the case names (head of the brief only)."""
    head = (case_content or "")[:800]
    notes = []
    for alias, _rng, note in _PLACES:
        if re.search(rf"\b(?:{alias})\b", head, re.I) and note not in notes:
            notes.append(note)
    if notes and not any(n.startswith("India") for n in notes) and re.search(r"\b(india|indian)\b", head, re.I):
        notes.append(_PLACES[0][2])
    return notes[:limit]


# ----------------------------------------------------------------- households

_HH_SIZE_RE = re.compile(
    r"(?:household|family|hh)\s*size\s*(?:of|is|=|:|as|~|around|about)?\s*(?P<a>\d(?:\.\d)?)\b"
    r"|(?P<b>\d(?:\.\d)?)\s*(?:people|persons|members|ppl|person|member)\s*(?:per|in a|in each|/|a)\s*"
    r"(?:house(?:hold)?|home|family|flat|hh)\b"
    r"|(?:household|family|home|house)s?\s+of\s+(?P<c>\d(?:\.\d)?)\b", re.I)
_HH_COUNT_RE = re.compile(r"(?:households?|homes?|houses|families|hh)\b", re.I)


def _latest_population(texts: List[str]) -> Optional[Qty]:
    """Most recent '... population ... N' statement (newest text first)."""
    for t in texts:
        tn = _normalize(t)
        best = None
        for m in re.finditer(_POP_WORD, tn, re.I):
            window_after = tn[m.end():m.end() + 45]
            q = first_quantity(window_after)
            if q is not None and not q.pct and (q.explicit or q.value >= 1e5):
                cand = (m.start(), q)
            else:
                before = tn[max(0, m.start() - 30):m.start()]
                qs = [x for x in parse_quantities(before) if not x.pct and (x.explicit or x.value >= 1e5)]
                cand = (m.start(), qs[-1]) if qs else None
            if cand and (best is None or cand[0] > best[0]):
                best = cand
        if best:
            return best[1]
    return None


def _check_households(texts_newest_first: List[str], msg: str) -> List[Dict[str, Any]]:
    m = _HH_SIZE_RE.search(_normalize(msg))
    if not m:
        return []
    size = float(m.group("a") or m.group("b") or m.group("c"))
    if not 2 <= size <= 10:
        return []
    tn = _normalize(msg)
    rest = tn[m.end():]
    hh = None
    for q in parse_quantities(rest):
        if q.pct or not (q.explicit or q.value >= 1e4):
            continue
        tail = rest[q.end:q.end + 25]
        lead = rest[max(0, q.start - 3):q.start]
        if _HH_COUNT_RE.match(tail.strip()) or (":" in lead and not re.match(r"\s*[a-z]", tail)):
            hh = q
        break  # only the first figure after the household size can be the household count
    if hh is None:
        return []
    before = tn[:m.start()]
    # The base may be a segment they worked out ("middle class = 1.2 crore", then
    # "family of 4 so 30 lakh households"): any recent figure that fits is fine.
    recent = [before] + texts_newest_first[:4]
    for t in recent:
        for q in parse_quantities(t):
            if not q.pct and q.value > 0 and _rel(q.value / size, hh.value) <= 0.25:
                return [{"kind": "households", "verified": True, "expr": f"{q.raw} / {_sig(size)}",
                         "stated": hh.raw, "pos": m.start()}]
    pop = _latest_population([before] + texts_newest_first[:6])
    if pop is None:
        return []
    expected = pop.value / size
    if _rel(expected, hh.value) <= 0.25:
        return [{"kind": "households", "verified": True, "expr": f"{pop.raw} / {_sig(size)}", "stated": hh.raw,
                 "pos": m.start()}]
    indian = hh.indian or pop.indian
    factor = expected / hh.value if hh.value else 0
    msg_ = (f"{pop.raw} people / {_sig(size)} per household = "
            f"{fmt_number(expected, indian_first=indian)} households, not {hh.raw}")
    if factor >= 3 or (factor and factor <= 1 / 3):
        msg_ += f" (about {_sig(factor if factor >= 1 else 1 / factor, 2)}x off)"
    return [{"kind": "households", "verified": False, "expr": f"{pop.raw} / {_sig(size)}", "pos": m.start(),
             "stated": hh.raw, "expected": expected, "stated_value": hh.value, "message": msg_}]


# ----------------------------------------------------------------- office week x 7

_SEVEN_RE = re.compile(r"(?:[*x×]|\binto\b|\btimes\b|\bmultipl\w*(?:\s+[a-z]+){0,3}\s+by)\s*7\b"
                       r"(?!\s*(?:%|lakh|crore|million|am\b|pm\b|days? a month))", re.I)


def _check_office_week(msg: str, case_content: str) -> List[Dict[str, Any]]:
    t = _normalize(msg).lower()
    ctx = (case_content or "")[:400].lower() + " " + t
    if not re.search(r"\bweek", t) or not _SEVEN_RE.search(t):
        return []
    if not re.search(r"office|commut", ctx):
        return []
    if re.search(r"\b(5|five|6|six)\s*(working\s*)?days\b|\b(all|every|7|seven)\s*days\b|\bweekends?\b|"
                 r"\bevery ?day\b|\bdaily including\b", t):
        return []  # they have reasoned about the days (or it really runs all week)
    return [{"kind": "logic", "verified": False, "expr": "daily x 7", "stated": "x 7",
             "pos": _SEVEN_RE.search(t).start(),
             "message": ("a weekly figure for office-goers multiplies by the days they actually go in "
                         "(about 5, maybe 6), not 7")}]


# ----------------------------------------------------------------- unit restatements
# Arena adversarial run (Oct 21): "so total is 5 crore ml which is like 5 crore litres"
# went through, and the model then "corrected" it to 0.5 crore litres (also wrong:
# 5 crore ml is 50,000 litres). Python does the conversion; the model words it.

_UNIT_TABLE = {
    "ml": ("vol", 1e-3, "ml"), "millilitre": ("vol", 1e-3, "ml"), "millilitres": ("vol", 1e-3, "ml"),
    "milliliter": ("vol", 1e-3, "ml"), "milliliters": ("vol", 1e-3, "ml"),
    "litre": ("vol", 1.0, "litres"), "litres": ("vol", 1.0, "litres"), "liter": ("vol", 1.0, "litres"),
    "liters": ("vol", 1.0, "litres"), "ltr": ("vol", 1.0, "litres"), "ltrs": ("vol", 1.0, "litres"),
    "mg": ("mass", 1e-6, "mg"), "g": ("mass", 1e-3, "g"), "gm": ("mass", 1e-3, "g"),
    "gms": ("mass", 1e-3, "g"), "gram": ("mass", 1e-3, "g"), "grams": ("mass", 1e-3, "g"),
    "kg": ("mass", 1.0, "kg"), "kgs": ("mass", 1.0, "kg"), "kilogram": ("mass", 1.0, "kg"),
    "kilograms": ("mass", 1.0, "kg"), "kilo": ("mass", 1.0, "kg"), "kilos": ("mass", 1.0, "kg"),
    "tonne": ("mass", 1e3, "tonnes"), "tonnes": ("mass", 1e3, "tonnes"), "ton": ("mass", 1e3, "tonnes"),
    "tons": ("mass", 1e3, "tonnes"), "quintal": ("mass", 100.0, "quintals"), "quintals": ("mass", 100.0, "quintals"),
    "sec": ("time", 1 / 3600, "seconds"), "secs": ("time", 1 / 3600, "seconds"),
    "second": ("time", 1 / 3600, "seconds"), "seconds": ("time", 1 / 3600, "seconds"),
    "min": ("time", 1 / 60, "minutes"), "mins": ("time", 1 / 60, "minutes"),
    "minute": ("time", 1 / 60, "minutes"), "minutes": ("time", 1 / 60, "minutes"),
    "hr": ("time", 1.0, "hours"), "hrs": ("time", 1.0, "hours"), "hour": ("time", 1.0, "hours"),
    "hours": ("time", 1.0, "hours"),
    "mm": ("len", 1e-3, "mm"), "cm": ("len", 1e-2, "cm"), "metre": ("len", 1.0, "metres"),
    "metres": ("len", 1.0, "metres"), "meter": ("len", 1.0, "metres"), "meters": ("len", 1.0, "metres"),
    "km": ("len", 1e3, "km"), "kms": ("len", 1e3, "km"), "kilometre": ("len", 1e3, "km"),
    "kilometres": ("len", 1e3, "km"), "kilometer": ("len", 1e3, "km"), "kilometers": ("len", 1e3, "km"),
}
_UNIT_NOTE = {
    ("ml", "litres"): "1,000 ml make a litre", ("litres", "ml"): "1,000 ml make a litre",
    ("g", "kg"): "1,000 g make a kg", ("kg", "g"): "1,000 g make a kg",
    ("kg", "tonnes"): "1,000 kg make a tonne", ("tonnes", "kg"): "1,000 kg make a tonne",
    ("minutes", "hours"): "60 minutes make an hour", ("hours", "minutes"): "60 minutes make an hour",
    ("seconds", "minutes"): "60 seconds make a minute", ("minutes", "seconds"): "60 seconds make a minute",
    ("metres", "km"): "1,000 metres make a km", ("km", "metres"): "1,000 metres make a km",
    ("cm", "metres"): "100 cm make a metre", ("metres", "cm"): "100 cm make a metre",
}
# What may sit between "5 crore ml" and "5 crore litres" in a restatement.
# ("or" and "will be" are left out on purpose: "30 min or 1 hr" are alternatives, and
# "a 30 minute walk will be 1 hour return" is a different quantity.)
_RESTATE_WORDS = {"which", "is", "like", "about", "around", "roughly", "approx", "approximately", "that",
                  "thats", "that's", "equals", "equal", "to", "means", "ie", "i.e", "so", "nearly",
                  "basically", "comes", "come", "gives", "makes", "almost", "close", "i", "guess", "say",
                  "it's", "its", "it", "in", "total", "~", "="}
_RESTATE_LINK = re.compile(r"(?i)(?:\bis\b|=|\bequals?\b|\bmeans\b|\bi\.?e\b|\bcomes? to\b|\bgives\b|"
                           r"\bmakes\b|~|\bthat'?s\b|\bthats\b|\bwhich is\b)")
_UNIT_RATE = re.compile(r"\s*(?:per|a|an|/|every|each)\s*(day|week|month|year|annum|person|household|head|"
                        r"capita|hour|minute|trip|visit|order)s?\b|\s*(daily|weekly|monthly|yearly|annually)\b",
                        re.I)


_CHAIN_OP = re.compile(r"(?i)\s*(?:[*×/+]|(?<![a-z])x(?![a-z])|\btimes\b|\binto\b|\bmultipl\w*|\bdivid\w*|\bplus\b)")


def _unit_word_at(tn: str, pos: int) -> Tuple[Optional[str], int]:
    m = re.match(r"\s*([a-z]+)\b", tn[pos:pos + 24].lower())
    if not m:
        return None, pos
    return m.group(1), pos + m.end()


def _rate_after(tn: str, pos: int) -> str:
    m = _UNIT_RATE.match(tn[pos:pos + 30])
    return (m.group(1) or m.group(2) or "").lower() if m else ""


def _check_unit_conversions(msg: str) -> List[Dict[str, Any]]:
    """'5 crore ml which is like 5 crore litres': a bare restatement between two units of
    the same kind (volume, mass, time, length). Nothing but linking words may sit
    between the two figures; a rate on either side must match the other."""
    tn = _normalize(msg)
    qs = [q for q in parse_quantities(msg) if not q.pct and not q.cur]
    out: List[Dict[str, Any]] = []
    for a, b in zip(qs, qs[1:]):
        ua, end_a = _unit_word_at(tn, a.end)
        ub, end_b = _unit_word_at(tn, b.end)
        if ua not in _UNIT_TABLE or ub not in _UNIT_TABLE:
            continue
        fa, fb = _UNIT_TABLE[ua], _UNIT_TABLE[ub]
        if fa[0] != fb[0] or fa[2] == fb[2]:
            continue
        # Inside a calculation ("2 lakh people x 30 minutes = 1 lakh hours") the left
        # figure is an operand, not a restatement: only bare restatements are checked.
        seg_start = max(tn.rfind(c, 0, a.start) for c in ".;\n")
        if _CHAIN_OP.search(tn[seg_start + 1:a.start]) or _CHAIN_OP.match(tn[end_b:end_b + 6]):
            continue
        between = tn[end_a:b.start]
        if len(between) > 40 or re.search(r"\d", between) or not _RESTATE_LINK.search(between):
            continue
        words = re.findall(r"[a-z.'~=]+", between.lower())
        if any(w.strip(".") not in _RESTATE_WORDS for w in words):
            continue
        if _rate_after(tn, end_a) != _rate_after(tn, end_b):
            continue  # "500 ml a day is 15 litres a month" changes the period too
        expected = a.value * fa[1] / fb[1]
        approx = bool(re.search(r"(?i)\b(like|about|around|roughly|approx\w*|nearly|almost|close)\b|~", between))
        if expected <= 0 or _rel(expected, b.value) <= (0.25 if approx else 0.1):
            out.append({"kind": "units", "verified": True, "expr": f"{a.raw} {ua}", "stated": f"{b.raw} {ub}",
                        "pos": a.start})
            continue
        note = _UNIT_NOTE.get((fa[2], fb[2]))
        out.append({"kind": "units", "verified": False, "expr": f"{a.raw} {ua}", "stated": f"{b.raw} {ub}",
                    "expected": expected, "stated_value": b.value, "pos": a.start,
                    "message": (f"{a.raw} {ua} is {fmt_in(expected, a.indian or b.indian)} {fb[2]}"
                                + (f" ({note})" if note else "") + f", not {b.raw} {ub}")})
    return out


# ----------------------------------------------------------------- one-variable equations
# "so if 0.95S - 2 = 0.80(S-2) then S comes to like 2.67 cr" (arena, Oct 21): the
# candidate's algebra was right and the interviewer kept asking how they got it.

_EQ_MAX = 120


class _LinParser:
    """number | V | + - * / ( ) with implicit multiplication ("0.95S", "0.80(S-2)")."""

    def __init__(self, text: str, var: str):
        self.toks: List[Any] = []
        i = 0
        t = text.replace(" ", "")
        while i < len(t):
            ch = t[i]
            m = re.match(r"\d+(?:\.\d+)?|\.\d+", t[i:])
            if m:
                self.toks.append(float(m.group(0)))
                i += m.end()
                continue
            if ch == var:
                self.toks.append("V")
            elif ch in "+-*/()":
                self.toks.append(ch)
            else:
                raise ValueError("bad char")
            i += 1
        # implicit multiplication
        out: List[Any] = []
        for tok in self.toks:
            if out and (isinstance(out[-1], float) or out[-1] in ("V", ")")) and (
                    isinstance(tok, float) or tok in ("V", "(")):
                out.append("*")
            out.append(tok)
        self.toks = out

    def value(self, v: float) -> float:
        self.i, self.v, self.depth = 0, v, 0
        r = self._expr()
        if self.i != len(self.toks):
            raise ValueError("trailing")
        return r

    def _peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _expr(self):
        r = self._term()
        while self._peek() in ("+", "-"):
            op = self.toks[self.i]
            self.i += 1
            t = self._term()
            r = r + t if op == "+" else r - t
        return r

    def _term(self):
        r = self._unary()
        while self._peek() in ("*", "/"):
            op = self.toks[self.i]
            self.i += 1
            u = self._unary()
            if op == "/" and u == 0:
                raise ZeroDivisionError
            r = r * u if op == "*" else r / u
        return r

    def _unary(self):
        if self._peek() == "-":
            self.i += 1
            return -self._unary()
        if self._peek() == "+":
            self.i += 1
            return self._unary()
        return self._atom()

    def _atom(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("end")
        self.i += 1
        if isinstance(tok, float):
            return tok
        if tok == "V":
            return self.v
        if tok == "(":
            self.depth += 1
            if self.depth > 12:
                raise ValueError("deep")
            r = self._expr()
            if self._peek() != ")":
                raise ValueError("paren")
            self.i += 1
            self.depth -= 1
            return r
        raise ValueError("unexpected")


# Single letters that are units or scales far more often than unknowns
# ("1 M = 10 L", "1 h = 60 min", "1 m = 100 cm", "100 k = 1 L").
_UNIT_LETTERS = set("mMbBlLkKhHgGtTsSdDwWyY") - set("SDY")
_SCALED_NUM = re.compile(r"(?i)(?<![\w.])(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?"
                         r"(lakh crores?|lakhs?|lacs?|crores?|cr|thousand|million|mn|billion|bn|k)?\b(?!\s?%)")
_EQ_SCALE = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "cr": 1e7,
             "thousand": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "k": 1e3,
             "lakh crore": 1e12, "lakh crores": 1e12}


def _absolute_numbers(t: str) -> str:
    """'1.2 crore' -> '12000000', '25,000' -> '25000' (the equation is solved in real units)."""
    def sub(m):
        num, unit = m.group(1), (m.group(2) or "").lower()
        if "," in num and not _valid_commas(num):
            return m.group(0)
        v = float(num.replace(",", ""))
        if unit:
            v *= _EQ_SCALE.get(unit, 1.0)
        return (f"{v:.10g}" if v < 1e15 else m.group(0)) + ("" if not unit else " ")
    return _SCALED_NUM.sub(sub, t)


_EQ_SCAN_CHARS = 600
_EQ_TOKEN = r"[0-9.,\s+\-*/()]|(?:lakh|lakhs|lac|lacs|crores?|cr|thousand|million|mn|billion|bn|k)\b"


def _raw_equation(raw_t: str, eq_at: int, var: str) -> Optional[Tuple[str, int, int]]:
    """The equation as the candidate wrote it around the '=' at eq_at, and its span."""
    tok = rf"(?:{_EQ_TOKEN}|(?<![A-Za-z]){var}(?![A-Za-z]))"
    for m in re.finditer(rf"((?:{tok})+)=((?:{tok})+)", raw_t):
        if m.start() <= eq_at < m.end():
            lhs = re.split(r",\s", m.group(1))[-1].strip()
            rhs = re.split(r",\s", m.group(2))[0].strip()
            return re.sub(r"\s+", " ", f"{lhs} = {rhs}").strip(" ,"), m.start(), m.end()
    return None


def _check_equations(msg: str) -> List[Dict[str, Any]]:
    """A linear equation in one letter and the value the candidate says it solves to.
    Precision first: unit letters are not unknowns, scales and comma groups are read as
    real numbers, and 'so 5 M = 50' (another equation) is never taken as the solution."""
    if "=" not in (msg or ""):
        return []
    raw_t = _normalize(msg)[:_EQ_SCAN_CHARS]
    t = _absolute_numbers(raw_t)
    out: List[Dict[str, Any]] = []
    cands = set(re.findall(r"(?:(?<=[\d.)])|(?<=[\d.)] ))([A-Za-z])(?![A-Za-z])", t))
    cands |= {v for v in re.findall(r"(?<![A-Za-z'])([A-Za-z])(?=\s?[-+*/=)])", t) if v.lower() not in ("a", "i")}
    cands -= _UNIT_LETTERS
    for var in sorted(cands)[:3]:
        if var.lower() == "x" and re.search(r"\d\s*x\s*\d", t):
            continue  # "4 x 5" is multiplication
        side = rf"(?:[0-9.\s+\-*/()]|(?<![A-Za-z]){var}(?![A-Za-z]))+"
        for m in re.finditer(rf"({side})=({side})", t):
            lhs, rhs = m.group(1).strip(), m.group(2).strip()
            eq = f"{lhs} = {rhs}"
            if len(eq) > _EQ_MAX or var not in eq or not re.search(r"\d", lhs) or not re.search(r"\d", rhs):
                continue
            try:
                pl, pr = _LinParser(lhs, var), _LinParser(rhs, var)
                f = [pl.value(v) - pr.value(v) for v in (0.0, 1.0, 2.0)]
            except (ValueError, ZeroDivisionError, OverflowError, RecursionError):
                continue
            slope = f[1] - f[0]
            if abs(slope) < 1e-12 or abs((f[2] - f[1]) - slope) > 1e-9 * max(1.0, abs(slope)):
                continue  # not linear in the letter (or no letter in effect)
            sol = -f[0] / slope
            # Read the stated solution in the candidate's own text ("S comes to like 2.67 cr"):
            # the k-th "=" is the same character in both texts.
            k = t[:m.start() + len(m.group(1))].count("=")
            eq_pos = [i for i, ch in enumerate(raw_t) if ch == "="]
            if k >= len(eq_pos):
                continue
            found = _raw_equation(raw_t, eq_pos[k], var)
            raw_eq, sp_start, sp_end = found if found else (f"{lhs} = {rhs}", eq_pos[k], eq_pos[k] + 1)
            tail = raw_t[eq_pos[k] + 1:eq_pos[k] + 90]
            sm = None
            for cand in re.finditer(rf"(?<![A-Za-z]){var}\s*(?:=|is|comes(?:\s+out)?\s+to|works\s+out(?:\s+to)?|"
                                    rf"equals|would\s+be|will\s+be|turns\s+out\s+to\s+be|~)\s*(?:like|about|"
                                    rf"around|roughly|approx\.?|approximately|~)?\s*", tail):
                if re.search(r"[\d.)]\s*$", tail[:cand.start()]):
                    continue  # "so 5 M = 50": a coefficient in front -> another equation
                sm = cand
                break
            if not sm:
                continue
            q = first_quantity(tail[sm.end():sm.end() + 25])
            if q is None or q.pct or tail[sm.end():].lstrip()[:1] not in "0123456789.":
                continue
            base = q.value / q.scale if q.scale else q.value
            head = q.raw.split()[0]
            digits = len(re.sub(r"\D", "", head.split(".")[1])) if "." in head else 0
            approx = bool(re.search(r"(?i)\b(like|about|around|roughly|approx\w*)\b|~", tail[:sm.end()]))
            # rounding to the digits they wrote is fine, up to a point (3 for 2.67 is not)
            tol = (0.15 * abs(sol) if approx
                   else max(0.02 * abs(sol), min(0.5 * 10 ** (-digits), 0.06 * abs(sol))) + 1e-9)
            ok = abs(base - sol) <= tol or abs(q.value - sol) <= tol
            shown = f"{var} ≈ {fmt_in(sol, q.indian) if abs(sol) >= 1e5 else _sig(sol, 3)}"
            rec = {"kind": "equation", "verified": ok, "expr": raw_eq, "stated": f"{var} = {q.raw}",
                   "pos": eq_pos[k], "span": (sp_start, eq_pos[k] + 1 + sm.end() + len(q.raw) + 2),
                   "text": f"{raw_eq} solves to {shown}"}
            if not ok:
                rec.update(expected=sol, stated_value=base,
                           message=f"solving {raw_eq} gives {shown}, not {q.raw}")
            out.append(rec)
            break
    return out


# ----------------------------------------------------------------- dangling expression

def _dangling_expression(cand_turns_newest_first: List[str]) -> Optional[Tuple[str, float]]:
    """The newest earlier candidate turn that ENDS on an unfinished product/sum
    ("estimated 80M * 2000") -> (raw, value)."""
    for t in cand_turns_newest_first[:3]:
        tn = _normalize((t or "")[-160:]).strip()
        m = re.search(r"((?:[\d.,]+\s*(?:%|[a-z]{1,8})?\s*[*x/+]\s*)+[\d.,]+\s*(?:%|[a-z]{1,8})?)\s*$", tn, re.I)
        if not m or "=" in tn[m.start():]:
            continue
        raw = m.group(1)
        qs = parse_quantities(raw)
        rawn = _normalize(raw)
        ops = []
        for a, b in zip(qs, qs[1:]):
            gap = re.findall(r"(?<![a-z])[*x/+](?![a-z])", rawn[a.end:b.start].lower())
            if len(gap) != 1:
                ops = []
                break
            ops.append(gap[0])
        if len(qs) < 2 or len(ops) != len(qs) - 1:
            continue
        expr = ""
        for k, q in enumerate(qs):
            expr += repr(q.value / 100.0 if q.pct else q.value)
            if k < len(ops):
                expr += "*" if ops[k] in "xX" else ops[k]
        v = safe_eval(expr)
        if v is not None:
            return raw.strip(), v
    return None


# ----------------------------------------------------------------- final answers

def final_context(transcript: List[Dict[str, str]], msg: str, case_content: str) -> Dict[str, Any]:
    """What the final answer is and what it implies (for the interviewer's sanity check).
    Never raises."""
    try:
        return _final_context([dict(t, content=(t.get("content") or "")[:2000]) for t in list(transcript)[-12:]],
                              (msg or "")[:_MAX_CHECK_CHARS], case_content or "")
    except Exception as e:  # noqa: BLE001
        print(f"[numbers_check] final_context skipped: {type(e).__name__}")
        return {}


def _final_context(transcript: List[Dict[str, str]], msg: str, case_content: str) -> Dict[str, Any]:
    cand = [(t.get("content") or "") for t in transcript if (t.get("role") or "user") == "user"]
    qs = [q for q in parse_quantities(msg) if not q.pct]
    final = None
    if qs:
        big = [q for q in qs if q.explicit or q.value >= 100]
        final = (big or qs)[-1]
    else:
        for t in reversed(cand[-3:]):
            tq = [q for q in parse_quantities(t) if not q.pct and (q.explicit or q.value >= 100)]
            if tq:
                final = tq[-1]
                break
    out: Dict[str, Any] = {}
    if final is None:
        return out
    out["final_answer"] = final.raw
    earlier = []
    for t in reversed(cand[-5:]):
        found = [c.result for c in _chains(t) if c.result is not None and not c.result.pct]
        if not found and len(t.split()) <= 6:
            # a bare figure they gave as an answer ("210 million")
            found = [q for q in parse_quantities(t) if not q.pct and (q.explicit or q.value >= 1e4)][-1:]
        for q in found:
            if q.raw not in earlier and q.raw != final.raw:
                earlier.append(q.raw)
        if len(earlier) >= 3:
            break
    if earlier:
        out["earlier_results"] = earlier[:3]
    texts = [msg] + list(reversed(cand)) + [t.get("content") or "" for t in reversed(transcript)
                                              if t.get("role") == "assistant"]
    pop = _latest_population(texts)
    if pop is None:
        for alias, (lo, hi), _note in _PLACES:
            if re.search(rf"\b(?:{alias})\b", (case_content or "")[:400], re.I):
                pop = Qty((lo + hi) / 2, f"~{fmt_number((lo + hi) / 2)}", 1.0, False, False, 0, 0, True)
                break
    tn = _normalize(msg)
    noun = _noun_after(tn, final) if final.end <= len(tn) and final.start < len(tn) else None
    after = tn[final.end:final.end + 40].lower() if final.end <= len(tn) else ""
    daily = bool(re.search(r"\b(?:per|a|every|each)\s+day\b|\bdaily\b|/\s?day\b", after))
    periodic = bool(re.search(r"\b(?:per|a|every|each|in a|this|last)\s+(?:year|month|week|annum)\b|\bannual(?:ly)?\b|"
                              r"\byearly\b|\bmonthly\b|\bweekly\b|\bvisit|\btrips?\b|\bflights?\b", after))
    if pop is None:
        # "90000 orders per day for this small town": no figure, but the place has a size.
        recent = " ".join([msg] + cand[-3:]).lower()
        for pat, rng, label in _PLACE_SIZE_HINTS:
            if re.search(pat, recent):
                lo, hi = rng
                if final.value > 0 and not final.cur:
                    per_lo, per_hi = final.value / hi, final.value / lo
                    per = " every day" if daily else ""
                    out["implied_check"] = (f"{final.raw}{' ' + noun if noun else ''}{per} in {label} of roughly "
                                            f"{fmt_number(lo)} to {fmt_number(hi)} people is about "
                                            f"{_sig(per_lo, 2)} to {_sig(per_hi, 2)} per person{per}")
                    verdict = _implied_verdict(noun, daily, per_lo, periodic)
                    if verdict:
                        out["implied_verdict"] = verdict
                break
        else:
            out["no_base"] = True
        return out
    if pop.value > 0 and final.value > 0:
        per_person = final.value / pop.value
        unit = "$" if final.cur == "usd" else "Rs " if final.cur else ""
        if per_person < 0.5 and not final.cur:
            out["implied_check"] = (f"{final.raw} over a population of {pop.raw} is about one for every "
                                    f"{_sig(1 / per_person, 2)} people")
        else:
            out["implied_check"] = (f"{final.raw} over a population of {pop.raw} is about "
                                    f"{unit}{_sig(per_person, 2)} per person"
                                    + (f" (about {unit}{_sig(per_person * 4, 2)} per household of 4)"
                                       if per_person < 50 else "") + (" every day" if daily else ""))
        verdict = None if final.cur else _implied_verdict(noun, daily, per_person, periodic)
        if verdict:
            out["implied_verdict"] = verdict
    return out


# A place named by size only. Ranges are deliberately wide: the verdict below fires only
# when even the most generous size makes the number implausible.
_PLACE_SIZE_HINTS = [
    (r"\bsmall town\b|\blittle town\b", (5e4, 2e5), "a small town"),
    (r"\bvillage\b", (1e3, 1e4), "a village"),
]
# Only shapes where no reasonable reading survives (adversarial review, Oct 21: rides,
# tickets and UPI transactions run at 0.3+ per person a day for real; phones in use are
# about one per person; passengers and visitors per YEAR exceed a city's population).
_DAILY_EVENT_NOUNS = {"orders", "order", "deliveries", "delivery"}
_DURABLE_NOUNS = {"laptops", "laptop", "cars", "car", "tvs", "tv", "televisions", "television", "fridges",
                  "fridge", "refrigerators", "refrigerator", "acs", "washing"}
_PEOPLE_NOUNS = {"people", "persons", "users", "customers", "households", "members", "students", "consumers",
                 "buyers", "subscribers", "families", "homes"}


def _implied_verdict(noun: Optional[str], daily: bool, per_person: float, periodic: bool = False) -> Optional[str]:
    """'looks high' only where no reasonable reading survives: more people than the place
    has (a count, not a yearly flow), most residents ordering every single day, or a
    durable good for most people."""
    n = (noun or "").lower()
    if n in _PEOPLE_NOUNS and not daily and not periodic and per_person > 1.3:
        return f"looks high: more {n} than there are people"
    if daily and n in _DAILY_EVENT_NOUNS and per_person > 0.3:
        return f"looks high: about {_sig(per_person, 2)} {n} per person every single day"
    if n in _DURABLE_NOUNS and not daily and per_person > 0.6:
        return f"looks high: about {_sig(per_person, 2)} {n} per person"
    return None


# ----------------------------------------------------------------- the turn check

def check_turn(transcript: Iterable[Dict[str, str]], msg: str, case_content: str = "",
               signals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deterministic checks on the candidate's latest message. Never raises."""
    try:
        return _check_turn(list(transcript or []), msg or "", case_content or "", signals or {})
    except Exception as e:  # noqa: BLE001 - a checker must never break a turn
        print(f"[numbers_check] skipped: {type(e).__name__}: {e}")
        return {"math_issue": None, "math_verified": [], "reference_facts": reference_facts(case_content)}


_MAX_CHECK_CHARS = 4000


def _check_turn(transcript, msg, case_content, signals) -> Dict[str, Any]:
    # A pasted essay is checked on its first few thousand characters: the cost of a
    # check must stay a few milliseconds whatever the candidate sends.
    msg = msg[:_MAX_CHECK_CHARS]
    transcript = [dict(t, content=(t.get("content") or "")[:2000]) for t in transcript[-12:]]
    results: List[Dict[str, Any]] = []
    for ch in _chains(msg):
        r = _check_chain(ch, msg)
        if r:
            r.setdefault("pos", ch.start or 0)
            results.append(r)
    results += _check_totals(msg)
    newest_first = [(t.get("content") or "") for t in reversed(transcript)]
    results += _check_households(newest_first, msg)
    results += _check_populations(msg)
    results += _check_office_week(msg, case_content)
    results += _check_unit_conversions(msg)
    results += _check_equations(msg)
    words = re.findall(r"[a-z0-9']+", msg.lower())
    if len(words) <= 4:
        q = first_quantity(msg)
        cand_prev = [(t.get("content") or "") for t in reversed(transcript) if (t.get("role") or "user") == "user"]
        dang = _dangling_expression(cand_prev) if q is not None and not q.pct else None
        if dang:
            raw, v = dang
            sib = [q.scale] if q.explicit else []
            # The earlier expression may have been written without its unit ("revenue = 500 x 2"
            # then "1000 cr"): any scale reading that fits counts.
            scales = (1.0, 1e3, 1e5, 1e6, 1e7, 1e9)
            ok = any(_rel(v * s1, q.value * s2) <= 0.12 for s1 in scales for s2 in scales
                     if s1 == 1.0 or s2 == 1.0)
            if ok:
                results.append({"kind": "arithmetic", "verified": True, "expr": raw, "stated": q.raw})
            else:
                results.append({"kind": "arithmetic", "verified": False, "expr": raw, "stated": q.raw,
                                "expected": v, "stated_value": q.value,
                                "message": f"{raw} = {fmt_number(v, indian_first=q.indian)}, not {q.raw}"})
    # A linear equation Python solved owns its span: the chain reader's partial readings
    # inside it ("(1-0.2) = 0.8, not 80", "0.8C + 20 = 20.8") are not issues; the
    # equation's own verdict stands.
    solved = [r["span"] for r in results if r.get("kind") == "equation" and r.get("span")]
    if solved:
        results = [r for r in results if r.get("verified") or r.get("kind") == "equation"
                   or not any(a <= (r.get("pos") or 0) <= b for a, b in solved)]
    issues = sorted((r for r in results if not r.get("verified")), key=lambda r: r.get("pos") or 0)
    verified = [r for r in results if r.get("verified")]
    out: Dict[str, Any] = {
        "math_issue": issues[0] if issues else None,
        "math_issues": issues[:3],
        "math_verified": [r.get("text") or f"{r['expr']} = {r['stated']}".strip(" =") for r in verified][:3],
        "reference_facts": reference_facts(case_content),
    }
    return out


# ----------------------------------------------------------------- model-assisted audit

AUDIT_MODEL_DEFAULT = "gpt-4o"

_AUDIT_PROMPT = """You extract arithmetic from a case interview so that a calculator can check it.

Look ONLY at the candidate's LATEST message. For each number in it that the candidate DERIVED by arithmetic from numbers already in the conversation (or earlier in the same message), give:
- "stated": that number exactly as the candidate wrote it in the latest message (copy the characters: "1.3%", "35 million", "5.6M", "20k").
- "expression": the arithmetic that produces it from those inputs, in plain numbers only: digits, + - * / ( ) and decimal points. Write lakh as 100000, crore as 10000000, million as 1000000, billion as 1000000000, k as 1000, and percentages as decimals (30% -> 0.3). If the stated number is a percentage, the expression must give it as a decimal (8.5% -> 0.085). If it is a percentage change, write (new - old) / old.
- "readable": the same step in a few words a candidate would recognise, with units (for example "47.72 crore / (70% of 800 crore)").
Rules:
- An ASSUMPTION is not arithmetic: an assumed share, price, frequency, population or rate given as an input is never a check.
- Use the candidate's own inputs and the interviewer's stated facts, not what you think the right assumption would be.
- At most 3 checks, the most important first. If nothing in the latest message is derived, return an empty list.
Output JSON only: {"checks": [{"stated": "...", "expression": "...", "readable": "..."}]}"""


def audit_enabled() -> bool:
    return os.getenv("INTERVIEWER_NUMBERS_AUDIT", "on").strip().lower() not in ("off", "0", "false", "no")


# A number the candidate DERIVED (not an assumption they are setting up): the cue
# word and the number sit in the same clause.
_DERIVED_CUE = re.compile(
    r"(?:=|\b(?:gives?|gets?|got|comes? (?:to|out)|works? out|totals?|in total|final|arrived at|"
    r"which is|that means|implies|hence|therefore|thus|should be|needs? to be|has to be|"
    r"(?:increase|decrease|rise|fall|grow|drop|cut|raise)[ds]? by)\b)[^.\n?]{0,28}?\d", re.I)


def should_audit(signals: Dict[str, Any], transcript: List[Dict[str, str]], msg: str, channel: str) -> bool:
    if channel != "text" or not audit_enabled():
        return False
    if signals.get("math_issue") or not re.search(r"\d", msg or ""):
        return False
    if signals.get("is_session_open") or signals.get("looks_garbage") or signals.get("is_meta"):
        return False
    if signals.get("asks_interviewer") and not signals.get("final_declared") and (signals.get("word_count") or 0) < 15:
        return False
    if signals.get("math_verified"):
        return False  # Python already checked this turn's arithmetic
    if not (_DERIVED_CUE.search(msg) or signals.get("final_declared") or signals.get("answers_interviewer_number")):
        return False
    # A derived figure is a NEW one: at least one number not already in the conversation.
    seen = set()
    for t in transcript[-8:]:
        for q in parse_quantities((t.get("content") or "")[:2000]):
            seen.add(round(q.value, 6))
    new = [q for q in parse_quantities(msg[:2000]) if round(q.value, 6) not in seen]
    if not new:
        return False
    return bool(seen) or len(parse_quantities(msg)) >= 3


def _audit_messages(transcript, msg, case_content):
    lines = []
    for t in transcript[-8:]:
        role = "INTERVIEWER" if t.get("role") == "assistant" else "CANDIDATE" if (t.get("role") or "user") == "user" else None
        c = re.sub(r"\s+", " ", (t.get("content") or "")).strip()
        if role and c:
            lines.append(f"{role}: {c[:600]}")
    user = (f"CASE BRIEF (start): {(case_content or '')[:700]}\n\nCONVERSATION SO FAR:\n" + "\n".join(lines)
            + f"\n\nLATEST CANDIDATE MESSAGE:\n{(msg or '')[:1500]}")
    return [{"role": "system", "content": _AUDIT_PROMPT}, {"role": "user", "content": user}]


def _norm_stated(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower()).replace(",", "")


def verify_audit(raw_checks: Any, msg: str) -> Dict[str, Any]:
    """Python decides. A check counts only if its stated number is in the message
    and its expression is plain arithmetic."""
    issues, verified = [], []
    if not isinstance(raw_checks, list):
        return {"issues": issues, "verified": verified}
    msg_n = _norm_stated(_normalize(msg))
    for c in raw_checks[:3]:
        if not isinstance(c, dict):
            continue
        stated = str(c.get("stated") or "").strip()
        expr = str(c.get("expression") or "").strip()
        readable = re.sub(r"\s+", " ", str(c.get("readable") or "")).strip()[:120]
        if not stated or not expr or _norm_stated(stated) not in msg_n:
            continue
        q = first_quantity(stated)
        v = safe_eval(expr)
        if q is None or v is None:
            continue
        r = q.value / 100.0 if q.pct else q.value
        idx = msg_n.find(_norm_stated(stated))
        approx = bool(re.search(r"(approx\w*|about|around|roughly|nearly|~|almost)$", msg_n[max(0, idx - 14):idx]))
        tol = 0.25 if approx else 0.15
        if _rel(v, r) <= tol or (q.pct and _rel(v, q.value) <= tol):
            verified.append(f"{readable or expr} = {stated}")
            continue
        ratio = (v / r) if r else 0
        if ratio and any(abs(ratio * k - 1) <= 0.03 or abs(ratio / k - 1) <= 0.03
                         for k in (100.0, 1e3, 1e5, 1e6, 1e7, 1e9)):
            continue  # a unit the extractor dropped, not evidence of a slip
        exp_text = fmt_pct(v) if q.pct else fmt_number(v, indian_first=q.indian or not re.search(r"million|billion|\d[mb]\b", stated, re.I))
        issues.append({"kind": "arithmetic", "source": "audit", "verified": False,
                       "expr": readable or expr, "stated": stated, "expected": v,
                       "message": f"{readable or 'that step'} comes to about {exp_text}, not {stated}"})
    return {"issues": issues, "verified": verified}


def audit_turn(transcript: List[Dict[str, str]], msg: str, case_content: str = "", *,
               client=None, model: Optional[str] = None, record=None) -> Dict[str, Any]:
    """One JSON model call that WRITES expressions; verify_audit() decides.
    `record(response, latency_ms, model)` logs usage. Never raises."""
    if client is None:
        return {"issues": [], "verified": []}
    model = model or (os.getenv("INTERVIEWER_AUDIT_MODEL", AUDIT_MODEL_DEFAULT) or AUDIT_MODEL_DEFAULT).strip()
    t0 = time.time()
    # No SDK retries: a slow audit must cost the candidate at most one short wait.
    cli = client
    try:
        if hasattr(client, "with_options"):
            cli = client.with_options(max_retries=0)
    except Exception:  # noqa: BLE001
        cli = client
    try:
        resp = cli.chat.completions.create(
            model=model, messages=_audit_messages(transcript, msg, case_content),
            temperature=0.0, max_tokens=300, response_format={"type": "json_object"}, timeout=4)
    except Exception as e:  # noqa: BLE001
        print(f"[numbers_check] audit_failed {type(e).__name__}")
        return {"issues": [], "verified": []}
    if record is not None:
        try:
            record(resp, int((time.time() - t0) * 1000), model)
        except Exception:  # noqa: BLE001
            pass
    try:
        data = json.loads((resp.choices[0].message.content or "").strip() or "{}")
    except (ValueError, AttributeError, IndexError, TypeError):
        return {"issues": [], "verified": []}
    return verify_audit((data or {}).get("checks") if isinstance(data, dict) else None, msg)


def merge_audit(signals: Dict[str, Any], audit: Dict[str, Any]) -> None:
    issues = (audit or {}).get("issues") or []
    ver = (audit or {}).get("verified") or []
    if issues and not signals.get("math_issue"):
        signals["math_issue"] = issues[0]
        signals["math_issues"] = issues[:3]
    if ver:
        signals["math_verified"] = (list(signals.get("math_verified") or []) + ver)[:3]
