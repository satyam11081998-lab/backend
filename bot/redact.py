"""C3 privacy redaction and post-redaction identifier leak verification."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


class RedactionLeakError(RuntimeError):
    """Raised when a redacted turn or report text still contains an identifier."""


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Phone numbers: international (+91 / +1 / etc.), separated groups, or 10-digit
# Indian mobile numbers [6-9]xxxxxxxxx not followed by a case-interview unit.
_UNIT_AFTER_NUM_RE = re.compile(
    r"^\s*(?:%|percent|per\s+cent|cr\b|crore|crores|lakh|lakhs|lac|lacs|"
    r"million|mn\b|billion|bn\b|thousand|k\b|rs\b|inr\b|usd\b|rupees|dollars|"
    r"orders|units|people|households|users|customers|bottles|cups|cars|litres|liters)\b",
    re.I,
)
_PHONE_CONTEXT_RE = re.compile(
    r"(?i)\b(?:phone|ph|mobile|mob|cell|whatsapp|wa|call|contact|tel|telephone|number\s+is|reach\s+me\s+(?:at|on))\s*[:\-]?\s*"
    r"(\+?\d[\d\s\-().]{7,18}\d)"
)
_INTL_OR_SEP_PHONE_RE = re.compile(
    r"(?<!\w)(?:"
    r"\+\d{1,3}[\s\-.]?(?:\(?\d{2,5}\)?[\s\-.]?)?\d{3,5}[\s\-.]?\d{3,5}"
    r"|\(\d{3}\)\s*\d{3}[\s\-.]?\d{4}"
    r"|\d{3}[\s\-]\d{3}[\s\-]\d{4}"
    r"|\d{5}[\s\-]\d{5}"
    r")(?!\w)"
)
_INDIAN_MOBILE_RE = re.compile(r"(?<![\w.,])(?:0?[6-9]\d{9})(?![\w])(?!\.\d)")

# College patterns — explicit Indian/global institutions + capitalized college suffixes.
_CASE_INSENSITIVE_COLLEGE_RES: Sequence[re.Pattern[str]] = (
    re.compile(
        r"\bIIM[\s\-]*(?:Ahmedabad|Bangalore|Bengaluru|Calcutta|Lucknow|Kozhikode|"
        r"Indore|Shillong|Rohtak|Udaipur|Trichy|Tiruchirappalli|Ranchi|Raipur|"
        r"Kashipur|Amritsar|Bodh\s*Gaya|Jammu|Nagpur|Sambalpur|Sirmaur|"
        r"Visakhapatnam|Vizag|Mumbai|CAP|[ABCLIKMSURTNPV])\b",
        re.I,
    ),
    re.compile(r"\bIIM[ABCLIKM]\b", re.I),
    re.compile(
        r"\bIIT[\s\-]*(?:Bombay|Mumbai|Delhi|Madras|Chennai|Kanpur|Kharagpur|KGP|"
        r"Roorkee|Guwahati|Hyderabad|BHU|Varanasi|Indore|Jodhpur|Dhanbad|ISM|"
        r"Bhubaneswar|Gandhinagar|Patna|Mandi|Ropar|Tirupati|Palakkad|Bhilai|"
        r"Goa|Jammu|Dharwad)\b",
        re.I,
    ),
    re.compile(
        r"\bNIT[\s\-]*(?:Trichy|Tiruchirappalli|Surathkal|Warangal|Calicut|Rourkela|"
        r"Allahabad|Jaipur|Bhopal|Nagpur|Kurukshetra|Durgapur|Jamshedpur|"
        r"Silchar|Hamirpur|Jalandhar|Patna|Raipur|Agartala|Srinagar|Surat)\b",
        re.I,
    ),
    re.compile(
        r"\bIIIT[\s\-]*(?:Hyderabad|Bangalore|Bengaluru|Allahabad|Delhi|Gwalior)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:ISB(?:\s+(?:Hyderabad|Mohali))?|FMS(?:\s+Delhi)?|XLRI(?:\s+Jamshedpur)?|"
        r"SPJIMR|SP\s*Jain(?:\s+Institute[^,.!?]{0,30})?|MDI(?:\s+Gurgaon)?|"
        r"IIFT(?:\s+(?:Delhi|Kolkata))?|JBIMS|NITIE|NMIMS(?:\s+Mumbai)?|"
        r"SIBM(?:\s+Pune)?|SCMHRD|SIIB(?:\s+Pune)?|XIMB|XIM\s*(?:Bhubaneswar|Jabalpur)|"
        r"XIME(?:\s+Bangalore)?|IMT[\s\-]*(?:Ghaziabad|Nagpur|Hyderabad)|"
        r"IMI[\s\-]*(?:Delhi|Kolkata|Bhubaneswar)|TAPMI(?:\s+Manipal)?|"
        r"MICA(?:\s+Ahmedabad)?|K\.?\s*J\.?\s*Somaiya|GIM\s+Goa|IRMA(?:\s+Anand)?|"
        r"LIBA(?:\s+Chennai)?|Welingkar(?:\s+Mumbai)?|FORE(?:\s+Delhi|\s+School[^,.!?]{0,20})?|"
        r"BIMTECH|TISS(?:\s+Mumbai)?|SRCC|St\.?\s*Stephen'?s(?:\s+College)?|"
        r"St\.?\s*Xavier'?s(?:\s+College)?|Hindu\s+College|Lady\s+Shri\s+Ram|LSR|"
        r"BITS[\s\-]*(?:Pilani|Goa|Hyderabad)|IISc(?:\s+Bangalore)?|"
        r"Great\s+Lakes(?:\s+(?:Chennai|Gurgaon))?|LBSIM|Ashoka\s+University|"
        r"Christ\s+University|Symbiosis(?:\s+Institute[^,.!?]{0,25})?|"
        r"Narsee\s+Monjee|Jamnalal\s+Bajaj|Delhi\s+University|Panjab\s+University|"
        r"Jadavpur\s+University|Anna\s+University|Amity\s+(?:University|Business\s+School)|"
        r"Chandigarh\s+University|Sharda\s+University|Manipal\s+University|"
        r"Wharton(?:\s+School)?|Harvard\s+Business\s+School|Stanford\s+GSB|"
        r"MIT\s+Sloan|Chicago\s+Booth|Kellogg\s+School(?:\s+of\s+Management)?|"
        r"Columbia\s+Business\s+School|Yale\s+SOM|Berkeley\s+Haas|UC\s+Berkeley|"
        r"Michigan\s+Ross|NYU\s+Stern|UCLA\s+Anderson|Cornell\s+Johnson|"
        r"Tuck\s+School|Darden\s+School|Duke\s+Fuqua|London\s+Business\s+School|"
        r"INSEAD|HEC\s+Paris|IESE|Oxford\s+Said|Cambridge\s+Judge)\b",
        re.I,
    ),
)
# Case-sensitive acronyms and capitalized "<Name> University / College / Institute"
_CASE_SENSITIVE_COLLEGE_RES: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(?:IIM|IIT|NIT|IIIT|BITS|DTU|NSUT|NSIT|SJMSOM|VGSOM|DOMS|HBS|CBS|LBS)\b"),
    re.compile(
        r"\b(?:[A-Z][A-Za-z.&'\-]+(?:\s+[A-Z][A-Za-z.&'\-]+){0,4})\s+"
        r"(?:University|Institute\s+of\s+(?:Management|Technology|Science)|"
        r"Business\s+School|School\s+of\s+Management|College)\b"
    ),
)

# Contextual personal-name patterns (case-sensitive on the captured name tokens).
_HONORIFIC_NAME_RE = re.compile(
    r"\b(?:Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})\b"
)
_INTRO_NAME_RE = re.compile(
    r"\b(?i:my\s+name\s+is|this\s+is|call\s+me|i\s+am|i'?m|hi,?\s+i'?m|hello,?\s+i'?m|"
    r"regards,|thanks,|sincerely,|signed\s+by)\s+"
    r"([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,2})\b"
)
_NON_NAME_WORDS: Set[str] = {
    "i", "the", "a", "an", "and", "or", "but", "so", "if", "at", "in", "on", "to", "for",
    "of", "from", "with", "here", "there", "now", "then", "case", "client", "market",
    "company", "firm", "team", "user", "admin", "guest", "test", "india", "indian", "us",
    "usa", "mumbai", "delhi", "chennai", "bangalore", "bengaluru", "kolkata", "hyderabad",
    "pune", "ahmedabad", "stuck", "confused", "lost", "done", "finished", "ready", "new",
    "sorry", "sure", "okay", "ok", "good", "great", "right", "wrong", "assuming", "taking",
    "going", "trying", "starting", "splitting", "estimating", "calculating", "looking",
    "thinking", "working", "breaking", "checking", "recommending", "revenue", "cost",
    "profit", "price", "volume", "demand", "supply", "growth", "margin", "share", "rate",
    "total", "number", "population", "households", "customers", "people", "students",
    "college", "university", "school", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october", "november", "december",
}


def build_college_terms(colleges: Iterable[Dict[str, Any]], extra_terms: Iterable[str] = ()) -> List[str]:
    """Build normalized college-name literal terms from DB rows + user fields."""
    terms: Set[str] = set()
    for row in colleges or ():
        for key in ("name", "short_name"):
            val = (row.get(key) or "").strip()
            if len(val) >= 3:
                terms.add(val)
    for t in extra_terms:
        val = (t or "").strip()
        if len(val) >= 3 and val.lower() not in {"other", "none", "na", "n/a", "-"}:
            terms.add(val)
    return sorted(terms, key=len, reverse=True)


def _redact_phones(text: str) -> str:
    def _ctx_repl(m: re.Match[str]) -> str:
        full = m.group(0)
        num = m.group(1)
        digits = re.sub(r"\D", "", num)
        if 7 <= len(digits) <= 15:
            return full.replace(num, "[PHONE]")
        return full

    out = _PHONE_CONTEXT_RE.sub(_ctx_repl, text)

    def _intl_repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if 8 <= len(digits) <= 15:
            return "[PHONE]"
        return raw

    out = _INTL_OR_SEP_PHONE_RE.sub(_intl_repl, out)

    def _mobile_repl(m: re.Match[str]) -> str:
        tail = out[m.end() : m.end() + 24]
        if _UNIT_AFTER_NUM_RE.match(tail):
            return m.group(0)
        return "[PHONE]"

    return _INDIAN_MOBILE_RE.sub(_mobile_repl, out)


def _redact_colleges(text: str, college_terms: Sequence[str]) -> str:
    out = text
    for term in college_terms:
        pat = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.I)
        out = pat.sub("[COLLEGE]", out)
    for pat in _CASE_INSENSITIVE_COLLEGE_RES:
        out = pat.sub("[COLLEGE]", out)
    for pat in _CASE_SENSITIVE_COLLEGE_RES:
        out = pat.sub("[COLLEGE]", out)
    return out


def _looks_like_person_name(candidate: str) -> bool:
    tokens = [tok.strip(".,!?:;") for tok in (candidate or "").split() if tok.strip(".,!?:;")]
    if not tokens:
        return False
    for tok in tokens:
        if len(tok) < 2 or tok.lower() in _NON_NAME_WORDS:
            return False
    return True


def _redact_names(text: str, known_names: Iterable[str]) -> str:
    out = text
    # 1. Exact full names first, then individual name tokens (>= 3 chars, non-stopword).
    exact_phrases: List[str] = []
    part_tokens: List[str] = []
    for raw_name in known_names:
        cleaned = re.sub(r"\s+", " ", (raw_name or "").strip())
        if len(cleaned) < 2:
            continue
        if cleaned.lower() not in _NON_NAME_WORDS:
            exact_phrases.append(cleaned)
        for part in re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", cleaned):
            if len(part) >= 3 and part.lower() not in _NON_NAME_WORDS:
                part_tokens.append(part)

    for phrase in sorted(set(exact_phrases), key=len, reverse=True):
        out = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.I).sub("[NAME]", out)
    for tok in sorted(set(part_tokens), key=len, reverse=True):
        out = re.compile(r"(?<!\w)" + re.escape(tok) + r"(?!\w)", re.I).sub("[NAME]", out)

    # 2. Contextual honorifics and introductions when no DB name matched.
    def _group_repl(m: re.Match[str]) -> str:
        cap = m.group(1)
        if not _looks_like_person_name(cap):
            return m.group(0)
        start, end = m.span(1)
        rel_start = start - m.start()
        rel_end = end - m.start()
        full = m.group(0)
        return full[:rel_start] + "[NAME]" + full[rel_end:]

    out = _HONORIFIC_NAME_RE.sub(_group_repl, out)
    out = _INTRO_NAME_RE.sub(_group_repl, out)
    return out


def redact_text(
    text: str,
    *,
    user_info: Optional[Dict[str, Any]] = None,
    known_ids: Iterable[str] = (),
    known_emails: Iterable[str] = (),
    college_terms: Sequence[str] = (),
    extra_names: Iterable[str] = (),
) -> str:
    """Redact emails, phone numbers, personal names, college names, and user_id."""
    out = text or ""
    if not out:
        return out

    u = user_info or {}
    ids_to_strip: List[str] = [str(x).strip() for x in known_ids if x and str(x).strip()]
    if u.get("id"):
        ids_to_strip.append(str(u["id"]).strip())
    for uid in sorted(set(ids_to_strip), key=len, reverse=True):
        if len(uid) >= 8:
            out = re.compile(re.escape(uid), re.I).sub("[USER_ID]", out)

    # Emails (exact user emails first, then general regex)
    emails = [str(x).strip() for x in known_emails if x and str(x).strip()]
    for key in ("email", "college_email"):
        if u.get(key):
            emails.append(str(u[key]).strip())
    for em in sorted(set(emails), key=len, reverse=True):
        out = re.compile(re.escape(em), re.I).sub("[EMAIL]", out)
    out = _EMAIL_RE.sub("[EMAIL]", out)

    # Phone numbers
    out = _redact_phones(out)

    # Colleges (DB terms + user's college_other + regexes)
    user_colleges = list(college_terms)
    if u.get("college_other"):
        user_colleges = build_college_terms([], [str(u["college_other"])]) + user_colleges
    out = _redact_colleges(out, user_colleges)

    # Personal names
    names = list(extra_names)
    for key in ("full_name", "name"):
        if u.get(key):
            names.append(str(u[key]))
    out = _redact_names(out, names)
    return out


def find_identifier_leaks(
    text: str,
    *,
    raw_ids: Iterable[str] = (),
    raw_emails: Iterable[str] = (),
    raw_names: Iterable[str] = (),
    raw_colleges: Iterable[str] = (),
) -> List[str]:
    """Detect any remaining identifier in a redacted string."""
    t = text or ""
    leaks: List[str] = []
    if _EMAIL_RE.search(t):
        leaks.append("email_pattern")
    if _INTL_OR_SEP_PHONE_RE.search(t):
        for m in _INTL_OR_SEP_PHONE_RE.finditer(t):
            if 8 <= len(re.sub(r"\D", "", m.group(0))) <= 15:
                leaks.append("phone_pattern")
                break
    for m in _INDIAN_MOBILE_RE.finditer(t):
        tail = t[m.end() : m.end() + 24]
        if not _UNIT_AFTER_NUM_RE.match(tail):
            leaks.append("mobile_pattern")
            break
    for pat in _CASE_INSENSITIVE_COLLEGE_RES:
        if pat.search(t):
            leaks.append("college_pattern")
            break

    low = t.lower()
    for rid in raw_ids:
        s = str(rid or "").strip()
        if len(s) >= 8 and s.lower() in low:
            leaks.append("raw_id")
            break
    for em in raw_emails:
        s = str(em or "").strip()
        if "@" in s and s.lower() in low:
            leaks.append("raw_email")
            break
    for nm in raw_names:
        s = re.sub(r"\s+", " ", str(nm or "").strip())
        if len(s) >= 4 and s.lower() not in _NON_NAME_WORDS:
            if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", t, re.I):
                leaks.append("raw_name")
                break
    for col in raw_colleges:
        s = str(col or "").strip()
        if len(s) >= 4 and re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", t, re.I):
            leaks.append("raw_college")
            break
    return leaks


def assert_no_identifiers(
    texts: Iterable[str],
    *,
    raw_ids: Iterable[str] = (),
    raw_emails: Iterable[str] = (),
    raw_names: Iterable[str] = (),
    raw_colleges: Iterable[str] = (),
) -> None:
    """Raise RedactionLeakError if any text still contains an identifier."""
    ids_l = list(raw_ids)
    emails_l = list(raw_emails)
    names_l = list(raw_names)
    colleges_l = list(raw_colleges)
    for txt in texts:
        leaks = find_identifier_leaks(
            txt,
            raw_ids=ids_l,
            raw_emails=emails_l,
            raw_names=names_l,
            raw_colleges=colleges_l,
        )
        if leaks:
            raise RedactionLeakError(f"Unredacted identifier detected after redaction: {', '.join(leaks)}")
