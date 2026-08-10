"""
Certificate copy drafter, turns an admin's plain-prose notes about what an
intern actually did into the two variable lines on a MECE certificate:

  role_title   a recognisable job title
  scope_line   one sentence, "Scope of work: ..."

This is a legal-adjacent document signed by the Founder and Co-Founder, so the
model is held to a stricter standard than the Resume Lab bullet coach:

  - it may not invent a metric, a team size or a timeframe (same
    placeholders-over-hallucination policy as services/resume_ai.py);
  - it may not upgrade seniority beyond what the notes support;
  - it may not emit an em dash or an en dash (house style);
  - everything it writes must be defensible if a recruiter rings MECE.

Every generation is post-validated in Python before it is returned. A model slip
must never reach the admin as a save error from the database CHECK constraint.
"""

import json
import os
import re
import time
from typing import List, Optional, TypedDict

from openai import OpenAI

from services.ai_usage import log_ai_usage

MODEL = "gpt-4o"
MAX_ROLE_TITLE_CHARS = 55
MAX_SCOPE_LINE_CHARS = 200
SCOPE_PREFIX = "Scope of work:"

EN_DASH = "–"
EM_DASH = "—"
DASH_RE = re.compile(f"[{EN_DASH}{EM_DASH}]")

# Words that describe a person rather than the work. A certificate states what
# was done; "passionate" is not verifiable and cannot appear on a signed record.
BANNED_WORDS = (
    "passionate", "hardworking", "hard-working", "dedicated", "excellent",
    "enthusiastic", "diligent", "sincere", "talented", "motivated",
    "helped", "assisted with", "was involved in", "participated in",
    "gained exposure to", "worked on",
)


class CertificateAIError(Exception):
    pass


class Draft(TypedDict):
    role_title: str
    scope_line: str
    alternatives: dict


_RULES = f"""You draft the two variable lines on a MECE live-project completion certificate.
MECE (mece.in) is an AI-led case, guesstimate and interview preparation platform for
Indian MBA and PGDM students. The certificate is signed by the Founder and the
Co-Founder and is shown to recruiters.

HARD RULES

1. NEVER invent a metric, percentage, timeframe, team size, revenue figure or user
   count. If the admin's notes contain no number, your output contains no number.
2. NEVER use an em dash or an en dash. Use commas, colons, parentheses, or a plain
   hyphen inside a date range. No exceptions.
3. Restate only what the notes support. Do not infer seniority. If the notes say
   "assisted", do not write "led". If they say "supported", do not write "owned".
4. British spelling: monetisation, optimise, organisation, specialise.
5. No character adjectives. Banned: passionate, hardworking, dedicated, excellent,
   enthusiastic, diligent, sincere, talented, motivated.
6. No filler verbs: helped, assisted with, was involved in, participated in,
   worked on, gained exposure to.
7. Plain ASCII punctuation only. No smart quotes, no ellipsis character.

role_title
- A job title a recruiter recognises, not a description of tasks.
- Title Case. At most {MAX_ROLE_TITLE_CHARS} characters INCLUDING any parenthetical.
- An optional parenthetical specialisation is good: "Associate Product Manager (AI & Growth)".
- Default to an "Associate" or "Intern" level unless the notes clearly describe
  ownership of a function end to end.
- Lean towards a title that reads well for the target roles supplied, but never at
  the cost of accuracy. If the work was marketing and the target is product, return
  the marketing title.

scope_line
- EXACTLY one sentence. At most {MAX_SCOPE_LINE_CHARS} characters including the prefix.
- MUST start with "{SCOPE_PREFIX} ".
- Two to four work areas, comma separated, as concrete NOUN PHRASES, most
  substantial first: "product discovery and PRD ownership", not "did discovery".
- It may end with a clause tying the work to production, for example
  "all shipped to live users", but ONLY if the notes support it.
- End with a full stop. No full stop anywhere else in the sentence.

Return ONLY a JSON object of this exact shape:
{{"role_title": "...", "scope_line": "...",
  "alternatives": {{"role_title": ["...", "..."], "scope_line": ["...", "..."]}}}}
The two alternatives for each field must be materially different choices, not
rewordings, so a human can pick rather than re-prompt."""


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise CertificateAIError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key, timeout=45.0, max_retries=1)


def _chat(endpoint: str, user_id: Optional[str], *, model: str, **kwargs):
    """Every certificate OpenAI call goes through here so admin usage lands in
    ai_usage_log alongside everything else. An admin tool that bypasses the cost
    monitor is exactly the leak the cost-hardening work closed."""
    t0 = time.time()
    resp = _client().chat.completions.create(model=model, **kwargs)
    log_ai_usage(
        user_id=user_id, endpoint=endpoint, model=model, response=resp,
        latency_ms=int((time.time() - t0) * 1000),
    )
    return resp


# ── validation ───────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Repair the punctuation the model was told not to use, rather than failing
    the whole generation over a single character."""
    text = (text or "").strip()
    text = re.sub(r"(\d)\s*[–—]\s*(\d)", r"\1-\2", text)   # "2025 – 27" -> "2025-27"
    text = DASH_RE.sub(",", text)
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
                .replace("…", "..."))
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+,", ",", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _problems(role_title: str, scope_line: str) -> List[str]:
    out: List[str] = []

    if not role_title:
        out.append("role_title is empty")
    elif len(role_title) > MAX_ROLE_TITLE_CHARS:
        out.append(f"role_title is {len(role_title)} chars (max {MAX_ROLE_TITLE_CHARS})")

    if not scope_line:
        out.append("scope_line is empty")
    else:
        if len(scope_line) > MAX_SCOPE_LINE_CHARS:
            out.append(f"scope_line is {len(scope_line)} chars (max {MAX_SCOPE_LINE_CHARS})")
        if not scope_line.startswith(SCOPE_PREFIX):
            out.append(f'scope_line must start with "{SCOPE_PREFIX}"')
        if ". " in scope_line[:-1]:
            out.append("scope_line must be a single sentence")

    blob = f"{role_title} {scope_line}".lower()
    for word in BANNED_WORDS:
        if word in blob:
            out.append(f'banned wording: "{word}"')
    if DASH_RE.search(role_title) or DASH_RE.search(scope_line):
        out.append("contains an em dash or en dash")
    return out


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


# ── public API ───────────────────────────────────────────────────────────────

def draft_certificate_copy(
    work_notes: str,
    *,
    recipient_program: str = "",
    target_roles: Optional[List[str]] = None,
    duration_label: str = "",
    user_id: Optional[str] = None,
) -> Draft:
    """Draft role_title + scope_line from the admin's notes.

    Retries once at temperature 0 if the first attempt breaks a rule. Raises
    CertificateAIError rather than returning copy that would fail the frontend
    validator or the database CHECK.
    """
    notes = (work_notes or "").strip()
    if len(notes) < 20:
        raise CertificateAIError(
            "Describe the work in a sentence or two first (at least 20 characters)."
        )
    notes = notes[:6000]

    targets = ", ".join(target_roles or []) or "not specified"
    user_msg = (
        f"Notes from the admin about what this person actually did:\n\"\"\"\n{notes}\n\"\"\"\n\n"
        f"Recipient's programme: {recipient_program or 'not specified'}\n"
        f"Roles they are targeting: {targets}\n"
        f"Engagement length: {duration_label or 'not specified'}\n\n"
        "Draft the certificate's role_title and scope_line."
    )

    last: List[str] = []
    for attempt, temperature in enumerate((0.4, 0.0)):
        messages = [{"role": "system", "content": _RULES}, {"role": "user", "content": user_msg}]
        if attempt and last:
            messages.append({
                "role": "user",
                "content": "Your previous attempt broke these rules: "
                           + "; ".join(last)
                           + ". Return corrected JSON in the same shape.",
            })

        resp = _chat(
            "/certificates/draft", user_id, model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=600,
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            last = ["response was not valid JSON"]
            continue

        role = _clean(str(parsed.get("role_title", "")))
        scope = _clean(str(parsed.get("scope_line", "")))
        alts = parsed.get("alternatives") or {}
        alt_roles = [_clean(str(x)) for x in (alts.get("role_title") or [])][:2]
        alt_scopes = [_clean(str(x)) for x in (alts.get("scope_line") or [])][:2]

        last = _problems(role, scope)

        # Numbers may only appear if the admin's own notes contained them. This
        # is the check that stops a plausible-sounding "cut cost by 40%" from
        # reaching a signed document.
        invented = (_numbers(role) | _numbers(scope)) - _numbers(notes)
        if invented:
            last.append(f"invented figures not present in the notes: {sorted(invented)}")

        if not last:
            return {
                "role_title": role,
                "scope_line": scope,
                "alternatives": {
                    "role_title": [a for a in alt_roles if a and a != role],
                    "scope_line": [a for a in alt_scopes if a and a != scope],
                },
            }

    raise CertificateAIError(
        "The draft kept breaking house style (" + "; ".join(last)
        + "). Write the two lines by hand, or add more detail to the notes."
    )
