"""
Resume Lab AI — original MECE bullet engine for Indian B-school resumes.

Three functions, all returning strict JSON:
  - refine_bullet(text, domain, max_chars): 3 stronger one-line options
  - generate_bullets(role, task, result, domain, count, max_chars): new bullets
  - fit_bullet(text, max_chars): trim to the line limit, keep the impact

Standards (industry-standard B-school resume conventions, not proprietary):
single line, strong action verb or number in the first few words, quantified
impact, no weak verbs (helped/supported/participated), no fabricated metrics
unless explicitly asked. Uses GPT-4o. Generated on demand.
"""

import os
import json
from typing import List, TypedDict, Optional
from openai import OpenAI


class BulletOption(TypedDict):
    text: str
    chars: int
    rationale: str


class ResumeAIError(Exception):
    pass


_SHARED_RULES = """You are MECE Resume Lab's bullet coach for elite Indian MBA/PGDM placement resumes
(IIM / IMI / XLRI / SPJIMR style one-pagers; recruiters: consulting, IB, FMCG, product).

Hard rules for every bullet:
- ONE line only. Start with a strong action verb OR a number/percentage in the first 3-4 words.
- Quantify impact (%, ₹/$ , #, time saved, scale) wherever plausible.
- Strong verbs: Led, Drove, Scaled, Built, Cut, Reduced, Generated, Launched, Orchestrated,
  Negotiated, Streamlined. NEVER weak verbs: helped, supported, participated, worked on, was responsible for.
- No first person, no articles padding, no buzzword filler ("synergy", "leverage" without specifics).
- Stay within the character limit given. Indian English (lakh/crore, ₹) where natural.
- Do NOT invent specific companies, titles, or precise metrics unless the user asks for plausible
  placeholders; if a metric is missing, use a tasteful placeholder like "X%" rather than a fake number.
"""


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ResumeAIError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key, timeout=45.0, max_retries=1)


def _parse_options(raw: Optional[str], max_chars: int) -> List[BulletOption]:
    if not raw:
        raise ResumeAIError("OpenAI returned empty response")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ResumeAIError(f"Invalid JSON: {e}")
    arr = parsed.get("options", [])
    if not isinstance(arr, list) or not arr:
        raise ResumeAIError("No options returned")
    out: List[BulletOption] = []
    for it in arr:
        text = str(it.get("text", "")).strip().rstrip(".")
        if not text:
            continue
        out.append({"text": text, "chars": len(text), "rationale": str(it.get("rationale", "")).strip()[:140]})
    if not out:
        raise ResumeAIError("No usable options")
    return out[:3]


def refine_bullet(text: str, domain: str, max_chars: int) -> List[BulletOption]:
    text = (text or "").strip()
    if not text:
        raise ResumeAIError("Empty bullet")
    sys = _SHARED_RULES + (
        f"\nTASK: Rewrite the user's bullet into EXACTLY 3 stronger options, each <= {max_chars} characters.\n"
        f"Domain flavour: {domain or 'general management'}.\n"
        'OUTPUT JSON: {"options":[{"text":"...","rationale":"one short why"}, ... x3]}'
    )
    resp = _client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": f"Bullet: {text}"}],
        response_format={"type": "json_object"},
        temperature=0.6,
    )
    return _parse_options(resp.choices[0].message.content, max_chars)


def generate_bullets(role: str, task: str, result: str, domain: str, count: int, max_chars: int) -> List[BulletOption]:
    role = (role or "").strip()
    if not role and not task and not result:
        raise ResumeAIError("Provide at least a role, task, or result")
    count = max(1, min(int(count or 3), 3))
    sys = _SHARED_RULES + (
        f"\nTASK: Generate {count} distinct bullet options from the context, each <= {max_chars} characters.\n"
        f"Domain flavour: {domain or 'general management'}.\n"
        'OUTPUT JSON: {"options":[{"text":"...","rationale":"one short why"}, ...]}'
    )
    user = f"Role: {role}\nTask: {task}\nResult: {result}"
    resp = _client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return _parse_options(resp.choices[0].message.content, max_chars)


def fit_bullet(text: str, max_chars: int) -> List[BulletOption]:
    text = (text or "").strip()
    if not text:
        raise ResumeAIError("Empty bullet")
    sys = _SHARED_RULES + (
        f"\nTASK: Tighten the user's bullet to <= {max_chars} characters WITHOUT losing the impact or the numbers.\n"
        'Return 1 option. OUTPUT JSON: {"options":[{"text":"...","rationale":"what you trimmed"}]}'
    )
    resp = _client().chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": f"Bullet: {text}"}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return _parse_options(resp.choices[0].message.content, max_chars)


# ── Full-resume rebuild ──────────────────────────────────────────────────

_REBUILD_SYSTEM = _SHARED_RULES + """
TASK: The user pastes their existing résumé as raw text. Parse it and RESTRUCTURE the whole
thing into the MECE one-page B-school format, rewriting every bullet to the rules above
(one line, strong verb / number first, quantified, <= 120 chars). Keep the person's real facts
(names, companies, roles, dates, numbers) — do NOT invent employers, degrees, or fake metrics;
if a metric is missing leave the achievement without a fabricated number. Improve weak wording.

Map content into THIS exact JSON shape (use empty arrays / "" when a section is absent):
{
  "header": {"name":"","program":"","email":"","phone":"","linkedin":"","extra":""},
  "education": [{"degree":"","institute":"","board":"","score":"","year":""}],
  "academicAchievements": ["..."],
  "workExperience": [{"org":"","role":"","dates":"","meta":"","bullets":["..."]}],
  "internships": [{"org":"","role":"","dates":"","meta":"","bullets":["..."]}],
  "projects": [{"org":"","role":"","dates":"","meta":"","bullets":["..."]}],
  "positionsOfResponsibility": [{"org":"","role":"","year":"","bullets":["..."]}],
  "awards": ["..."],
  "certifications": [{"provider":"","title":"","year":""}],
  "extracurricular": [{"category":"","bullets":["..."]}],
  "additionalInfo": {"examScores":["..."], "skills":["..."], "hobbies":["..."]}
}
Return ONLY this JSON object. "program" = the MBA/PGDM line if present. "meta" = duration like "13 months".
"""


def _norm_str(v) -> str:
    return str(v).strip() if v is not None else ""


def _norm_bullets(v, cap: int = 8):
    if not isinstance(v, list):
        return []
    return [_norm_str(x).rstrip(".") for x in v if _norm_str(x)][:cap]


def _norm_exp(v):
    out = []
    if not isinstance(v, list):
        return out
    for it in v:
        if not isinstance(it, dict):
            continue
        out.append({
            "org": _norm_str(it.get("org")),
            "role": _norm_str(it.get("role")),
            "dates": _norm_str(it.get("dates")),
            "meta": _norm_str(it.get("meta")),
            "bullets": _norm_bullets(it.get("bullets")),
        })
    return out[:8]


def rebuild_resume(text: str) -> dict:
    """Parse a raw résumé into the full MECE ResumeData JSON. Raises ResumeAIError."""
    text = (text or "").strip()
    if len(text) < 40:
        raise ResumeAIError("Paste your full résumé text (it looks too short).")
    if len(text) > 16000:
        text = text[:16000]

    resp = _client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _REBUILD_SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    raw = resp.choices[0].message.content
    if not raw:
        raise ResumeAIError("OpenAI returned empty response")
    try:
        p = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ResumeAIError(f"Invalid JSON: {e}")

    h = p.get("header") or {}
    ai = p.get("additionalInfo") or {}
    edu = []
    for e in (p.get("education") or []):
        if isinstance(e, dict):
            edu.append({
                "degree": _norm_str(e.get("degree")), "institute": _norm_str(e.get("institute")),
                "board": _norm_str(e.get("board")), "score": _norm_str(e.get("score")),
                "year": _norm_str(e.get("year")),
            })
    certs = []
    for c in (p.get("certifications") or []):
        if isinstance(c, dict):
            certs.append({"provider": _norm_str(c.get("provider")), "title": _norm_str(c.get("title")), "year": _norm_str(c.get("year"))})
    extra = []
    for g in (p.get("extracurricular") or []):
        if isinstance(g, dict):
            extra.append({"category": _norm_str(g.get("category")), "bullets": _norm_bullets(g.get("bullets"))})

    def _list_str(v, cap=10):
        return [_norm_str(x) for x in v if _norm_str(x)][:cap] if isinstance(v, list) else []

    return {
        "header": {
            "name": _norm_str(h.get("name")), "program": _norm_str(h.get("program")),
            "email": _norm_str(h.get("email")), "phone": _norm_str(h.get("phone")),
            "linkedin": _norm_str(h.get("linkedin")), "extra": _norm_str(h.get("extra")),
        },
        "education": edu[:6],
        "academicAchievements": _norm_bullets(p.get("academicAchievements")),
        "workExperience": _norm_exp(p.get("workExperience")),
        "internships": _norm_exp(p.get("internships")),
        "projects": _norm_exp(p.get("projects")),
        "positionsOfResponsibility": [
            {"org": _norm_str(x.get("org")), "role": _norm_str(x.get("role")), "year": _norm_str(x.get("year")), "bullets": _norm_bullets(x.get("bullets"))}
            for x in (p.get("positionsOfResponsibility") or []) if isinstance(x, dict)
        ][:8],
        "awards": _norm_bullets(p.get("awards"), 12),
        "certifications": certs[:12],
        "extracurricular": extra[:8],
        "additionalInfo": {
            "examScores": _norm_bullets(ai.get("examScores"), 6),
            "skills": _list_str(ai.get("skills")),
            "hobbies": _list_str(ai.get("hobbies")),
        },
    }
