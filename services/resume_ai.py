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
import math
import time
from typing import List, TypedDict, Optional
from openai import OpenAI

from services.ai_usage import log_ai_usage


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


def _chat(endpoint: str, user_id: Optional[str], *, model: str, **kwargs):
    """Single place every Resume Lab OpenAI call goes through, so all of them are
    logged to ai_usage_log (cost dashboard + daily-budget backstop)."""
    t0 = time.time()
    resp = _client().chat.completions.create(model=model, **kwargs)
    log_ai_usage(
        user_id=user_id, endpoint=endpoint, model=model, response=resp,
        latency_ms=int((time.time() - t0) * 1000),
    )
    return resp


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


def refine_bullet(text: str, domain: str, max_chars: int, user_id: Optional[str] = None) -> List[BulletOption]:
    text = (text or "").strip()
    if not text:
        raise ResumeAIError("Empty bullet")
    sys = _SHARED_RULES + (
        f"\nTASK: Rewrite the user's bullet into EXACTLY 3 stronger options, each <= {max_chars} characters.\n"
        f"Domain flavour: {domain or 'general management'}.\n"
        'OUTPUT JSON: {"options":[{"text":"...","rationale":"one short why"}, ... x3]}'
    )
    resp = _chat(
        "/resume/refine-bullet", user_id, model="gpt-4o",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": f"Bullet: {text}"}],
        response_format={"type": "json_object"},
        temperature=0.6,
        max_tokens=500,
    )
    return _parse_options(resp.choices[0].message.content, max_chars)


def generate_bullets(role: str, task: str, result: str, domain: str, count: int, max_chars: int, user_id: Optional[str] = None) -> List[BulletOption]:
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
    resp = _chat(
        "/resume/generate-bullets", user_id, model="gpt-4o",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=600,
    )
    return _parse_options(resp.choices[0].message.content, max_chars)


def fit_bullet(text: str, max_chars: int, user_id: Optional[str] = None) -> List[BulletOption]:
    text = (text or "").strip()
    if not text:
        raise ResumeAIError("Empty bullet")
    sys = _SHARED_RULES + (
        f"\nTASK: Tighten the user's bullet to <= {max_chars} characters WITHOUT losing the impact or the numbers.\n"
        'Return 1 option. OUTPUT JSON: {"options":[{"text":"...","rationale":"what you trimmed"}]}'
    )
    # Tightening is a compress task — mini handles it well at a fraction of the cost.
    resp = _chat(
        "/resume/fit-bullet", user_id, model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": f"Bullet: {text}"}],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=220,
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


def rebuild_resume(text: str, user_id: Optional[str] = None) -> dict:
    """Parse a raw résumé into the full MECE ResumeData JSON. Raises ResumeAIError."""
    text = (text or "").strip()
    if len(text) < 40:
        raise ResumeAIError("Paste your full résumé text (it looks too short).")
    if len(text) > 16000:
        text = text[:16000]

    resp = _chat(
        "/resume/rebuild", user_id, model="gpt-4o",
        messages=[
            {"role": "system", "content": _REBUILD_SYSTEM},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.4,
        max_tokens=4000,  # a densely-packed CV rebuilds into a large JSON; avoid mid-object truncation
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


# ── Strict character-band engine (95–100% of the limit, never over) ──────────

def _band_lo(max_chars: int) -> int:
    """Lower bound of the fill band: at least 95% of the limit."""
    return max(1, math.ceil(0.95 * max_chars))


def _trim_to_words(text: str, max_chars: int) -> str:
    """Deterministic, never mid-word: drop trailing words until it fits."""
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    out = ""
    for w in t.split():
        cand = (out + " " + w).strip()
        if len(cand) > max_chars:
            break
        out = cand
    return out.strip().rstrip(",;:-– ")


def _one_line(instruction: str, user_text: str, max_chars: int, domain: str, temp: float = 0.4,
              user_id: Optional[str] = None) -> str:
    sys = _SHARED_RULES + "\n" + instruction + (
        f"\nDomain flavour: {domain or 'general management'}.\n"
        'Return ONE option. OUTPUT JSON: {"options":[{"text":"...","rationale":"..."}]}'
    )
    # Band coercion is compress/expand — a mini-strength task. Was gpt-4o (up to 2
    # calls PER option = the single most expensive path in the app); now mini + at
    # most ONE call per option (see _enforce_band).
    resp = _chat(
        "/resume/band-fix", user_id, model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user_text}],
        response_format={"type": "json_object"},
        temperature=temp,
        max_tokens=220,
    )
    raw = resp.choices[0].message.content or ""
    try:
        arr = json.loads(raw).get("options", [])
        return str(arr[0].get("text", "")).strip().rstrip(".") if arr else ""
    except Exception:
        return ""


def _enforce_band(text: str, max_chars: int, domain: str, user_id: Optional[str] = None) -> str:
    """Coerce a single bullet toward [95%, 100%] of max_chars, never exceeding it.

    COST GUARD: at most ONE AI (mini) call per option. Over-length is fixed for FREE
    with a deterministic word-boundary trim (no quality loss — it only drops trailing
    filler); the one AI call is spent only on the harder EXPAND case, where creativity
    actually helps. Previously this made up to TWO gpt-4o calls per option."""
    lo = _band_lo(max_chars)
    t = (text or "").strip().rstrip(".")

    # Over the limit -> deterministic trim first (free, never mid-word).
    if len(t) > max_chars:
        t = _trim_to_words(t, max_chars)

    # Comfortably short -> spend the one allowed AI call to expand with a concrete detail.
    if len(t) < lo:
        e = _one_line(
            f"TASK: Expand this into ONE line between {lo} and {max_chars} characters by adding ONE concrete, "
            "plausible detail (scope, %, count, timeframe). Use a tasteful placeholder like X% only if no number "
            "exists. Do not pad with filler words.",
            f"Bullet: {t}", max_chars, domain, 0.5, user_id=user_id,
        )
        if e and len(e) > len(t):
            t = e if len(e) <= max_chars else _trim_to_words(e, max_chars)
    return t.strip().rstrip(".")


def generate_points(achievement: str, domain: str, max_chars: int, count: int = 3, instructions: str = "", user_id: Optional[str] = None) -> dict:
    """Achievement -> {"options":[...], "clarify": None} where each option is within 95-100% of
    max_chars. If the achievement is too vague, returns {"options":[], "clarify": "<question>"}.
    `instructions` are extra user guidance the model must follow strictly."""
    achievement = (achievement or "").strip()
    if len(achievement) < 3:
        raise ResumeAIError("Describe your achievement in a few words first.")
    count = max(1, min(int(count or 3), 3))
    instructions = (instructions or "").strip()
    lo = _band_lo(max_chars)
    extra = ""
    if instructions:
        extra = (
            "\nUSER INSTRUCTIONS (follow these STRICTLY; when they conflict with the default style, "
            f"the user's instructions win):\n{instructions[:800]}\n"
        )
    sys = _SHARED_RULES + (
        f"\nTASK: Turn the user's achievement into {count} DISTINCT one-line CV bullet options.\n"
        f"CHARACTER TARGET (critical): each option MUST be between {lo} and {max_chars} characters — as close "
        f"to {max_chars} as possible WITHOUT ever exceeding it. Count characters carefully before answering.\n"
        f"Domain flavour: {domain or 'general management'}.\n"
        + extra +
        "If (and ONLY if) the achievement is too vague or missing a key fact (what you actually did, the "
        "outcome, or a number) to write an ACCURATE bullet, do NOT invent details — instead ask ONE short "
        "clarifying question.\n"
        'OUTPUT JSON: either {"clarify":"<one short question>"} '
        'OR {"options":[{"text":"...","rationale":"one short why"}, ...]}'
    )
    resp = _chat(
        "/resume/point", user_id, model="gpt-4o",
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": f"Achievement: {achievement}"}],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=600,
    )
    raw = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ResumeAIError(f"Invalid JSON: {e}")
    clar = str(parsed.get("clarify", "")).strip()
    arr = parsed.get("options")
    if clar and not arr:
        return {"options": [], "clarify": clar[:240]}
    if not isinstance(arr, list) or not arr:
        raise ResumeAIError("Could not produce a bullet. Try adding a little more detail.")
    out: List[BulletOption] = []
    seen = set()
    for it in arr:
        text = str(it.get("text", "")).strip().rstrip(".")
        if not text:
            continue
        t = _enforce_band(text, max_chars, domain, user_id=user_id)
        key = t.lower()
        if t and key not in seen:
            seen.add(key)
            out.append({"text": t, "chars": len(t), "rationale": str(it.get("rationale", ""))[:140]})
    if not out:
        raise ResumeAIError("Could not produce a bullet that fits. Try a shorter achievement or a larger limit.")
    return {"options": out[:count], "clarify": None}
