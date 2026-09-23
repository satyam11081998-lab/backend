"""
Prep Copilot v2 - GEMINI-GROUNDED RESEARCHER (corpus builder).

Given ANY role (e.g. 'asset management') and optionally a company (e.g. 'BNY
asset management'), this scours the web with Gemini's google_search grounding,
then structures the findings into a Pack, then runs an ADVERSARIAL expansion
pass ('what would a real interview ALSO probe that this draft misses?'). Every
factual, company-specific claim is asked to carry a source; where none exists the
pack is marked role-typical rather than inventing an employer's hiring process.

GEMINI for research/generation, per owner directive (GPT is reserved for scoring,
services/copilot/scoring.py). This module NEVER raises to its caller: any failure
degrades to an honest, low-confidence role-typical fallback pack so the copilot
keeps working. The self-improvement + persistence lives in corpus.py; this file
only builds a fresh Pack object.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import Pack, Rubric, RubricDimension, Framework, Scenario, Source, normalize_key

RESEARCH_MODEL = os.getenv("COPILOT_RESEARCH_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
_GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def research_available() -> bool:
    if not _GEMINI_KEY:
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except Exception:
        return False


def _client():
    from google import genai
    return genai.Client(api_key=_GEMINI_KEY)


def _extract_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {}
    t = re.sub(r"^```(?:json)?", "", t.strip()).rstrip("`").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


# ---------------------------------------------------------------------------
# Stage 1 - grounded web research (google_search tool)
# ---------------------------------------------------------------------------
def _grounded_research(role: str, company: str) -> Tuple[str, List[Source]]:
    from google.genai import types
    client = _client()
    tgt = f"the {role} role" + (f" at {company}" if company else "")
    prompt = (
        f"Research, from the live web, how candidates are interviewed and assessed for {tgt}. "
        "Be concrete and cite real sources. Cover:\n"
        f"1. The core COMPETENCIES a strong {role} must demonstrate, and the real FRAMEWORKS, METRICS and "
        "domain concepts they apply (name them precisely - the things a domain expert would test, not generic 'communication').\n"
        f"2. What {company or 'employers for this role'} SPECIFICALLY assess in the interview - drawn from "
        "careers pages, published interview experiences/transcripts, Glassdoor-style reports, and "
        f"{'the company annual report / investor materials' if company else 'industry hiring guides'}.\n"
        "3. Whether QUANTITATIVE / numerical problem-solving is expected, and what kind (mental math, case math, "
        "modelling, estimation).\n"
        "4. The typical interview FORMAT and rounds.\n"
        "Prefer primary and recent sources. Where you are not sure something is company-specific, say it is role-typical."
    )
    resp = client.models.generate_content(
        model=RESEARCH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.3,
        ),
    )
    text = getattr(resp, "text", "") or ""
    sources: List[Source] = []
    try:
        cand = (resp.candidates or [None])[0]
        gm = getattr(cand, "grounding_metadata", None)
        for ch in (getattr(gm, "grounding_chunks", None) or []):
            web = getattr(ch, "web", None)
            if web is None:
                continue
            uri = getattr(web, "uri", "") or ""
            title = getattr(web, "title", "") or uri
            if uri:
                sources.append(Source(title=title[:160], url=uri, snippet="", confidence="medium"))
    except Exception:
        pass
    # dedupe by url
    seen, uniq = set(), []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            uniq.append(s)
    return text, uniq[:12]


# ---------------------------------------------------------------------------
# Stage 2 - structure the research into a Pack JSON (no tool, JSON out)
# ---------------------------------------------------------------------------
_STRUCT_INSTR = """You are building a rigorous, domain-expert PREP PACK for one job role (optionally at one company), \
from the research notes provided. Output ONLY JSON, no prose, in this exact shape:

{
  "rubric": {"dimensions": [
     {"key":"snake_case","label":"Human label","max":<int>,
      "what_good_looks_like":"one concrete sentence",
      "anchors":["<band: high> ...","<band: mid> ...","<band: low> ..."],
      "red_flags":["<what a weak candidate does>", "..."]}
  ]},
  "frameworks": [{"name":"...","summary":"one or two sentences a candidate can actually use","why_it_matters":"..."}],
  "assessment": {"what_they_test":["...","..."],"numericals_expected":<true|false>,
                 "formats":["e.g. 45-min case + mental math"],"notes":"role-typical vs company-specific caveats"},
  "scenarios": [{"title":"short","prompt":"a realistic ON-THE-JOB dilemma (not trivia), 3-6 sentences with a clear decision",
                 "focus":"<one rubric dimension key>","numerical_ask":"specific quant task or empty",
                 "solution_outline":"4-8 lines of how a strong hire works it (shown AFTER the attempt)","difficulty":"easy|medium|hard"}]
}

HARD RULES:
- 5-7 rubric dimensions whose "max" values SUM TO EXACTLY 100. The dimensions must be SPECIFIC to this role \
(e.g. for an Area Sales Manager: distributor economics, PJP/beat design, trade-scheme ROI, sales math, negotiation, \
market execution) - never the generic consulting six unless the role IS consulting.
- Frameworks must be the REAL ones for this role, named precisely, usable in an answer.
- 4-6 scenarios, at least two 'hard'. If numericals are expected, at least two scenarios MUST carry a numerical_ask.
- GROUNDING: only put a company-specific claim in "assessment" if the research supports it; otherwise write role-typical \
guidance and say so in notes. NEVER invent an employer's internal hiring process."""


def _structure_pack(role: str, company: str, research_text: str) -> Dict[str, Any]:
    from google.genai import types
    client = _client()
    user = (f"ROLE: {role}\nCOMPANY: {company or '(none - role-only pack)'}\n\n"
            f"RESEARCH NOTES (ground everything in this):\n{research_text[:12000]}")
    resp = client.models.generate_content(
        model=RESEARCH_MODEL,
        contents=[_STRUCT_INSTR, user],
        config=types.GenerateContentConfig(temperature=0.4, response_mime_type="application/json"),
    )
    return _extract_json(getattr(resp, "text", "") or "")


# ---------------------------------------------------------------------------
# Stage 3 - adversarial expansion ('what more would they ask?')
# ---------------------------------------------------------------------------
_ADV_INSTR = """You are a SKEPTICAL, senior hiring manager for this exact role. The DRAFT pack below may be shallow, \
generic, or missing what a real interview digs into. Harden it. Return ONLY the improved pack JSON in the SAME shape.

Do all of:
- Add or sharpen any rubric dimension a real interviewer would weigh but the draft under-weights (keep 5-7 dims, max still sums to 100).
- Add the harder, less-obvious FRAMEWORKS/metrics an expert would expect.
- Add 1-2 HARDER scenarios that separate a strong hire from an average one, and make sure numerical_ask is set wherever the role expects math.
- Tighten assessment.what_they_test to the things that actually decide the hire.
- Keep every company-specific claim grounded; downgrade anything speculative to role-typical."""


def _adversarial_expand(role: str, company: str, draft: Dict[str, Any]) -> Dict[str, Any]:
    from google.genai import types
    client = _client()
    user = (f"ROLE: {role}\nCOMPANY: {company or '(none)'}\n\nDRAFT PACK JSON:\n{json.dumps(draft)[:12000]}")
    resp = client.models.generate_content(
        model=RESEARCH_MODEL,
        contents=[_ADV_INSTR, user],
        config=types.GenerateContentConfig(temperature=0.5, response_mime_type="application/json"),
    )
    improved = _extract_json(getattr(resp, "text", "") or "")
    return improved or draft


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _pack_from_json(role: str, company: str, d: Dict[str, Any], sources: List[Source],
                    confidence: str) -> Pack:
    rubric = Rubric.from_dict(d.get("rubric") or {})
    # normalise dimension maxima to sum to 100 if the model drifted
    total = rubric.total
    if rubric.dimensions and total and total != 100:
        scale = 100.0 / total
        acc = 0
        for i, dim in enumerate(rubric.dimensions):
            if i == len(rubric.dimensions) - 1:
                dim.max = max(1, 100 - acc)
            else:
                dim.max = max(1, round(dim.max * scale))
                acc += dim.max
    frameworks = [Framework.from_dict(x) for x in (d.get("frameworks") or []) if isinstance(x, dict)]
    scenarios = [Scenario.from_dict(x) for x in (d.get("scenarios") or []) if isinstance(x, dict)]
    assessment = dict(d.get("assessment") or {})
    return Pack(
        role_key=normalize_key(role),
        company_key=(normalize_key(company) or None),
        display_role=role.strip(),
        display_company=(company.strip() or None),
        rubric=rubric, frameworks=frameworks, assessment=assessment,
        scenarios=scenarios, sources=sources,
        confidence=confidence, version=1,
        status="ready" if rubric.is_valid() and scenarios else "failed",
    )


def _fallback_pack(role: str, company: str, note: str) -> Pack:
    """Honest, un-grounded, LOW-confidence default so the copilot still runs when
    research is unavailable. Generic competency spine, clearly flagged."""
    dims = [
        RubricDimension("problem_structuring", "Problem structuring", 25,
                        "Breaks the problem into the few drivers that matter for this role."),
        RubricDimension("domain_knowledge", "Domain knowledge & frameworks", 25,
                        "Applies the role's real concepts, not generic ones."),
        RubricDimension("quantitative", "Quantitative reasoning", 20,
                        "Does the numbers the role lives on and sanity-checks them."),
        RubricDimension("judgment", "Judgment & recommendation", 20,
                        "Commits to a defensible call under real-world constraints."),
        RubricDimension("communication", "Communication", 10,
                        "Answer-first, structured, concise."),
    ]
    return Pack(
        role_key=normalize_key(role), company_key=(normalize_key(company) or None),
        display_role=role.strip() or "Target role", display_company=(company.strip() or None),
        rubric=Rubric(dimensions=dims), frameworks=[],
        assessment={"what_they_test": [], "numericals_expected": True,
                    "formats": [], "notes": f"Role-typical fallback ({note}). Not yet grounded in sources."},
        scenarios=[Scenario(
            title=f"{role.strip() or 'Role'} - core dilemma",
            prompt=f"Work a realistic decision a {role.strip() or 'candidate in this role'} would face this quarter. "
                   "State your recommendation first, then the structure and the key numbers behind it.",
            focus="judgment", numerical_ask="Size the key number your recommendation depends on.",
            solution_outline="", difficulty="medium")],
        confidence="low", version=1, status="ready", notes=f"fallback:{note}",
    )


def research_pack(role: str, company: str = "") -> Pack:
    """Build a fresh, grounded Pack for (role[, company]). Never raises."""
    role = (role or "").strip()
    company = (company or "").strip()
    if not role and not company:
        return _fallback_pack("target role", "", "no role provided")
    if not research_available():
        return _fallback_pack(role or company, company if role else "", "gemini unavailable")
    try:
        research_text, sources = _grounded_research(role, company)
    except Exception as e:  # noqa: BLE001
        return _fallback_pack(role or company, company if role else "", f"research error {type(e).__name__}")
    if not research_text.strip():
        return _fallback_pack(role or company, company if role else "", "empty research")
    try:
        draft = _structure_pack(role, company, research_text)
        improved = _adversarial_expand(role, company, draft) if draft else {}
        chosen = improved or draft
    except Exception as e:  # noqa: BLE001
        return _fallback_pack(role or company, company if role else "", f"structuring error {type(e).__name__}")
    if not chosen or not (chosen.get("rubric") or {}).get("dimensions"):
        return _fallback_pack(role or company, company if role else "", "unstructured result")
    # confidence: company pack with real sources = high; role pack with sources = medium; else low
    conf = "high" if (company and sources) else ("medium" if sources else "low")
    pack = _pack_from_json(role, company, chosen, sources, conf)
    if pack.status != "ready":
        fb = _fallback_pack(role or company, company if role else "", "invalid rubric")
        fb.sources = sources
        return fb
    return pack
