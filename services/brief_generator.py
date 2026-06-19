"""
GD Brief generator service.
Takes a classified news headline and generates a full Group Discussion brief
using OpenAI GPT-4o (premium model — this is user-facing output).

The brief includes:
- GD type classification (Abstract / Case-based / Opinion / Trend-analytical)
- 2-3 likely question framings
- 4-5 smart angles (non-obvious perspectives)
- 4-5 concrete data points (numbers + sources)
- 2-3 opening lines (how to start the discussion)
- 3-4 counter-arguments (what the other side would say)
- 2-3 closing lines (how to wrap)
- Neutral context summary (a substantive 5-7 sentence paragraph)

Designed for Indian MBA / PGDM students preparing for placement GDs at
top consulting, IB, and FMCG firms.
"""

import os
import json
from typing import TypedDict, List, Optional
from openai import OpenAI


class GeneratedBrief(TypedDict):
    """The structured brief returned to the frontend."""
    summary: str
    gd_type: str                    # "Abstract" | "Case-based" | "Opinion" | "Trend-analytical"
    likely_questions: List[str]     # 2-3 question framings
    smart_angles: List[str]         # 4-5 non-obvious perspectives
    data_points: List[str]          # 4-5 concrete numbers with attribution
    opening_lines: List[str]        # 2-3 ways to start
    counter_arguments: List[str]    # 3-4 opposing views
    closing_lines: List[str]        # 2-3 ways to wrap


class BriefGenerationError(Exception):
    """Raised when OpenAI fails to produce a valid brief."""
    pass


BRIEF_SYSTEM_PROMPT = """You are an elite GD coach for top Indian MBA/PGDM placement preparation.
Your students target McKinsey, BCG, Bain, top investment banks, and FMCG strategy roles.

Given a news headline, you generate a complete GD brief that helps a candidate walk into
a 10-minute Group Discussion with:
- A clear understanding of how this topic might be framed as a GD question
- Non-obvious perspectives that make them stand out
- Concrete data they can cite with confidence  
- Strong opening and closing lines
- Counter-arguments so they aren't blindsided

QUALITY STANDARDS — read carefully:

1. SMART ANGLES must be NON-OBVIOUS. Anyone can say "this is good for the economy." 
   A smart angle is "this disproportionately benefits tier-1 cities at the cost of tier-3,
   creating a hidden regional imbalance" — specific, counterintuitive, defensible.

2. DATA POINTS must be REAL and CITED. If unsure of exact figure, give a defensible
   ballpark with the source. Example: "India's fintech market reached ~$84B in 2024
   (per BCG-Matrix report)." NEVER fabricate specific numbers — if you don't know
   precise figures, use phrases like "industry estimates suggest" with a reasonable
   range.

3. COUNTER-ARGUMENTS must be the STRONGEST possible version of opposition. Don't
   make straw men. Steelman the other side.

4. OPENING LINES must be assertive but not aggressive. Should set a frame that
   guides the discussion. Examples of good frames: "Let me start by reframing this
   as a question about X..." or "Three forces are converging here..."

5. CLOSING LINES must SYNTHESIZE, not summarize. Pull together threads, name the
   central tension, leave the panel with a memorable takeaway.

6. SUMMARY — this is the candidate's context anchor. It must be a substantive,
   NEUTRAL paragraph (about 5-7 sentences, 120-180 words), not a one-liner. Give:
   (a) what actually happened, (b) the essential background needed to understand it,
   (c) why it is GD-relevant, and (d) the central tension. Represent the issue
   FAIRLY — state the genuine positions on each side at a factual level; do NOT take
   a side, editorialise, pre-judge the "right" answer, or flatten the debate into a
   single viewpoint. A candidate reading only the summary must come away with an
   accurate, balanced picture of the topic — never a distorted or mis-framed one.

GD TYPE CLASSIFICATION:
- "Abstract" — topic is broad/philosophical (e.g., "Is regulation always bad?")
- "Case-based" — topic requires applying frameworks (e.g., "Should Reliance enter X?")
- "Opinion" — explicit ban/should/allow framing (e.g., "Should crypto be banned in India?")
- "Trend-analytical" — about predicting/explaining a shift (e.g., "Where is Indian retail going?")

LANGUAGE: Confident, sharp, formal. Indian English (lakh/crore where appropriate).
Avoid corporate clichés. No "synergies", "leverage", "unlock value" without specificity.

OUTPUT FORMAT: valid JSON object with these exact keys:
{
  "summary": "A substantive, neutral context paragraph (about 5-7 sentences / 120-180 words): what happened, the essential background, why it matters for the GD, and the central tension — fairly stated, without taking a side or omitting the opposing position",
  "gd_type": "Abstract" | "Case-based" | "Opinion" | "Trend-analytical",
  "likely_questions": ["question 1", "question 2", "question 3"],
  "smart_angles": ["angle 1", "angle 2", "angle 3", "angle 4"],
  "data_points": ["data point 1 with source", "data point 2 with source", ...],
  "opening_lines": ["opening 1", "opening 2"],
  "counter_arguments": ["counter 1", "counter 2", "counter 3"],
  "closing_lines": ["closing 1", "closing 2"]
}
"""


def generate_brief(
    headline_title: str,
    headline_description: Optional[str],
    headline_source: str,
    headline_keywords: List[str],
    headline_category: str,
) -> GeneratedBrief:
    """
    Generate a full GD brief for a specific news headline.
    
    Uses GPT-4o (premium model) because output is shown directly to the user.
    Typical cost: ₹2-4 per brief.
    
    Raises BriefGenerationError on any failure. Caller should handle gracefully.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise BriefGenerationError("OPENAI_API_KEY not set")
    
    if not headline_title or not headline_title.strip():
        raise BriefGenerationError("Empty headline title")
    
    # Build user message with all context the AI needs
    context_parts = [
        f"HEADLINE: {headline_title}",
    ]
    if headline_description:
        context_parts.append(f"DESCRIPTION: {headline_description}")
    if headline_source:
        context_parts.append(f"SOURCE: {headline_source}")
    if headline_keywords:
        context_parts.append(f"TOPIC TAGS: {', '.join(headline_keywords)}")
    if headline_category:
        context_parts.append(f"CATEGORY: {headline_category}")
    
    context_parts.append(
        "\nGenerate a complete GD brief for an MBA candidate preparing to discuss "
        "this topic in a 10-minute placement Group Discussion."
    )
    
    user_message = "\n".join(context_parts)
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
    except Exception as e:
        raise BriefGenerationError(f"OpenAI API call failed: {type(e).__name__}: {e}")
    
    raw_content = response.choices[0].message.content
    if not raw_content:
        raise BriefGenerationError("OpenAI returned empty response")
    
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise BriefGenerationError(f"OpenAI returned invalid JSON: {e}")
    
    # Validate required fields and provide defensive fallbacks
    required_keys = [
        "summary", "gd_type", "likely_questions", "smart_angles",
        "data_points", "opening_lines", "counter_arguments", "closing_lines"
    ]
    missing = [k for k in required_keys if k not in parsed]
    if missing:
        raise BriefGenerationError(f"Brief missing keys: {missing}")
    
    # Validate gd_type is one of the allowed values
    allowed_types = {"Abstract", "Case-based", "Opinion", "Trend-analytical"}
    gd_type = str(parsed["gd_type"]).strip()
    if gd_type not in allowed_types:
        # Defensive: pick closest match or default
        gd_type_lower = gd_type.lower()
        if "abstract" in gd_type_lower:
            gd_type = "Abstract"
        elif "case" in gd_type_lower:
            gd_type = "Case-based"
        elif "opinion" in gd_type_lower or "ban" in gd_type_lower:
            gd_type = "Opinion"
        elif "trend" in gd_type_lower or "predict" in gd_type_lower:
            gd_type = "Trend-analytical"
        else:
            gd_type = "Case-based"  # safe default
    
    # Build validated brief
    return {
        "summary": str(parsed["summary"]).strip(),
        "gd_type": gd_type,
        "likely_questions": [str(q).strip() for q in parsed["likely_questions"] if str(q).strip()][:3],
        "smart_angles": [str(a).strip() for a in parsed["smart_angles"] if str(a).strip()][:5],
        "data_points": [str(d).strip() for d in parsed["data_points"] if str(d).strip()][:5],
        "opening_lines": [str(o).strip() for o in parsed["opening_lines"] if str(o).strip()][:3],
        "counter_arguments": [str(c).strip() for c in parsed["counter_arguments"] if str(c).strip()][:4],
        "closing_lines": [str(c).strip() for c in parsed["closing_lines"] if str(c).strip()][:3],
    }