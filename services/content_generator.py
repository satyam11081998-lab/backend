"""
Content generator service using GPT-4o.
Generates unique, challenging daily cases and guesstimates.
"""

import os
import json
from typing import TypedDict, List
from openai import OpenAI
from services.supabase_client import get_supabase_client

class GeneratedCase(TypedDict):
    code: str
    title: str
    sector: str
    source: str
    problem: str
    rootCause: str
    keyInsight: str
    framework: str
    resolution: str
    math: str
    risks: str

class GeneratedGuesstimate(TypedDict):
    code: str
    title: str
    approach: str
    keyDetail: str
    result: str

class GeneratorError(Exception):
    pass

SYSTEM_PROMPT = """You are an expert McKinsey/BCG interviewer creating practice materials for Indian MBA students.

Your task is to generate one highly challenging, MECE-compliant Case Study and one Guesstimate.
These must be unique, non-repetitive, and deeply grounded in Indian business realities or global macroeconomics.

# 1. CASE STUDY REQUIREMENTS
- Must have a clear 'code' (e.g., 'GEN-C-01').
- Sector: E.g., FMCG, Fintech, SaaS, Logistics, Retail, etc.
- Source: "Generated (Expert Mode)"
- Problem: The core prompt given to the candidate (2-3 sentences).
- RootCause: The underlying reason for the problem.
- KeyInsight: The non-obvious realization the candidate must reach.
- Framework: Recommended structure (e.g., Profitability Tree, Market Entry, Value Chain).
- Resolution: How the case is solved.
- Math: A quantitative aspect they must solve (e.g., "Calculate break-even if fixed costs are 50Cr...").
- Risks: 2-3 risks of the recommended resolution.

# 2. GUESSTIMATE REQUIREMENTS
- Must have a clear 'code' (e.g., 'GEN-G-01').
- Title: E.g., "Estimate the daily revenue of a bustling Mumbai local train station."
- Approach: Step-by-step logic (Top-down or Bottom-up).
- KeyDetail: Specific assumptions (e.g., "Assume 30% peak hour traffic, average ticket Rs 15").
- Result: The final estimated number.

OUTPUT FORMAT: Return a valid JSON object strictly matching this shape:
{
    "case": {
        "code": "...", "title": "...", "sector": "...", "source": "...", "problem": "...",
        "rootCause": "...", "keyInsight": "...", "framework": "...", "resolution": "...", "math": "...", "risks": "..."
    },
    "guesstimate": {
        "code": "...", "title": "...", "approach": "...", "keyDetail": "...", "result": "..."
    }
}
"""

def generate_daily_content(recent_themes: List[str]) -> dict:
    """Generate a new case and guesstimate using GPT-4o."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise GeneratorError("OPENAI_API_KEY not set")
        
    client = OpenAI(api_key=api_key)
    
    user_prompt = "Generate a challenging MBA-level Case Study and Guesstimate.\n"
    if recent_themes:
        user_prompt += f"DO NOT REPEAT or use themes closely related to these recent ones:\n{', '.join(recent_themes)}\n"
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)
    except Exception as e:
        raise GeneratorError(f"Failed to generate content: {e}")

def save_generated_content():
    """End-to-end: generate content, save to DB, and return IDs."""
    supabase = get_supabase_client()
    
    # Get recent cases to avoid repetition
    recent_cases = supabase.table("cases").select("title").order("created_at", desc=True).limit(10).execute()
    recent_themes = [row["title"] for row in (recent_cases.data or [])]
    
    content = generate_daily_content(recent_themes)
    case_data = content.get("case")
    guess_data = content.get("guesstimate")
    
    if not case_data or not guess_data:
        raise GeneratorError("Missing case or guesstimate in AI response")
        
    # Append timestamp to code to ensure uniqueness
    import time
    ts = str(int(time.time()))
    case_data["code"] = f"GEN-C-{ts}"
    guess_data["code"] = f"GEN-G-{ts}"
    
    # Insert case
    case_res = supabase.table("cases").insert({
        "code": case_data["code"],
        "title": case_data["title"],
        "sector": case_data["sector"],
        "source": case_data["source"],
        "problem": case_data["problem"],
        "root_cause": case_data.get("rootCause"),
        "key_insight": case_data.get("keyInsight"),
        "framework": case_data.get("framework"),
        "resolution": case_data.get("resolution"),
        "math": case_data.get("math"),
        "risks": case_data.get("risks"),
        "is_active": True
    }).execute()
    
    case_id = case_res.data[0]["id"]
    
    # Insert guesstimate
    supabase.table("guesstimates").insert({
        "code": guess_data["code"],
        "title": guess_data["title"],
        "approach": guess_data["approach"],
        "key_detail": guess_data.get("keyDetail"),
        "result": str(guess_data.get("result")),
        "is_active": True
    }).execute()
    
    return {
        "case_id": case_id,
        "guesstimate_code": guess_data["code"]
    }
