"""
Headline classifier service.
Takes raw headlines from news_fetcher and uses OpenAI to:
1. Score each for GD-worthiness (0-10)
2. Extract keyword tags (2-4 per headline)
3. Categorize (business/macro/policy/etc.)
4. Identify the "star" headline of the day

Uses GPT-4o-mini for cost efficiency (this is bulk classification,
not user-facing output, so we don't need GPT-4o quality).
"""

import os
import json
from typing import List, TypedDict, Optional
from openai import OpenAI


class ClassifiedHeadline(TypedDict):
    """A headline after AI classification."""
    title: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    source_url: str
    source_name: str
    published_at: str
    
    # AI-added fields
    gd_worthiness_score: int  # 0-10
    keywords: List[str]        # 2-4 topic tags
    category: str              # "business", "macro", "policy", "tech", "global", "other"
    is_star: bool              # only ONE headline in batch has is_star=True


class ClassificationError(Exception):
    """Raised when OpenAI fails to classify headlines properly."""
    pass


CLASSIFIER_SYSTEM_PROMPT = """You are an expert curator for MBA Group Discussion (GD) topics in India.

Your job: rate news headlines for how likely they are to become GD topics in MBA/PGDM placement interviews at IIMs, IMI, FMS, SP Jain, and top consulting/IB recruiters.

A high GD-worthy headline (8-10) has these traits:
- Has TWO defensible sides (room for debate)
- Touches business strategy, economics, policy, regulation, or sector dynamics
- Affects multiple stakeholders (consumers, companies, government, society)
- Recent enough to be relevant but substantive enough for 10-min discussion
- Indian context strongly preferred but global business is also valid

A LOW GD-worthy headline (0-3):
- Sports, entertainment, celebrity news
- Crime, accidents, individual tragedies  
- Political rallies or partisan content (unless business-relevant policy)
- Generic press releases or product launches
- Single-company minor news with no broader implications

Mid-range (4-7):
- Tech product launches with sector implications
- Earnings results of major companies
- Court rulings affecting business

For EACH headline you receive, output:
- gd_worthiness_score (0-10 integer)
- keywords (2-4 short topic tags, e.g., "RBI policy", "fintech regulation", "EV adoption")
- category (one of: "business", "macro", "policy", "tech", "global", "sector", "other")

Then identify the single STAR headline — the most GD-worthy one in the batch — and set is_star=true for it.

OUTPUT FORMAT: valid JSON object with key "classified" containing an array. Each item must include the original title plus your additions. Preserve the original order.

Example output structure:
{
  "classified": [
    {
      "title": "...",
      "gd_worthiness_score": 8,
      "keywords": ["RBI policy", "inflation"],
      "category": "macro",
      "is_star": true
    },
    ...
  ]
}
"""


def classify_headlines(raw_headlines: List[dict]) -> List[ClassifiedHeadline]:
    """
    Classify a batch of raw headlines using OpenAI.
    
    Input: list of dicts from news_fetcher (RawHeadline shape)
    Output: list of ClassifiedHeadline with AI-added fields
    
    Strategy: send all headlines in ONE API call (batch) to minimize cost.
    GPT-4o-mini at ~40 headlines costs ~₹1-2 per classification run.
    
    Raises ClassificationError on AI failure. Caller should fall back gracefully.
    """
    if not raw_headlines:
        return []
    
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ClassificationError("OPENAI_API_KEY not set")
    
    # Prepare input: just titles + descriptions for AI context (save tokens)
    headlines_for_ai = [
        {
            "index": i,
            "title": h["title"],
            "description": h.get("description") or "",
            "source": h["source_name"],
        }
        for i, h in enumerate(raw_headlines)
    ]
    
    user_message = (
        f"Classify these {len(raw_headlines)} headlines for MBA GD-worthiness. "
        f"Identify the single most GD-worthy as the star.\n\n"
        f"Headlines:\n{json.dumps(headlines_for_ai, ensure_ascii=False, indent=2)}"
    )
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # cheap, plenty smart for this task
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except Exception as e:
        raise ClassificationError(f"OpenAI API call failed: {type(e).__name__}: {e}")
    
    raw_content = response.choices[0].message.content
    if not raw_content:
        raise ClassificationError("OpenAI returned empty response")
    
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ClassificationError(f"OpenAI returned invalid JSON: {e}")
    
    classified_array = parsed.get("classified", [])
    if not isinstance(classified_array, list) or len(classified_array) == 0:
        raise ClassificationError(f"Expected 'classified' array, got: {parsed}")
    
    # Merge AI output back with original headline data
    # AI may not return exact same count if it got confused — match by index/title
    classified_by_title = {item.get("title", "").strip().lower(): item for item in classified_array}
    
    results: List[ClassifiedHeadline] = []
    star_found = False
    
    for original in raw_headlines:
        title_key = original["title"].strip().lower()
        ai_data = classified_by_title.get(title_key)
        
        if not ai_data:
            # Fallback: use neutral defaults if AI missed this headline
            print(f"WARNING: AI didn't classify headline: {original['title'][:60]}")
            ai_data = {
                "gd_worthiness_score": 5,
                "keywords": ["business"],
                "category": "other",
                "is_star": False,
            }
        
        # Ensure we only have ONE star (defensive — AI sometimes flags multiple)
        is_star = bool(ai_data.get("is_star", False)) and not star_found
        if is_star:
            star_found = True
        
        results.append({
            "title": original["title"],
            "description": original.get("description"),
            "thumbnail_url": original.get("thumbnail_url"),
            "source_url": original["source_url"],
            "source_name": original["source_name"],
            "published_at": original["published_at"],
            "gd_worthiness_score": max(0, min(10, int(ai_data.get("gd_worthiness_score", 5)))),
            "keywords": [str(k).strip() for k in ai_data.get("keywords", [])][:4],
            "category": str(ai_data.get("category", "other")).lower(),
            "is_star": is_star,
        })
    
    # If AI didn't pick a star (rare), promote highest scorer
    if not star_found and results:
        top = max(range(len(results)), key=lambda i: results[i]["gd_worthiness_score"])
        results[top]["is_star"] = True
    
    return results


def filter_top_headlines(classified: List[ClassifiedHeadline], top_n: int = 20) -> List[ClassifiedHeadline]:
    """
    From a batch of classified headlines, return the top N by score.
    Always includes the star headline.
    Sorts result by: star first, then descending score.
    """
    if not classified:
        return []
    
    # Sort by gd_worthiness_score descending, with star always at top
    sorted_h = sorted(
        classified,
        key=lambda h: (not h["is_star"], -h["gd_worthiness_score"])
    )
    
    return sorted_h[:top_n]