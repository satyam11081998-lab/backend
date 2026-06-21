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
    gd_worthiness_score: int  # 0-100
    keywords: List[str]        # 2-4 topic tags
    category: str              # "business", "macro", "micro", "policy", "tech", "geopolitics", "jobs", "other"
    is_star: bool              # only ONE headline in batch has is_star=True


class ClassificationError(Exception):
    """Raised when OpenAI fails to classify headlines properly."""
    pass


CLASSIFIER_SYSTEM_PROMPT = """You are an expert curator for MBA Group Discussion (GD) topics in India.

Your job: rate news headlines for how likely they are to become GD topics in MBA/PGDM placement interviews at IIMs, IMI, FMS, SP Jain, and top consulting/IB recruiters.

You must score each headline on a scale of 0 to 100 based on the following specific parameters:
- India-specific relevance (Does it affect the Indian economy or society?)
- Management/Strategy impact (Does it involve corporate governance, mergers, leadership, or business strategy?)
- Jobs & Recruiting context (Does it impact MBA hiring, tech layoffs, or workforce trends?)
- Macroeconomics (Inflation, GDP, RBI policy, central budgets)
- Microeconomics (Specific sector dynamics, supply chain, pricing strategies)
- Geopolitics (Trade wars, international relations affecting Indian business)

A high GD-worthy headline (80-100) has these traits:
- Has TWO defensible sides (room for debate)
- Touches heavily on one or more of the above parameters.
- Affects multiple stakeholders (consumers, companies, government, society).

You MUST instantly score 0 for any news about:
- Bollywood, celebrity gossip, or entertainment.
- Sports (unless it's a major business acquisition, e.g., IPL broadcasting rights).
- Local crimes, accidents, or individual tragedies.
- Purely political posturing (e.g., party rallies) without economic/policy substance.
- Penny stock movements or "Top 5 stocks to buy" clickbait.
- Generic PR product launches with no strategic depth.

For EACH headline you receive, output:
- gd_worthiness_score (0-100 integer)
- keywords (2-4 short topic tags, e.g., "RBI policy", "fintech regulation")
- category (one of: "business", "macro", "micro", "policy", "tech", "geopolitics", "jobs", "other")

Then identify the single STAR headline — the most GD-worthy one in the batch — and set is_star=true for it.

OUTPUT FORMAT: valid JSON object with key "classified" containing an array. Each item must include the original "index" (the integer index given in the input for that headline) and the original title, plus your additions. Preserve the original order.

Example output structure:
{
  "classified": [
    {
      "index": 0,
      "title": "...",
      "gd_worthiness_score": 85,
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
    
    # Bounded so a hung classify call fails fast (run_news_refresh treats a page
    # failure as non-fatal) instead of stalling the news cron.
    client = OpenAI(api_key=api_key, timeout=45.0, max_retries=2)
    
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
    
    # Merge AI output back with original headline data.
    # PRIMARY KEY = index (each input headline was given an integer "index";
    # the model echoes it back). GPT lightly rewrites titles, so a title-only
    # join silently misses → headlines fell to the fallback score → got filtered
    # out → "saved 0". We now match by index first, then fall back to a
    # normalized title, then to positional order, then to the neutral default.
    def _norm(t: str) -> str:
        return " ".join((t or "").strip().lower().split())

    classified_by_index = {}
    for item in classified_array:
        idx = item.get("index")
        try:
            classified_by_index[int(idx)] = item
        except (TypeError, ValueError):
            pass  # item had no usable index; title/position fallback will cover it

    classified_by_title = {_norm(item.get("title", "")): item for item in classified_array}

    results: List[ClassifiedHeadline] = []
    star_found = False
    
    for i, original in enumerate(raw_headlines):
        # 1) match by index, 2) by normalized title, 3) by position, 4) default
        ai_data = classified_by_index.get(i)
        if not ai_data:
            ai_data = classified_by_title.get(_norm(original["title"]))
        if not ai_data and i < len(classified_array):
            ai_data = classified_array[i]
        
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
            "gd_worthiness_score": max(0, min(100, int(ai_data.get("gd_worthiness_score", 50)))),
            "keywords": [str(k).strip() for k in ai_data.get("keywords", [])][:4],
            "category": str(ai_data.get("category", "other")).lower(),
            "is_star": is_star,
        })
    
    # If AI didn't pick a star (rare), promote highest scorer
    if not star_found and results:
        top = max(range(len(results)), key=lambda i: results[i]["gd_worthiness_score"])
        results[top]["is_star"] = True
    
    return results


def filter_top_headlines(classified: List[ClassifiedHeadline], top_n: int = 10, min_score: int = 75) -> List[ClassifiedHeadline]:
    """
    From a batch of classified headlines, return the top N by score that meet the min_score.
    Always includes the star headline if it exists.
    Sorts result by: star first, then descending score.
    """
    if not classified:
        return []
    
    # Filter by minimum score (unless it's the star)
    qualified = [h for h in classified if h["gd_worthiness_score"] >= min_score or h["is_star"]]
    
    # Sort by gd_worthiness_score descending, with star always at top
    sorted_h = sorted(
        qualified,
        key=lambda h: (not h["is_star"], -h["gd_worthiness_score"])
    )
    
    return sorted_h[:top_n]