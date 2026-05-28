import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

print("Testing News Fetcher...")
try:
    from services.news_fetcher import fetch_all_headlines
    from services.headline_classifier import classify_headlines, filter_top_headlines
    
    raw = fetch_all_headlines(max_results=20, page=1)
    if raw:
        classified = classify_headlines(raw)
        top = filter_top_headlines(classified, top_n=5, min_score=75)
        print(f"News fetch success! Found {len(raw)} raw, {len(top)} top GD-worthy headlines.")
        for h in top[:2]:
            print(f"- [{h['gd_worthiness_score']}/100] {h['title']}")
    else:
        print("News fetch returned 0 headlines.")
except Exception as e:
    print(f"News fetch failed: {e}")

print("\nTesting Content Generator...")
try:
    from services.content_generator import generate_daily_content
    # Test generation without saving to DB (so we don't pollute DB if schema isn't ready)
    generated = generate_daily_content([])
    print("AI Content Generation success!")
    print(f"CASE: {generated.get('case', {}).get('title')}")
    print(f"GUESSTIMATE: {generated.get('guesstimate', {}).get('title')}")
except Exception as e:
    print(f"Content generation failed: {e}")
