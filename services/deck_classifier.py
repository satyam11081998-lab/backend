"""
4-Stage Deck Taxonomy Classifier & Inference Engine.

Hierarchical inference with field-level confidence scores:
- Stage 1: Deterministic Extraction (folder path, filename, title slide)
- Stage 2: Rule-Based Taxonomy & Keyword Matching
- Stage 3: Cached LLM Classification (only when confidence is below threshold)
- Stage 4: Confidence Scoring & Human Review Routing
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from services.deck_taxonomy import (
    BSCHOOLS,
    CASE_TYPES,
    DIFFICULTY_LEVELS,
    FUNCTIONS,
    INDUSTRIES,
    KNOWN_COMPETITIONS,
    RESULTS,
    ROUND_TYPES,
    SOURCE_KINDS,
    STANDARD_TAGS,
)

CONFIDENCE_HIGH_THRESHOLD = 0.85
CONFIDENCE_REVIEW_THRESHOLD = 0.70

# Keyword heuristics for Case Type (Domain)
CASE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "supply chain": ["supply chain", "logistics", "warehousing", "inventory", "drp", "procurement", "reverse logistics", "distribution network"],
    "sustainability": ["sustainability", "sustainable", "carbon neutral", "net zero", "circular economy", "recycling", "esg", "green", "emissions"],
    "digital transformation": ["digital transformation", "automation", "cloud", "ai", "modernization", "erp", "digital infrastructure", "api"],
    "marketing": ["brand strategy", "brand positioning", "consumer insights", "campaign", "awareness", "ad spend", "influencer", "stp", "target persona"],
    "growth": ["growth strategy", "market expansion", "revenue growth", "scale", "penetration", "user acquisition", "go to market", "gtm"],
    "market entry": ["market entry", "new market", "geographical expansion", "entry strategy", "mode of entry"],
    "pricing": ["pricing strategy", "price elasticity", "monetization", "subscription", "dynamic pricing", "unit economics"],
    "BFSI": ["banking", "npa", "lending", "credit", "fintech", "insurance", "rrb", "deposits", "underwriting", "wealth management", "upi"],
    "product": ["product strategy", "feature roadmap", "ui/ux", "wireframe", "mvp", "user journey", "retention", "engagement", "product spotlight"],
    "operations": ["operations", "process optimization", "throughput", "bottleneck", "lean", "six sigma", "capacity utilization", "plant"],
    "finance": ["financial modeling", "dcf", "valuation", "npv", "irr", "ebitda", "capital allocation", "cost reduction"],
    "analytics": ["analytics", "machine learning", "data science", "dashboard", "predictive modeling", "clustering"],
    "healthcare": ["pharma", "clinical", "hospital", "healthcare", "patient", "medical device"],
    "retail": ["retail", "fmcg", "store format", "kirana", "visual merchandising", "footfall"],
    "strategy": ["strategy", "competitive advantage", "swot", "pestel", "porter", "strategic priorities", "core competencies"],
}

# Industry keywords
INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "FMCG": ["fmcg", "packaged goods", "unilever", "itc", "mondelez", "colgate", "marico", "nestle", "p&g", "shampoo", "soap", "food", "snacks", "personal care"],
    "Paints & Coatings": ["paint", "coatings", "asian paints", "berger", "dulux", "waterproofing", "dealers", "contractors"],
    "Consumer Electronics": ["electronics", "audio", "boat", "samsung", "smartphones", "wearables", "appliances", "hearables", "hardware"],
    "BFSI & Banking": ["bank", "banking", "icici", "union bank", "hdfc", "sbi", "loan", "fintech", "insurance", "deposits", "credit"],
    "E-Commerce & Retail": ["e-commerce", "ecommerce", "meesho", "myntra", "flipkart", "amazon", "d2c", "fashion", "marketplace", "orders"],
    "Consulting & Professional Services": ["accenture", "mckinsey", "bcg", "bain", "consulting", "advisory", "deloitte", "pwc", "kpmg", "ey"],
    "Manufacturing & Industrial": ["tata steel", "pidilite", "cummins", "steel", "manufacturing", "automotive", "machinery", "adhesives", "fevicol"],
    "Tech, SaaS & IT": ["saas", "software", "publicis sapient", "cloud", "developer", "platform", "b2b saas"],
    "Hospitality & Travel": ["travel", "tourism", "hotel", "makemytrip", "mmt", "booking", "flights", "hospitality"],
}


def _match_keywords(text_lower: str, keyword_map: Dict[str, List[str]]) -> Tuple[Optional[str], float]:
    """Score text against a dictionary of keywords and return top match with confidence."""
    best_key = None
    max_score = 0
    total_matches = 0

    for key, words in keyword_map.items():
        score = 0
        for w in words:
            # Word boundary regex search
            count = len(re.findall(r"\b" + re.escape(w) + r"\b", text_lower))
            score += count
        if score > 0:
            total_matches += score
            if score > max_score:
                max_score = score
                best_key = key

    if not best_key or max_score == 0:
        return None, 0.0

    confidence = min(0.95, 0.50 + (max_score / (total_matches + 2)) * 0.45)
    return best_key, round(confidence, 2)


def classify_deck_rules(extracted_deck: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform Stage 1 and Stage 2 deterministic & rule-based classification.
    """
    path_signals = extracted_deck.get("path_signals", {})
    full_text = extracted_deck.get("full_text", "")
    full_text_lower = full_text.lower()
    slides = extracted_deck.get("slides", [])
    title_slide_text = (slides[0]["text"] if slides else "").lower()
    first_3_slides_text = " ".join([s["text"] for s in slides[:3]]).lower()

    field_confidences: Dict[str, float] = {}

    # 1. Competition & Organizer & Company
    competition = path_signals.get("detected_competition")
    company = path_signals.get("detected_company")
    organizer = path_signals.get("detected_company") or ""
    source_kind = path_signals.get("source_kind", "corporate")

    if competition:
        field_confidences["competition"] = 0.95
        field_confidences["company"] = 0.92
        field_confidences["organizer"] = 0.90
    else:
        # Check text in first 3 slides for known competitions
        for comp_key, comp_info in KNOWN_COMPETITIONS.items():
            if comp_key in first_3_slides_text:
                competition = comp_info["canonical_name"]
                company = comp_info["company"]
                organizer = comp_info["organizer"]
                source_kind = comp_info["source_kind"]
                field_confidences["competition"] = 0.88
                field_confidences["company"] = 0.88
                field_confidences["organizer"] = 0.85
                break

    # If company still unknown, look for "Company Name : XY" in slides
    if not company:
        comp_regex = re.search(r"company\s*name\s*[:\-]\s*([a-zA-Z0-9\s&]+?)(?:\n|$|\.|\r)", full_text, re.IGNORECASE)
        if comp_regex:
            company = comp_regex.group(1).strip()
            organizer = company
            field_confidences["company"] = 0.80

    # 2. Result
    result = path_signals.get("detected_result")
    if result:
        field_confidences["result"] = 0.95
    else:
        # Scan slide 1 and full text
        for r in RESULTS:
            if r.lower() in title_slide_text or r.lower() in full_text_lower[:1000]:
                result = r
                field_confidences["result"] = 0.85
                break
        if not result:
            result = "National Finalist"  # Safe default for curated competition repo
            field_confidences["result"] = 0.60

    # 3. Round Type
    round_type = path_signals.get("detected_round_type")
    if round_type:
        field_confidences["round_type"] = 0.90
    else:
        if "semi" in result.lower():
            round_type = "semi-final"
            field_confidences["round_type"] = 0.85
        elif "final" in result.lower() or "winner" in result.lower() or "runner" in result.lower():
            round_type = "finale"
            field_confidences["round_type"] = 0.85
        elif "campus" in result.lower():
            round_type = "campus"
            field_confidences["round_type"] = 0.85
        else:
            round_type = "finale"
            field_confidences["round_type"] = 0.60

    # 4. Year
    year = path_signals.get("detected_year")
    if year:
        field_confidences["year"] = 0.95
    else:
        # Search text for 2020..2026
        years_found = re.findall(r"\b(202[0-6]|201[8-9])\b", full_text)
        if years_found:
            year = int(max(set(years_found), key=years_found.count))
            field_confidences["year"] = 0.80
        else:
            year = 2024  # default
            field_confidences["year"] = 0.50

    # 5. Case Type (Domain)
    case_type = path_signals.get("detected_case_type")
    if case_type:
        field_confidences["case_type"] = 0.88
    else:
        matched_type, conf = _match_keywords(full_text_lower, CASE_TYPE_KEYWORDS)
        if matched_type:
            case_type = matched_type
            field_confidences["case_type"] = conf
        else:
            case_type = "strategy"
            field_confidences["case_type"] = 0.50

    # 6. Industry
    industry = None
    if competition:
        for comp_key, comp_info in KNOWN_COMPETITIONS.items():
            if comp_info["canonical_name"] == competition:
                industry = comp_info.get("industry")
                field_confidences["industry"] = 0.92
                break

    if not industry:
        matched_ind, conf = _match_keywords(full_text_lower, INDUSTRY_KEYWORDS)
        if matched_ind:
            industry = matched_ind
            field_confidences["industry"] = conf
        else:
            industry = "FMCG"
            field_confidences["industry"] = 0.50

    # 7. Function
    function_map = {
        "strategy": "Strategy",
        "growth": "Strategy",
        "market entry": "Strategy",
        "marketing": "Marketing & Brand Strategy",
        "pricing": "Marketing & Brand Strategy",
        "supply chain": "Operations & Supply Chain",
        "operations": "Operations & Supply Chain",
        "finance": "Finance & M&A",
        "M&A": "Finance & M&A",
        "product": "Product Management",
        "digital transformation": "Product Management",
        "technology": "Product Management",
        "analytics": "Analytics & AI Strategy",
        "BFSI": "Finance & M&A",
        "sustainability": "Strategy",
    }
    function = function_map.get(case_type, "Strategy")
    field_confidences["function"] = field_confidences.get("case_type", 0.70)

    # 8. Difficulty & Geography
    slide_count = extracted_deck.get("slide_count", 0)
    difficulty = "hard" if slide_count > 15 else "medium" if slide_count >= 7 else "easy"
    field_confidences["difficulty"] = 0.90
    geography = "India"
    field_confidences["geography"] = 0.95

    # Overall Confidence Calculation
    weights = {
        "competition": 0.25,
        "company": 0.20,
        "case_type": 0.20,
        "result": 0.15,
        "year": 0.10,
        "industry": 0.10,
    }
    overall_confidence = sum(field_confidences.get(k, 0.5) * w for k, w in weights.items())

    # Build Tags
    tags: List[str] = []
    if case_type:
        tags.append(case_type.title())
    if function:
        tags.append(function.split("&")[0].strip())
    if industry and industry != "Other":
        tags.append(industry.split("&")[0].strip())

    for candidate_tag in STANDARD_TAGS:
        if candidate_tag.lower() in full_text_lower and candidate_tag not in tags:
            tags.append(candidate_tag)
            if len(tags) >= 6:
                break

    # Determine title
    team_name = path_signals.get("detected_team")
    college = path_signals.get("detected_college")
    title_suffix = f" · {team_name}" if team_name else (f" · {college}" if college else "")
    lead_name = competition or company or "Case Competition"
    title = f"{lead_name} {year} — {result}{title_suffix}"

    return {
        "competition": competition or "Corporate Case Challenge",
        "company": company or organizer or "Enterprise",
        "organizer": organizer or company or "",
        "source_kind": source_kind,
        "result": result,
        "round_type": round_type,
        "case_type": case_type,
        "industry": industry,
        "function": function,
        "year": year,
        "difficulty": difficulty,
        "geography": geography,
        "tags": tags[:6],
        "title": title,
        "confidence": round(overall_confidence, 2),
        "field_confidences": field_confidences,
        "needs_llm_classification": overall_confidence < CONFIDENCE_HIGH_THRESHOLD,
        "needs_review": overall_confidence < CONFIDENCE_REVIEW_THRESHOLD,
    }
