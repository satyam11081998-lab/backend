"""
Abstract GD brief generator.

Unlike news briefs (fact-driven), abstract GD topics ("Black or White",
"The glass is half full", "Zero") are NOT about facts — they test lateral,
critical thinking. This generator produces a brief that TEACHES the candidate
how to crack THIS abstract topic AND, by repetition, how to crack any abstract
topic: multiple interpretations, a divergent idea-pool, structured lenses,
balanced positions, analogies to cite, a model structure, and pitfalls.

Uses GPT-4o (user-facing). Generated on demand; the caller may cache.
"""

import os
import json
from typing import TypedDict, List
from openai import OpenAI


class GeneratedAbstractBrief(TypedDict):
    topic: str
    interpretations: List[str]    # 2-3 ways to read the abstract phrase
    idea_pool: List[str]          # 6-8 divergent associations to seed discussion
    lenses: List[str]             # structured perspectives applied to THIS topic
    balanced_for: List[str]       # points on one side
    balanced_against: List[str]   # points on the other side
    analogies: List[str]          # examples / stories / analogies to cite
    sample_structure: List[str]   # a model open -> build -> conclude flow
    pitfalls: List[str]           # what weak candidates do
    opening_lines: List[str]
    closing_lines: List[str]


class AbstractBriefError(Exception):
    pass


ABSTRACT_SYSTEM_PROMPT = """You are an elite Group Discussion coach for top Indian MBA/PGDM placements
(targets: McKinsey, BCG, Bain, IB, FMCG strategy).

You are given a GD topic. It may be ABSTRACT (a word/phrase/proverb with no factual answer,
e.g. "Black or White", "Culture eats strategy for breakfast") or DOMAIN-BASED (HR, business,
economy, technology, society — e.g. "Should AI replace recruiters?", "Is a 4-day week the future?").
For abstract topics, lead with multiple interpretations of the phrase. For domain topics, the
"interpretations" become the key sub-questions / framings of the debate, and the idea_pool and
analogies should include concrete, real examples and figures (cited fairly) where they exist.
Either way, reward lateral thinking, structure, and balance — never a one-sided take.

Your job: produce a brief that teaches a candidate how to crack THIS topic, in a way that also
trains the repeatable METHOD for any abstract topic. Be concrete and specific to the given topic
— never generic filler.

Produce, as strict JSON:
- interpretations: 2-3 genuinely different ways to read the phrase (literal, metaphorical,
  contrarian). This is the most important skill — show how the same prompt opens many doors.
- idea_pool: 6-8 short, DIVERGENT associations/angles (business, society, psychology, history,
  ethics, personal) — the raw material a candidate would brainstorm in the first 30 seconds.
- lenses: 4-6 structured perspectives APPLIED to this topic, each as "Lens: the point".
  Use lenses like Stakeholders, Time horizon (past/present/future), Scale (individual/org/society),
  Opposing values, PESTLE, Business example. Make each specific to the topic.
- balanced_for / balanced_against: 3 strong points for EACH side of the central tension (steelman
  both — abstract GDs reward balance, not a hot take).
- analogies: 3-4 vivid analogies, short stories, historical or current examples a candidate can cite
  to sound concrete instead of vague.
- sample_structure: a 4-6 step model flow for the discussion (how to OPEN with a frame/definition,
  how to BUILD breadth then depth, how to bring in others, how to CONCLUDE with synthesis).
- pitfalls: 3-4 specific mistakes weak candidates make on this kind of topic.
- opening_lines: 2-3 strong ways to open (assertive, sets a frame, not aggressive).
- closing_lines: 2-3 ways to synthesize and close memorably.

STYLE: sharp, confident, Indian English (lakh/crore where natural). No clichés. Stay NEUTRAL —
present both sides fairly; never tell the candidate the "right" answer (abstract topics have none).

OUTPUT: a valid JSON object with EXACTLY these keys:
{
  "interpretations": [...],
  "idea_pool": [...],
  "lenses": [...],
  "balanced_for": [...],
  "balanced_against": [...],
  "analogies": [...],
  "sample_structure": [...],
  "pitfalls": [...],
  "opening_lines": [...],
  "closing_lines": [...]
}
"""


def generate_abstract_brief(topic: str) -> GeneratedAbstractBrief:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AbstractBriefError("OPENAI_API_KEY not set")
    topic = (topic or "").strip()
    if not topic:
        raise AbstractBriefError("Empty topic")
    if len(topic) > 160:
        raise AbstractBriefError("Topic too long")

    user_message = (
        f'ABSTRACT GD TOPIC: "{topic}"\n\n'
        "Generate the abstract GD brief for an MBA candidate preparing to discuss this in a "
        "10-minute placement Group Discussion. Be specific to this exact topic."
    )

    client = OpenAI(api_key=api_key, timeout=60.0, max_retries=1)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": ABSTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.6,
        )
    except Exception as e:
        raise AbstractBriefError(f"OpenAI API call failed: {type(e).__name__}: {e}")

    raw = response.choices[0].message.content
    if not raw:
        raise AbstractBriefError("OpenAI returned empty response")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise AbstractBriefError(f"OpenAI returned invalid JSON: {e}")

    def _list(key: str, cap: int) -> List[str]:
        vals = parsed.get(key, [])
        if not isinstance(vals, list):
            return []
        return [str(v).strip() for v in vals if str(v).strip()][:cap]

    result: GeneratedAbstractBrief = {
        "topic": topic,
        "interpretations": _list("interpretations", 3),
        "idea_pool": _list("idea_pool", 8),
        "lenses": _list("lenses", 6),
        "balanced_for": _list("balanced_for", 3),
        "balanced_against": _list("balanced_against", 3),
        "analogies": _list("analogies", 4),
        "sample_structure": _list("sample_structure", 6),
        "pitfalls": _list("pitfalls", 4),
        "opening_lines": _list("opening_lines", 3),
        "closing_lines": _list("closing_lines", 3),
    }

    # Minimal validity: must have at least interpretations + idea_pool.
    if not result["interpretations"] or not result["idea_pool"]:
        raise AbstractBriefError("Brief missing core fields (interpretations/idea_pool)")
    return result
