"""
Defensive JSON parsing for model replies.

WHY THIS EXISTS
---------------
`response_format={"type": "json_object"}` is an OpenAI/Groq feature. Gemini's
OpenAI-compatibility layer does not document support for it, so a Gemini reply
can come back as a ```json fenced block, with prose around it, or with a trailing
comma — all of which make a bare `json.loads()` throw. That failure happens AFTER
a successful HTTP call, so the provider chain in `ai_providers.chat_with_fallback`
cannot rescue it: the request succeeded, the parse did not.

`deck_ai_gemini._parse_model_json` solved the same problem for the deck pipeline.
This is that logic, lifted so the GD generators can share it. Deck Vault keeps its
own copy on purpose — it is a live path and not worth disturbing for a refactor.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def parse_model_json(raw: str) -> Dict[str, Any]:
    """Parse a model's JSON reply, tolerating fences, prose and trailing commas.

    Raises ValueError if it still cannot be parsed, so callers can keep their
    existing "the model returned garbage" error path.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response")

    txt = raw.strip()

    # ```json ... ```  or  ``` ... ```
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z0-9]*\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt).strip()

    # isolate the outermost object when the model wrapped it in commentary
    if not txt.startswith("{"):
        i = txt.find("{")
        if i != -1:
            txt = txt[i:]
    j = txt.rfind("}")
    if j != -1:
        txt = txt[: j + 1]

    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        repaired = re.sub(r",(\s*[}\]])", r"\1", txt)  # trailing commas
        return json.loads(repaired)
