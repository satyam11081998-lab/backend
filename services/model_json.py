"""
Defensive JSON parsing for model replies.

WHY THIS EXISTS
---------------
`response_format={"type": "json_object"}` is an OpenAI/Groq feature. Gemini's
OpenAI-compatibility layer can return ```json fences, commentary, trailing
commas, or unescaped control characters. In addition, reasoning models may
occasionally truncate near max_tokens.

This parser handles:
- Code fences (```json ... ```)
- Surrounding commentary before/after the JSON block
- Trailing commas before } and ]
- Unescaped newlines/control characters (strict=False)
- Truncated strings/objects/arrays by repairing unclosed brackets and quotes
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def _repair_truncated_json(s: str) -> str:
    """Attempt to balance brackets and quotes for slightly truncated JSON."""
    s = s.strip()
    if not s.startswith("{"):
        i = s.find("{")
        if i != -1:
            s = s[i:]
        else:
            return s

    in_string = False
    escape = False
    stack = []

    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch in ("{", "["):
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()

    # If terminated inside string, close quote
    if in_string:
        s += '"'

    # Strip any trailing comma before closing
    s = re.sub(r",\s*$", "", s.strip())

    # Close unclosed containers
    for opener in reversed(stack):
        s = re.sub(r",\s*$", "", s.strip())
        if opener == "{":
            s += "}"
        elif opener == "[":
            s += "]"

    return s


def parse_model_json(raw: str) -> Dict[str, Any]:
    """Parse a model's JSON reply, tolerating fences, prose, trailing commas,
    and partial truncation.

    Raises ValueError if it still cannot be parsed.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response")

    txt = raw.strip()

    # Strip markdown code fences anywhere in the string
    txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"\s*```$", "", txt)

    # Isolate outermost JSON object
    i = txt.find("{")
    if i != -1:
        j = txt.rfind("}")
        if j != -1 and j > i:
            txt = txt[i : j + 1]
        else:
            txt = txt[i:]

    # Attempt 1: direct parse with strict=False (allows unescaped control characters)
    try:
        return json.loads(txt, strict=False)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip trailing commas
    repaired = re.sub(r",(\s*[}\]])", r"\1", txt)
    try:
        return json.loads(repaired, strict=False)
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair unclosed strings and brackets/braces (e.g. from truncation)
    try:
        balanced = _repair_truncated_json(repaired)
        return json.loads(balanced, strict=False)
    except (json.JSONDecodeError, Exception) as e:
        raise ValueError(f"Model returned invalid JSON: {e}")
