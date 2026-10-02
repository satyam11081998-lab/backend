"""Compact, token-cheap rendering of a Pydantic model as a JSON shape for prompts.

Example:  {"claims": [{"id": string, "type": "impact"|"scale"|..., "quantified": boolean}]}
"""

from __future__ import annotations

import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel


def _render(tp: Any, depth: int = 0) -> str:
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is typing.Union or (origin is not None and str(origin) == "types.UnionType"):
        parts = [a for a in args if a is not type(None)]
        r = " | ".join(_render(a, depth) for a in parts)
        return r + (" | null" if len(parts) != len(args) else "")
    if origin is typing.Literal:
        return "|".join(f'"{a}"' if isinstance(a, str) else str(a) for a in args)
    if origin in (list, typing.List):
        return f"[{_render(args[0], depth) if args else 'any'}]"
    if origin in (dict, typing.Dict):
        return "{string: " + (_render(args[1], depth) if len(args) > 1 else "any") + "}"
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        if depth > 6:
            return "{...}"
        fields = []
        for name, f in tp.model_fields.items():
            desc = f" /* {f.description} */" if f.description else ""
            fields.append(f'"{name}": {_render(f.annotation, depth + 1)}{desc}')
        return "{" + ", ".join(fields) + "}"
    return {str: "string", int: "integer", float: "number", bool: "boolean"}.get(tp, "any")


def schema_text(model: type[BaseModel]) -> str:
    return _render(model)
