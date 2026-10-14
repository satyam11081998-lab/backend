"""The variant registry: id -> prompt-layer implementation + provenance.

Adding a variant is the whole extension point of the arena. Snapshot the prompt
file into this folder, add a row below, and both arms can select it.

INTERFACE every variant must satisfy (the engine's existing seam):

    build_adaptive_interviewer_messages(
        case_content: str, case_type: str, transcript: Iterable[dict],
        new_user_message: str, teaching_policy: str = "coached",
        signals_block: str = "", clarifications_exhausted: bool = False,
    ) -> list[dict]        # OpenAI messages array

tests/test_arena_variants.py enforces conformance AND pins each frozen variant's
SHA256, so an accidental edit to a baseline fails loudly instead of quietly
invalidating every vote recorded against it.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable, Dict, List, Optional

# `live` tracks the working tree: the arm you iterate with.
from prompts.interview_prompts_v2 import (
    build_adaptive_interviewer_messages as _live_builder,
)
from prompts.variants import baseline_main_v12 as _baseline_main
from prompts.variants import patched_v12_2026_10_14 as _patched


def _sha256_of(module: Any) -> str:
    """Content hash of a variant module's source file, for drift detection."""
    path = getattr(module, "__file__", None)
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_SPECS: List[Dict[str, Any]] = [
    {
        "id": "live",
        "label": "Live working tree",
        "description": (
            "prompts/interview_prompts_v2.py exactly as it is on disk right now. "
            "Moves as you edit, so it is the arm to iterate with -- and the one arm "
            "whose results are NOT reproducible later."
        ),
        "origin": "working tree",
        "frozen": False,
        "builder": _live_builder,
        "module": None,
    },
    {
        "id": "baseline-main",
        "label": "Baseline - main (pre-patch)",
        "description": (
            "Prompt stack on backend main @ 5f66b3b. Claims a human senior-consultant "
            "identity, has the original YOU OWN THE FACTS block, plain-text case context."
        ),
        "origin": "git main @ 5f66b3b",
        "frozen": True,
        "builder": _baseline_main.build_adaptive_interviewer_messages,
        "module": _baseline_main,
    },
    {
        "id": "patched-2026-10-14",
        "label": "Reviewed patch (c4674d7)",
        "description": (
            "ChatGPT's AI-behaviour patch plus the six review fixes: single AI identity, "
            "restored banned-phrase list, system_note un-demoted, plain-text case context, "
            "anti-clustering calibration."
        ),
        "origin": "git test/ai-behaviour-patch @ c4674d7",
        "frozen": True,
        "builder": _patched.build_adaptive_interviewer_messages,
        "module": _patched,
    },
]

VARIANTS: Dict[str, Dict[str, Any]] = {s["id"]: s for s in _SPECS}

# What the arena opens with: the reviewed state against the state it replaced.
DEFAULT_ARM_A = "baseline-main"
DEFAULT_ARM_B = "patched-2026-10-14"


def variant_ids() -> List[str]:
    return [s["id"] for s in _SPECS]


def get_variant(variant_id: str) -> Dict[str, Any]:
    """Look up a variant. Raises KeyError for an unknown id -- the caller turns that
    into a 400 rather than silently falling back, because a silent fallback would
    attribute one variant's replies to another."""
    if variant_id not in VARIANTS:
        raise KeyError(variant_id)
    return VARIANTS[variant_id]


def get_builder(variant_id: str) -> Callable[..., List[Dict[str, str]]]:
    return get_variant(variant_id)["builder"]


def describe_variants() -> List[Dict[str, Any]]:
    """Public, JSON-safe catalogue for the admin UI. No callables."""
    out = []
    for s in _SPECS:
        out.append({
            "id": s["id"],
            "label": s["label"],
            "description": s["description"],
            "origin": s["origin"],
            "frozen": bool(s["frozen"]),
            "sha256": _sha256_of(s["module"])[:12] if s["module"] is not None else None,
        })
    return out


def fingerprint(variant_id: str) -> Optional[str]:
    """Full content hash of a frozen variant, stored with every recorded vote so a
    later reader can tell whether the baseline was the same one they think it was."""
    spec = get_variant(variant_id)
    if spec["module"] is None:
        return None
    return _sha256_of(spec["module"])
