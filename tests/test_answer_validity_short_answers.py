"""Cheap, dependency-isolated checks for short genuine case answers."""
from __future__ import annotations
import importlib
import sys
from types import ModuleType
from unittest.mock import patch


class _OpenAIStub:
    def __init__(self, *args, **kwargs):
        pass


def _load_answer_validity():
    openai_stub = ModuleType("openai")
    openai_stub.OpenAI = _OpenAIStub
    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    usage_stub = ModuleType("services.ai_usage")
    usage_stub.log_ai_usage = lambda **kwargs: None
    providers_stub = ModuleType("services.ai_providers")
    providers_stub.chat_with_fallback = lambda *args, **kwargs: None
    with patch.dict(sys.modules, {
        "openai": openai_stub,
        "dotenv": dotenv_stub,
        "services.ai_usage": usage_stub,
        "services.ai_providers": providers_stub,
    }):
        sys.modules.pop("services.answer_validity", None)
        module = importlib.import_module("services.answer_validity")
        # Do not leave the isolated stubbed module in sys.modules for other tests.
        sys.modules.pop("services.answer_validity", None)
        return module


def test_short_numeric_answers_are_deferred_to_case_aware_screen():
    module = _load_answer_validity()
    assert module.deterministic_verdict("2 crore") is None
    assert module.deterministic_verdict("500") is None


def test_unmistakable_greetings_and_placeholders_still_fail_fast():
    module = _load_answer_validity()
    assert module.deterministic_verdict("hi")["verdict"] == "gibberish"
    assert module.deterministic_verdict("I don't know")["verdict"] == "gibberish"
