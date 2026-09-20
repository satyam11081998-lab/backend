"""Let the stdlib-only test suites import the deterministic interviewer layer.

WHY THIS FILE EXISTS
--------------------
`tests/test_session_signals.py` and `tests/test_interviewer_mode.py` are
documented to run "on the standard library alone (no OpenAI, no network)", and
that is the point of them: the signal detectors, the mode selector and the
guardrails are pure functions, so they must be testable on any machine in under
a second without keys, without a venv and without the network.

The chatbot -> pipeline rework added
    services/interviewer_decision.py:  from services.ai_providers import openai_client
at MODULE scope, and `services/ai_providers.py` does `from openai import OpenAI`
at import time. That one line makes the whole deterministic layer unimportable
without the SDK, so both suites died at the import, not at an assertion.

The obvious fix is a lazy import inside the one function that needs a client.
It is deliberately NOT taken here: those five interviewer files are maintained
directly on GitHub and the local copies must stay byte-identical to origin, so
this repo does not get to edit them. The fix therefore lives on the TEST side,
where it belongs anyway — a test's environment is the test's problem.

WHAT THIS DOES
--------------
Installs a minimal stand-in for each third-party module the import chain needs,
and ONLY when the real one is absent. If the SDK is installed (CI, the server,
anyone's venv) nothing here is used and the real code path is exercised.

The stubs satisfy imports and nothing else. Nothing in these suites calls a
provider — `assess_context_with_llm` is the only consumer and no test reaches
it — so a stub that is never invoked is honest: there is still no OpenAI and
still no network. Any test that DID try to make a call would get an
AttributeError rather than a silent pass, which is the failure mode we want.
"""
from __future__ import annotations

import sys
import types


def _stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__["__stubbed_by_tests__"] = True
    sys.modules[name] = mod
    return mod


def install() -> list[str]:
    """Stub what is missing. Returns the names stubbed, for the test banner."""
    stubbed: list[str] = []

    try:  # openai — services/ai_providers.py builds clients at import time
        import openai  # noqa: F401
    except ModuleNotFoundError:
        m = _stub("openai")

        class _Client:  # minimal: constructed at import, never called here
            def __init__(self, *args, **kwargs):
                self._args, self._kwargs = args, kwargs

            def __getattr__(self, item):
                raise AttributeError(
                    f"openai client is stubbed for tests; nothing should call .{item}. "
                    "If a test now needs a real provider it belongs in tools/eval_*, not here."
                )

        m.OpenAI = _Client
        m.APIError = type("APIError", (Exception,), {})
        stubbed.append("openai")

    try:  # supabase — services/supabase_client.py
        import supabase  # noqa: F401
    except ModuleNotFoundError:
        m = _stub("supabase")

        def _create_client(*args, **kwargs):
            raise AttributeError("supabase is stubbed for tests; no DB access in this suite.")

        m.create_client = _create_client
        m.Client = type("Client", (), {})
        stubbed.append("supabase")

    try:  # dotenv — usually present, cheap to cover
        import dotenv  # noqa: F401
    except ModuleNotFoundError:
        m = _stub("dotenv")
        m.load_dotenv = lambda *a, **k: False
        stubbed.append("dotenv")

    return stubbed
