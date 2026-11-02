"""Session-wide test setup for the backend suite.

pytest imports this before any test module, so everything here is guaranteed to be in
place for every file — including the ones that do not set themselves up.

WHY (2026-11-02)
----------------
Setup was per-file and inconsistent. `tests/test_daily_blog.py` inserts the repo root on
sys.path and calls `_sdk_stubs.install()` at its top; most others rely on pytest having
done the right thing. That worked only as long as the files happened to be imported in a
helpful order, and it left eight suites failing at collection with

    ImportError: cannot import name 'openai_client' from 'services.ai_providers'
                 (unknown location)

"(unknown location)" means the name resolved to a module object with no __file__ — i.e.
the import system did not find the real services/ai_providers.py. Doing both steps here,
once, before any test module is imported, removes the ordering dependency entirely.

If a suite still fails after this, the cause is NOT import ordering and the diagnostic in
the repo notes applies.
"""
from __future__ import annotations

import os
import sys

# 1. The repo root, so `services.*`, `routes.*` and `prompts.*` resolve to THIS checkout
#    no matter which directory pytest was started from or what rootdir it picked.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 2. Stand-ins for third-party SDKs that services/ai_providers.py imports at module
#    scope. Installed ONLY where the real package is missing (see tests/_sdk_stubs.py),
#    so a machine with the real SDKs still exercises the real path.
try:
    from tests import _sdk_stubs

    _sdk_stubs.install()
except Exception as _e:  # noqa: BLE001 - never let setup hide the real test failures
    print(f"[conftest] SDK stubs not installed: {type(_e).__name__}: {_e}")

# 3. A test must never reach a real provider, a real database or real money. These are
#    set before any module is imported, because several read their config at import time.
os.environ.setdefault("ADAPTIVE_INTERVIEWER", "true")
for _k, _v in {
    "OPENAI_API_KEY": "sk-test",
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test",
}.items():
    os.environ.setdefault(_k, _v)
