"""Test doubles and isolated git repo fixtures for bot tests."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from bot.state import default_state, save_state


class _FakeQuery:
    def __init__(self, rows: List[Dict[str, Any]], raise_on_execute: Optional[Exception] = None):
        self._rows = list(rows)
        self._raise = raise_on_execute
        self._filters: List[Any] = []
        self._order_col: Optional[str] = None
        self._order_desc: bool = False
        self._limit: Optional[int] = None
        self._range: Optional[tuple[int, int]] = None

    def select(self, *_args, **_kwargs) -> "_FakeQuery":
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters.append(lambda r: r.get(col) == val)
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        self._filters.append(lambda r: str(r.get(col) or "") >= str(val))
        return self

    def lte(self, col: str, val: Any) -> "_FakeQuery":
        self._filters.append(lambda r: str(r.get(col) or "") <= str(val))
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        allowed = set(vals)
        self._filters.append(lambda r: r.get(col) in allowed)
        return self

    def order(self, col: str, desc: bool = False) -> "_FakeQuery":
        self._order_col = col
        self._order_desc = desc
        return self

    def limit(self, n: int) -> "_FakeQuery":
        self._limit = n
        return self

    def range(self, start: int, end: int) -> "_FakeQuery":
        self._range = (start, end)
        return self

    def execute(self) -> SimpleNamespace:
        if self._raise is not None:
            raise self._raise
        data = [dict(r) for r in self._rows if all(fn(r) for fn in self._filters)]
        if self._order_col:
            data.sort(key=lambda r: str(r.get(self._order_col) or ""), reverse=self._order_desc)
        if self._range is not None:
            s, e = self._range
            data = data[s : e + 1]
        elif self._limit is not None:
            data = data[: self._limit]
        return SimpleNamespace(data=data)


class FakeSupabase:
    """Minimal in-memory Supabase table double for bot tests."""

    def __init__(self, tables: Optional[Dict[str, List[Dict[str, Any]]]] = None, fail_with: Optional[Exception] = None):
        self.tables: Dict[str, List[Dict[str, Any]]] = {
            "attempts": [],
            "attempt_messages": [],
            "case_attempts": [],
            "submissions": [],
            "users": [],
            "cases": [],
            "colleges": [],
        }
        if tables:
            self.tables.update(tables)
        self.fail_with = fail_with
        self.queries: List[str] = []

    def table(self, name: str) -> _FakeQuery:
        self.queries.append(name)
        return _FakeQuery(self.tables.get(name, []), raise_on_execute=self.fail_with)


class FakeGeminiClient:
    """OpenAI-compatible Gemini client double recording every call."""

    def __init__(self, reply_json: str = '{"items": []}', raise_exc: Optional[Exception] = None):
        self.reply_json = reply_json
        self.raise_exc = raise_exc
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> Any:
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40, total_tokens=160)
        msg = SimpleNamespace(content=self.reply_json)
        return SimpleNamespace(
            id="gemini-test-id-1",
            usage=usage,
            choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        )


def init_temp_git_repo(tmp_path: Path, *, copy_offline_baseline: Optional[Path] = None) -> Path:
    """Create a real, committed temporary git repo with services/ai_scorer.py and bot/."""
    repo = tmp_path / "repo"
    (repo / "services").mkdir(parents=True)
    (repo / "tools").mkdir(parents=True)
    (repo / "bot" / "reports").mkdir(parents=True)
    (repo / "bot" / "logs").mkdir(parents=True)

    (repo / "services" / "ai_scorer.py").write_text("# product scorer\nSCORE = 1\n", encoding="utf-8")
    if copy_offline_baseline and copy_offline_baseline.is_file():
        shutil.copy2(copy_offline_baseline, repo / "tools" / "eval_result_offline.txt")
    else:
        (repo / "tools" / "eval_result_offline.txt").write_text("OFFLINE: ok\n", encoding="utf-8")

    save_state(repo / "bot" / "state.json", default_state())

    subprocess.run(["git", "init"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Bot Test"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "bot@example.com"], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(repo), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=str(repo), check=True, capture_output=True)
    return repo


def stub_suite_ok(_repo_root: Path) -> Dict[str, Any]:
    return {
        "suite": {"pytest_passed": 517, "pytest_failed": 0, "offline_eval_drift": []},
        "notes": [],
    }
