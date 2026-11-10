"""C4 Step 4: run pytest and offline eval_interviewer_behavior, diff against baseline."""
from __future__ import annotations

import difflib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from bot.git_guard import read_committed_file

_PASSED_RE = re.compile(r"(\d+)\s+passed\b")
_FAILED_RE = re.compile(r"(\d+)\s+failed\b")
_ERROR_RE = re.compile(r"(\d+)\s+errors?\b")

_OFFLINE_BASELINE_REL = "tools/eval_result_offline.txt"


def parse_pytest_counts(output: str) -> tuple[int, int]:
    """Extract (passed, failed_plus_errors) counts from pytest output."""
    text = output or ""
    passed_m = _PASSED_RE.findall(text)
    failed_m = _FAILED_RE.findall(text)
    error_m = _ERROR_RE.findall(text)
    passed = int(passed_m[-1]) if passed_m else 0
    failed = (int(failed_m[-1]) if failed_m else 0) + (int(error_m[-1]) if error_m else 0)
    return passed, failed


def _utf8_env() -> Dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_pytest_suite(repo_root: Path, timeout_sec: int = 300) -> Dict[str, Any]:
    """Run `python -m pytest -q` in repo_root without modifying repo files."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(repo_root),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )
    combined = "\n".join(x for x in (proc.stdout, proc.stderr) if x)
    passed, failed = parse_pytest_counts(combined)
    if proc.returncode != 0 and failed == 0:
        failed = 1
    return {
        "pytest_passed": passed,
        "pytest_failed": failed,
        "returncode": proc.returncode,
        "tail": "\n".join(combined.splitlines()[-20:]),
    }


def run_offline_eval_and_diff(repo_root: Path, timeout_sec: int = 120) -> List[str]:
    """Run offline eval without overwriting tools/eval_result_offline.txt, and diff vs committed."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from tools.eval_interviewer_behavior import run_offline; "
                "sys.exit(0 if run_offline() else 1)"
            ),
        ],
        cwd=str(repo_root),
        env=_utf8_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )
    actual_text = proc.stdout or ""
    baseline_text = read_committed_file(repo_root, _OFFLINE_BASELINE_REL)
    if baseline_text is None:
        return [f"Missing baseline file {_OFFLINE_BASELINE_REL}"]

    expected_lines = baseline_text.replace("\r\n", "\n").splitlines()
    actual_lines = actual_text.replace("\r\n", "\n").splitlines()
    diff_lines = list(
        difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile=f"a/{_OFFLINE_BASELINE_REL}",
            tofile="offline_eval_current",
            lineterm="",
        )
    )
    if proc.returncode != 0 and not diff_lines:
        diff_lines = [f"offline eval exited non-zero (rc={proc.returncode})"]
    return diff_lines


def run_all_checks(repo_root: Path, timeout_sec: int = 420) -> Dict[str, Any]:
    """Run both pytest and offline eval diff; return C5 `suite` block plus notes."""
    pt = run_pytest_suite(repo_root, timeout_sec=min(300, timeout_sec))
    drift = run_offline_eval_and_diff(repo_root, timeout_sec=min(120, timeout_sec))
    notes: List[str] = []
    if pt["pytest_passed"] < 172:
        notes.append(f"pytest passed count ({pt['pytest_passed']}) is below expected 172+")
    if pt["pytest_failed"] > 0:
        notes.append(f"pytest reported {pt['pytest_failed']} failure(s)/error(s) (rc={pt['returncode']})")
    if drift:
        notes.append(f"offline eval drifted from {_OFFLINE_BASELINE_REL} ({len(drift)} diff lines)")
    return {
        "suite": {
            "pytest_passed": pt["pytest_passed"],
            "pytest_failed": pt["pytest_failed"],
            "offline_eval_drift": drift,
        },
        "notes": notes,
    }
