"""Git working-tree guard and committed baseline reader."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


def _unquote_git_path(raw: str) -> str:
    p = raw.strip()
    if p.startswith('"') and p.endswith('"') and len(p) >= 2:
        p = p[1:-1].encode("utf-8").decode("unicode_escape", errors="replace")
    return p.replace("\\", "/")


def parse_porcelain_paths(porcelain_output: str) -> List[str]:
    """Extract repo-relative paths from `git status --porcelain` stdout."""
    paths: List[str] = []
    for line in (porcelain_output or "").splitlines():
        if len(line) < 4:
            continue
        body = line[3:]
        if " -> " in body:
            old_p, new_p = body.split(" -> ", 1)
            paths.append(_unquote_git_path(old_p))
            paths.append(_unquote_git_path(new_p))
        else:
            paths.append(_unquote_git_path(body))
    return [p for p in paths if p]


def is_inside_bot(rel_path: str) -> bool:
    """True only when rel_path resides strictly inside the bot/ directory."""
    norm = rel_path.replace("\\", "/").lstrip("./")
    return norm == "bot" or norm.startswith("bot/")


def check_git_clean_outside_bot(repo_root: Path) -> Tuple[bool, List[str], Optional[str]]:
    """Run `git status --porcelain` and return (ok, changed_outside_bot, error_message)."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return False, [], f"Failed to execute git status: {type(exc).__name__}: {exc}"

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, [], f"git status --porcelain failed (rc={proc.returncode}): {err}"

    all_paths = parse_porcelain_paths(proc.stdout)
    outside = [p for p in all_paths if not is_inside_bot(p)]
    return len(outside) == 0, outside, None


def read_committed_file(repo_root: Path, rel_path: str) -> Optional[str]:
    """Read `rel_path` from HEAD via git show, falling back to working copy."""
    git_path = rel_path.replace("\\", "/")
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{git_path}"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        if proc.returncode == 0:
            return proc.stdout
    except Exception:  # noqa: BLE001
        pass

    disk_path = repo_root / rel_path
    if disk_path.is_file():
        return disk_path.read_text(encoding="utf-8", errors="replace")
    return None
