"""Git working-tree baseline guard and committed file reader."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _unquote_git_path(raw: str) -> str:
    p = raw.strip()
    if p.startswith('"') and p.endswith('"') and len(p) >= 2:
        p = p[1:-1].encode("utf-8").decode("unicode_escape", errors="replace")
    return p.replace("\\", "/")


def is_inside_bot(rel_path: str) -> bool:
    """True only when rel_path resides strictly inside the bot/ directory."""
    norm = rel_path.replace("\\", "/").lstrip("./")
    return norm == "bot" or norm.startswith("bot/")


def parse_porcelain_entries(porcelain_output: str) -> Dict[str, str]:
    """Parse `git status --porcelain` into {repo_rel_path: xy_status} without stripping leading spaces."""
    entries: Dict[str, str] = {}
    for line in (porcelain_output or "").splitlines():
        if len(line) < 4:
            continue
        xy = line[:2]
        body = line[3:]
        if " -> " in body:
            old_p, new_p = body.split(" -> ", 1)
            for p in (_unquote_git_path(old_p), _unquote_git_path(new_p)):
                if p:
                    entries[p] = xy
        else:
            p = _unquote_git_path(body)
            if p:
                entries[p] = xy
    return entries


def parse_porcelain_paths(porcelain_output: str) -> List[str]:
    """Extract repo-relative paths from `git status --porcelain` stdout."""
    return list(parse_porcelain_entries(porcelain_output).keys())


def _file_digest(repo_root: Path, rel_path: str) -> Optional[str]:
    p = repo_root / rel_path
    if not p.is_file():
        return None
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def capture_outside_bot_baseline(
    repo_root: Path,
) -> Tuple[bool, Dict[str, Tuple[str, Optional[str]]], Optional[str]]:
    """Capture dirty paths outside bot/ at run start as {path: (xy_status, sha256)}."""
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return False, {}, f"Failed to execute git status: {type(exc).__name__}: {exc}"

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, {}, f"git status --porcelain failed (rc={proc.returncode}): {err}"

    entries = parse_porcelain_entries(proc.stdout)
    snapshot = {
        path: (xy, _file_digest(repo_root, path))
        for path, xy in entries.items()
        if not is_inside_bot(path)
    }
    return True, snapshot, None


def compare_outside_bot_snapshot(
    repo_root: Path,
    baseline: Dict[str, Tuple[str, Optional[str]]],
) -> Tuple[bool, List[str], Optional[str]]:
    """Abort only if a path outside bot/ is newly dirty or changed status/content vs baseline."""
    ok, current, err = capture_outside_bot_baseline(repo_root)
    if not ok:
        return False, [], err

    changed: List[str] = []
    for path, cur_state in sorted(current.items()):
        if path not in baseline:
            changed.append(f"{path} (new: {cur_state[0].strip() or '?'})")
        elif baseline[path] != cur_state:
            changed.append(f"{path} (changed: {baseline[path][0].strip()} -> {cur_state[0].strip()})")
    return len(changed) == 0, changed, None


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
