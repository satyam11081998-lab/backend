"""Freeze one git ref's interviewer prompt layer as an arena variant.

    python -m tools.arena_snapshot --ref test/ai-behaviour-patch --id branch-oct20 \
        --label "AI-behaviour branch (Oct 20)"
    python -m tools.arena_snapshot --check prompts/variants/patched_v12_2026_10_14.py

WHY: the arena's two arms are frozen snapshots, not git branches (two branches cannot
be imported into one process, and a vote against "the branch" means nothing once the
branch moves). Hand-copying a snapshot is how v1 ended up with only ONE of the three
prompt-layer pieces a text turn uses. This tool copies all three, verbatim, from git:

  1. prompts/interview_prompts_v2.py                  -- the interviewer's messages
  2. services/interviewer_mode.py   _PACKET_RULES     -- rules that brief the model
                                                         alongside the control packet
  3. services/interviewer_decision.py _ASSESSOR_PROMPT + assess_context_with_llm
                                                       -- the context assessor, which
                                                         feeds the response-function
                                                         decision itself

Anything else a branch changes (signals, decision logic, scoring, voice) is CODE and
cannot be a variant; the tool says so when it sees such a diff, so you are never told
"the branch arm" when it is only part of the branch.

The output is a pure function of (variant id, label, commit): --check regenerates a
file in memory and compares it byte for byte, which is how the test suite proves each
frozen arm really is the commit it names.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

V2 = "prompts/interview_prompts_v2.py"
MODE = "services/interviewer_mode.py"
DECISION = "services/interviewer_decision.py"
# Files whose diffs DO change a text turn but are code, not prompt text. Listed so the
# tool can warn instead of silently pretending a snapshot covers them.
CODE_THAT_SHAPES_A_TURN = [
    "services/interview_engine.py", "services/session_signals.py",
    "services/interviewer_decision.py", "services/interviewer_mode.py",
    "services/learning_model.py", "prompts/voice_renderer.py",
]
OUT_DIR = "prompts/variants"
# What the frozen assessor may reference. Anything else means the assessor grew a
# dependency the snapshot cannot carry -- fail loudly rather than freeze a broken arm.
ASSESSOR_ALLOWED_GLOBALS = {"openai_client", "_ASSESSOR_PROMPT", "json", "Dict", "Any", "List", "Optional"}


def _git(*args: str) -> str:
    return subprocess.run(["git", "--no-optional-locks", *args], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def _show(sha: str, path: str) -> str:
    return _git("show", f"{sha}:{path}").replace("\r\n", "\n")


def _assignment(src: str, name: str) -> Tuple[ast.Assign, str]:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return node, ast.get_source_segment(src, node)
    raise SystemExit(f"{name} not found -- the file's shape changed; extend the tool")


def _function(src: str, name: str) -> Tuple[ast.FunctionDef, str]:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, ast.get_source_segment(src, node)
    raise SystemExit(f"def {name} not found -- the file's shape changed; extend the tool")


def _free_globals(fn: ast.FunctionDef) -> set:
    local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    if fn.args.vararg:
        local.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        local.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local.add(node.id)
        elif isinstance(node, ast.arg):
            local.add(node.arg)
        elif isinstance(node, (ast.ExceptHandler,)) and node.name:
            local.add(node.name)
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - local - set(dir(builtins))


def _check_v2_standalone(src: str) -> None:
    for node in ast.walk(ast.parse(src)):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        bad = [m for m in mods if m not in ("typing", "__future__")]
        if bad:
            raise SystemExit(f"{V2} imports {bad}: a snapshot must not depend on repo modules")


def build(variant_id: str, label: str, sha: str, ref_name: str) -> str:
    subject = _git("log", "-1", "--format=%s", sha).strip()
    date = _git("log", "-1", "--format=%cs", sha).strip()
    v2 = _show(sha, V2)
    _check_v2_standalone(v2)
    mode_src = _show(sha, MODE)
    dec_src = _show(sha, DECISION)

    pr_node, pr_src = _assignment(mode_src, "_PACKET_RULES")
    if not isinstance(ast.literal_eval(pr_node.value), str):  # a plain string literal, nothing computed
        raise SystemExit("_PACKET_RULES is no longer a plain string literal; extend the tool")
    if not pr_src.startswith("_PACKET_RULES"):
        raise SystemExit("unexpected _PACKET_RULES assignment shape; extend the tool")
    packet_rules_src = "PACKET_RULES" + pr_src[len("_PACKET_RULES"):]

    _, assessor_prompt_src = _assignment(dec_src, "_ASSESSOR_PROMPT")
    fn_node, assessor_fn_src = _function(dec_src, "assess_context_with_llm")
    extra = _free_globals(fn_node) - ASSESSOR_ALLOWED_GLOBALS
    if extra:
        raise SystemExit(f"assess_context_with_llm @ {sha[:7]} references {sorted(extra)}; "
                         "the snapshot cannot carry that -- extend the tool")

    banner = f'''"""FROZEN SNAPSHOT -- GENERATED, DO NOT EDIT BY HAND.

Arena variant:  {variant_id}
Label:          {label}
Source ref:     {ref_name}
Source commit:  {sha}
Commit:         {date} {subject}
Generated by:   tools/arena_snapshot.py  (python -m tools.arena_snapshot --check <this file>)

The interviewer's whole TEXT-TURN prompt layer at that commit, copied verbatim:
  1. {V2}
  2. {MODE} :: _PACKET_RULES          (exported here as PACKET_RULES)
  3. {DECISION} :: _ASSESSOR_PROMPT + assess_context_with_llm

Regenerate, never edit: an edit changes the SHA pinned in tests/test_arena_variants.py,
and every vote recorded against this file belongs to the text it had then.
Standalone: imports only the standard library. The assessor's `openai_client` is
injected by prompts/variants/registry.py at call time.
"""
'''
    parts = [
        banner,
        f"# {'=' * 76}\n# 1. {V2} @ {sha[:7]}\n# {'=' * 76}\n",
        v2.rstrip("\n") + "\n",
        f"\n\n# {'=' * 76}\n# 2. {MODE} :: _PACKET_RULES @ {sha[:7]}\n# {'=' * 76}\n",
        packet_rules_src.rstrip("\n") + "\n",
        f"\n\n# {'=' * 76}\n# 3. {DECISION} :: context assessor @ {sha[:7]}\n# {'=' * 76}\n",
        "import json  # noqa: E402  (standard library)\n",
        "from typing import Any, Dict, List, Optional  # noqa: E402,F401\n\n",
        "openai_client = None  # injected by prompts/variants/registry.py; see the banner\n\n",
        assessor_prompt_src.rstrip("\n") + "\n\n\n",
        assessor_fn_src.rstrip("\n") + "\n",
    ]
    return "".join(parts)


_HEADER_RE = {
    "id": re.compile(r"^Arena variant:\s+(.+)$", re.M),
    "label": re.compile(r"^Label:\s+(.+)$", re.M),
    "ref": re.compile(r"^Source ref:\s+(.+)$", re.M),
    "sha": re.compile(r"^Source commit:\s+([0-9a-f]{40})$", re.M),
}


def check(path: str) -> Tuple[bool, str]:
    if not os.path.exists(path):
        return False, f"{path}: no such file"
    have = open(path, encoding="utf-8", newline="").read()
    meta = {}
    for k, rx in _HEADER_RE.items():
        m = rx.search(have)
        if not m:
            return False, f"{path}: no '{k}' in the header -- not a generated snapshot"
        meta[k] = m.group(1).strip()
    try:
        want = build(meta["id"], meta["label"], meta["sha"], meta["ref"])
    except subprocess.CalledProcessError:
        return False, f"commit {meta['sha'][:7]} is not in this clone (shallow?)"
    if want == have:
        return True, f"{path}: identical to {meta['ref']} @ {meta['sha'][:7]}"
    return False, f"{path}: DIFFERS from {meta['ref']} @ {meta['sha'][:7]} -- edited by hand, or the tool changed"


def _module_name(variant_id: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", variant_id).strip("_").lower()


def _residual(path: str, src: str) -> str:
    """The file with the pieces a snapshot CARRIES cut out, so what is left is the
    code it cannot carry. Two residuals that differ mean a real code change."""
    tree = ast.parse(src)
    carried = {MODE: {"_PACKET_RULES"}, DECISION: {"_ASSESSOR_PROMPT", "assess_context_with_llm"}}.get(path, set())
    keep = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = next((getattr(t, "id", None) for t in node.targets), None)
        if name in carried:
            continue
        keep.append(ast.dump(node, include_attributes=False))
    return "\n".join(keep)


def _code_diff_warning(sha: str, base: str = "main") -> List[str]:
    """Files that shape a text turn as CODE and differ between `base` and this commit,
    ignoring the pieces the snapshot carries."""
    try:
        base_sha = _git("merge-base", base, sha).strip()
    except subprocess.CalledProcessError:
        return []
    out = []
    for f in CODE_THAT_SHAPES_A_TURN:
        try:
            a, b = _show(base_sha, f), _show(sha, f)
        except subprocess.CalledProcessError:
            continue
        if _residual(f, a) != _residual(f, b):
            out.append(f)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", help="git ref to freeze (branch, tag or sha)")
    ap.add_argument("--id", help="arena variant id, e.g. branch-oct20")
    ap.add_argument("--label", help="label shown in the arena")
    ap.add_argument("--module", help="output module name (default: from --id)")
    ap.add_argument("--check", nargs="+", metavar="FILE", help="verify generated snapshots")
    a = ap.parse_args(argv)

    if a.check:
        ok_all = True
        for p in a.check:
            ok, msg = check(p)
            print(("OK    " if ok else "FAIL  ") + msg)
            ok_all &= ok
        return 0 if ok_all else 1

    if not (a.ref and a.id and a.label):
        ap.error("--ref, --id and --label are required (or use --check)")
    sha = _git("rev-parse", "--verify", f"{a.ref}^{{commit}}").strip()
    text = build(a.id, a.label, sha, a.ref)
    module = a.module or _module_name(a.id)
    path = os.path.join(OUT_DIR, f"{module}.py")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists. A frozen snapshot is never overwritten -- pick a new --id.")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {path}  ({a.ref} @ {sha[:7]})")
    print(f"\n1. prompts/variants/registry.py -- import it and add a spec row:")
    print(f"     from prompts.variants import {module} as _{module}")
    print(f'     {{"id": "{a.id}", "label": "{a.label}", "description": "...", '
          f'"origin": "git {a.ref} @ {sha[:7]}", "frozen": True, "module": _{module}}},')
    print(f"\n2. tests/test_arena_variants.py -- pin it:\n     \"{a.id}\": \"{digest}\",")
    warn = _code_diff_warning(sha)
    if warn:
        base = _git("merge-base", "main", sha).strip()
        print(f"\nNOTE: since it left main (at {base[:7]}) this commit also changes CODE that "
              "shapes a text turn, which a snapshot cannot carry -- the arm runs main's version:")
        for f in warn:
            print(f"  {f}, changed by:")
            for line in _git("log", "--oneline", f"{base}..{sha}", "--", f).splitlines():
                print(f"      {line}")
        print("If those commits are arena plumbing main already has, ignore this. If they "
              "change how the interviewer decides, the arm is not the whole branch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
