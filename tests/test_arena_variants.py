"""Adversarial checks on the arena's variant layer.

An eval harness that silently mismeasures is worse than no harness, so these
tests attack the measurement itself, not just the code:

  provenance   each frozen arm IS the commit it names -- regenerated from git by
               tools/arena_snapshot.py and compared byte for byte
  completeness each arm carries the WHOLE text-turn prompt layer (interviewer
               prompt, packet rules, context assessor); v1 carried one of three
  drift        a frozen file must never change -- its SHA is pinned
  isolation    a frozen file imports only the standard library
  distinctness the two arms must actually differ (else "winners" are noise)
  conformance  every variant returns a well-formed messages array
  no-fallback  an unknown variant id raises, never silently substitutes
"""
from __future__ import annotations

import ast
import io
import subprocess

import pytest

from prompts.variants import registry

# Pinned 2026-10-16 (re-snapshot: v1's files of 2026-10-14 lacked the packet rules and
# the assessor). If a frozen variant is edited these fail: that is the point. Re-pin
# ONLY when deliberately re-snapshotting; votes recorded against an old hash stay in
# their own bucket in /admin/arena/results.
PINNED = {
    "baseline-main": "de82906fa94823b5f96e354db3f68ca4e698ea249d40601b5a9837807bc1c948",
    "patched-2026-10-14": "501454afd2af95ffae51c62ffe889bb5f72591acf26a2ab3f8c4510a90b6eef7",
}

FIXTURE = dict(
    case_content="Spice Express sales fell 12% YoY.\n- 240 outlets\n- AOV Rs 420",
    case_type="profitability",
    transcript=[
        {"role": "system", "kind": "system_note", "content": "Case ready: Spice Express."},
        {"role": "assistant", "content": "Where would you start?"},
        {"role": "user", "content": "I'd split revenue and cost."},
    ],
    new_user_message="What's the delivery commission?",
    teaching_policy="coached",
    signals_block="SESSION SIGNALS: intent=answering",
    clarifications_exhausted=False,
)


def test_frozen_variants_have_not_drifted():
    for vid, expected in PINNED.items():
        assert registry.fingerprint(vid) == expected, (
            f"frozen variant {vid} changed; every vote recorded against it is now "
            f"against a different baseline. Re-pin only on a deliberate re-snapshot."
        )


def _commit_available(sha: str) -> bool:
    try:
        subprocess.run(["git", "--no-optional-locks", "cat-file", "-e", f"{sha}^{{commit}}"],
                       check=True, capture_output=True)
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("vid", sorted(PINNED))
def test_each_frozen_arm_is_exactly_the_commit_it_names(vid):
    """The claim "arm B is the branch" checked against git itself, not a description."""
    from tools.arena_snapshot import _HEADER_RE, check
    path = registry.get_variant(vid)["module"].__file__
    head = io.open(path, encoding="utf-8").read(2000)
    m = _HEADER_RE["sha"].search(head)
    assert m, f"{vid} is not a generated snapshot (no Source commit header)"
    if not _commit_available(m.group(1)):
        pytest.skip(f"commit {m.group(1)[:7]} not in this clone (shallow checkout)")
    ok, msg = check(path)
    assert ok, msg


@pytest.mark.parametrize("vid", sorted(PINNED))
def test_each_frozen_arm_carries_the_whole_prompt_layer(vid):
    mod = registry.get_variant(vid)["module"]
    assert callable(mod.build_adaptive_interviewer_messages)
    assert isinstance(mod.PACKET_RULES, str) and "PACKET" in mod.PACKET_RULES
    assert isinstance(mod._ASSESSOR_PROMPT, str) and "JSON" in mod._ASSESSOR_PROMPT
    assert callable(mod.assess_context_with_llm)
    ov = registry.get_overrides(vid, client_factory=lambda: None)
    assert ov["packet_rules"] is mod.PACKET_RULES
    assert ov["assessor"] is mod.assess_context_with_llm


def test_live_has_no_overrides_because_it_is_the_deployed_layer():
    assert registry.get_overrides("live") is None


def test_the_frozen_assessor_uses_the_client_it_is_given():
    """A snapshot imports nothing from the repo, so its assessor must use the client
    the arena hands it -- that is also what lets tests fake it."""
    sent = []

    class _C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.append(kw)
                    msg = type("M", (), {"content": '{"intent": "working"}'})()
                    return type("R", (), {"choices": [type("Ch", (), {"message": msg})()]})()

    for vid in sorted(PINNED):
        ov = registry.get_overrides(vid, client_factory=lambda: _C())
        out = ov["assessor"]([{"role": "user", "content": "hi"}], "next", "case text")
        assert out == {"intent": "working"}
        assert sent[-1]["messages"][0]["content"] == registry.get_variant(vid)["module"]._ASSESSOR_PROMPT


def test_frozen_variants_are_standalone():
    """A baseline that imported a repo module would move when that module is edited.
    The standard library cannot drift with the repo, so json is allowed."""
    for vid in PINNED:
        mod = registry.get_variant(vid)["module"]
        tree = ast.parse(io.open(mod.__file__, encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= {"typing", "__future__", "json"}, f"{vid} imports {imported}"


def test_the_two_arms_actually_differ_in_all_three_pieces():
    """If both arms render the same layer, any preference is pure noise."""
    a = registry.get_variant("baseline-main")["module"]
    b = registry.get_variant("patched-2026-10-14")["module"]
    assert a.build_adaptive_interviewer_messages(**FIXTURE)[0]["content"] != \
        b.build_adaptive_interviewer_messages(**FIXTURE)[0]["content"]
    assert a.PACKET_RULES != b.PACKET_RULES
    assert a._ASSESSOR_PROMPT != b._ASSESSOR_PROMPT


@pytest.mark.parametrize("vid", registry.variant_ids())
def test_variant_conformance(vid):
    messages = registry.get_builder(vid)(**FIXTURE)
    assert isinstance(messages, list) and len(messages) >= 3
    assert all(set(m) == {"role", "content"} for m in messages)
    assert all(m["role"] in ("system", "user", "assistant") for m in messages)
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)
    assert messages[0]["role"] == "system"
    # The candidate's new turn is always last, so nothing can be appended after it.
    assert messages[-1] == {"role": "user", "content": FIXTURE["new_user_message"]}
    # The case must reach the model readably, in every variant.
    assert any("240 outlets" in m["content"] for m in messages)


def test_unknown_variant_raises_instead_of_falling_back():
    with pytest.raises(KeyError):
        registry.get_variant("no-such-variant")
    with pytest.raises(KeyError):
        registry.get_builder("")
    with pytest.raises(KeyError):
        registry.get_overrides("no-such-variant")


def test_provenance_of_main_and_the_branch():
    main = registry.get_variant("baseline-main")["module"]
    branch = registry.get_variant("patched-2026-10-14")["module"]
    before = main.build_adaptive_interviewer_messages(**FIXTURE)[0]["content"]
    after = branch.build_adaptive_interviewer_messages(**FIXTURE)[0]["content"]
    # main: the human-consultant identity and no AI disclosure
    assert "warm, human interviewer" in before and "IDENTITY LOCK" in before
    # branch: one consistent AI identity, and the banned-phrase list restored
    assert "warm, human interviewer" not in after
    assert "MECE's AI interviewer" in after
    assert "YOU OWN THE CASE FACTS" in after
    assert "isn't specified" in " ".join(after.split())
    # the two pieces v1 left behind
    assert "untrusted evidence" in branch.PACKET_RULES and "untrusted" not in main.PACKET_RULES
    assert "untrusted data" in branch._ASSESSOR_PROMPT and "untrusted" not in main._ASSESSOR_PROMPT


def test_catalogue_is_json_safe_and_flags_the_unreproducible_arm():
    rows = registry.describe_variants()
    assert any(r["id"] == "live" and r["frozen"] is False for r in rows)
    assert all(not callable(v) for r in rows for v in r.values())
    for r in rows:
        assert r["sha256"] is None or len(r["sha256"]) == 12


def test_live_matches_names_the_snapshot_the_deployment_equals():
    """Am I comparing against what is actually serving users? All three pieces are
    compared. On a pre-patch deployment this is 'baseline-main'; mid-edit it is None.
    Either is correct -- it must simply not lie."""
    from prompts import interview_prompts_v2 as live_v2
    from services import interviewer_decision as live_dec
    from services import interviewer_mode as live_mode
    match = registry.live_matches()
    assert match in (None, "baseline-main", "patched-2026-10-14")
    if match is not None:
        mod = registry.get_variant(match)["module"]
        assert mod.CASE_INTERVIEWER_V2 == live_v2.CASE_INTERVIEWER_V2
        assert mod.GUESSTIMATE_INTERVIEWER_V2 == live_v2.GUESSTIMATE_INTERVIEWER_V2
        assert mod.PACKET_RULES == live_mode._PACKET_RULES
        assert mod._ASSESSOR_PROMPT == live_dec._ASSESSOR_PROMPT
