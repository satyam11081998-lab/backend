"""Adversarial checks on the arena's variant layer.

An eval harness that silently mismeasures is worse than no harness, so these
tests attack the measurement itself, not just the code:

  drift        a frozen baseline must never change -- its SHA is pinned
  isolation    a frozen baseline must not import the live module
  distinctness the two baselines must actually differ (else "winners" are noise)
  conformance  every variant must return a well-formed messages array
  no-fallback  an unknown variant id must raise, never silently substitute
  provenance   the "before" baseline must really be the before
"""
from __future__ import annotations

import ast
import io

import pytest

from prompts.variants import registry

# Pinned 2026-10-14. If a frozen variant is edited these fail: that is the point.
# Re-pin ONLY when deliberately re-snapshotting, and treat every vote recorded
# against the old hash as belonging to a different baseline.
PINNED = {
    "baseline-main": "63a57ae0d94b0884d3d8fece954e7409d70aa67b049af0cea4c080852d14c148",
    "patched-2026-10-14": "8a34b4044cf2c09867b33566e8d9733dd8a43246079dc54c38b3900fb5761904",
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


def test_frozen_variants_are_standalone():
    """A baseline that imported the live module would move when you edit live.

    Checks real import statements via the AST, not raw text -- prose in a docstring
    that merely mentions a module name is not an import."""
    for vid in PINNED:
        mod = registry.get_variant(vid)["module"]
        tree = ast.parse(io.open(mod.__file__, encoding="utf-8").read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert imported <= {"typing", "__future__"}, f"{vid} imports {imported}"


def test_the_two_baselines_actually_differ():
    """If both arms render the same prompt, any preference is pure noise."""
    a = registry.get_builder("baseline-main")(**FIXTURE)
    b = registry.get_builder("patched-2026-10-14")(**FIXTURE)
    assert a[0]["content"] != b[0]["content"]


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


def test_provenance_of_the_before_and_after_baselines():
    before = registry.get_builder("baseline-main")(**FIXTURE)[0]["content"]
    after = registry.get_builder("patched-2026-10-14")(**FIXTURE)[0]["content"]
    # before: the human-consultant identity and no AI disclosure
    assert "warm, human interviewer" in before
    assert "IDENTITY LOCK" in before
    # after: one consistent AI identity, and the banned-phrase list restored
    assert "warm, human interviewer" not in after
    assert "MECE's AI interviewer" in after
    assert "YOU OWN THE CASE FACTS" in after
    assert "isn't specified" in " ".join(after.split())


def test_catalogue_is_json_safe_and_flags_the_unreproducible_arm():
    rows = registry.describe_variants()
    assert any(r["id"] == "live" and r["frozen"] is False for r in rows)
    assert all(not callable(v) for r in rows for v in r.values())
    for r in rows:
        assert r["sha256"] is None or len(r["sha256"]) == 12


def test_live_matches_names_the_snapshot_the_deployment_equals():
    """The question that otherwise silently wrecks a session: am I comparing against
    what is actually serving users? On a pre-patch deployment this is
    'baseline-main'; after the fixes land it becomes 'patched-2026-10-14'; mid-edit
    it is None. Any of the three is correct -- it must simply not lie."""
    match = registry.live_matches()
    assert match in (None, "baseline-main", "patched-2026-10-14")
    assert len(registry.live_fingerprint()) == 64
    # Matching is on composed prompt TEXT, because a frozen snapshot carries a
    # banner its original does not and so never matches byte-for-byte.
    from prompts import interview_prompts_v2 as live_mod
    if match is not None:
        mod = registry.get_variant(match)["module"]
        assert mod.CASE_INTERVIEWER_V2 == live_mod.CASE_INTERVIEWER_V2
        assert mod.GUESSTIMATE_INTERVIEWER_V2 == live_mod.GUESSTIMATE_INTERVIEWER_V2
    else:
        for v in PINNED:
            assert registry.get_variant(v)["module"].CASE_INTERVIEWER_V2 != live_mod.CASE_INTERVIEWER_V2
