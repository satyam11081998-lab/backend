"""Adversarial checks on the arena's HTTP surface.

Attacks the things that would quietly ruin the tool:
  blinding    a variant id leaking into a turn response defeats the whole method
  auth        a non-admin (or a guest) reaching an endpoint that spends money
  ownership   one admin voting into another's blinded session
  spend       validation and the budget cap must run BEFORE any model call
  honesty     a vote that was not saved must say so, loudly
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ["ADAPTIVE_INTERVIEWER"] = "true"

CASE = {"id": "case-1", "title": "Spice Express", "type": "profitability",
        "difficulty": "medium", "content": "Sales fell 12% YoY.\n- 240 outlets", "market": "IN"}


class _Exec:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return type("R", (), {"data": self._data})()


class _Query(_Exec):
    calls: list = []          # every builder call, so a test can see the SQL it built
    fail_on_columns = None    # a column name whose presence in select() raises

    def select(self, *a, **k):
        _Query.calls.append(("select", a, k))
        if _Query.fail_on_columns and a and _Query.fail_on_columns in a[0]:
            raise RuntimeError(f'column cases.{_Query.fail_on_columns} does not exist')
        return self

    def eq(self, *a, **k):
        _Query.calls.append(("eq", a))
        return self

    def ilike(self, *a, **k):
        _Query.calls.append(("ilike", a))
        return self

    def range(self, *a, **k):
        _Query.calls.append(("range", a))
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        _Query.calls.append(("order", a, k))
        return self

    def single(self):
        return _Exec(self._data[0] if isinstance(self._data, list) and self._data else self._data)

    def insert(self, row):
        _FakeSB.inserted.append(row)
        if _FakeSB.votes_table_missing:
            raise RuntimeError('relation "arena_votes" does not exist')
        return _Exec([row])


class _FakeSB:
    inserted: list = []
    votes_table_missing = False
    vote_rows: list = []

    def table(self, name):
        if name == "cases":
            return _Query([CASE])
        if name == "users":
            return _Query([{"is_admin": True}])
        if name == "arena_votes":
            if _FakeSB.votes_table_missing:
                raise RuntimeError('relation "arena_votes" does not exist')
            return _Query(list(_FakeSB.vote_rows))
        return _Query([])


TURN_OUT = {
    "variant_id": "SHOULD-BE-STRIPPED",
    "reply": "Which unit are you sizing in?",
    "raw_reply": "Which unit are you sizing in?",
    "silent": False, "error": None, "prompt_consulted": True, "enforced": False,
    # Mirrors services.arena.run_turn exactly: a stub narrower than the real shape
    # would let a contract test pass while the page renders blanks.
    "control": {"mode": "HINT", "function": "MICRO_HINT", "render": "model",
                "reason": "help_requested", "intervention": "micro_hint",
                "materiality": "material", "skill": None, "error_kind": None,
                "hint": "1", "fallback": False},
    "violations": [], "metrics": {"chars": 30, "words": 6, "sentences": 1, "questions": 1},
    "latency_ms": 700, "model": "gpt-4o-mini", "provider": "openai", "next_state": {"hint_level": 1},
    "voice": None,
}


@pytest.fixture()
def client(monkeypatch):
    from routes import arena as ar

    _FakeSB.inserted = []
    _FakeSB.votes_table_missing = False
    _FakeSB.vote_rows = []
    _Query.calls = []
    _Query.fail_on_columns = None
    ar._SESSIONS.clear()
    monkeypatch.setattr(ar, "get_supabase_client", lambda: _FakeSB())
    monkeypatch.setattr(ar, "_require_admin", lambda authorization: "admin-1")
    monkeypatch.setattr(ar, "check_rate_limit", lambda *a, **k: None)
    spent = {"budget_checked": 0}
    monkeypatch.setattr(ar, "assert_daily_budget", lambda: spent.__setitem__("budget_checked", spent["budget_checked"] + 1))
    calls = []

    def _fake_run_turn(**kw):
        calls.append(kw)
        return dict(TURN_OUT, variant_id=kw["variant_id"])

    monkeypatch.setattr(ar, "run_turn", _fake_run_turn)
    app = FastAPI()
    app.include_router(ar.router)
    c = TestClient(app)
    c.mod, c.calls, c.spent = ar, calls, spent
    return c


def _start(client, a="baseline-main", b="patched-2026-10-14"):
    r = client.post("/admin/arena/session", json={
        "case_id": "case-1", "variant_a": a, "variant_b": b,
        "model_mode": "openai-mini", "teaching_policy": "coached"})
    assert r.status_code == 200, r.text
    return r.json()


def test_config_reports_variants_seeds_and_whether_votes_are_saved(client):
    r = client.get("/admin/arena/config")
    assert r.status_code == 200
    body = r.json()
    assert {v["id"] for v in body["variants"]} >= {"live", "baseline-main", "patched-2026-10-14"}
    assert body["default_model_mode"] == "openai-mini"
    assert "votes_persisted" in body and body["persistence_note"]
    # Seeding from the adversarial scenario set, not the happy path.
    assert len(body["seed_scenarios"]) == 50
    assert any(s["id"] == "wants_to_stop_frustrated" for s in body["seed_scenarios"])


def test_session_does_not_tell_the_client_which_side_is_which(client):
    body = _start(client)
    blob = json.dumps(body)
    assert body["blinded"] is True
    for vid in ("baseline-main", "patched-2026-10-14"):
        assert vid not in blob, f"{vid} leaked into the session response"


def test_turn_response_is_fully_blind(client):
    s = _start(client)
    r = client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "help na i'm not getting this",
        "left": {"transcript": [], "prior_state": None},
        "right": {"transcript": [], "prior_state": None}})
    assert r.status_code == 200, r.text
    body = r.json()
    blob = json.dumps(body)
    for vid in ("baseline-main", "patched-2026-10-14", "SHOULD-BE-STRIPPED"):
        assert vid not in blob, f"{vid} leaked into the turn response"
    assert "variant_id" not in body["left"] and "variant_id" not in body["right"]
    assert body["comparable"] is True
    assert body["left"]["control"]["intervention"] == "micro_hint"
    # Both arms really were run, each with its own variant.
    assert {c["variant_id"] for c in client.calls} == {"baseline-main", "patched-2026-10-14"}


def test_a_deterministic_turn_is_marked_not_comparable(client, monkeypatch):
    s = _start(client)
    monkeypatch.setattr(client.mod, "run_turn",
                        lambda **kw: dict(TURN_OUT, variant_id=kw["variant_id"],
                                          prompt_consulted=False, reply="Mm-hm."))
    r = client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "ok", "left": {}, "right": {}})
    body = r.json()
    assert body["comparable"] is False
    assert "deterministic lane" in body["not_comparable_reason"]
    assert body["replies_identical"] is True


def test_budget_and_validation_run_before_any_model_call(client):
    s = _start(client)
    # oversized message: rejected with no arm executed and no spend
    r = client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "x" * 5000, "left": {}, "right": {}})
    assert r.status_code == 400
    assert client.calls == []
    # empty message likewise
    assert client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "   ", "left": {}, "right": {}}).status_code == 400
    assert client.calls == []
    # a real turn checks the daily cap first
    client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "help", "left": {}, "right": {}})
    assert client.spent["budget_checked"] == 1


def test_unknown_variant_is_refused_at_session_start(client):
    r = client.post("/admin/arena/session", json={
        "case_id": "case-1", "variant_a": "baseline-main", "variant_b": "nope"})
    assert r.status_code == 400 and "Unknown variant" in r.text


def test_bad_policy_and_missing_session_are_refused(client):
    assert client.post("/admin/arena/session", json={
        "case_id": "case-1", "variant_a": "live", "variant_b": "live",
        "teaching_policy": "freestyle"}).status_code == 400
    r = client.post("/admin/arena/turn", json={
        "session_id": "does-not-exist", "message": "hi", "left": {}, "right": {}})
    assert r.status_code == 404 and "blinding cannot be recovered" in r.text


def test_vote_reveals_only_after_the_choice_is_recorded(client):
    s = _start(client)
    r = client.post("/admin/arena/vote", json={
        "session_id": s["session_id"], "turn_index": 1, "choice": "left",
        "note": "stated the number flatly"})
    assert r.status_code == 200, r.text
    body = r.json()
    revealed = {body["reveal"]["left"]["variant_id"], body["reveal"]["right"]["variant_id"]}
    assert revealed == {"baseline-main", "patched-2026-10-14"}
    assert body["reveal"]["chosen_variant"] == body["reveal"]["left"]["variant_id"]
    assert body["small_n_warning"], "a single vote must be labelled as noise"
    row = _FakeSB.inserted[-1]
    # The fingerprint of each baseline is stored, so a re-snapshot can't silently
    # pool two different experiments.
    assert row["left_fingerprint"] and row["right_fingerprint"]
    assert row["choice"] == "left" and row["chosen_variant"] in revealed


def test_tie_and_both_bad_record_no_winner(client):
    s = _start(client)
    for choice in ("tie", "both_bad"):
        r = client.post("/admin/arena/vote", json={
            "session_id": s["session_id"], "turn_index": 1, "choice": choice})
        assert r.status_code == 200
        assert r.json()["reveal"]["chosen_variant"] is None
    assert client.post("/admin/arena/vote", json={
        "session_id": s["session_id"], "turn_index": 1, "choice": "whichever"}).status_code == 400


def test_an_unsaved_vote_says_so_instead_of_pretending(client):
    _FakeSB.votes_table_missing = True
    s = _start(client)
    r = client.post("/admin/arena/vote", json={
        "session_id": s["session_id"], "turn_index": 1, "choice": "right"})
    body = r.json()
    assert body["persisted"] is False
    assert "2026-10-14_arena_votes.sql" in body["detail"]
    # Still reveals, so the turn was not wasted.
    assert body["reveal"]["left"]["variant_id"]


def test_one_admin_cannot_act_in_another_admins_session(client, monkeypatch):
    s = _start(client)
    monkeypatch.setattr(client.mod, "_require_admin", lambda authorization: "admin-2")
    for path, payload in (
        ("/admin/arena/turn", {"session_id": s["session_id"], "message": "hi", "left": {}, "right": {}}),
        ("/admin/arena/vote", {"session_id": s["session_id"], "turn_index": 1, "choice": "left"}),
    ):
        assert client.post(path, json=payload).status_code == 403


def test_same_variant_on_both_arms_is_flagged_as_a_self_noise_run(client):
    """Comparing a variant with itself measures the JUDGE, not the prompts. Useful on
    purpose -- but it must never be mistaken for a real comparison."""
    s = _start(client, a="live", b="live")
    assert s["self_noise_run"] is True
    r = client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "help", "left": {}, "right": {}})
    assert r.json()["self_noise_run"] is True


def test_every_endpoint_is_admin_gated():
    """The gate is patched out in the other tests, so assert it is really wired."""
    import inspect
    from routes import arena as ar
    for route in ar.router.routes:
        src = inspect.getsource(route.endpoint)
        assert "_require_admin" in src, f"{route.path} is not admin-gated"


# --------------------------------------------------------------------- contract

# The admin page (frontend app/(app)/admin/arena/page.tsx) reads these exact keys.
# They are pinned here because the two repos deploy independently: a renamed field
# would ship a blank panel, not a build error.
ARM_KEYS = {
    "reply", "raw_reply", "silent", "error", "prompt_consulted", "enforced",
    "control", "violations", "metrics", "latency_ms", "model", "provider", "next_state",
    "voice",
}
CONTROL_KEYS = {
    "mode", "function", "render", "reason", "intervention", "materiality",
    "skill", "error_kind", "hint", "fallback",
}
TURN_KEYS = {
    "session_id", "turn_index", "left", "right", "comparable",
    "not_comparable_reason", "replies_identical", "self_noise_run", "clarification",
}
TURN_CLARIFICATION_KEYS = {"counted", "quota", "used", "remaining", "exhausted_this_turn"}
SESSION_KEYS = {"session_id", "case", "teaching_policy", "model_mode", "blinded", "self_noise_run",
                "opening_transcript", "clarification"}
SESSION_CLARIFICATION_KEYS = {"tier", "quota", "used", "remaining"}
VOTE_KEYS = {"persisted", "detail", "reveal", "session_votes", "small_n_warning"}
CONFIG_KEYS = {
    "variants", "live_matches", "default_arm_a", "default_arm_b", "model_modes",
    "default_model_mode", "teaching_policies", "seed_scenarios", "votes_persisted",
    "persistence_note", "limits", "tiers", "default_tier", "build", "candidate_url",
}


def test_response_shapes_match_what_the_admin_page_reads(client):
    assert set(client.get("/admin/arena/config").json()) == CONFIG_KEYS
    s = _start(client)
    assert set(s) == SESSION_KEYS
    assert set(s["case"]) == {"id", "title", "type", "difficulty", "content"}
    assert set(s["clarification"]) == SESSION_CLARIFICATION_KEYS
    assert all(set(r) == {"role", "kind", "content"} for r in s["opening_transcript"])
    t = client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "help", "left": {}, "right": {}}).json()
    assert set(t) == TURN_KEYS
    assert set(t["clarification"]) == TURN_CLARIFICATION_KEYS
    for side in ("left", "right"):
        assert set(t[side]) == ARM_KEYS, side
        assert set(t[side]["control"]) == CONTROL_KEYS, side
        assert set(t[side]["metrics"]) == {"chars", "words", "sentences", "questions"}
    v = client.post("/admin/arena/vote", json={
        "session_id": s["session_id"], "turn_index": 1, "choice": "tie"}).json()
    assert set(v) == VOTE_KEYS
    assert set(v["reveal"]) == {"left", "right", "chosen_variant"}
    assert set(v["reveal"]["left"]) == {"variant_id", "label", "origin", "frozen"}


def test_the_real_run_turn_emits_exactly_the_contracted_arm_keys():
    """The route strips variant_id; everything else must match the pinned contract,
    checked against services.arena itself rather than the test's stub."""
    import inspect
    from services import arena as arena_mod
    src = inspect.getsource(arena_mod.run_turn)
    returned = {line.split('"')[1] for line in src.splitlines()
                if line.strip().startswith('"') and '":' in line}
    assert ARM_KEYS <= returned, ARM_KEYS - returned
    assert "variant_id" in returned  # stripped by the route, present at source


# ----------------------------------------------------------------------- results

def _vote(left, right, choice, chosen=None, lfp="aaaaaaaaaaaa", rfp="bbbbbbbbbbbb", ct="profitability"):
    return {"left_variant": left, "right_variant": right, "choice": choice,
            "chosen_variant": chosen, "left_fingerprint": lfp, "right_fingerprint": rfp,
            "case_type": ct}


def test_results_say_so_when_there_is_no_table(client):
    _FakeSB.votes_table_missing = True
    body = client.get("/admin/arena/results").json()
    assert body["available"] is False
    assert "2026-10-14_arena_votes.sql" in body["note"]
    assert body["total_votes"] == 0 and body["pairs"] == []


def test_self_noise_votes_are_never_pooled_into_win_rates(client):
    """Votes on two identical arms measure the judge. Counting them as wins would
    invent a winner out of pure noise."""
    _FakeSB.vote_rows = (
        [_vote("live", "live", "left", "live") for _ in range(5)]
        + [_vote("live", "live", "tie") for _ in range(3)]
    )
    body = client.get("/admin/arena/results").json()
    assert body["pairs"] == [], "self-noise leaked into the pair tally"
    assert body["self_noise"]["votes"] == 8
    assert body["self_noise"]["left"] == 5
    assert "noise floor" in body["self_noise"]["note"]


def test_a_clean_self_noise_run_is_reported_as_reassurance(client):
    _FakeSB.vote_rows = [_vote("live", "live", "tie") for _ in range(6)]
    body = client.get("/admin/arena/results").json()
    assert "not inventing differences" in body["self_noise"]["note"]


def test_no_winner_is_named_below_the_minimum_n(client):
    _FakeSB.vote_rows = [
        _vote("baseline-main", "patched-2026-10-14", "right", "patched-2026-10-14")
        for _ in range(3)
    ]
    body = client.get("/admin/arena/results").json()
    pair = body["pairs"][0]
    assert pair["decided_votes"] == 3
    assert "Too few to call" in pair["verdict"]
    assert body["min_n_for_a_verdict"] == 12


def test_a_clear_lead_is_named_and_a_narrow_one_is_hedged(client):
    _FakeSB.vote_rows = (
        [_vote("baseline-main", "patched-2026-10-14", "right", "patched-2026-10-14") for _ in range(12)]
        + [_vote("baseline-main", "patched-2026-10-14", "left", "baseline-main") for _ in range(2)]
    )
    body = client.get("/admin/arena/results").json()
    v = body["pairs"][0]["verdict"]
    assert "patched-2026-10-14 leads 12-2" in v and "86%" in v
    assert "noise floor" not in v

    _FakeSB.vote_rows = (
        [_vote("baseline-main", "patched-2026-10-14", "right", "patched-2026-10-14") for _ in range(7)]
        + [_vote("baseline-main", "patched-2026-10-14", "left", "baseline-main") for _ in range(6)]
    )
    v2 = client.get("/admin/arena/results").json()["pairs"][0]["verdict"]
    assert "noise floor could explain it" in v2


def test_the_same_pair_pools_regardless_of_which_side_it_rendered_on(client):
    """Sides are randomised per session; if A-vs-B and B-vs-A made separate buckets
    every tally would be split in half and never reach a verdict."""
    _FakeSB.vote_rows = (
        [_vote("baseline-main", "patched-2026-10-14", "right", "patched-2026-10-14",
               lfp="aaaaaaaaaaaa", rfp="bbbbbbbbbbbb") for _ in range(6)]
        + [_vote("patched-2026-10-14", "baseline-main", "left", "patched-2026-10-14",
                 lfp="bbbbbbbbbbbb", rfp="aaaaaaaaaaaa") for _ in range(6)]
    )
    body = client.get("/admin/arena/results").json()
    assert len(body["pairs"]) == 1
    assert body["pairs"][0]["decided_votes"] == 12
    assert "patched-2026-10-14 leads 12-0" in body["pairs"][0]["verdict"]


def test_votes_against_a_different_snapshot_are_not_averaged_together(client):
    """A re-pinned baseline is a different baseline. Pooling it would hide the change."""
    _FakeSB.vote_rows = (
        [_vote("baseline-main", "patched-2026-10-14", "right", "patched-2026-10-14",
               lfp="oldoldoldold") for _ in range(6)]
        + [_vote("baseline-main", "patched-2026-10-14", "right", "patched-2026-10-14",
                 lfp="newnewnewnew") for _ in range(6)]
    )
    body = client.get("/admin/arena/results").json()
    assert len(body["pairs"]) == 2, "two different snapshots were pooled as one experiment"
    assert all("Too few to call" in p["verdict"] for p in body["pairs"])


def test_position_bias_is_surfaced(client):
    _FakeSB.vote_rows = [
        _vote("baseline-main", "patched-2026-10-14", "left", "baseline-main") for _ in range(14)
    ] + [_vote("baseline-main", "patched-2026-10-14", "right", "patched-2026-10-14") for _ in range(2)]
    body = client.get("/admin/arena/results").json()
    assert body["position"]["left"] == 14 and body["position"]["right"] == 2
    assert "position bias" in body["position"]["note"]


def test_ties_and_both_bad_are_counted_but_do_not_decide(client):
    _FakeSB.vote_rows = (
        [_vote("baseline-main", "patched-2026-10-14", "tie") for _ in range(4)]
        + [_vote("baseline-main", "patched-2026-10-14", "both_bad") for _ in range(2)]
    )
    pair = client.get("/admin/arena/results").json()["pairs"][0]
    assert pair["votes"] == 6 and pair["ties"] == 4 and pair["both_bad"] == 2
    assert pair["decided_votes"] == 0
    assert all(v["wins"] == 0 for v in pair["variants"])


RESULTS_KEYS = {"available", "note", "total_votes", "min_n_for_a_verdict", "pairs",
                "self_noise", "position"}
PAIR_KEYS = {"variants", "votes", "ties", "both_bad", "by_case_type", "decided_votes", "verdict"}


def test_results_shape_matches_what_the_panel_reads(client):
    _FakeSB.vote_rows = [_vote("baseline-main", "patched-2026-10-14", "left", "baseline-main")]
    body = client.get("/admin/arena/results").json()
    assert set(body) == RESULTS_KEYS
    assert set(body["pairs"][0]) == PAIR_KEYS
    assert set(body["pairs"][0]["variants"][0]) == {"id", "fingerprint", "wins"}
    assert set(body["position"]) == {"left", "right", "tie", "both_bad", "note"}
    # the unavailable branch keeps the fields the panel dereferences
    _FakeSB.votes_table_missing = True
    missing = client.get("/admin/arena/results").json()
    assert {"available", "note", "total_votes", "pairs", "self_noise", "position"} <= set(missing)


# ------------------------------------------------------------- production parity

def test_session_opens_with_the_note_every_real_attempt_starts_with(client):
    s = _start(client)
    assert s["opening_transcript"] == [{
        "role": "system", "kind": "system_note",
        "content": "Case ready: Spice Express. Ask any clarifying questions before structuring."}]
    # Free is the default: it is the plan most real candidates are on.
    assert s["clarification"] == {"tier": "free", "quota": 7, "used": 0, "remaining": 7}


def test_the_opening_note_reaches_both_arms_as_a_system_turn(client):
    s = _start(client)
    note = s["opening_transcript"]
    client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "help",
        "left": {"transcript": note}, "right": {"transcript": note}})
    for call in client.calls:
        assert call["transcript"][0] == note[0], "system_note lost on the way to the engine"


def test_quota_is_spent_like_production_and_exhaustion_reaches_both_arms(client):
    s = _start(client)  # free: 7
    def turn(msg):
        client.calls.clear()
        return client.post("/admin/arena/turn", json={
            "session_id": s["session_id"], "message": msg, "left": {}, "right": {}}).json()

    t1 = turn("What is the market? Which cities? How many outlets? Delivery share? AOV? Margin? Growth?")
    assert t1["clarification"]["counted"] == 7
    assert t1["clarification"]["remaining"] == 0
    # The quota is spent BY this turn, so this turn is still answered normally.
    assert t1["clarification"]["exhausted_this_turn"] is False
    assert all(c["clarifications_exhausted"] is False for c in client.calls)

    t2 = turn("And what is the rent per outlet?")
    assert t2["clarification"]["exhausted_this_turn"] is True
    assert len(client.calls) == 2
    assert all(c["clarifications_exhausted"] is True for c in client.calls)

    # A statement asks for nothing, so it never triggers the directive.
    t3 = turn("I'll assume rent is 15% of revenue and move on.")
    assert t3["clarification"]["counted"] == 0
    assert t3["clarification"]["exhausted_this_turn"] is False


def test_a_higher_plan_gets_the_higher_quota(client):
    r = client.post("/admin/arena/session", json={
        "case_id": "case-1", "variant_a": "live", "variant_b": "baseline-main", "tier": "pro"})
    assert r.status_code == 200
    assert r.json()["clarification"]["quota"] == 20
    assert client.post("/admin/arena/session", json={
        "case_id": "case-1", "variant_a": "live", "variant_b": "live",
        "tier": "platinum"}).status_code == 400


def test_both_arms_run_at_the_same_time(client, monkeypatch):
    """Each arm waits at a two-party barrier. Run one after the other, the first arm
    would wait alone until the barrier times out -- so this passes only if the two
    arms really are in flight together."""
    import threading
    barrier = threading.Barrier(2, timeout=3)

    def _concurrent_arm(**kw):
        barrier.wait()
        return dict(TURN_OUT, variant_id=kw["variant_id"])

    monkeypatch.setattr(client.mod, "run_turn", _concurrent_arm)
    s = _start(client)
    r = client.post("/admin/arena/turn", json={
        "session_id": s["session_id"], "message": "help", "left": {}, "right": {}})
    assert r.status_code == 200, r.text
    assert r.json()["left"]["reply"] and r.json()["right"]["reply"]


# ------------------------------------------------------------------- case search
# "Any case or guesstimate": v1 fetched 200 arbitrary rows and filtered THOSE in
# Python, so most of a 500+ item library could never be picked.

CASES_KEYS = {"cases", "count", "total", "offset", "limit", "has_more", "markets_supported"}
CASE_ROW_KEYS = {"id", "title", "type", "difficulty", "market", "unlisted", "is_active", "created_at"}


def test_case_search_runs_in_the_database_over_the_whole_library(client):
    r = client.get("/admin/arena/cases?q=EV chargers&case_type=guesstimate&market=IN&limit=50&offset=100")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == CASES_KEYS
    assert all(set(row) == CASE_ROW_KEYS for row in body["cases"])
    calls = _Query.calls
    assert ("ilike", ("title", "%EV chargers%")) in calls
    assert ("eq", ("type", "guesstimate")) in calls
    assert ("eq", ("market", "IN")) in calls
    assert ("range", (100, 149)) in calls
    assert any(c[0] == "order" and c[1] == ("created_at",) and c[2].get("desc") for c in calls)
    select = next(c for c in calls if c[0] == "select")
    assert "content" not in select[1][0], "the picker must not ship every brief"
    assert select[2].get("count") == "exact"
    # no is_active / unlisted filter: inactive and unlisted cases are testable too
    assert not any(c[0] == "eq" and c[1][0] in ("is_active", "unlisted") for c in calls)


def test_a_pasted_case_id_matches_that_case_exactly(client):
    cid = "3f2c1a9e-4b7d-4c1e-9a2b-1234567890ab"
    client.get(f"/admin/arena/cases?q={cid}")
    assert ("eq", ("id", cid)) in _Query.calls
    assert not any(c[0] == "ilike" for c in _Query.calls)


def test_search_text_is_not_treated_as_a_pattern(client):
    client.get("/admin/arena/cases?q=20%25_off")
    assert ("ilike", ("title", "%20\\%\\_off%")) in _Query.calls


def test_page_size_is_clamped_and_no_filter_means_everything(client):
    client.get("/admin/arena/cases?limit=5000")
    assert ("range", (0, 199)) in _Query.calls
    assert not any(c[0] in ("eq", "ilike") for c in _Query.calls)


def test_a_database_without_the_market_column_still_lists_cases(client):
    _Query.fail_on_columns = "market"
    r = client.get("/admin/arena/cases?market=US")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["markets_supported"] is False
    assert not any(c[0] == "eq" and c[1][0] == "market" for c in _Query.calls)


# ------------------------------------------------------------- current vs new builds

BUILD_KEYS = {"commit", "branch", "service", "role", "prompt_layer", "started_at"}
ARM_TURN_KEYS = {"build", "channel", "arm", "clarification"}


def _arm(client, **over):
    body = {"message": "help na i'm not getting this", "case_id": "case-1"}
    body.update(over)
    return client.post("/admin/arena/arm-turn", json=body)


def test_arm_turn_runs_this_builds_engine_and_says_which_build_answered(client):
    r = _arm(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == ARM_TURN_KEYS
    assert set(body["build"]) == BUILD_KEYS
    assert set(body["arm"]) == ARM_KEYS | {"variant_id"}   # labelled panel: not stripped
    call = client.calls[-1]
    assert call["variant_id"] == "live", "a build compares its OWN deployed prompt layer by default"
    assert call["channel"] == "text"
    assert client.spent["budget_checked"] == 1


def test_arm_turn_is_stateless_about_the_question_count(client):
    """Each pane keeps its own count, so a restart or a second worker loses nothing."""
    r = _arm(client, message="What is the market size?", tier="free", clar_used=7).json()
    assert r["clarification"]["exhausted_this_turn"] is True
    assert client.calls[-1]["clarifications_exhausted"] is True
    r = _arm(client, message="What is the market size? And the growth?", tier="pro", clar_used=3).json()
    assert r["clarification"] == {"counted": 2, "quota": 20, "used": 5, "remaining": 15,
                                  "exhausted_this_turn": False}
    # a forged count cannot go negative or past the quota
    r = _arm(client, message="ok", tier="free", clar_used=999).json()
    assert r["clarification"]["used"] == 7


def test_arm_turn_voice_channel_reaches_the_engine(client):
    _arm(client, channel="voice")
    assert client.calls[-1]["channel"] == "voice"


def test_arm_turn_can_run_an_adversarial_scenario_with_the_evals_own_case_text(client):
    r = _arm(client, case_id=None, scenario_id="wants_to_stop_frustrated",
             message="ugh this is getting irritating, leave it na i don't want to continue")
    assert r.status_code == 200, r.text
    call = client.calls[-1]
    assert call["case_content"] == "A profitability case for practice on MECE."
    assert call["case_type"] == "profitability"


def test_arm_turn_validates_before_any_spend(client):
    for over in ({"message": "  "}, {"channel": "telepathy"}, {"tier": "gold"},
                 {"variant": "nope"}, {"case_id": None}, {"message": "x" * 5000},
                 {"teaching_policy": "freestyle"}):
        assert _arm(client, **over).status_code == 400, over
    assert client.calls == [] and client.spent["budget_checked"] == 0
    assert _arm(client, case_id=None, scenario_id="no-such-scenario").status_code == 404


def test_build_endpoint_and_config_say_which_code_is_running(client):
    b = client.get("/admin/arena/build").json()["build"]
    assert set(b) == BUILD_KEYS and b["role"] in ("current", "candidate")
    cfg = client.get("/admin/arena/config").json()
    assert set(cfg["build"]) == BUILD_KEYS
    assert "candidate_url" in cfg
    assert all(s["required"] for s in cfg["seed_scenarios"]), "the check needs each scenario's rule"


def test_the_scenario_case_text_is_the_evals(client):
    src = open("tools/eval_interviewer_behavior.py", encoding="utf-8").read()
    assert 'case_content=f"A {s[\'case_type\']} case for practice on MECE."' in src
    assert client.mod.scenario_case_content("growth") == "A growth case for practice on MECE."


class _JudgeClient:
    def __init__(self, verdict, sink):
        outer = self
        self.verdict, self.sink = verdict, sink

        class _C:
            @staticmethod
            def create(**kw):
                outer.sink.append(kw)
                msg = type("M", (), {"content": json.dumps(outer.verdict)})()
                return type("R", (), {"choices": [type("Ch", (), {"message": msg})()], "usage": None, "id": "j"})()
        self.chat = type("Chat", (), {"completions": _C()})()


def test_judge_grades_with_the_evals_prompt_and_rule(client, monkeypatch):
    sink = []
    monkeypatch.setattr(client.mod, "openai_client", lambda: _JudgeClient({"pass": True, "why": "closed"}, sink))
    monkeypatch.setattr(client.mod, "log_ai_usage", lambda **kw: None)
    from tools.eval_interviewer_behavior import judge_prompt, SCENARIOS
    s = next(x for x in SCENARIOS if x["id"] == "wants_to_stop_frustrated")
    r = client.post("/admin/arena/judge", json={"scenario_id": s["id"], "reply": "Okay, we can stop here."}).json()
    assert r["pass"] is True and r["judge_pass"] is True and r["violations"] == []
    assert sink[-1]["messages"][0]["content"] == judge_prompt(s, "Okay, we can stop here.")
    assert sink[-1]["temperature"] == 0
    # a deterministic guardrail breach fails it even when the judge says pass
    r = client.post("/admin/arena/judge", json={
        "scenario_id": s["id"], "reply": "That isn't specified. What now? And then?"}).json()
    assert r["judge_pass"] is True and r["pass"] is False and r["violations"]


def test_judge_failure_is_reported_not_passed(client, monkeypatch):
    class _Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("down")
    monkeypatch.setattr(client.mod, "openai_client", lambda: _Boom())
    r = client.post("/admin/arena/judge", json={"scenario_id": "wants_to_stop_frustrated", "reply": "ok"}).json()
    assert r["pass"] is False and r["judge_pass"] is None and "judge error" in r["why"]
    assert client.post("/admin/arena/judge", json={"scenario_id": "nope", "reply": "x"}).status_code == 404
