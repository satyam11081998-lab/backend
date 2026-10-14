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
    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
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

    def table(self, name):
        if name == "cases":
            return _Query([CASE])
        if name == "users":
            return _Query([{"is_admin": True}])
        if name == "arena_votes":
            if _FakeSB.votes_table_missing:
                return _Query([])
            return _Query([])
        return _Query([])


TURN_OUT = {
    "variant_id": "SHOULD-BE-STRIPPED",
    "reply": "Which unit are you sizing in?",
    "raw_reply": "Which unit are you sizing in?",
    "silent": False, "error": None, "prompt_consulted": True, "enforced": False,
    "control": {"mode": "HINT", "function": "MICRO_HINT", "render": "model",
                "intervention": "micro_hint", "materiality": "material"},
    "violations": [], "metrics": {"chars": 30, "words": 6, "sentences": 1, "questions": 1},
    "latency_ms": 700, "model": "gpt-4o-mini", "provider": "openai", "next_state": {"hint_level": 1},
}


@pytest.fixture()
def client(monkeypatch):
    from routes import arena as ar

    _FakeSB.inserted = []
    _FakeSB.votes_table_missing = False
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
