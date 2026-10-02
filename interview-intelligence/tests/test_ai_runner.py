"""Stage M plumbing — structured outputs, repair, provider fallback, observability, budgets
(spec §71–§74, §95, §100)."""

import json

import pytest
from pydantic import BaseModel
from sqlalchemy import select

from interview_intelligence import config as cfg
from interview_intelligence.ai import routing, runner
from interview_intelligence.ai.prompts import PromptSpec, prompt_versions
from interview_intelligence.ai.provider import CompletionResult, ProviderError
from interview_intelligence.db.models import ModelRun
from interview_intelligence.db.session import db_session
from tests.conftest import make_settings

SPEC = PromptSpec(id="unit_test_prompt", version="7", stage="unit", route="strong", temperature=0.0, max_tokens=200,
                  system="Return the answer.", schema_version="3")


class Out(BaseModel):
    answer: str
    n: int


class Scripted:
    def __init__(self, name, outputs):
        self.name = name
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, *, model, temperature, max_tokens, json_mode, timeout_s, meta=None):
        self.calls.append({"messages": messages, "model": model, "json_mode": json_mode})
        o = self.outputs.pop(0)
        if isinstance(o, Exception):
            raise o
        return CompletionResult(text=o, provider=self.name, model=model, input_tokens=1000, output_tokens=500)


@pytest.fixture
def chain():
    routes = {"strong": [{"provider": "p1", "model": "gpt-4o"}, {"provider": "p2", "model": "mystery-model-x"}]}
    cfg.set_settings_for_tests(make_settings(model_routes_json=json.dumps(routes)))

    def install(o1, o2=()):
        p1, p2 = Scripted("p1", o1), Scripted("p2", o2)
        routing.set_provider_override("p1", p1)
        routing.set_provider_override("p2", p2)
        return p1, p2
    return install


def runs():
    with db_session() as db:
        return db.execute(select(ModelRun).order_by(ModelRun.created_at, ModelRun.attempt)).scalars().all()


GOOD = '{"answer": "ok", "n": 2}'


def test_fenced_json_is_accepted_and_run_recorded_with_versions(chain):
    p1, _ = chain(["```json\n" + GOOD + "\n```"])
    out = runner.run_structured(SPEC, "q", Out, runner.RunContext())
    assert out.n == 2 and p1.calls[0]["json_mode"] is True
    r = runs()
    assert len(r) == 1 and r[0].status == "ok" and r[0].prompt_id == "unit_test_prompt"
    assert r[0].prompt_version == "7" and r[0].schema_version == "3" and float(r[0].cost_usd) > 0
    system = p1.calls[0]["messages"][0]["content"]
    assert "SECURITY RULES" in system and "OUTPUT FORMAT" in system, "data rules + schema always injected"


def test_invalid_output_is_repaired_once_on_the_same_provider(chain):
    p1, p2 = chain(["Sure! here you go", GOOD])
    assert runner.run_structured(SPEC, "q", Out, runner.RunContext()).answer == "ok"
    assert len(p1.calls) == 2 and not p2.calls
    assert "previous output was rejected" in p1.calls[1]["messages"][-1]["content"]
    assert [x.status for x in runs()] == ["invalid_output", "repaired"]


def test_schema_violation_falls_back_to_next_provider_and_flags_unpriced_model(chain):
    p1, p2 = chain(['{"answer": "x"}', '{"answer": "x", "n": "many"}'], [GOOD])
    assert runner.run_structured(SPEC, "q", Out, runner.RunContext()).n == 2
    r = runs()
    assert [(x.provider, x.status) for x in r] == [("p1", "invalid_output"), ("p1", "invalid_output"), ("p2", "ok")]
    assert r[0].validation["problems"], "validation problems are stored for admins"
    assert r[-1].validation.get("unpriced_model") == "mystery-model-x" and float(r[-1].cost_usd) == 0


def test_provider_error_skips_straight_to_next_provider(chain):
    p1, p2 = chain([ProviderError("HTTP 503")], [GOOD])
    runner.run_structured(SPEC, "q", Out, runner.RunContext())
    assert len(p1.calls) == 1 and len(p2.calls) == 1
    assert [x.status for x in runs()] == ["error", "ok"]


def test_every_provider_failing_raises_structured_error(chain):
    chain([ProviderError("timeout", timeout=True)], ["nope", "still nope"])
    with pytest.raises(runner.StructuredOutputError):
        runner.run_structured(SPEC, "q", Out, runner.RunContext())
    assert [x.status for x in runs()] == ["timeout", "invalid_output", "invalid_output"]


def test_semantic_validation_triggers_repair(chain):
    p1, _ = chain(['{"answer": "bad", "n": 1}', '{"answer": "good", "n": 1}'])
    out = runner.run_structured(SPEC, "q", Out, runner.RunContext(),
                                validate=lambda o: ["answer must be good"] if o.answer != "good" else [])
    assert out.answer == "good" and "answer must be good" in p1.calls[1]["messages"][-1]["content"]


def test_runs_are_persisted_even_when_the_callers_transaction_fails(chain):
    chain([GOOD])
    with pytest.raises(RuntimeError):
        with runner.collect_runs():
            runner.run_structured(SPEC, "q", Out, runner.RunContext())
            assert runs() == [], "deferred until the caller finishes"
            raise RuntimeError("caller rolled back")
    assert len(runs()) == 1


def test_daily_budget_refuses_new_spend(chain):
    chain([GOOD])
    with db_session() as db:
        db.add(ModelRun(stage="x", provider="p", model="m", status="ok", cost_usd=5))
    runner.reset_budget_cache()
    with pytest.raises(runner.BudgetExceeded):
        runner.run_structured(SPEC, "q", Out, runner.RunContext(daily_budget_usd=1.0))
    # a live turn (no daily budget on its context) is never cut off by the global switch
    assert runner.run_structured(SPEC, "q", Out, runner.RunContext()).n == 2


def test_run_text_returns_empty_when_all_fail_so_callers_fall_back(chain):
    chain([ProviderError("down")], [ProviderError("down too")])
    assert runner.run_text(SPEC, "q", runner.RunContext()) == ""


def test_parse_json_object_tolerates_prose():
    assert runner.parse_json_object('Here it is: {"a": 1} hope that helps') == {"a": 1}
    with pytest.raises(json.JSONDecodeError):
        runner.parse_json_object("no json at all")


def test_simulation_is_refused_in_production():
    cfg.set_settings_for_tests(make_settings(env="production"))
    routing.reset_for_tests()
    from interview_intelligence.errors import Unavailable
    with pytest.raises(Unavailable):
        routing.resolve("cv_parser", "fast")


def test_every_prompt_is_versioned():
    v = prompt_versions()
    assert len(v) >= 14 and all(x for x in v.values())
    assert v["turn_analyzer"] == "3"


def test_unset_environment_fails_safe_to_production(monkeypatch):
    """A deploy that forgets II_ENV must not get simulated scores or an ephemeral key."""
    monkeypatch.delenv("II_ENV", raising=False)
    monkeypatch.delenv("II_ALLOW_SIMULATION", raising=False)
    s = cfg.load_settings()
    assert s.env == "prod" and s.is_production
    monkeypatch.setenv("II_ALLOW_SIMULATION", "true")
    cfg.set_settings_for_tests(make_settings(env="prod"))
    routing.reset_for_tests()
    assert routing.simulation_allowed() is False
    from interview_intelligence.security import crypto
    crypto.reset_for_tests()
    with pytest.raises(Exception):
        crypto.encrypt_text("x")


def test_production_refuses_to_start_on_sqlite():
    from fastapi.testclient import TestClient
    from interview_intelligence.main import create_app
    cfg.set_settings_for_tests(make_settings(env="prod", database_url="sqlite+pysqlite:///:memory:",
                                             encryption_key="x" * 44))
    with pytest.raises(RuntimeError, match="Postgres in production"):
        with TestClient(create_app()):
            pass
