"""Per-stage model routing: Gemini (free tier) for reading documents and light checks, OpenAI for
judgement; admins change any stage; env overrides still win; a missing key falls through."""

from interview_intelligence import config as cfg
from interview_intelligence.ai import routing
from tests.conftest import ADMIN_EMAIL, auth, make_settings, mint


def _names(prompt_id, route="fast"):
    return [(p.name, m) for p, m in routing.resolve(prompt_id, route)]


def _keys(**k):
    cfg.set_settings_for_tests(make_settings(**k))
    routing.reset_for_tests()


def test_default_split_light_work_on_gemini_judgement_on_openai():
    _keys(openai_api_key="sk-x", gemini_api_key="g-x", gemini_model="gemini-test-flash")
    try:
        assert _names("cv_parser") == [("gemini", "gemini-test-flash"), ("openai", "gpt-4o-mini")]
        assert _names("jd_parser")[0][0] == "gemini"
        assert _names("role_classifier")[0][0] == "gemini"
        assert _names("turn_analyzer") == [("openai", "gpt-4o-mini")]
        assert _names("interviewer")[0] == ("openai", "gpt-4o-mini")
        assert _names("evidence_extractor", "strong")[0] == ("openai", "gpt-4o")
        assert _names("competency_evaluator", "strong")[0] == ("openai", "gpt-4o")
        assert _names("feedback_writer", "strong")[0] == ("openai", "gpt-4o")
    finally:
        _keys()


def test_without_a_gemini_key_light_work_falls_through_to_openai():
    _keys(openai_api_key="sk-x")
    try:
        assert _names("cv_parser") == [("openai", "gpt-4o-mini")]
    finally:
        _keys()


def test_no_keys_in_test_env_uses_the_simulator():
    assert _names("cv_parser") == [("simulated", "simulated")]


def test_admin_can_move_any_stage_and_reset_it(client):
    _keys(openai_api_key="sk-x", gemini_api_key="g-x")
    try:
        admin = auth(mint(email=ADMIN_EMAIL, tier="free"))
        view = client.get("/v1/admin/ai-routing", headers=admin).json()
        by_id = {s["id"]: s for s in view["stages"]}
        assert by_id["cv_parser"]["current"] == "gemini" and by_id["turn_analyzer"]["current"] == "openai_fast"
        assert view["providers"] == {"openai": True, "gemini": True, "groq": False, "anthropic": False}
        assert "gemini" in view["presets"]
        r = client.patch("/v1/admin/ai-routing", json={"values": {"turn_analyzer": "gemini", "cv_parser": "openai_strong"}},
                         headers=admin)
        assert r.status_code == 200, r.text
        assert _names("turn_analyzer")[0][0] == "gemini"
        assert _names("cv_parser")[0] == ("openai", "gpt-4o")
        r = client.patch("/v1/admin/ai-routing", json={"values": {"turn_analyzer": "default"}}, headers=admin)
        assert {s["id"]: s["current"] for s in r.json()["stages"]}["turn_analyzer"] == "openai_fast"
        assert client.patch("/v1/admin/ai-routing", json={"values": {"turn_analyzer": "skynet"}},
                            headers=admin).status_code == 422
        assert client.patch("/v1/admin/ai-routing", json={"values": {"nope": "gemini"}}, headers=admin).status_code == 422
        user = auth(mint(email="pro@example.invalid"))
        assert client.get("/v1/admin/ai-routing", headers=user).status_code == 403
        # the generic settings endpoint validates the same way
        assert client.patch("/v1/admin/config", json={"values": {"ai.routes": {"cv_parser": "bogus"}}},
                            headers=admin).status_code == 422
    finally:
        _keys()


def test_env_routes_still_win(monkeypatch):
    _keys(openai_api_key="sk-x", gemini_api_key="g-x",
          model_routes_json='{"cv_parser":[{"provider":"openai","model":"gpt-4.1"}]}')
    try:
        assert _names("cv_parser") == [("openai", "gpt-4.1")]
    finally:
        _keys()
