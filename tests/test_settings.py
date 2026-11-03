import json
from types import SimpleNamespace

import openai
import pytest
from fastapi.testclient import TestClient

import app.config as config
from app import llm as llm_mod
from app.main import app


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "ENV_FILE", str(tmp_path / ".env"))
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "ARXIV_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_saved_settings_are_overridden_per_field_by_env(monkeypatch, tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({
        "llm_base_url": "https://saved.example/v1",
        "llm_api_key": "saved-key",
        "llm_model": "saved-model",
        "arxiv_timeout_seconds": 77,
    }))
    (tmp_path / ".env").write_text(
        "LLM_MODEL=env-model\nLLM_BASE_URL=https://env.example/v1\n")

    settings = config.get_settings()
    assert settings.llm_model == "env-model"
    assert settings.llm_base_url == "https://env.example/v1"
    assert settings.llm_api_key == "saved-key"
    assert settings.arxiv_timeout_seconds == 77

    sources = config.setting_sources()
    assert sources["llm_base_url"] == ".env"
    assert sources["llm_model"] == ".env"
    assert sources["llm_api_key"] == "settings"
    assert sources["arxiv_timeout_seconds"] == "settings"

    monkeypatch.setenv("LLM_API_KEY", "environment-key")
    config.get_settings.cache_clear()
    assert config.get_settings().llm_api_key == "environment-key"
    assert config.setting_sources()["llm_api_key"] == "environment"


def test_save_mask_and_clear_api_key(tmp_path):
    client = TestClient(app)
    r = client.post("/settings", data={
        "llm_base_url": "https://api.example/v1",
        "llm_api_key": "super-secret-key",
        "llm_model": "gpt-4o-mini",
        "arxiv_timeout_seconds": "30",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/settings"

    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved == {
        "llm_base_url": "https://api.example/v1",
        "llm_api_key": "super-secret-key",
        "llm_model": "gpt-4o-mini",
        "arxiv_timeout_seconds": 30,
    }

    page = client.get("/settings")
    assert page.status_code == 200
    assert "https://api.example/v1" in page.text
    assert "gpt-4o-mini" in page.text
    assert "super-secret-key" not in page.text
    assert "sup…-key" in page.text

    r = client.post("/settings", data={
        "llm_base_url": "https://api.example/v1",
        "llm_api_key": "",
        "llm_model": "gpt-4o-mini",
        "arxiv_timeout_seconds": "30",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert json.loads((tmp_path / "settings.json").read_text())["llm_api_key"] == "super-secret-key"

    r = client.post("/settings", data={
        "llm_base_url": "https://api.example/v1",
        "llm_api_key": "ignored-new-key",
        "clear_llm_api_key": "1",
        "llm_model": "gpt-4o-mini",
        "arxiv_timeout_seconds": "30",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert "llm_api_key" not in json.loads((tmp_path / "settings.json").read_text())

    before = (tmp_path / "settings.json").read_text()
    r = client.post("/settings", data={
        "llm_base_url": "https://api.example/v1",
        "llm_model": "gpt-4o-mini",
        "arxiv_timeout_seconds": "0",
    }, follow_redirects=False)
    assert r.status_code == 422
    assert "arXiv timeout must be a whole number between 1 and 3600." in r.text
    assert (tmp_path / "settings.json").read_text() == before


def test_save_ignores_field_overridden_by_dotenv(tmp_path):
    (tmp_path / ".env").write_text("LLM_BASE_URL=https://env.example/v1\n")
    client = TestClient(app)
    r = client.post("/settings", data={
        "llm_base_url": "https://different.example/v1",
        "llm_model": "gpt-4o-mini",
        "arxiv_timeout_seconds": "30",
    }, follow_redirects=False)
    assert r.status_code == 303

    saved = json.loads((tmp_path / "settings.json").read_text())
    assert "llm_base_url" not in saved
    assert config.get_settings().llm_base_url == "https://env.example/v1"


def test_listing_models_uses_form_values_and_falls_back_to_saved_key(monkeypatch):
    captured = {}

    def fake_list_models(base_url, api_key):
        captured.update(base_url=base_url, api_key=api_key)
        return ["gpt-4o-mini", "gpt-4o"]

    monkeypatch.setattr(llm_mod, "list_models", fake_list_models)
    config.update_saved_settings({
        "llm_base_url": "https://saved.example/v1",
        "llm_api_key": "saved-key",
    })
    client = TestClient(app)

    r = client.post("/settings/models", data={
        "llm_base_url": "https://api.example/v1",
        "llm_api_key": "typed-key",
    })
    assert r.status_code == 200
    assert r.json() == {"models": ["gpt-4o-mini", "gpt-4o"]}
    assert captured == {"base_url": "https://api.example/v1", "api_key": "typed-key"}

    r = client.post("/settings/models", data={"llm_base_url": "", "llm_api_key": ""})
    assert r.status_code == 200
    assert captured == {"base_url": "https://saved.example/v1", "api_key": "saved-key"}

    def broken_list_models(base_url, api_key):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(llm_mod, "list_models", broken_list_models)
    r = client.post("/settings/models", data={
        "llm_base_url": "https://api.example/v1",
        "llm_api_key": "typed-key",
    })
    assert r.status_code == 502
    assert "error" in r.json()


def test_list_models_openai_helper_sorts_ids_and_passes_credentials(monkeypatch):
    captured = {}

    class FakeModels:
        def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="b"), SimpleNamespace(id="a")])

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setattr(llm_mod, "OpenAI", FakeClient)
    assert llm_mod.list_models("https://api.example/v1", "test-key") == ["a", "b"]
    assert captured == {
        "base_url": "https://api.example/v1",
        "api_key": "test-key",
        "timeout": 15.0,
    }


def test_connection_test_reports_success_and_failure_reasons(monkeypatch):
    monkeypatch.setattr(llm_mod, "list_models", lambda base_url, api_key: ["a", "b"])
    client = TestClient(app)

    r = client.post("/settings/test-connection", data={
        "llm_base_url": "https://api.example/v1", "llm_api_key": "typed-key"})
    assert r.status_code == 200
    assert r.json() == {"message": "Connection successful — 2 models available."}

    def unauthorized(base_url, api_key):
        raise openai.AuthenticationError(
            "Error code: 401",
            response=SimpleNamespace(status_code=401, headers={}, request=object()),
            body={"error": {"message": "Incorrect API key provided."}})

    monkeypatch.setattr(llm_mod, "list_models", unauthorized)
    r = client.post("/settings/test-connection", data={
        "llm_base_url": "https://api.example/v1", "llm_api_key": "bad"})
    assert r.status_code == 502
    assert r.json() == {"error": "Authentication failed (HTTP 401): Incorrect API key provided."}

    r = client.post("/settings/test-connection", data={"llm_base_url": "ftp://x", "llm_api_key": "k"})
    assert r.status_code == 422
    assert r.json() == {"error": "Base URL must start with http:// or https://."}
