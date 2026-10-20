def test_settings_read_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    from app.config import get_settings
    s = get_settings()
    assert s.llm_base_url == "http://localhost:8001/v1"
    assert s.llm_api_key == "test-key"
    assert s.llm_model == "test-model"
