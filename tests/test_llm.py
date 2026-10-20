import json
import logging

from app import llm as llm_mod

logging.basicConfig(level=logging.INFO)


def test_parse_schedule_builds_messages_and_parses_json(monkeypatch):
    captured = {}

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": json.dumps({"sessions": [
                {"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need"}
            ]})})()})()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    rows = llm_mod.parse_schedule("27.10. Alice - Attention Is All You Need", assume_year=2026)
    assert rows == [{"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need"}]
    assert "sessions" in captured["messages"][1]["content"] or captured["response_format"] == {"type": "json_object"}
    assert captured["model"] == llm_mod.get_settings().llm_model


def test_parse_schedule_logs_request_and_response(monkeypatch, caplog):
    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": json.dumps({"sessions": [
                {"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need"}
            ]})})()})()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())

    with caplog.at_level(logging.INFO, logger="seminar.llm"):
        llm_mod.parse_schedule("27.10. Alice - Attention Is All You Need", assume_year=2026)

    messages = [r.message for r in caplog.records]
    assert any("LLM request" in m and "model=" in m for m in messages)
    assert any("LLM response" in m and "duration=" in m for m in messages)
    assert any("1 session" in m for m in messages)
