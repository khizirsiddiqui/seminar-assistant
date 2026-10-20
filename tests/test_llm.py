import json

from app import llm as llm_mod


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
