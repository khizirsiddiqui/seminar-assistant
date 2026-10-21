import json
import logging

import pytest

from app import llm as llm_mod

logging.basicConfig(level=logging.INFO)


def test_parse_schedule_builds_messages_and_parses_json(monkeypatch):
    captured = {}

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": json.dumps({"course_title": "Seminar: Advanced Machine Learning",
                                   "sessions": [
                {"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need",
                 "paper_url": "https://arxiv.org/abs/1706.03762"}
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
    result = llm_mod.parse_schedule("27.10. Alice - Attention Is All You Need", assume_year=2026)
    assert result == {"course_title": "Seminar: Advanced Machine Learning",
                      "sessions": [{"date": "2026-10-27", "student": "Alice",
                                    "paper": "Attention Is All You Need",
                                    "paper_url": "https://arxiv.org/abs/1706.03762"}]}
    assert "sessions" in captured["messages"][1]["content"] or captured["response_format"] == {"type": "json_object"}
    assert "course_title" in captured["messages"][0]["content"]
    assert captured["model"] == llm_mod.get_settings().llm_model


def test_parse_schedule_extracts_paper_url_from_hyperlinks(monkeypatch):
    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": json.dumps({"course_title": None, "sessions": [
                {"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need",
                 "paper_url": "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)"},
                {"date": "2026-10-27", "student": "Bob", "paper": "BERT",
                 "paper_url": '<a href="https://arxiv.org/abs/1706.03762">BERT</a>'},
            ]})})()})()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    result = llm_mod.parse_schedule("schedule", assume_year=2026)
    assert [r["paper_url"] for r in result["sessions"]] == [
        "https://arxiv.org/abs/1706.03762",
        "https://arxiv.org/abs/1706.03762",
    ]


def test_parse_schedule_logs_request_and_response(monkeypatch, caplog):
    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": json.dumps({"course_title": None, "sessions": [
                {"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need"}
            ]})})()})()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())

    with caplog.at_level(logging.INFO, logger="seminar.llm"):
        result = llm_mod.parse_schedule("27.10. Alice - Attention Is All You Need", assume_year=2026)
    assert result["course_title"] is None
    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["paper_url"] == ""

    messages = [r.message for r in caplog.records]
    assert any("LLM request" in m and "model=" in m for m in messages)
    assert any("LLM response" in m and "duration=" in m for m in messages)
    assert any("1 session" in m for m in messages)


def test_preparation_chat_reports_token_usage(monkeypatch):
    captured = []

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class FakeMessage:
        content = json.dumps({"sections": [{"title": "Background"}]})

    class FakeChoice:
        message = FakeMessage()

    class FakeResp:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    llm_mod.set_usage_callback(captured.append)
    try:
        titles = llm_mod.generate_preparation_section_titles({
            "title": "Test paper",
            "authors": [],
            "published": "",
            "categories": [],
            "abstract": "",
            "paper_text": "Some paper text.",
        })
    finally:
        llm_mod.clear_usage_callback()

    assert titles == ["Background"]
    assert captured == [{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                         "cached_tokens": 0}]


def test_generate_preparation_section_titles_caps_at_five(monkeypatch):
    seven_titles = [f"Section {i}" for i in range(1, 8)]
    content = json.dumps({"sections": [{"title": title} for title in seven_titles]})

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {"content": content})})()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    titles = llm_mod.generate_preparation_section_titles({
        "title": "Test paper",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
        "paper_text": "Some paper text.",
    })

    assert titles == seven_titles[:5]


def test_preparation_section_content_includes_other_titles(monkeypatch):
    captured = []

    class FakeResp:
        def __init__(self):
            self.choices = [type("C", (), {"message": type("M", (), {
                "content": json.dumps({"content": "Section content."})
            })})()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    paper = {
        "title": "Test paper",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
        "paper_text": "Some paper text.",
    }
    ai_other = ["Background", "Results", "Questions Worth Asking the Speaker", "Key Take-Away"]

    llm_mod.generate_preparation_section_content(paper, "Method", ai_other)

    assert len(captured) == 1
    ai_user = captured[0]["messages"][1]["content"]
    assert "The other sections of this guide are: " + ", ".join(ai_other) in ai_user
    assert "do not produce content that belongs in those sections" in ai_user


def test_preparation_closing_is_one_request_and_includes_other_titles(monkeypatch):
    captured = []

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {"content": json.dumps({
            "questions": "- Why?",
            "take_away": "The idea.",
            "short_briefing": "A briefing.",
            "prerequisites": "Some background.",
        })})})()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    paper = {"title": "Test paper", "authors": [], "published": "", "categories": [],
             "abstract": "", "paper_text": "Some paper text."}

    closing = llm_mod.generate_preparation_closing(paper, ["Background", "Method"])

    assert closing == {"questions": "- Why?", "take_away": "The idea.",
                       "short_briefing": "A briefing.", "prerequisites": "Some background."}
    assert len(captured) == 1
    assert "The other sections of this guide are: Background, Method" in captured[0]["messages"][1]["content"]


def test_preparation_closing_requires_every_text_field(monkeypatch):
    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {"content": json.dumps({
            "questions": "- Why?", "take_away": "", "short_briefing": "A briefing."})})})()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    with pytest.raises(ValueError, match="take_away"):
        llm_mod.generate_preparation_closing({"paper_text": "Some paper text."})


def test_cached_tokens_are_reported_from_prompt_token_details(monkeypatch):
    captured = []

    class FakeUsage:
        prompt_tokens = 100
        completion_tokens = 5
        total_tokens = 105
        prompt_tokens_details = type("D", (), {"cached_tokens": 80})()

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": json.dumps({"sections": [{"title": "Background"}]})})})()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    llm_mod.set_usage_callback(captured.append)
    try:
        llm_mod.generate_preparation_section_titles({"paper_text": "Some paper text."})
    finally:
        llm_mod.clear_usage_callback()
    assert captured[0]["cached_tokens"] == 80


def test_preparation_section_content_omits_other_titles_suffix_by_default(monkeypatch):
    captured = []

    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {
            "content": json.dumps({"content": "Section content."})
        })})()]

    class FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm_mod, "_client", lambda: FakeClient())
    llm_mod.generate_preparation_section_content({
        "title": "Test paper",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
        "paper_text": "Some paper text.",
    }, "Method")

    assert "The other sections of this guide are:" not in captured[0]["messages"][1]["content"]
