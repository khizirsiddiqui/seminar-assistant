import pytest

from app import arxiv, llm, main, storage
from tests.test_prepare import _create_session, _make_client, _prepare_mocks, _wait_preparation_done

BRIEFING = "BRIEFING-MARKER: the paper replaces recurrence with attention."
PAPER_MARKER = "The Transformer is an architecture based on self-attention."
LINK_BODY = "Link body sentence about scaling laws. " * 500


def _stub_answer(monkeypatch, reply="**Answer** about the paper."):
    """Record every (context, messages) pair the route hands to the model."""
    calls = []

    def fake_answer_question(context, messages):
        calls.append({"context": context, "messages": [dict(m) for m in messages]})
        return reply

    monkeypatch.setattr(llm, "answer_question", fake_answer_question)
    return calls


def _generated_session(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)
    calls = _prepare_mocks(monkeypatch, generation={
        "short_briefing": BRIEFING,
        "prerequisites": "Linear algebra basics.",
        "sections": [{"title": "Key ideas", "content": "Self-attention replaces recurrence."}],
    })
    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""}, follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)
    return client, sid, calls


def test_chat_available_after_generate_and_paper_context_used(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    answer_calls = _stub_answer(
        monkeypatch, reply="Self-attention lets every position attend to all others. "
                           "<script>alert(1)</script>")

    r = client.post(f"/prepare/{sid}/0/chat",
                    data={"message": "Why self-attention?", "use_paper": "on"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["messages"]) == 2
    assert body["messages"][0] == {"role": "user", "content": "Why self-attention?", "html": ""}
    assistant = body["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["html"]
    assert "<script>" not in assistant["html"]

    assert len(answer_calls) == 1
    assert PAPER_MARKER in answer_calls[0]["context"]
    assert BRIEFING not in answer_calls[0]["context"]
    assert answer_calls[0]["messages"] == [{"role": "user", "content": "Why self-attention?"}]

    saved = storage.load_chat(sid, 0)
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant"]


def test_chat_persists_across_requests(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    answer_calls = _stub_answer(monkeypatch)

    client.post(f"/prepare/{sid}/0/chat", data={"message": "First?", "use_paper": "on"})
    second = client.post(f"/prepare/{sid}/0/chat",
                         data={"message": "Second?", "use_paper": "on"})
    assert second.status_code == 200
    assert [m["content"] for m in answer_calls[1]["messages"]] == [
        "First?", "**Answer** about the paper.", "Second?"]

    listing = client.get(f"/prepare/{sid}/0/chat").json()
    assert [m["content"] for m in listing["messages"]][::2] == ["First?", "Second?"]
    assert listing["paper_available"] is True
    assert listing["summary_available"] is True

    page = client.get(f"/prepare/{sid}/0")
    assert page.status_code == 200
    assert 'id="chat-panel"' in page.text
    assert 'id="chat-use-paper"' in page.text


def test_chat_summary_context_only(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    answer_calls = _stub_answer(monkeypatch)

    r = client.post(f"/prepare/{sid}/0/chat", data={"message": "Recap?", "use_summary": "on"})
    assert r.status_code == 200
    context = answer_calls[0]["context"]
    assert BRIEFING in context
    assert "### Key ideas" in context
    assert PAPER_MARKER not in context


def test_chat_paper_text_missing_returns_409(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)
    storage.save_preparation(sid, 0, {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "source": {"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
        "short_briefing": "A legacy briefing.",
        "prerequisites": "",
        "sections": [{"title": "Key ideas", "content": "Attention only."}],
        "tokens": {},
        "duration_seconds": 0,
    })
    answer_calls = _stub_answer(monkeypatch)

    r = client.post(f"/prepare/{sid}/0/chat", data={"message": "What?", "use_paper": "on"})
    assert r.status_code == 409
    assert "Regenerate the preparation guide" in r.json()["error"]

    assert client.get(f"/prepare/{sid}/0/chat").json()["paper_available"] is False

    ok = client.post(f"/prepare/{sid}/0/chat", data={"message": "What?", "use_summary": "on"})
    assert ok.status_code == 200
    assert "A legacy briefing." in answer_calls[0]["context"]


def test_chat_link_add_remove_and_context(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    answer_calls = _stub_answer(monkeypatch)
    url = "https://example.org/scaling-laws"
    second_url = "https://example.org/benchmarks"

    monkeypatch.setattr(arxiv, "extract_paper", lambda paper: {
        "paper_text": LINK_BODY, "paper_text_truncated": False})

    added = client.post(f"/prepare/{sid}/0/chat/links", data={"url": url})
    assert added.status_code == 200
    links = added.json()["links"]
    assert len(links) == 1
    assert links[0]["url"] == url
    assert links[0]["truncated"] is False
    assert links[0]["chars"] == len(LINK_BODY)

    duplicate = client.post(f"/prepare/{sid}/0/chat/links", data={"url": url})
    assert duplicate.status_code == 409

    second = client.post(f"/prepare/{sid}/0/chat/links", data={"url": second_url})
    assert [link["url"] for link in second.json()["links"]] == [url, second_url]

    r = client.post(f"/prepare/{sid}/0/chat",
                    data={"message": "What limits scaling?", "use_summary": "on",
                          "links": [url, second_url]})
    assert r.status_code == 200
    context = answer_calls[0]["context"]
    assert LINK_BODY[:200] in context
    assert f"## Source: {url}" in context
    assert f"## Source: {second_url}" in context

    removed = client.post(f"/prepare/{sid}/0/chat/links/remove", data={"url": url})
    assert removed.status_code == 200
    assert [link["url"] for link in removed.json()["links"]] == [second_url]
    assert [link["url"] for link in storage.load_chat(sid, 0)["links"]] == [second_url]


def test_chat_llm_failure_rolls_back_user_message(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)

    def failing(context, messages):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(llm, "answer_question", failing)

    r = client.post(f"/prepare/{sid}/0/chat", data={"message": "Broken?", "use_paper": "on"})
    assert r.status_code == 502
    assert r.json()["error"] == "RuntimeError: provider exploded"
    assert storage.load_chat(sid, 0)["messages"] == []


def test_chat_link_limit_and_truncation(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    long_text = "x" * (main.LINK_TEXT_LIMIT + 50)
    monkeypatch.setattr(arxiv, "extract_paper", lambda paper: {
        "paper_text": long_text, "paper_text_truncated": False})

    for index in range(main.MAX_CHAT_LINKS):
        r = client.post(f"/prepare/{sid}/0/chat/links",
                        data={"url": f"https://example.org/{index}"})
        assert r.status_code == 200
        assert r.json()["links"][-1]["chars"] == main.LINK_TEXT_LIMIT
        assert r.json()["links"][-1]["truncated"] is True

    over = client.post(f"/prepare/{sid}/0/chat/links", data={"url": "https://example.org/extra"})
    assert over.status_code == 409
    assert over.json()["error"] == "You can add at most 5 links."

    missing = client.post(f"/prepare/{sid}/0/chat/links/remove",
                          data={"url": "https://example.org/absent"})
    assert missing.status_code == 404


def test_chat_clear_keeps_links(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    _stub_answer(monkeypatch)
    url = "https://example.org/paper"
    monkeypatch.setattr(arxiv, "extract_paper", lambda paper: {
        "paper_text": LINK_BODY, "paper_text_truncated": False})
    client.post(f"/prepare/{sid}/0/chat/links", data={"url": url})
    client.post(f"/prepare/{sid}/0/chat", data={"message": "Hello?", "use_summary": "on"})

    cleared = client.post(f"/prepare/{sid}/0/chat/clear")
    assert cleared.status_code == 200
    assert cleared.json()["messages"] == []
    saved = storage.load_chat(sid, 0)
    assert saved["messages"] == []
    assert [link["url"] for link in saved["links"]] == [url]


def test_regenerate_clears_chat_and_paper_text(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    _stub_answer(monkeypatch)
    client.post(f"/prepare/{sid}/0/chat", data={"message": "Hello?", "use_paper": "on"})
    assert storage.load_chat(sid, 0)["messages"]
    assert storage.load_paper_text(sid, 0)

    r = client.post(f"/prepare/{sid}/0/regenerate", data={"arxiv_query": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    chat = storage.load_chat(sid, 0)
    assert chat["messages"] == []
    assert chat["links"] == []
    assert storage.load_paper_text(sid, 0)


def test_chat_rejects_empty_and_invalid_input(tmp_path, monkeypatch):
    client, sid, _ = _generated_session(tmp_path, monkeypatch)
    _stub_answer(monkeypatch)

    empty = client.post(f"/prepare/{sid}/0/chat", data={"message": "   ", "use_paper": "on"})
    assert empty.status_code == 422
    assert empty.json()["error"] == "Type a question first."

    too_long = client.post(f"/prepare/{sid}/0/chat",
                           data={"message": "x" * 4001, "use_paper": "on"})
    assert too_long.status_code == 422

    no_context = client.post(f"/prepare/{sid}/0/chat", data={"message": "Hello?"})
    assert no_context.status_code == 409

    bad_link = client.post(f"/prepare/{sid}/0/chat/links", data={"url": "notaurl"})
    assert bad_link.status_code == 422
    assert bad_link.json()["error"] == "Enter a full http(s) link."

    unknown_row = client.get(f"/prepare/{sid}/9/chat")
    assert unknown_row.status_code == 404
