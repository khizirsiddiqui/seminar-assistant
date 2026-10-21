import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import arxiv, llm, main, storage

ROWS = [{"date": "2026-10-27", "student": "Alice",
         "paper": "Attention Is All You Need"}]

MOCK_PAPER = {
    "arxiv_id": "1706.03762",
    "url": "https://arxiv.org/abs/1706.03762",
    "pdf_url": "https://arxiv.org/pdf/1706.03762",
    "title": "Attention Is All You Need",
    "abstract": "We propose the Transformer model.",
    "authors": ["Ashish Vaswani"],
    "published": "2017-06-12",
    "categories": ["cs.CL"],
}


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        llm, "parse_schedule",
        lambda text, assume_year=None: {"course_title": "", "sessions": ROWS})
    from app.main import app
    return TestClient(app)


def _create_session(client) -> str:
    r = client.post("/parse", data={"pasted_text": "27.10. Alice - Attention Is All You Need",
                                    "venue": "Sand 14", "assume_year": "2026",
                                    "start_time": "14:15", "end_time": "15:45"},
                    follow_redirects=False)
    sid = r.headers["location"].split("/")[-1]
    r = client.post(f"/generate/{sid}", data={"row_index": "0", "venue": "Sand 14",
                                              "start_time": "14:15", "end_time": "15:45",
                                              "course_title": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    return sid


def _paper_with_text(paper=None):
    base = dict(paper or MOCK_PAPER)
    base["paper_text"] = "The Transformer is an architecture based on self-attention. " * 100
    base["paper_text_truncated"] = False
    return base


def _prepare_mocks(monkeypatch, generation=None):
    calls = {"find": [], "extract": [], "generate": 0}
    paper = _paper_with_text()
    fixed_titles = {"Questions Worth Asking the Speaker", "Key Take-Away"}

    def resolved(_paper):
        if callable(generation):
            return generation(_paper) or {}
        return generation or {}

    def fake_find(query, title="", presenter=""):
        calls["find"].append(query)
        return dict(paper)

    def fake_extract(found_paper):
        calls["extract"].append(found_paper)
        return _paper_with_text(found_paper)

    def fake_generate_section_titles(_paper):
        calls["generate"] += 1
        result = resolved(_paper)
        return [s["title"] for s in result.get("sections", [])
                if s["title"] not in fixed_titles]

    def fake_generate_section_content(_paper, title, other_titles=None):
        result = resolved(_paper)
        for section in result.get("sections", []):
            if section["title"] == title:
                return section["content"]
        return f"{title} content."

    def fake_generate_closing(_paper, other_titles=None):
        result = resolved(_paper)
        by_title = {section["title"]: section["content"] for section in result.get("sections", [])}
        return {
            "questions": by_title.get("Questions Worth Asking the Speaker",
                                      "Questions Worth Asking the Speaker content."),
            "take_away": by_title.get("Key Take-Away", "Key Take-Away content."),
            "short_briefing": result.get("short_briefing", ""),
            "prerequisites": result.get("prerequisites", ""),
        }

    monkeypatch.setattr(arxiv, "find_paper", fake_find)
    monkeypatch.setattr(arxiv, "extract_paper", fake_extract)
    monkeypatch.setattr(llm, "extract_paper_metadata", lambda _paper: {})
    monkeypatch.setattr(llm, "generate_preparation_section_titles", fake_generate_section_titles)
    monkeypatch.setattr(llm, "generate_preparation_section_content", fake_generate_section_content)
    monkeypatch.setattr(llm, "generate_preparation_closing", fake_generate_closing)
    return calls


def _wait_preparation_done(tmp_path, sid, row_index, timeout=2.0):
    prep = tmp_path / sid / "preparation" / f"{row_index}.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = storage.load_generation_state(sid, row_index)
        if state and state.get("status") == "complete" and prep.exists():
            return True
        time.sleep(0.01)
    state = storage.load_generation_state(sid, row_index)
    raise AssertionError(
        f"preparation not done: prep_exists={prep.exists()} "
        f"status={state!r}")


def _wait_state_status(tmp_path, sid, row_index, status, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = storage.load_generation_state(sid, row_index)
        if state and state.get("status") == status:
            return state
        time.sleep(0.01)
    raise AssertionError(
        f"state did not become {status!r}: {storage.load_generation_state(sid, row_index)!r}")


def test_generate_prepare_uses_arxiv_and_saves_cache(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    before = client.get(f"/prepare/{sid}/0")
    assert before.status_code == 200
    assert "Generate preparation" in before.text
    assert "No preparation guide yet." in before.text

    prep = {
        "short_briefing": "This paper introduces the Transformer architecture.",
        "sections": [
            {"title": "Key ideas", "content": "Self-attention replaces recurrence."},
            {"title": "Results", "content": "Transformers achieve strong translation results."},
        ],
    }
    _prepare_mocks(monkeypatch, generation=prep)

    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == f"/prepare/{sid}/0"
    assert _wait_preparation_done(tmp_path, sid, 0)

    page = client.get(f"/prepare/{sid}/0")
    assert page.status_code == 200
    assert "Executive briefing" in page.text
    assert "Preparation guide" in page.text
    assert MOCK_PAPER["url"] in page.text
    assert "Key ideas" in page.text
    assert (tmp_path / sid / "preparation" / "0.json").exists()


def test_prepare_page_reads_saved_content_without_calling_llm(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Cached briefing",
        "sections": [{"title": "Cached section", "content": "Cached content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)

    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)

    first = client.get(f"/prepare/{sid}/0").text
    second = client.get(f"/prepare/{sid}/0").text
    assert "Cached briefing" in first
    assert "Cached section" in first
    assert "Cached content" in first
    assert "Cached briefing" in second
    assert "Cached section" in second
    assert calls["generate"] == 1


def test_regenerate_preparation_overwrites_cache(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    first = {
        "short_briefing": "First briefing",
        "sections": [{"title": "First section", "content": "First content."}],
    }
    _prepare_mocks(monkeypatch, generation=first)
    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)

    second = {
        "short_briefing": "Second briefing",
        "sections": [{"title": "Second section", "content": "Second content."}],
    }
    _prepare_mocks(monkeypatch, generation=second)
    r = client.post(f"/prepare/{sid}/0/regenerate", data={"arxiv_query": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)

    page = client.get(f"/prepare/{sid}/0").text
    assert "Second briefing" in page
    assert "Second section" in page
    assert "First briefing" not in page
    assert "First section" not in page


def test_manual_arxiv_identifier_skips_search(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing",
        "sections": [{"title": "Section", "content": "Content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)

    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": "1706.03762"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert calls["find"] == ["1706.03762"]


def test_manual_pdf_url_skips_search(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing",
        "sections": [{"title": "Section", "content": "Content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)

    pdf_url = "https://arxiv.org/pdf/1706.03762v3"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": pdf_url},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert calls["find"] == [pdf_url]


def test_find_paper_accepts_pdf_url(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        content = (
            "<feed><entry>"
            "<id>http://arxiv.org/abs/1706.03762v3</id>"
            "<title>Attention Is All You Need</title>"
            "<summary>Abstract text.</summary>"
            "<author><name>Ashish Vaswani</name></author>"
            "<published>2017-06-12T00:00:00Z</published>"
            "<link title=\"pdf\" href=\"https://arxiv.org/pdf/1706.03762v3\"/>"
            "</entry></feed>"
        ).encode()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params):
            captured["url"] = url
            captured["params"] = params
            return FakeResponse()

    monkeypatch.setattr(arxiv.httpx, "Client", FakeClient)
    result = arxiv.find_paper("https://arxiv.org/pdf/1706.03762v3")
    assert captured["params"]["id_list"] == "1706.03762v3"
    assert captured["params"]["max_results"] == 1
    assert result["arxiv_id"] == "1706.03762v3"


def test_external_pdf_url_is_used_directly(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "External PDF briefing",
        "sections": [{"title": "External section", "content": "External content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)

    pdf_url = "https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": pdf_url},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert calls["find"] == []
    assert len(calls["extract"]) == 1
    assert calls["extract"][0]["pdf_url"] == pdf_url

    page = client.get(f"/prepare/{sid}/0").text
    assert "External PDF briefing" in page
    assert "Executive briefing" in page
    assert '>PDF</a>' in page


def test_generate_preparation_makes_multiple_requests(monkeypatch):
    responses = [
        {"sections": [
            {"title": "Background"},
            {"title": "Questions Worth Asking the Speaker"},
            {"title": "Method"},
            {"title": "Key Take-Away"},
        ]},
        {"content": "Background content."},
        {"content": "Method content."},
        {"questions": "Questions to ask the speaker.", "take_away": "The key take-away is X.",
         "short_briefing": "A short briefing.", "prerequisites": "Know linear algebra."},
    ]
    captured = []

    class FakeResp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})})()]

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            captured.append(kwargs)
            content = json.dumps(responses[self.calls])
            self.calls += 1
            return FakeResp(content)

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr(llm, "_client", lambda: FakeClient())

    paper = {
        "title": "Deep Learning via Hessian-Free Optimization",
        "authors": ["James Martens"],
        "published": "2010-01-01",
        "categories": ["cs.LG"],
        "abstract": "Abstract.",
        "paper_text": "Paper text " * 100,
        "paper_text_truncated": False,
    }
    result = llm.generate_preparation(paper)

    assert len(captured) == 4
    assert result["short_briefing"] == "A short briefing."
    assert result["prerequisites"] == "Know linear algebra."
    assert result["sections"] == [
        {"title": "Background", "content": "Background content."},
        {"title": "Method", "content": "Method content."},
        {"title": "Questions Worth Asking the Speaker", "content": "Questions to ask the speaker."},
        {"title": "Key Take-Away", "content": "The key take-away is X."},
    ]
    # First call asks for section titles only.
    assert "sections" in json.loads(json.dumps(responses[0]))
    # Section content calls each include its own title in the user message.
    assert "Background" in captured[1]["messages"][1]["content"]
    assert "Method" in captured[2]["messages"][1]["content"]
    # Every call shares the same system prompt and paper text; only the trailing task differs.
    assert len({c["messages"][0]["content"] for c in captured}) == 1
    paper_prefix = captured[0]["messages"][1]["content"].split("\n\nTask:")[0]
    assert all(c["messages"][1]["content"].startswith(paper_prefix) for c in captured)
    # One closing request writes both fixed sections and the briefing.
    assert '"questions"' in captured[3]["messages"][1]["content"]


def test_prerequisites_rendered_from_cache(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing",
        "prerequisites": "Linear algebra and optimization basics.",
        "sections": [{"title": "Section", "content": "Content."}],
    }
    _prepare_mocks(monkeypatch, generation=prep)
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)

    page = client.get(f"/prepare/{sid}/0").text
    assert "Prerequisites" in page
    assert "Linear algebra and optimization basics." in page


def test_preparation_markdown_rendered(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing **bold**.",
        "prerequisites": "- prerequisite one\n- prerequisite two",
        "sections": [
            {"title": "Key ideas",
             "content": "**Key idea**: transformers use *attention*.\n\n- item one\n- item two"},
        ],
    }
    _prepare_mocks(monkeypatch, generation=prep)
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)

    page = client.get(f"/prepare/{sid}/0").text
    assert "<strong>Key idea</strong>" in page
    assert "<em>attention</em>" in page
    assert "<li>item one</li>" in page


def test_markdown_filter_escapes_html():
    rendered = main.render_markdown("<script>alert(1)</script>**bold**")
    assert "<script>" not in rendered
    assert "<strong>bold</strong>" in rendered


def test_prompt_files_load_from_markdown():
    filenames = [
        "schedule.md",
        "preparation_sections.md",
        "preparation_section.md",
        "preparation_closing.md",
        "preparation_common.md",
    ]
    for filename in filenames:
        prompt_file = llm.PROMPT_DIR / filename
        assert prompt_file.is_file(), f"missing prompt file: {prompt_file}"

    constants = [
        llm.SYSTEM_PROMPT,
        llm.PREPARATION_SYSTEM_PROMPT,
        llm.PREPARATION_SECTION_SYSTEM_PROMPT,
        llm.PREPARATION_CLOSING_SYSTEM_PROMPT,
        llm.PREPARATION_COMMON_PROMPT,
    ]
    assert all(isinstance(constant, str) and constant.strip() for constant in constants)


def test_sections_are_saved_individually(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing",
        "sections": [
            {"title": "First section", "content": "First body."},
            {"title": "Second section", "content": "Second body."},
        ],
    }
    _prepare_mocks(monkeypatch, generation=prep)
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)

    section_dir = tmp_path / sid / "preparation" / "0.sections"
    assert (section_dir / "0.json").is_file()
    assert (section_dir / "1.json").is_file()
    first = json.loads((section_dir / "0.json").read_text())
    second = json.loads((section_dir / "1.json").read_text())
    assert first == {"title": "First section", "content": "First body."}
    assert second == {"title": "Second section", "content": "Second body."}
    assert storage.load_sections(sid, 0)[:2] == [first, second]


def test_progress_state_tracks_requests(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing",
        "prerequisites": "Know linear algebra.",
        "sections": [
            {"title": "Alpha", "content": "Alpha content."},
            {"title": "Beta", "content": "Beta content."},
        ],
    }
    _prepare_mocks(monkeypatch, generation=prep)
    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)

    state = storage.load_generation_state(sid, 0)
    assert state["status"] == "complete"
    assert state["completed_requests"] == 4
    assert state["total_requests"] == 4
    assert state["section_done"] == [True, True, True, True]
    assert [section["title"] for section in storage.load_sections(sid, 0)] == [
        "Alpha",
        "Beta",
        "Questions Worth Asking the Speaker",
        "Key Take-Away",
    ]


def test_section_content_requests_receive_other_titles(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing",
        "sections": [
            {"title": "Alpha", "content": "Alpha content."},
            {"title": "Beta", "content": "Beta content."},
        ],
    }
    _prepare_mocks(monkeypatch, generation=prep)
    captured = {}

    def content(paper, title, other_titles=None):
        captured[("ai", title)] = other_titles
        return f"{title} content."

    def closing(paper, other_titles=None):
        captured["closing"] = other_titles
        return {"questions": "Q.", "take_away": "T.", "short_briefing": "Briefing",
                "prerequisites": ""}

    monkeypatch.setattr(llm, "generate_preparation_section_content", content)
    monkeypatch.setattr(llm, "generate_preparation_closing", closing)

    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)

    fixed_titles = ["Questions Worth Asking the Speaker", "Key Take-Away"]
    assert captured[("ai", "Alpha")] == ["Beta", *fixed_titles]
    assert captured[("ai", "Beta")] == ["Alpha", *fixed_titles]
    assert captured["closing"] == ["Alpha", "Beta"]


def test_prepare_generates_sections_concurrently(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    ai_sections = [
        {"title": "Alpha", "content": "Alpha content."},
        {"title": "Beta", "content": "Beta content."},
        {"title": "Gamma", "content": "Gamma content."},
    ]
    prep = {
        "short_briefing": "Briefing",
        "sections": ai_sections,
    }
    _prepare_mocks(monkeypatch, generation=prep)
    base_section_content = llm.generate_preparation_section_content
    ai_barrier = threading.Barrier(len(ai_sections))

    def concurrent_section_content(paper, title, other_titles=None):
        try:
            ai_barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            raise AssertionError("AI sections did not run concurrently")
        return base_section_content(paper, title, other_titles)

    monkeypatch.setattr(llm, "generate_preparation_section_content", concurrent_section_content)

    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)

    state = storage.load_generation_state(sid, 0)
    assert state["status"] == "complete"
    assert state["section_done"] == [True, True, True, True, True]
    page = client.get(f"/prepare/{sid}/0").text
    for title in [s["title"] for s in ai_sections]:
        assert title in page


def test_pause_during_generation_resumes_remaining(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    ai_sections = [
        {"title": "Alpha", "content": "Alpha content."},
        {"title": "Beta", "content": "Beta content."},
        {"title": "Gamma", "content": "Gamma content."},
        {"title": "Delta", "content": "Delta content."},
    ]
    prep = {
        "short_briefing": "Briefing",
        "sections": ai_sections,
    }
    _prepare_mocks(monkeypatch, generation=prep)
    base_section_content = llm.generate_preparation_section_content
    ai_titles = {s["title"] for s in ai_sections}
    ai_barrier = threading.Barrier(len(ai_sections))
    release_ai = threading.Event()
    all_ai_started = threading.Event()

    def gated_section_content(paper, title, other_titles=None):
        if title in ai_titles:
            try:
                arrived = ai_barrier.wait(timeout=5)
            except threading.BrokenBarrierError:
                raise AssertionError("AI sections did not run concurrently")
            if arrived == 0:
                all_ai_started.set()
            if not release_ai.wait(timeout=5):
                raise TimeoutError("timed out waiting to release AI sections")
        return base_section_content(paper, title, other_titles)

    monkeypatch.setattr(llm, "generate_preparation_section_content", gated_section_content)

    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                follow_redirects=False)
    assert all_ai_started.wait(timeout=5)

    r = client.post(f"/prepare/{sid}/0/pause", follow_redirects=False)
    assert r.status_code == 303

    release_ai.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        state = storage.load_generation_state(sid, 0)
        if state and state.get("status") == "paused" and len(storage.load_sections(sid, 0)) == len(ai_sections):
            break
        time.sleep(0.01)
    else:
        raise AssertionError(
            "generation did not pause after the in-flight AI sections completed: "
            f"state={storage.load_generation_state(sid, 0)!r} "
            f"sections={storage.load_sections(sid, 0)!r}")

    state = storage.load_generation_state(sid, 0)
    assert state["status"] == "paused"
    assert state["section_done"] == [True] * len(ai_sections) + [False, False]

    r = client.post(f"/prepare/{sid}/0/resume", follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)

    state = storage.load_generation_state(sid, 0)
    assert state["status"] == "complete"
    assert (tmp_path / sid / "preparation" / "0.json").is_file()
    assert [section["title"] for section in storage.load_sections(sid, 0)] == [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Questions Worth Asking the Speaker",
        "Key Take-Away",
    ]


def test_pause_without_running_generation_returns_409(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    r = client.post(f"/prepare/{sid}/0/pause", follow_redirects=False)
    assert r.status_code == 409


def test_prepare_page_shows_progress_and_sections(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    storage.save_generation_state(sid, 0, {
        "status": "running",
        "completed_requests": 3,
        "total_requests": 8,
        "section_done": [True, True, False, False],
        "section_titles": ["Background", "Method"],
        "duration_seconds": 2.5,
    })
    storage.save_section(sid, 0, 0, {"title": "Background", "content": "Background content."})
    storage.save_section(sid, 0, 1, {"title": "Method", "content": "Method content."})

    page = client.get(f"/prepare/{sid}/0").text
    assert "3 of 8 AI requests complete" in page
    assert "progress-bar-fill" in page
    assert "Background" in page
    assert "Method" in page
    assert "Duration" in page
    assert "2.5s" in page


def test_prepare_page_shows_section_titles_before_content(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    storage.save_generation_state(sid, 0, {
        "status": "running",
        "completed_requests": 1,
        "total_requests": 8,
        "section_done": [False, False, False, False],
        "section_titles": ["Background", "Method"],
    })

    page = client.get(f"/prepare/{sid}/0").text
    assert "Sections" in page
    assert "Background" in page
    assert "Method" in page


def test_prepare_page_shows_tokens_consumed(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    storage.save_preparation(sid, 0, {
        "source": {"title": "Attention Is All You Need", "authors": []},
        "sections": [{"title": "Section", "content": "Content."}],
        "tokens": {"prompt_tokens": 2000, "completion_tokens": 2321, "total_tokens": 4321},
        "duration_seconds": 4.3,
    })

    page = client.get(f"/prepare/{sid}/0").text
    assert "Tokens consumed" in page
    assert "4321" in page
    assert "Duration" in page
    assert "4.3s" in page


def test_clear_preparation_removes_content(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Cached briefing",
        "sections": [{"title": "Clear me", "content": "Generated content."}],
    }
    _prepare_mocks(monkeypatch, generation=prep)
    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""},
                follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert storage.load_preparation(sid, 0) is not None
    assert storage.load_sections(sid, 0) != []

    r = client.post(f"/prepare/{sid}/0/clear", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == f"/prepare/{sid}/0"

    assert storage.load_preparation(sid, 0) is None
    assert storage.load_sections(sid, 0) == []
    assert storage.load_generation_state(sid, 0) is None

    page = client.get(f"/prepare/{sid}/0").text
    assert "No preparation guide yet." in page
    assert "Generated content." not in page


def _fake_html_client(monkeypatch, body, content_type="text/html"):
    class FakeResponse:
        status_code = 200
        headers = {"content-type": content_type}
        content = body

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(arxiv.httpx, "Client", FakeClient)


def test_extract_paper_extracts_text_from_html_response(monkeypatch):
    body = (
        "<html><body><article><h1>Article body</h1><p>"
        + "This is the article text. " * 60
        + "</p></article></body></html>"
    ).encode()
    _fake_html_client(monkeypatch, body, content_type="text/html; charset=UTF-8")

    result = arxiv.extract_paper({
        "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4.pdf",
        "title": "A Nature article",
        "arxiv_id": "",
        "url": "",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
    })

    assert "Article body" in result["paper_text"]
    assert "This is the article text." in result["paper_text"]
    assert len(result["paper_text"]) >= 500
    assert result["paper_text_truncated"] is False


def test_extract_paper_rejects_small_html_response(monkeypatch):
    body = b"<html><body><p>This page does not contain a full paper.</p></body></html>"
    _fake_html_client(monkeypatch, body)

    with pytest.raises(RuntimeError, match="not a readable PDF"):
        arxiv.extract_paper({
            "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4.pdf",
            "title": "A Nature article",
        })


def test_extract_paper_doctype_html_has_clear_error(monkeypatch):
    body = (
        b"<!DOCTYPE html><html><body>"
        b"<p>JavaScript is required to view this page.</p>"
        b"</body></html>"
    )
    _fake_html_client(monkeypatch, body)

    with pytest.raises(RuntimeError, match="not a readable PDF") as excinfo:
        arxiv.extract_paper({
            "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4.pdf",
            "title": "A Nature article",
        })
    assert "text/html" in str(excinfo.value)


def test_prepare_error_saves_generation_state_without_name_error(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    def raise_unreadable_pdf(_paper):
        raise RuntimeError("Downloaded file is not a readable PDF (content-type: text/html).")

    monkeypatch.setattr(arxiv, "extract_paper", raise_unreadable_pdf)

    pdf_url = "https://www.nature.com/articles/s41586-025-09142-4.pdf"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": pdf_url},
                    follow_redirects=False)
    assert r.status_code == 303

    state = _wait_state_status(tmp_path, sid, 0, "error")
    assert state["status"] == "error"
    assert "not a readable PDF" in state["error"]


def test_is_article_url(monkeypatch):
    assert arxiv.is_article_url("https://www.nature.com/articles/s41586-025-09142-4")
    assert arxiv.is_article_url("https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf")
    assert not arxiv.is_article_url("https://arxiv.org/abs/1706.03762")
    assert not arxiv.is_article_url("https://arxiv.org/pdf/1706.03762v3")
    assert not arxiv.is_article_url("Attention Is All You Need")


def test_extract_paper_fetches_with_browser_headers(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html"}
        content = (
            "<html><head><title>Attention Is All You Need</title></head><body><p>"
            + "This is the article text. " * 60
            + "</p></body></html>"
        ).encode()

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(arxiv.httpx, "Client", FakeClient)

    arxiv.extract_paper({
        "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4",
        "title": "Attention Is All You Need",
    })

    assert captured["kwargs"]["follow_redirects"] is True
    assert "User-Agent" in captured["kwargs"]["headers"]
    assert "Mozilla/5.0" in captured["kwargs"]["headers"]["User-Agent"]


def test_extract_paper_html_article_accepts_matching_title(monkeypatch):
    title = "Attention Is All You Need"
    body = (
        f"<html><head><title>{title} | Nature</title></head><body><p>"
        + "This is the article text. " * 60
        + "</p></body></html>"
    ).encode()
    _fake_html_client(monkeypatch, body)

    result = arxiv.extract_paper({
        "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4",
        "title": title,
        "arxiv_id": "",
        "url": "",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
    })

    assert "This is the article text." in result["paper_text"]
    assert len(result["paper_text"]) >= 500


def test_extract_paper_html_article_rejects_mismatched_title(monkeypatch):
    body = (
        "<html><head><title>A Different Paper | Nature</title></head><body><p>"
        + "This is the article text. " * 60
        + "</p></body></html>"
    ).encode()
    _fake_html_client(monkeypatch, body)

    with pytest.raises(RuntimeError, match="does not match"):
        arxiv.extract_paper({
            "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4",
            "title": "Attention Is All You Need",
        })


def test_html_article_url_skips_search(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "HTML article briefing",
        "sections": [{"title": "Section", "content": "Content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)

    html_url = "https://www.nature.com/articles/s41586-025-09142-4"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": html_url},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert calls["find"] == []
    assert len(calls["extract"]) == 1
    assert calls["extract"][0]["pdf_url"] == html_url

    page = client.get(f"/prepare/{sid}/0").text
    assert "HTML article briefing" in page
    assert "Executive briefing" in page


def test_non_arxiv_article_fills_metadata_from_llm(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Non-arXiv briefing",
        "sections": [{"title": "Section", "content": "Content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)
    captured = {}

    def fake_metadata(paper):
        captured["paper"] = paper
        return {
            "title": "",
            "authors": ["Jane Doe"],
            "published": "2025-03-04",
            "categories": ["Machine Learning"],
        }

    monkeypatch.setattr(llm, "extract_paper_metadata", fake_metadata)

    html_url = "https://www.nature.com/articles/s41586-025-09142-4"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": html_url},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert calls["find"] == []
    assert len(calls["extract"]) == 1
    assert captured["paper"]["paper_text"]

    state = storage.load_generation_state(sid, 0)
    assert state["status"] == "complete"
    assert state["source"]["authors"] == ["Jane Doe"]
    assert state["source"]["published"] == "2025-03-04"
    assert state["source"]["categories"] == ["Machine Learning"]

    page = client.get(f"/prepare/{sid}/0").text
    assert "Jane Doe" in page
    assert "2025-03-04" in page
    assert "Machine Learning" in page


def test_non_arxiv_article_uses_html_metadata_when_llm_empty(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "HTML metadata briefing",
        "sections": [{"title": "Section", "content": "Content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)

    def fake_extract_with_meta(found):
        extracted = _paper_with_text(found)
        extracted["html_metadata"] = {
            "authors": ["Qianyue Hao", "Fengli Xu"],
            "published": "2026-01-14",
            "categories": ["Machine Learning"],
        }
        return extracted

    monkeypatch.setattr(arxiv, "extract_paper", fake_extract_with_meta)
    monkeypatch.setattr(
        llm, "extract_paper_metadata",
        lambda _paper: {"title": "", "authors": [], "published": "", "categories": []},
    )

    html_url = "https://www.nature.com/articles/s41586-025-09922-y"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": html_url},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert calls["find"] == []

    state = storage.load_generation_state(sid, 0)
    assert state["status"] == "complete"
    assert state["source"]["authors"] == ["Qianyue Hao", "Fengli Xu"]
    assert state["source"]["published"] == "2026-01-14"
    assert state["source"]["categories"] == ["Machine Learning"]


def test_non_arxiv_article_metadata_failure_keeps_empty_fields(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    prep = {
        "short_briefing": "Briefing despite metadata failure",
        "sections": [{"title": "Section", "content": "Content."}],
    }
    calls = _prepare_mocks(monkeypatch, generation=prep)

    def fail_metadata(paper):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(llm, "extract_paper_metadata", fail_metadata)

    html_url = "https://www.nature.com/articles/s41586-025-09142-4"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": html_url},
                    follow_redirects=False)
    assert r.status_code == 303
    assert _wait_preparation_done(tmp_path, sid, 0)
    assert calls["find"] == []

    state = storage.load_generation_state(sid, 0)
    assert state["status"] == "complete"
    assert state["source"]["authors"] == []
    assert state["source"]["published"] == ""
    assert state["source"]["categories"] == []


def test_html_article_title_mismatch_is_persisted(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    body = (
        "<html><head><title>A Different Paper | Nature</title></head><body><p>"
        + "This is the article text. " * 60
        + "</p></body></html>"
    ).encode()
    _fake_html_client(monkeypatch, body)

    html_url = "https://www.nature.com/articles/s41586-025-09142-4"
    r = client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": html_url},
                    follow_redirects=False)
    assert r.status_code == 303

    state = _wait_state_status(tmp_path, sid, 0, "error")
    assert state["status"] == "error"
    assert "does not match" in state["error"]


def test_strip_references_cuts_reference_list_and_appendix():
    body = "Intro and method text. " * 100
    text = f"# Title\n\n{body}\n\n## References\n\n[1] A. Author. A paper.\n\n## Appendix A\n\nProofs."
    stripped = arxiv._strip_references(text)
    assert stripped.endswith("method text.")
    assert "[1] A. Author" not in stripped
    assert "Proofs" not in stripped


def test_strip_references_ignores_early_and_missing_headings():
    early = "References\n" + "Body text. " * 200
    assert arxiv._strip_references(early) == early
    plain = "Body text. " * 200
    assert arxiv._strip_references(plain) == plain
    inline = "Body text. " * 100 + "see the references section of this paper. " + "More body. " * 100
    assert arxiv._strip_references(inline) == inline


def test_closing_request_runs_once_per_guide(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)
    prep = {"short_briefing": "Briefing", "sections": [{"title": "Alpha", "content": "Alpha content."}]}
    _prepare_mocks(monkeypatch, generation=prep)
    closing_calls = []
    base_closing = llm.generate_preparation_closing

    def counting_closing(paper, other_titles=None):
        closing_calls.append(other_titles)
        return base_closing(paper, other_titles)

    monkeypatch.setattr(llm, "generate_preparation_closing", counting_closing)
    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""}, follow_redirects=False)
    assert _wait_preparation_done(tmp_path, sid, 0)

    assert closing_calls == [["Alpha"]]
    state = storage.load_generation_state(sid, 0)
    assert state["short_briefing"] == "Briefing"
    assert state["completed_requests"] == state["total_requests"] == 3


def test_prepare_status_endpoint_reports_progress(tmp_path, monkeypatch):
    client = _make_client(tmp_path, monkeypatch)
    sid = _create_session(client)

    idle = client.get(f"/prepare/{sid}/0/status").json()
    assert idle["status"] is None and idle["active"] is False

    prep = {
        "short_briefing": "Brief.",
        "sections": [{"title": "Key ideas", "content": "Self-attention."}],
    }
    _prepare_mocks(monkeypatch, generation=prep)
    client.post(f"/prepare/{sid}/0/generate", data={"arxiv_query": ""}, follow_redirects=False)
    _wait_preparation_done(tmp_path, sid, 0)

    done = client.get(f"/prepare/{sid}/0/status").json()
    assert done["status"] == "complete" and done["active"] is False
    assert done["completed_requests"] == done["total_requests"] > 0
    assert client.get(f"/prepare/{sid}/9/status").status_code == 404
