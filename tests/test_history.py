from fastapi.testclient import TestClient

from app import storage
from app import llm as llm_mod


def make_client(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        llm_mod, "parse_schedule",
        lambda text, assume_year=None: {"course_title": "", "sessions": rows})
    from app.main import app
    return TestClient(app)


def _parse(client):
    r = client.post("/parse", data={"pasted_text": "schedule text", "venue": "Sand 14",
                                    "assume_year": "2026", "start_time": "14:15", "end_time": "15:45"},
                    follow_redirects=False)
    return r.headers["location"].split("/")[-1]


def test_parsed_only_session_is_resumable(tmp_path, monkeypatch):
    """A session parsed but never generated: appears in history, /session/{sid} works,
    shows parsed rows, offers Resume (confirm) and Delete."""
    rows = [{"date": "2026-10-27", "student": "Alice", "paper": "P1"}]
    client = make_client(tmp_path, monkeypatch, rows)
    sid = _parse(client)

    # history lists it with status "parsed"
    page = client.get("/history").text
    assert sid in page and "parsed" in page.lower()

    # session detail works without selected.json
    detail = client.get(f"/session/{sid}")
    assert detail.status_code == 200
    assert "Alice" in detail.text
    assert "parsed" in detail.text.lower() and "resume" in detail.text.lower()
    assert f'/confirm/{sid}' in detail.text
    assert "delete" in detail.text.lower()
    # generated-state assertions belong in test_generated_session_detail
    detail = client.get(f"/session/{sid}")
    assert detail.status_code == 200
    assert "Alice" in detail.text
    assert "delete" in detail.text.lower()


def test_delete_session(tmp_path, monkeypatch):
    rows = [{"date": "2026-10-27", "student": "Alice", "paper": "P1"}]
    client = make_client(tmp_path, monkeypatch, rows)
    sid = _parse(client)

    # delete via POST
    r = client.post(f"/delete/{sid}", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/history"

    # gone from history and disk
    assert sid not in client.get("/history").text
    assert not (tmp_path / sid).exists()
    assert client.get(f"/session/{sid}").status_code == 404
    assert client.get(f"/confirm/{sid}").status_code == 404


def test_delete_missing_session(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, [])
    assert client.post("/delete/nope").status_code == 404


def test_delete_uses_post_only(tmp_path, monkeypatch):
    """GET /delete must not delete (CSRF/browser-prefetch safety)."""
    rows = [{"date": "2026-10-27", "student": "Alice", "paper": "P1"}]
    client = make_client(tmp_path, monkeypatch, rows)
    sid = _parse(client)
    assert client.get(f"/delete/{sid}").status_code == 405
    assert (tmp_path / sid).exists()


def test_history_is_newest_first_with_counts(tmp_path, monkeypatch):
    rows = [{"date": "2026-10-27", "student": "Alice", "paper": "P1"},
            {"date": "2026-11-03", "student": "Bob", "paper": "P2"}]
    client = make_client(tmp_path, monkeypatch, rows)
    old = _parse(client)
    new = _parse(client)
    storage.update_session_meta(old, created_at="2026-01-01T00:00:00+00:00")
    storage.update_session_meta(new, created_at="2026-06-01T00:00:00+00:00")

    from app.main import _session_summaries
    summaries = _session_summaries()
    assert [s["id"] for s in summaries] == [new, old]
    assert summaries[0]["talk_count"] == 2
    assert summaries[0]["first_date"] == "2026-10-27"
    assert summaries[0]["last_date"] == "2026-11-03"

    page = client.get("/history").text
    assert page.index(new) < page.index(old)
    assert "2 talks" in page
    assert new in client.get("/").text
