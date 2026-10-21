from fastapi.testclient import TestClient

from app import storage
from app import llm as llm_mod


def make_client(tmp_path, monkeypatch, rows, course_title=""):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        llm_mod, "parse_schedule",
        lambda text, assume_year=None: {"course_title": course_title, "sessions": rows})
    from app.main import app
    return TestClient(app)


def test_full_flow(tmp_path, monkeypatch):
    rows = [{"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need",
             "paper_url": "https://arxiv.org/abs/1706.03762"},
            {"date": "2026-10-27", "student": "Carol", "paper": "BERT"},
            {"date": "2026-11-03", "student": "Bob", "paper": "GPT-3"}]
    client = make_client(tmp_path, monkeypatch, rows, course_title="Seminar: Advanced ML")
    r = client.post("/parse", data={"pasted_text": "schedule text", "venue": "Sand 14",
                                    "assume_year": "2026", "start_time": "14:15", "end_time": "15:45"},
                    follow_redirects=False)
    assert r.status_code == 303
    sid = r.headers["location"].split("/")[-1]
    # confirm page shows all rows checked and the LLM course title pre-filled
    page = client.get(f"/confirm/{sid}").text
    assert "Alice" in page and "checked" in page
    assert 'value="Seminar: Advanced ML"' in page
    assert 'href="https://arxiv.org/abs/1706.03762"' in page
    assert ">link</a>" in page
    # generate with both same-date rows selected
    r = client.post(f"/generate/{sid}", data={"row_index": ["0", "1"], "venue": "Sand 14",
                                              "start_time": "14:15", "end_time": "15:45",
                                              "course_title": "Seminar: Advanced ML"},
                    follow_redirects=False)
    assert r.status_code == 303
    ics = client.get(f"/download/{sid}").text
    assert ics.count("BEGIN:VEVENT") == 1
    assert "Attention Is All You Need" in ics and "BERT" in ics and "GPT-3" not in ics
    assert "SUMMARY:Seminar: Advanced ML" in ics
    result = client.get(f"/session/{sid}").text
    assert "Prepare" in result
    old_result = client.get(f"/result/{sid}", follow_redirects=False)
    assert old_result.status_code == 303
    assert old_result.headers["location"] == f"/session/{sid}"
    prep = client.get(f"/prepare/{sid}/0").text
    assert "Alice" in prep and "guide" in prep
    assert 'value="https://arxiv.org/abs/1706.03762"' in prep
    assert client.get("/history").status_code == 200


def test_prepare_button_color_by_status(tmp_path, monkeypatch):
    rows = [{"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need"}]
    client = make_client(tmp_path, monkeypatch, rows)
    r = client.post("/parse", data={"pasted_text": "schedule text", "venue": "Sand 14",
                                    "assume_year": "2026", "start_time": "14:15", "end_time": "15:45"},
                    follow_redirects=False)
    sid = r.headers["location"].split("/")[-1]
    r = client.post(f"/generate/{sid}", data={"row_index": "0", "venue": "Sand 14",
                                              "start_time": "14:15", "end_time": "15:45",
                                              "course_title": ""},
                    follow_redirects=False)
    assert r.status_code == 303

    page = client.get(f"/session/{sid}").text
    assert 'class="button secondary small" href="/prepare/{}/0"'.format(sid) in page

    storage.save_generation_state(sid, 0, {"status": "running"})
    page = client.get(f"/session/{sid}").text
    assert 'class="button prepare-fetching small" href="/prepare/{}/0"'.format(sid) in page

    storage.save_preparation(sid, 0, {"sections": []})
    page = client.get(f"/session/{sid}").text
    assert 'class="button success small" href="/prepare/{}/0"'.format(sid) in page


def test_parse_requires_input(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, [])
    r = client.post("/parse", data={"pasted_text": "", "venue": "", "assume_year": "2026",
                                    "start_time": "14:15", "end_time": "15:45"})
    assert r.status_code == 422
