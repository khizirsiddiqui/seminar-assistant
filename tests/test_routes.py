from fastapi.testclient import TestClient

from app import storage
from app import llm as llm_mod


def make_client(tmp_path, monkeypatch, rows):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    monkeypatch.setattr(llm_mod, "parse_schedule", lambda text, assume_year=None: rows)
    from app.main import app
    return TestClient(app)


def test_full_flow(tmp_path, monkeypatch):
    rows = [{"date": "2026-10-27", "student": "Alice", "paper": "Attention Is All You Need"},
            {"date": "2026-11-03", "student": "Bob", "paper": "BERT"}]
    client = make_client(tmp_path, monkeypatch, rows)
    r = client.post("/parse", data={"pasted_text": "schedule text", "venue": "Sand 14",
                                    "assume_year": "2026", "start_time": "14:15", "end_time": "15:45"},
                    follow_redirects=False)
    assert r.status_code == 303
    sid = r.headers["location"].split("/")[-1]
    # confirm page shows all rows checked
    page = client.get(f"/confirm/{sid}").text
    assert "Alice" in page and "checked" in page
    # generate with only row 0 selected
    r = client.post(f"/generate/{sid}", data={"row_index": ["0"], "venue": "Sand 14",
                                              "start_time": "14:15", "end_time": "15:45"},
                    follow_redirects=False)
    assert r.status_code == 303
    ics = client.get(f"/download/{sid}").text
    assert "Attention Is All You Need" in ics and "BERT" not in ics
    result = client.get(f"/result/{sid}").text
    assert "Prepare" in result
    prep = client.get(f"/prepare/{sid}/0").text
    assert "Alice" in prep and "later" in prep
    assert client.get("/history").status_code == 200


def test_parse_requires_input(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, [])
    r = client.post("/parse", data={"pasted_text": "", "venue": "", "assume_year": "2026",
                                    "start_time": "14:15", "end_time": "15:45"})
    assert r.status_code == 422
