import pytest

from app import storage


def test_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    sid = storage.create_session(raw_text="raw", venue="V", start_time="14:15",
                                 end_time="15:45", source_filename="s.pdf",
                                 course_title="Seminar: Advanced ML")
    rows = [{"date": "2026-10-27", "student": "A", "paper": "P"}]
    storage.save_parsed(sid, rows)
    storage.save_selection(sid, rows)
    storage.save_ics(sid, "BEGIN:VCALENDAR...")
    meta = storage.get_session(sid)
    assert meta["venue"] == "V"
    assert meta["course_title"] == "Seminar: Advanced ML"
    assert storage.load_parsed(sid) == rows
    assert storage.load_ics(sid) == "BEGIN:VCALENDAR..."
    assert storage.list_sessions() == [sid]


def test_missing_session_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DATA_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.get_session("nope")
