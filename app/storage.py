import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data/sessions")


def _dir(sid: str) -> Path:
    d = DATA_DIR / sid
    if not d.is_dir():
        raise FileNotFoundError(f"session {sid} not found")
    return d


def create_session(raw_text: str, venue: str, start_time: str, end_time: str, source_filename: str) -> str:
    sid = uuid.uuid4().hex[:12]
    d = DATA_DIR / sid
    d.mkdir(parents=True, exist_ok=False)
    meta = {"id": sid, "created_at": datetime.now(timezone.utc).isoformat(),
            "venue": venue, "start_time": start_time, "end_time": end_time,
            "source_filename": source_filename}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    (d / "raw.txt").write_text(raw_text, encoding="utf-8")
    return sid


def save_parsed(sid: str, rows: list[dict]) -> None:
    (_dir(sid) / "parsed.json").write_text(json.dumps(rows, indent=2))


def load_parsed(sid: str) -> list[dict]:
    return json.loads((_dir(sid) / "parsed.json").read_text())


def save_selection(sid: str, rows: list[dict]) -> None:
    (_dir(sid) / "selected.json").write_text(json.dumps(rows, indent=2))


def load_selection(sid: str) -> list[dict]:
    return json.loads((_dir(sid) / "selected.json").read_text())


def save_ics(sid: str, ics_text: str) -> None:
    (_dir(sid) / "calendar.ics").write_text(ics_text, encoding="utf-8")


def load_ics(sid: str) -> str:
    return (_dir(sid) / "calendar.ics").read_text()


def get_session(sid: str) -> dict:
    return json.loads((_dir(sid) / "meta.json").read_text())


def update_session_meta(sid: str, **fields) -> None:
    meta = get_session(sid)
    meta.update(fields)
    (_dir(sid) / "meta.json").write_text(json.dumps(meta, indent=2))


def list_sessions() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted((p.name for p in DATA_DIR.iterdir() if p.is_dir()), reverse=True)


def has_selection(sid: str) -> bool:
    return (_dir(sid) / "selected.json").exists()


def delete_session(sid: str) -> None:
    import shutil
    shutil.rmtree(_dir(sid))
