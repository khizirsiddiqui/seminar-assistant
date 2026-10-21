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


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def create_session(raw_text: str, venue: str, start_time: str, end_time: str,
                   source_filename: str, course_title: str = "") -> str:
    sid = uuid.uuid4().hex[:12]
    d = DATA_DIR / sid
    d.mkdir(parents=True, exist_ok=False)
    meta = {"id": sid, "created_at": datetime.now(timezone.utc).isoformat(),
            "venue": venue, "start_time": start_time, "end_time": end_time,
            "source_filename": source_filename, "course_title": course_title}
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


def save_preparation(sid: str, row_index: int, preparation: dict) -> None:
    prep_dir = _dir(sid) / "preparation"
    prep_dir.mkdir(parents=True, exist_ok=True)
    _write_json(prep_dir / f"{row_index}.json", preparation)


def load_preparation(sid: str, row_index: int) -> dict | None:
    prep_file = _dir(sid) / "preparation" / f"{row_index}.json"
    if not prep_file.is_file():
        return None
    return json.loads(prep_file.read_text())


def clear_preparation(sid: str, row_index: int) -> None:
    prep_file = _dir(sid) / "preparation" / f"{row_index}.json"
    if prep_file.is_file():
        prep_file.unlink()


def _preparation_dir(sid: str, row_index: int) -> Path:
    prep_dir = _dir(sid) / "preparation"
    prep_dir.mkdir(parents=True, exist_ok=True)
    return prep_dir


def save_generation_state(sid: str, row_index: int, state: dict) -> None:
    state_file = _preparation_dir(sid, row_index) / f"{row_index}.state.json"
    _write_json(state_file, state)


def load_generation_state(sid: str, row_index: int) -> dict | None:
    state_file = _dir(sid) / "preparation" / f"{row_index}.state.json"
    if not state_file.is_file():
        return None
    return json.loads(state_file.read_text())


def save_section(sid: str, row_index: int, section_index: int, section: dict) -> None:
    sections_dir = _preparation_dir(sid, row_index) / f"{row_index}.sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    _write_json(sections_dir / f"{section_index}.json", section)


def load_sections(sid: str, row_index: int) -> list[dict]:
    sections_dir = _dir(sid) / "preparation" / f"{row_index}.sections"
    if not sections_dir.is_dir():
        return []
    files = sorted(sections_dir.glob("*.json"), key=lambda p: int(p.stem))
    return [json.loads(p.read_text()) for p in files]


def clear_sections(sid: str, row_index: int) -> None:
    sections_dir = _dir(sid) / "preparation" / f"{row_index}.sections"
    if sections_dir.is_dir():
        import shutil
        shutil.rmtree(sections_dir)


def clear_generation_state(sid: str, row_index: int) -> None:
    state_file = _dir(sid) / "preparation" / f"{row_index}.state.json"
    if state_file.is_file():
        state_file.unlink()


def pause_generation(sid: str, row_index: int) -> bool:
    state = load_generation_state(sid, row_index)
    if state is None or state.get("status") != "running":
        return False
    state["status"] = "paused"
    save_generation_state(sid, row_index, state)
    return True


def resume_generation(sid: str, row_index: int) -> bool:
    state = load_generation_state(sid, row_index)
    if state is None or state.get("status") != "paused":
        return False
    state["status"] = "running"
    save_generation_state(sid, row_index, state)
    return True


def start_generation(sid: str, row_index: int) -> None:
    prep_dir = _dir(sid) / "preparation"
    prep_dir.mkdir(parents=True, exist_ok=True)
    prep_file = prep_dir / f"{row_index}.json"
    if prep_file.exists():
        prep_file.unlink()
    (prep_dir / f"{row_index}.status.json").write_text(
        json.dumps({"status": "generating"}))


def save_generation_progress(sid: str, row_index: int, sections_count: int) -> None:
    prep_dir = _dir(sid) / "preparation"
    status_file = prep_dir / f"{row_index}.status.json"
    status = {"status": "generating", "sections_count": sections_count}
    status_file.write_text(json.dumps(status))


def complete_generation(sid: str, row_index: int) -> None:
    status_file = _dir(sid) / "preparation" / f"{row_index}.status.json"
    if status_file.exists():
        status_file.unlink()


def fail_generation(sid: str, row_index: int, message: str) -> None:
    prep_dir = _dir(sid) / "preparation"
    prep_dir.mkdir(parents=True, exist_ok=True)
    (prep_dir / f"{row_index}.status.json").write_text(
        json.dumps({"status": "error", "message": message}))


def load_generation_status(sid: str, row_index: int) -> dict | None:
    status_file = _dir(sid) / "preparation" / f"{row_index}.status.json"
    if not status_file.is_file():
        return None
    return json.loads(status_file.read_text())


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
