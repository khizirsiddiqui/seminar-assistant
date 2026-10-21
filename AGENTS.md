# Repository Guidelines

## Project Overview

Seminar Guide is a FastAPI web app that turns an uploaded/pasted seminar schedule into an ICS calendar and per-paper AI reading guides. The pipeline is:
parse a schedule with an LLM → let the user pick sessions → generate a calendar → generate a per-paper preparation guide with AI-generated sections.

The app is server-rendered with Jinja2 templates and stores all data as JSON files under `data/sessions/` (no database). LLM calls go through an OpenAI-compatible HTTP API.

## Architecture & Data Flow

High-level flow: **parse → generate (.ics) → prepare (per-paper guide)**.

```
index (/) → POST /parse → /confirm/{sid}
/src/confirm → POST /generate/{sid} → /session/{sid}
/session/{sid} → POST /prepare/{sid}/{row}/generate  → /prepare/{sid}/{row}
```

Key flows:

1. **Parse**: uploaded file or pasted text → `extract_text` (PDF via PyMuPDF, otherwise raw text) → `llm.parse_schedule` (JSON mode) → parsed rows persisted.
2. **Generate**: user selects rows → `ics.build_calendar` groups same-date rows into one `.ics` VEVENT → selection + ICS saved.
3. **Prepare**: a background `threading.Thread` resolves the paper (arXiv search or direct article/PDF URL), downloads/truncates paper text, and:
   - asks the LLM for section titles,
   - appends two fixed sections (`Questions Worth Asking the Speaker`, `Key Take-Away`),
   - runs the AI sections concurrently with `ThreadPoolExecutor(max_workers=4)` via `_generate_one_section`, alongside one closing request (`_generate_closing` → `llm.generate_preparation_closing`) that writes both fixed sections and the executive briefing,
   - saves each section individually under `preparation/<row>.sections/<i>.json`,
   - if not paused, writes the final `preparation/<row>.json` and marks state `complete`.

### Preparation state machine

Generation state lives in `preparation/<row_index>.state.json`:
- `status`: `running` → `paused` / `complete` / `error`
- `section_done`: boolean list, one entry per section including the 2 fixed ones (resume skips `True` AI entries; the two fixed entries are set together by the closing request)
- `completed_requests = 1 + done AI sections + 1 if the closing request is done` (`_completed_requests`)
- `total_requests = 1 + len(section_titles) + 1` (titles + AI sections + closing)

Pause/resume:
- `pause` only flips `running → paused`; in-flight section workers finish and save.
- not-yet-started workers check `status == "paused"` before calling the LLM and skip.
- `resume` flips to `running` and starts a new thread that reuses `section_done`.
- `regenerate` clears cached preparation/sections when the state is terminal.

## Key Directories

| Path | Purpose |
|---|---|
| `app/` | All application code (FastAPI app, LLM client, storage, parsers). |
| `app/prompts/` | Markdown prompt templates consumed by `app/llm.py`. |
| `templates/` | Jinja2 server-rendered UI (extends `base.html`). |
| `tests/` | pytest suite, one module per feature area; `tests/fixtures/sample.pdf` is a binary fixture. |
| `data/sessions/` | Runtime JSON storage (gitignored; created per session as 12-hex dirs). |

## Development Commands

From the repo root (`.venv` is the project virtualenv; `python` may resolve to system Python, prefer `.venv/bin/python`):

```bash
# Install dependencies
.venv/bin/python -m pip install -r requirements.txt

# Run the app (hot reload)
.venv/bin/python -m uvicorn app.main:app --reload

# Run tests
.venv/bin/python -m pytest tests/ -q

# Run one module / one test
.venv/bin/python -m pytest tests/test_prepare.py -q
.venv/bin/python -m pytest tests/test_prepare.py::test_pause_during_generation_resumes_remaining
```

Required env vars (see `.env.example`; loaded by `app/config.py` from `.env`):
- `LLM_BASE_URL` (default `https://api.openai.com/v1`)
- `LLM_API_KEY` (required for real LLM calls)
- `LLM_MODEL` (default `gpt-4o-mini`)
- `ARXIV_TIMEOUT_SECONDS` (default `20`)

## Code Conventions & Common Patterns

### Error handling
- LLM/parse failures that should reach the user are caught in route handlers and rendered with `error.html` (statuses like `404`, `422`, `502`).
- Internal background failures in preparation are caught in `_run_preparation`, stored as `state["status"] = "error"` + `state["error"]`, and rendered on the prepare page.
- Surface user-facing messages (e.g. "not a readable PDF"), not raw tracebacks.

### LLM / threading patterns
- All LLM calls use an OpenAI-compatible client with `response_format={"type": "json_object"}`.
- Token usage is reported through a `threading.local` callback (`llm.set_usage_callback` / `clear_usage_callback`). Worker threads must set/clear the callback themselves because `_usage_state` is thread-local.
- Preparation calls share `PREPARATION_COMMON_PROMPT` + paper text as a common prefix; only the trailing task prompt varies (keeps provider prompt-prefix caches reusable).

### Concurrency / state management
- `_active_generations` + `_active_generations_lock` prevent duplicating a preparation thread per `(sid, row_index)`.
- `_preparation_state_locks` is a per-`(sid, row_index)` lock. **All state read-modify-save operations (including pause/resume endpoints and `_mark_section_done`) must hold this lock; never hold it during LLM calls.**
- A local `tokens_lock` guards the shared token accumulator when multiple section workers report usage.
- `storage.save_section` writes index-named files (`0.json`, `1.json`, …) which are safe for concurrent writers; `storage.save_generation_state` relies on the per-run lock for same-file writes.
- Background preparation runs in a daemon thread; tests poll persisted state rather than waiting on the thread object.

### Rendering
- `render_markdown` in `app/main.py` converts Markdown and sanitizes with `bleach` (allowed tags/attrs/protocols; fenced code + sane lists). Registered as the Jinja `markdown` filter.
- LLM prompt guidance matches the allowlist: no headings above level 4, no tables/images/LaTeX.
- Templates expect `request` in context (Starlette/FastAPI style) and rely on `303` redirects for POST actions.

### Storage
- Flat-file JSON under `DATA_DIR = data/sessions/<sid>/`; writes use a temp-file-then-replace atomic pattern (`storage._write_json`).
- Per-session files: `meta.json`, `raw.txt`, `parsed.json`, `selected.json`, `calendar.ics`, plus `preparation/` with per-row state/sections/final record.

## Important Files

| File | Why it matters |
|---|---|
| `app/main.py` | FastAPI app, all routes, background preparation thread orchestration, concurrency locks, markdown/bleach rendering. |
| `app/llm.py` | All LLM calls, prompt loading, `threading.local` usage callback, preparation generators. |
| `app/storage.py` | JSON persistence, session/preparation CRUD, atomic writes. |
| `app/config.py` | `pydantic-settings` `Settings`; `get_settings()` is `@lru_cache`d. |
| `app/arxiv.py` | arXiv search, PDF/HTML fetching, paper text extraction (`PDF_TEXT_LIMIT = 100_000` chars). |
| `app/document_parser.py` | PDF→Markdown (PyMuPDF) and HTML→Markdown (trafilatura) extraction. |
| `app/ics.py` | Calendar generation. |
| `app/prompts/*.md` | Prompt templates (schedule parsing and all preparation steps). |
| `templates/prepare.html` | Most backend-coupled template; polls `/prepare/{sid}/{row}` by full page reload every 1s while running/paused. |

## Runtime/Tooling Preferences

- **Python**: CPython 3.14 (project venv at `.venv`); code uses modern syntax (`str | None`, built-in generics) — keep it Python 3.10+ compatible.
- **Package manager**: plain `requirements.txt` (unpinned) + virtualenv; no Poetry/Pipenv. There is no `pyproject.toml` or `setup.py`.
- **Web framework**: FastAPI + Uvicorn; run with `uvicorn app.main:app`.
- **Lint/format**: none configured (no ruff/black/mypy/pre-commit). Match existing style (PEP 8-ish, 4-space indent, type hints on signatures).
- **No README, docs dir, scripts dir, Makefile, or CI config** currently exists; templates + `app/` are the source of truth for backend/frontend contracts.
- **No coverage tooling** is configured; coverage is not an enforced gate.

## Testing & QA

- **Framework**: pytest (only test dependency; no `conftest.py`, no pytest config file).
- **Isolation**: tests use pytest built-ins — `monkeypatch` and `tmp_path`. Route tests point `storage.DATA_DIR` at `tmp_path` and stub `llm.parse_schedule` before importing/using `app.main.app`.
- **`TestClient`**: from `fastapi.testclient`; use `follow_redirects=False` to assert `303` + `Location`.
- **Mocking patterns**:
  - LLM client: patch `app.llm._client` with a fake returning canned JSON.
  - arXiv HTTP: patch `arxiv.httpx.Client` with a context-manager fake returning canned `FakeResponse`.
  - Preparation pipeline: `tests/test_prepare.py::_prepare_mocks` patches `arxiv.find_paper`, `arxiv.extract_paper`, and all `llm.generate_preparation_*` functions; returns a call-tracking dict.
- **Concurrency tests**: `tests/test_prepare.py` uses `threading.Barrier` / `threading.Event` to prove sections run in parallel and to test pause/resume; helpers like `_wait_preparation_done` and `_wait_state_status` poll persisted state because preparation runs on a background thread.
- **Suite size**: 53 tests across `test_config`, `test_ics`, `test_storage`, `test_extract`, `test_llm`, `test_routes`, `test_history`, `test_prepare` (largest).
- **Expectation**: run `pytest tests/` before finishing changes; all tests should pass.