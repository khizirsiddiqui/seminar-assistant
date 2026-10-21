import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import bleach
import markdown as markdown_lib
from fastapi import FastAPI, File, Form, Request, UploadFile
from markupsafe import Markup
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app import arxiv, llm, storage
from app.extract import extract_text
from app.ics import build_calendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger("seminar.main")

app = FastAPI(title="Seminar Assistant")
templates = Jinja2Templates(directory="templates")

_MD_EXTENSIONS = ["fenced_code", "sane_lists"]
_ALLOWED_TAGS = [
    "p", "br", "strong", "em", "b", "i", "code", "pre",
    "ul", "ol", "li", "blockquote", "a", "h4", "h5", "h6", "hr",
]
_ALLOWED_ATTRS = {"a": ["href", "title", "rel", "target"]}


def render_markdown(text: str) -> str:
    if not text:
        return ""
    html = markdown_lib.markdown(text, extensions=_MD_EXTENSIONS)
    cleaned = bleach.clean(
        html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=["http", "https", "mailto"],
    )
    return Markup(cleaned)


templates.env.filters["markdown"] = render_markdown

_active_generations: dict[tuple[str, int], bool] = {}
_active_generations_lock = threading.Lock()
_preparation_state_locks: dict[tuple[str, int], threading.Lock] = {}


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html",
                                      context={"current_year": date.today().year,
                                               "recent": _session_summaries()[:3]})


@app.post("/parse")
async def parse(request: Request,
                file: UploadFile | None = File(None),
                pasted_text: str = Form(""),
                venue: str = Form(""),
                assume_year: int | None = Form(None),
                start_time: str = Form("14:15"),
                end_time: str = Form("15:45"),
                course_title: str = Form("")):
    if file is not None and file.filename:
        text = extract_text(file.file, file.filename, file.content_type or "")
        source_filename = file.filename
    elif pasted_text.strip():
        text = pasted_text
        source_filename = "pasted-text"
    else:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=422,
            context={"message": "Please upload a file or paste the schedule text."})

    if not venue.strip() or not start_time or not end_time:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=422,
            context={"message": "Venue, start time and end time are required."})

    try:
        parsed = llm.parse_schedule(text, assume_year=assume_year)
    except Exception as e:  # surface API errors to the user
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=502,
            context={"message": f"LLM parsing failed: {e}"})

    guessed_course_title = parsed.get("course_title") or ""
    rows = parsed.get("sessions", [])
    if not rows:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=422,
            context={"message": "No sessions could be parsed from the input. "
                                "Please check the schedule text and try again."})

    sid = storage.create_session(raw_text=text, venue=venue.strip(),
                                 start_time=start_time, end_time=end_time,
                                 source_filename=source_filename,
                                 course_title=(course_title.strip() or guessed_course_title))
    storage.save_parsed(sid, rows)
    return RedirectResponse(f"/confirm/{sid}", status_code=303)


@app.get("/confirm/{sid}")
async def confirm(request: Request, sid: str):
    try:
        meta = storage.get_session(sid)
        rows = storage.load_parsed(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})
    return templates.TemplateResponse(request=request, name="confirm.html",
                                      context={"sid": sid, "rows": rows, "meta": meta})


@app.post("/generate/{sid}")
async def generate(request: Request, sid: str,
                   venue: str = Form(""),
                   start_time: str = Form("14:15"),
                   end_time: str = Form("15:45"),
                   course_title: str = Form("")):
    try:
        rows = storage.load_parsed(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})

    form = await request.form()
    indexes = [int(v) for v in form.getlist("row_index") if v.isdigit()]
    selected = [rows[i] for i in indexes if 0 <= i < len(rows)]

    if not selected:
        meta = storage.get_session(sid)
        return templates.TemplateResponse(
            request=request, name="confirm.html", status_code=422,
            context={"sid": sid, "rows": rows, "meta": meta,
                     "error": "Select at least one session."})

    storage.update_session_meta(sid, venue=venue.strip(), start_time=start_time,
                                end_time=end_time, course_title=course_title.strip())
    ics_text = build_calendar(selected, venue=venue.strip(), start_time=start_time,
                              end_time=end_time, course_title=course_title.strip())
    storage.save_selection(sid, selected)
    storage.save_ics(sid, ics_text)
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.get("/result/{sid}")
async def result(sid: str):
    return RedirectResponse(f"/session/{sid}", status_code=303)


@app.get("/download/{sid}")
async def download(sid: str):
    try:
        ics_text = storage.load_ics(sid)
    except FileNotFoundError:
        return Response(content="Session not found", status_code=404)
    return Response(content=ics_text, media_type="text/calendar",
                    headers={"Content-Disposition": f'attachment; filename="seminar-{sid}.ics"'})


async def _load_prepare_row(sid: str, row_index: int) -> tuple[list[dict], dict, str | None]:
    try:
        rows = storage.load_selection(sid)
    except FileNotFoundError:
        return [], {}, f"Session {sid} not found."
    if not (0 <= row_index < len(rows)):
        return [], {}, f"Row {row_index} not found."
    return rows, rows[row_index], None


def _source_from_paper(paper: dict) -> dict:
    return {
        "arxiv_id": paper["arxiv_id"],
        "url": paper["url"],
        "pdf_url": paper["pdf_url"],
        "title": paper["title"],
        "authors": paper["authors"],
        "published": paper["published"],
        "categories": paper["categories"],
        "abstract": paper["abstract"],
        "paper_text_chars": len(paper["paper_text"]),
        "paper_text_truncated": paper["paper_text_truncated"],
    }


def _completed_requests(section_done: list[bool], ai_count: int) -> int:
    """Titles request, plus each finished AI section, plus the closing request."""
    return 1 + sum(section_done[:ai_count]) + (1 if all(section_done[ai_count:]) else 0)


def _new_preparation_state(query: str) -> dict:
    return {
        "status": "running",
        "arxiv_query": query,
        "source": {},
        "section_titles": [],
        "section_done": [],
        "completed_requests": 0,
        "total_requests": 0,
        "short_briefing": "",
        "prerequisites": "",
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0},
        "duration_seconds": 0,
    }


def _start_preparation_thread(sid: str, row_index: int, query: str, row: dict) -> None:
    key = (sid, row_index)
    with _active_generations_lock:
        if _active_generations.get(key):
            return
        _active_generations[key] = True
        _preparation_state_locks.setdefault(key, threading.Lock())
    threading.Thread(target=_run_preparation, args=(sid, row_index, query, row),
                     daemon=True).start()


def _generate_one_section(sid: str, row_index: int, paper: dict, section: dict,
                          idx: int, usage_callback, mark_done) -> None:
    """Generate and persist one AI preparation section in a worker thread."""
    title = section["title"]
    try:
        llm.set_usage_callback(usage_callback)
        state = storage.load_generation_state(sid, row_index)
        if state and state.get("status") == "paused":
            return
        content = llm.generate_preparation_section_content(paper, title, section.get("other_titles"))
        storage.save_section(sid, row_index, idx, {"title": title, "content": content})
        mark_done(idx)
    finally:
        llm.clear_usage_callback()


def _generate_closing(sid: str, row_index: int, paper: dict, section_titles: list[str],
                      usage_callback, mark_closing_done) -> None:
    """Generate the fixed sections and the executive briefing with a single request."""
    ai_count = len(section_titles)
    try:
        llm.set_usage_callback(usage_callback)
        state = storage.load_generation_state(sid, row_index)
        if state and state.get("status") == "paused":
            return
        closing = llm.generate_preparation_closing(paper, section_titles)
        for offset, (title, key) in enumerate(zip(llm.FIXED_SECTION_TITLES, ("questions", "take_away"))):
            storage.save_section(sid, row_index, ai_count + offset,
                                 {"title": title, "content": closing[key]})
        mark_closing_done(closing["short_briefing"], closing["prerequisites"])
    finally:
        llm.clear_usage_callback()


def _run_preparation(sid: str, row_index: int, query: str, row: dict) -> None:
    key = (sid, row_index)
    state_lock = _preparation_state_locks[key]
    started = time.monotonic()
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    tokens_lock = threading.Lock()

    def _on_usage(usage):
        with tokens_lock:
            for token_key in tokens:
                tokens[token_key] += usage.get(token_key, 0)

    llm.set_usage_callback(_on_usage)
    try:
        if arxiv.is_article_url(query):
            paper = {
                "arxiv_id": "",
                "url": query,
                "pdf_url": query,
                "title": row["paper"],
                "abstract": "",
                "authors": [],
                "published": "",
                "categories": [],
            }
            paper = arxiv.extract_paper(paper)
            html_meta = paper.get("html_metadata") or {}
            for key in ("title", "authors", "published", "categories"):
                if not paper.get(key) and html_meta.get(key):
                    paper[key] = html_meta[key]
            missing = [key for key in ("title", "authors", "published", "categories")
                       if not paper.get(key)]
            meta = {}
            if missing:
                try:
                    meta = llm.extract_paper_metadata(paper)
                except Exception:
                    logger.warning("Could not extract article metadata; keeping empty fields", exc_info=True)
                    meta = {}
            for key in missing:
                if meta.get(key):
                    paper[key] = meta[key]
            paper["published"] = (paper.get("published") or "")[:10]
        else:
            paper = arxiv.find_paper(query, title=row["paper"], presenter=row["student"])
            if paper is None:
                with state_lock:
                    state = storage.load_generation_state(sid, row_index) or _new_preparation_state(query)
                    state["status"] = "error"
                    state["error"] = "No confident arXiv match was found. Paste an arXiv URL or ID and try again."
                    storage.save_generation_state(sid, row_index, state)
                return
            paper = arxiv.extract_paper(paper)

        source = _source_from_paper(paper)

        state = storage.load_generation_state(sid, row_index)
        if state is None:
            storage.clear_preparation(sid, row_index)
            storage.clear_sections(sid, row_index)
            state = _new_preparation_state(query)

        with state_lock:
            state = storage.load_generation_state(sid, row_index) or state
            state["arxiv_query"] = query
            state["source"] = source
            state["status"] = "running" if state.get("status") == "running" else state.get("status", "running")
            if state.get("status") not in ("running", "paused"):
                return
            state["tokens"] = dict(tokens)
            state["duration_seconds"] = time.monotonic() - started
            storage.save_generation_state(sid, row_index, state)

        state = storage.load_generation_state(sid, row_index) or state
        if state["status"] == "paused":
            return

        if not state.get("section_titles"):
            section_titles = llm.generate_preparation_section_titles(paper)
            with state_lock:
                persisted = storage.load_generation_state(sid, row_index) or state
                state["status"] = "paused" if persisted.get("status") == "paused" else "running"
                state["section_titles"] = section_titles
                state["completed_requests"] = state.get("completed_requests", 0) + 1
                storage.save_generation_state(sid, row_index, state)
        else:
            section_titles = state["section_titles"]

        with state_lock:
            state = storage.load_generation_state(sid, row_index) or state
            state["total_requests"] = 1 + len(section_titles) + 1
            if state["status"] != "paused":
                state["status"] = "running"
            state["tokens"] = dict(tokens)
            state["duration_seconds"] = time.monotonic() - started
            storage.save_generation_state(sid, row_index, state)
        if state["status"] == "paused":
            return

        ai_count = len(section_titles)
        section_count = ai_count + len(llm.FIXED_SECTION_TITLES)
        all_titles = section_titles + list(llm.FIXED_SECTION_TITLES)

        with state_lock:
            state = storage.load_generation_state(sid, row_index) or state
            if "section_done" not in state:
                if "completed_sections" in state:
                    state["section_done"] = [
                        i < state["completed_sections"] for i in range(section_count)
                    ]
                else:
                    state["section_done"] = [False] * section_count
            if len(state["section_done"]) != section_count:
                done = state["section_done"]
                state["section_done"] = (done + [False] * section_count)[:section_count]
            state.pop("completed_sections", None)
            state["tokens"] = dict(tokens)
            state["duration_seconds"] = time.monotonic() - started
            storage.save_generation_state(sid, row_index, state)

        def _mark_section_done(idx):
            with state_lock:
                state = storage.load_generation_state(sid, row_index) or _new_preparation_state(query)
                done = state.setdefault("section_done", [False] * section_count)
                if len(done) != section_count:
                    done = (done + [False] * section_count)[:section_count]
                    state["section_done"] = done
                state["section_done"][idx] = True
                state["completed_requests"] = _completed_requests(state["section_done"], ai_count)
                with tokens_lock:
                    state["tokens"] = dict(tokens)
                state["duration_seconds"] = time.monotonic() - started
                storage.save_generation_state(sid, row_index, state)

        def _mark_closing_done(short_briefing, prerequisites):
            with state_lock:
                state = storage.load_generation_state(sid, row_index) or _new_preparation_state(query)
                done = state.setdefault("section_done", [False] * section_count)
                if len(done) != section_count:
                    done = (done + [False] * section_count)[:section_count]
                    state["section_done"] = done
                for i in range(ai_count, section_count):
                    done[i] = True
                state["short_briefing"] = short_briefing
                state["prerequisites"] = prerequisites
                state["completed_requests"] = _completed_requests(done, ai_count)
                with tokens_lock:
                    state["tokens"] = dict(tokens)
                state["duration_seconds"] = time.monotonic() - started
                storage.save_generation_state(sid, row_index, state)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for idx, title in enumerate(section_titles):
                if state["section_done"][idx]:
                    continue
                section = {"title": title,
                           "other_titles": [t for t in all_titles if t != title]}
                futures.append(pool.submit(_generate_one_section, sid, row_index, paper, section,
                                           idx, _on_usage, _mark_section_done))
            if not (all(state["section_done"][ai_count:]) and state.get("short_briefing")):
                futures.append(pool.submit(_generate_closing, sid, row_index, paper, section_titles,
                                           _on_usage, _mark_closing_done))
            for future in futures:
                future.result()

        state = storage.load_generation_state(sid, row_index) or state
        if state["status"] == "paused":
            return

        record = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "short_briefing": state.get("short_briefing", ""),
            "prerequisites": state.get("prerequisites", ""),
            "sections": storage.load_sections(sid, row_index),
            "tokens": dict(tokens),
            "duration_seconds": time.monotonic() - started,
        }
        storage.save_preparation(sid, row_index, record)
        with state_lock:
            state = storage.load_generation_state(sid, row_index) or state
            state["status"] = "complete"
            state["tokens"] = dict(tokens)
            state["duration_seconds"] = time.monotonic() - started
            storage.save_generation_state(sid, row_index, state)
    except Exception as e:
        logger.exception("preparation generation failed for %s/%s", sid, row_index)
        with state_lock:
            state = storage.load_generation_state(sid, row_index) or _new_preparation_state(query)
            state["status"] = "error"
            state["error"] = str(e)
            state["tokens"] = dict(tokens)
            state["duration_seconds"] = time.monotonic() - started
            storage.save_generation_state(sid, row_index, state)
    finally:
        llm.clear_usage_callback()
        with _active_generations_lock:
            _active_generations[key] = False


async def _prepare_start(request: Request, sid: str, row_index: int,
                         arxiv_query: str) -> Response:
    _, row, error = await _load_prepare_row(sid, row_index)
    if error is not None:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": error})
    query = arxiv_query.strip() or (row.get("paper_url") or "").strip() or row["paper"]
    state = storage.load_generation_state(sid, row_index)
    if state is None or state.get("status") not in ("running", "paused"):
        storage.clear_preparation(sid, row_index)
        storage.clear_sections(sid, row_index)
        state = _new_preparation_state(query)
        storage.save_generation_state(sid, row_index, state)
    _start_preparation_thread(sid, row_index, query, row)
    return RedirectResponse(f"/prepare/{sid}/{row_index}", status_code=303)


async def _prepare_response(request: Request, sid: str, row_index: int,
                             status_code: int = 200) -> Response:
    rows, row, error = await _load_prepare_row(sid, row_index)
    if error is not None:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": error})
    state = storage.load_generation_state(sid, row_index)
    sections = storage.load_sections(sid, row_index)
    preparation = storage.load_preparation(sid, row_index)
    if preparation is None and state and sections:
        preparation = {
            "source": state.get("source") or {},
            "short_briefing": state.get("short_briefing", ""),
            "prerequisites": state.get("prerequisites", ""),
            "sections": sections,
            "tokens": state.get("tokens", {}),
            "duration_seconds": state.get("duration_seconds", 0),
        }
    section_titles = state.get("section_titles", []) if state else []
    generation_error = state.get("error") if state and state.get("status") == "error" else None
    return templates.TemplateResponse(request=request, name="prepare.html",
                                      status_code=status_code,
                                      context={"sid": sid, "row": row,
                                               "row_index": row_index,
                                               "preparation": preparation,
                                               "sections": sections,
                                               "section_titles": section_titles,
                                               "state": state,
                                               "generation_error": generation_error})


@app.get("/prepare/{sid}/{row_index}")
async def prepare(request: Request, sid: str, row_index: int):
    return await _prepare_response(request, sid, row_index)


@app.get("/prepare/{sid}/{row_index}/status")
async def prepare_status(sid: str, row_index: int):
    _, _, error = await _load_prepare_row(sid, row_index)
    if error is not None:
        return JSONResponse({"error": error}, status_code=404)
    state = storage.load_generation_state(sid, row_index) or {}
    done = sum(1 for flag in state.get("section_done", []) if flag)
    status = state.get("status")
    return {
        "status": status,
        "active": status in ("running", "paused"),
        "completed_requests": state.get("completed_requests", 0),
        "total_requests": state.get("total_requests", 0),
        # Changes whenever the rendered page would change, so the client only refetches then.
        "version": f"{status}:{state.get('completed_requests', 0)}:{state.get('total_requests', 0)}:{done}:{len(state.get('section_titles', []))}",
    }


@app.post("/prepare/{sid}/{row_index}/generate")
async def prepare_generate(request: Request, sid: str, row_index: int,
                           arxiv_query: str = Form("")):
    return await _prepare_start(request, sid, row_index, arxiv_query)


@app.post("/prepare/{sid}/{row_index}/regenerate")
async def prepare_regenerate(request: Request, sid: str, row_index: int,
                             arxiv_query: str = Form("")):
    return await _prepare_start(request, sid, row_index, arxiv_query)


@app.post("/prepare/{sid}/{row_index}/clear")
async def prepare_clear(request: Request, sid: str, row_index: int):
    _, row, error = await _load_prepare_row(sid, row_index)
    if error is not None:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": error})
    key = (sid, row_index)
    with _active_generations_lock:
        state_lock = _preparation_state_locks.setdefault(key, threading.Lock())
    with state_lock:
        state = storage.load_generation_state(sid, row_index)
        if state and state.get("status") in ("running", "paused"):
            return await _prepare_response(request, sid, row_index, status_code=409)
        storage.clear_preparation(sid, row_index)
        storage.clear_sections(sid, row_index)
        storage.clear_generation_state(sid, row_index)
    return RedirectResponse(f"/prepare/{sid}/{row_index}", status_code=303)


@app.post("/prepare/{sid}/{row_index}/pause")
async def prepare_pause(request: Request, sid: str, row_index: int):
    _, row, error = await _load_prepare_row(sid, row_index)
    if error is not None:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": error})
    key = (sid, row_index)
    with _active_generations_lock:
        state_lock = _preparation_state_locks.setdefault(key, threading.Lock())
    with state_lock:
        paused = storage.pause_generation(sid, row_index)
    if not paused:
        return await _prepare_response(request, sid, row_index, status_code=409)
    return RedirectResponse(f"/prepare/{sid}/{row_index}", status_code=303)


@app.post("/prepare/{sid}/{row_index}/resume")
async def prepare_resume(request: Request, sid: str, row_index: int):
    _, row, error = await _load_prepare_row(sid, row_index)
    if error is not None:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": error})
    key = (sid, row_index)
    with _active_generations_lock:
        state_lock = _preparation_state_locks.setdefault(key, threading.Lock())
    with state_lock:
        resumed = storage.resume_generation(sid, row_index)
    if not resumed:
        return await _prepare_response(request, sid, row_index, status_code=409)
    query = storage.load_generation_state(sid, row_index)["arxiv_query"] or row["paper"]
    _start_preparation_thread(sid, row_index, query, row)
    return RedirectResponse(f"/prepare/{sid}/{row_index}", status_code=303)


def _session_summaries() -> list[dict]:
    """Session metas with status, talk/guide counts and date range, newest first."""
    summaries = []
    for sid in storage.list_sessions():
        try:
            meta = storage.get_session(sid)
        except FileNotFoundError:
            continue
        generated = storage.has_selection(sid)
        try:
            rows = storage.load_selection(sid) if generated else storage.load_parsed(sid)
        except FileNotFoundError:
            rows = []
        dates = sorted(r["date"] for r in rows if r.get("date"))
        meta["status"] = "generated" if generated else "parsed"
        meta["talk_count"] = len(rows)
        meta["first_date"] = dates[0] if dates else None
        meta["last_date"] = dates[-1] if dates else None
        meta["prepared_count"] = (
            sum(1 for i in range(len(rows)) if storage.load_preparation(sid, i) is not None)
            if generated else 0)
        summaries.append(meta)
    summaries.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return summaries


@app.get("/history")
async def history(request: Request):
    return templates.TemplateResponse(request=request, name="history.html",
                                      context={"sessions": _session_summaries()})


@app.get("/session/{sid}")
async def session_detail(request: Request, sid: str):
    try:
        meta = storage.get_session(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})

    if storage.has_selection(sid):
        rows = storage.load_selection(sid)
        status = "generated"
    else:
        rows = storage.load_parsed(sid)
        status = "parsed"

    prepare_statuses = []
    if status == "generated":
        for idx, _row in enumerate(rows):
            state = storage.load_generation_state(sid, idx)
            if storage.load_preparation(sid, idx) is not None:
                prepare_statuses.append("generated")
            elif state and state.get("status") in ("running", "paused"):
                prepare_statuses.append("fetching")
            else:
                prepare_statuses.append("pending")

    today = date.today().isoformat()
    upcoming = [r["date"] for r in rows if r.get("date", "") >= today]
    next_date = min(upcoming) if upcoming else None
    dates = sorted(r["date"] for r in rows if r.get("date"))
    return templates.TemplateResponse(request=request, name="session.html",
                                      context={"sid": sid, "meta": meta, "rows": rows,
                                               "status": status,
                                               "prepare_statuses": prepare_statuses,
                                               "prepared_count": prepare_statuses.count("generated"),
                                               "today": today, "next_date": next_date,
                                               "first_date": dates[0] if dates else None,
                                               "last_date": dates[-1] if dates else None})



async def _edit_form(request: Request, sid: str):
    try:
        meta = storage.get_session(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})
    return templates.TemplateResponse(request=request, name="edit.html",
                                      context={"sid": sid, "meta": meta})


async def _edit_submit(request: Request, sid: str):
    try:
        meta = storage.get_session(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})

    form = await request.form()
    venue = form.get("venue", meta.get("venue", ""))
    start_time = form.get("start_time", meta.get("start_time", ""))
    end_time = form.get("end_time", meta.get("end_time", ""))
    course_title = form.get("course_title", meta.get("course_title", ""))

    venue = (venue or "").strip() if isinstance(venue, str) else venue
    course_title = (course_title or "").strip() if isinstance(course_title, str) else course_title

    storage.update_session_meta(sid, venue=venue, start_time=start_time,
                                end_time=end_time, course_title=course_title)

    if storage.has_selection(sid):
        selected = storage.load_selection(sid)
        ics_text = build_calendar(selected, venue=venue, start_time=start_time,
                                  end_time=end_time, course_title=course_title)
        storage.save_ics(sid, ics_text)

    return RedirectResponse(f"/session/{sid}", status_code=303)



@app.get("/edit/{sid}")
@app.get("/session/{sid}/edit")
@app.get("/sessions/{sid}/edit")
@app.get("/confirm/{sid}/edit")
async def edit_form(request: Request, sid: str):
    return await _edit_form(request, sid)


@app.post("/edit/{sid}")
@app.post("/session/{sid}/edit")
@app.post("/sessions/{sid}/edit")
@app.post("/confirm/{sid}/edit")
async def edit_submit(request: Request, sid: str):
    return await _edit_submit(request, sid)


@app.post("/delete/{sid}")
async def delete_session(request: Request, sid: str):
    try:
        storage.delete_session(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})
    return RedirectResponse("/history", status_code=303)
