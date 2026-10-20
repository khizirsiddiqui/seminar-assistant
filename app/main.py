import logging
from datetime import date

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app import llm, storage
from app.extract import extract_text
from app.ics import build_calendar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Seminar Guide")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html",
                                      context={"current_year": date.today().year})


@app.post("/parse")
async def parse(request: Request,
                file: UploadFile | None = File(None),
                pasted_text: str = Form(""),
                venue: str = Form(""),
                assume_year: int | None = Form(None),
                start_time: str = Form("14:15"),
                end_time: str = Form("15:45")):
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
        rows = llm.parse_schedule(text, assume_year=assume_year)
    except Exception as e:  # surface API errors to the user
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=502,
            context={"message": f"LLM parsing failed: {e}"})

    if not rows:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=422,
            context={"message": "No sessions could be parsed from the input. "
                                "Please check the schedule text and try again."})

    sid = storage.create_session(raw_text=text, venue=venue.strip(),
                                 start_time=start_time, end_time=end_time,
                                 source_filename=source_filename)
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
                   end_time: str = Form("15:45")):
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

    storage.update_session_meta(sid, venue=venue.strip(), start_time=start_time, end_time=end_time)
    ics_text = build_calendar(selected, venue=venue.strip(), start_time=start_time, end_time=end_time)
    storage.save_selection(sid, selected)
    storage.save_ics(sid, ics_text)
    return RedirectResponse(f"/result/{sid}", status_code=303)


@app.get("/result/{sid}")
async def result(request: Request, sid: str):
    try:
        rows = storage.load_selection(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})
    return templates.TemplateResponse(request=request, name="result.html",
                                      context={"sid": sid, "rows": rows})


@app.get("/download/{sid}")
async def download(sid: str):
    try:
        ics_text = storage.load_ics(sid)
    except FileNotFoundError:
        return Response(content="Session not found", status_code=404)
    return Response(content=ics_text, media_type="text/calendar",
                    headers={"Content-Disposition": f'attachment; filename="seminar-{sid}.ics"'})


@app.get("/prepare/{sid}/{row_index}")
async def prepare(request: Request, sid: str, row_index: int):
    try:
        rows = storage.load_selection(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})
    if not (0 <= row_index < len(rows)):
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Row {row_index} not found."})
    return templates.TemplateResponse(request=request, name="prepare.html",
                                      context={"sid": sid, "row": rows[row_index]})


@app.get("/history")
async def history(request: Request):
    sessions = []
    for sid in storage.list_sessions():
        try:
            sessions.append(storage.get_session(sid))
        except FileNotFoundError:
            continue
    return templates.TemplateResponse(request=request, name="history.html",
                                      context={"sessions": sessions})


@app.get("/session/{sid}")
async def session_detail(request: Request, sid: str):
    try:
        meta = storage.get_session(sid)
        rows = storage.load_selection(sid)
    except FileNotFoundError:
        return templates.TemplateResponse(
            request=request, name="error.html", status_code=404,
            context={"message": f"Session {sid} not found."})
    return templates.TemplateResponse(request=request, name="session.html",
                                      context={"sid": sid, "meta": meta, "rows": rows})
