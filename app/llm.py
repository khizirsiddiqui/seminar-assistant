import json
import logging
import time

from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger("seminar.llm")


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)

SYSTEM_PROMPT = (
    "You extract seminar schedules from raw text. Return ONLY a JSON object: "
    '{"sessions": [{"date": "YYYY-MM-DD", "student": "<presenter full name>", "paper": "<paper title>"}]}. '
    "Dates use the ISO format. If the source text lacks a year, use the year given by the user. "
    "One entry per scheduled presentation. Keep paper titles verbatim from the source. "
    "If a row is unparseable, skip it."
)


def parse_schedule(text: str, assume_year: int | None = None) -> list[dict]:
    s = get_settings()
    user = f"Assume the year is {assume_year}. " if assume_year else ""
    user += f"Raw schedule text:\n\n{text}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    request_kwargs = dict(
        model=s.llm_model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    logger.info("LLM request: model=%s base_url=%s messages=%d chars=%d",
                s.llm_model, s.llm_base_url, len(messages), sum(len(m["content"]) for m in messages))
    start = time.monotonic()
    try:
        resp = _client().chat.completions.create(**request_kwargs)
    except Exception:
        logger.exception("LLM request failed after %.2fs", time.monotonic() - start)
        raise
    duration = time.monotonic() - start
    raw_content = resp.choices[0].message.content
    logger.info("LLM response: duration=%.2fs response_chars=%d content=%s",
                duration, len(raw_content), raw_content)
    data = json.loads(raw_content)
    sessions = data.get("sessions", [])
    rows = [
        {"date": r["date"], "student": r["student"], "paper": r["paper"]}
        for r in sessions if all(k in r for k in ("date", "student", "paper"))
    ]
    logger.info("LLM parsed %d session(s) in %.2fs", len(rows), duration)
    return rows
