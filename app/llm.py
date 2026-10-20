import json

from openai import OpenAI

from app.config import get_settings


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
    resp = _client().chat.completions.create(
        model=s.llm_model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(resp.choices[0].message.content)
    sessions = data.get("sessions", [])
    return [
        {"date": r["date"], "student": r["student"], "paper": r["paper"]}
        for r in sessions if all(k in r for k in ("date", "student", "paper"))
    ]
