import html
import json
import logging
import re
import threading
import time
from pathlib import Path

from openai import OpenAI

from app.config import get_settings

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    file = PROMPT_DIR / filename
    try:
        return file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(f"Prompt file not found: {file}") from None

logger = logging.getLogger("seminar.llm")

_usage_state = threading.local()


def set_usage_callback(callback):
    _usage_state.callback = callback


def clear_usage_callback():
    _usage_state.callback = None


def _cached_tokens(usage) -> int:
    """Prompt tokens served from the provider's prefix cache (0 when not reported)."""
    details = getattr(usage, "prompt_tokens_details", None)
    return getattr(details, "cached_tokens", 0) or 0


def _report_usage(usage) -> None:
    callback = getattr(_usage_state, "callback", None)
    if callback is None or usage is None:
        return
    callback({
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "cached_tokens": _cached_tokens(usage),
    })


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key)

SYSTEM_PROMPT = load_prompt("schedule.md")

_HTML_LINK_RE = re.compile(r"<a\s+[^>]*href=[\"'](?P<url>[^\"']+)[\"'][^>]*>", re.IGNORECASE)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(?P<url>[^)\s]+)(?:\s+[\"'][^)]*[\"'])?\s*\)")


def _extract_paper_url(value) -> str:
    """Return the URL from a plain URL, markdown link, or HTML anchor."""
    if not value or not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    m = _HTML_LINK_RE.search(text)
    if m:
        return m.group("url").strip()
    m = _MARKDOWN_LINK_RE.search(text)
    if m:
        return m.group("url").strip()
    return text


def parse_schedule(text: str, assume_year: int | None = None) -> dict:
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
    usage = getattr(resp, "usage", None)
    _report_usage(usage)
    logger.info("LLM response: duration=%.2fs prompt_tokens=%s cached_tokens=%s completion_tokens=%s content=%s",
                duration, getattr(usage, "prompt_tokens", "?"), _cached_tokens(usage),
                getattr(usage, "completion_tokens", "?"), raw_content)
    data = json.loads(raw_content)
    course_title = data.get("course_title") or None
    sessions = data.get("sessions", [])
    rows = [
        {"date": r["date"], "student": r["student"], "paper": r["paper"],
         "paper_url": _extract_paper_url(r.get("paper_url"))}
        for r in sessions if all(k in r for k in ("date", "student", "paper"))
    ]
    logger.info("LLM parsed %d session(s) in %.2fs", len(rows), duration)
    return {"course_title": course_title, "sessions": rows}

PREPARATION_SYSTEM_PROMPT = load_prompt("preparation_sections.md")
PREPARATION_SECTION_SYSTEM_PROMPT = load_prompt("preparation_section.md")
PREPARATION_CLOSING_SYSTEM_PROMPT = load_prompt("preparation_closing.md")
PREPARATION_COMMON_PROMPT = load_prompt("preparation_common.md")


def _preparation_user_message(paper: dict) -> str:
    user = (
        "Paper title: {title}\n"
        "Authors: {authors}\n"
        "Published: {published}\n"
        "Categories: {categories}\n"
        "Abstract:\n{abstract}\n\n"
        "Paper text:\n{paper_text}\n"
    ).format(
        title=paper.get("title", ""),
        authors=", ".join(paper.get("authors") or []),
        published=paper.get("published", ""),
        categories=", ".join(paper.get("categories") or []),
        abstract=paper.get("abstract", ""),
        paper_text=paper.get("paper_text", ""),
    )
    if paper.get("paper_text_truncated"):
        user += (
            "\n(The paper text above was truncated. Mention this in the guide only if it "
            "materially affects the attendee's understanding.)\n"
        )
    return user


def _chat_json(task_prompt: str, paper_message: str, *, temperature: float, extra: str = "") -> dict:
    """Run one preparation task. The system prompt and the paper text come first and are
    identical across a guide's calls, so a provider that caches prompt prefixes can reuse
    them; only the trailing task instruction varies."""
    s = get_settings()
    messages = [
        {"role": "system", "content": PREPARATION_COMMON_PROMPT},
        {"role": "user", "content": f"{paper_message}\n\n{task_prompt}{extra}"},
    ]
    request_kwargs = dict(
        model=s.llm_model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
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
    usage = getattr(resp, "usage", None)
    _report_usage(usage)
    logger.info("LLM response: duration=%.2fs prompt_tokens=%s cached_tokens=%s completion_tokens=%s content=%s",
                duration, getattr(usage, "prompt_tokens", "?"), _cached_tokens(usage),
                getattr(usage, "completion_tokens", "?"), raw_content)
    return json.loads(raw_content)


PAPER_METADATA_SYSTEM_PROMPT = load_prompt("paper_metadata.md")


def extract_paper_metadata(paper: dict) -> dict:
    """Extract bibliographic metadata from a paper's text for non-arXiv sources.

    Returns a dict with ``title``, ``authors``, ``published`` and ``categories``.
    Missing or ill-typed values fall back to empty values. API and JSON errors
    are intentionally allowed to propagate; callers decide how to handle them.
    """
    text = paper.get("paper_text") or ""
    head = text[:8000]
    title = (paper.get("title") or "").strip()
    url = (paper.get("url") or paper.get("pdf_url") or "").strip()
    user = f"Title: {title}\nURL: {url}\n\nPaper text:\n{head}"
    messages = [
        {"role": "system", "content": PAPER_METADATA_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    s = get_settings()
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
    usage = getattr(resp, "usage", None)
    _report_usage(usage)
    logger.info("LLM response: duration=%.2fs prompt_tokens=%s cached_tokens=%s completion_tokens=%s content=%s",
                duration, getattr(usage, "prompt_tokens", "?"), _cached_tokens(usage),
                getattr(usage, "completion_tokens", "?"), raw_content)
    data = json.loads(raw_content)

    title_value = data.get("title")
    authors_value = data.get("authors")
    published_value = data.get("published")
    categories_value = data.get("categories")

    return {
        "title": html.unescape(title_value) if isinstance(title_value, str) else "",
        "authors": [html.unescape(a) for a in authors_value if isinstance(a, str)]
        if isinstance(authors_value, list) else [],
        "published": html.unescape(published_value) if isinstance(published_value, str) else "",
        "categories": [html.unescape(c) for c in categories_value if isinstance(c, str)]
        if isinstance(categories_value, list) else [],
    }


FIXED_SECTION_TITLES = ("Questions Worth Asking the Speaker", "Key Take-Away")


def _is_fixed_section(title: str) -> bool:
    normalized = " ".join(title.strip().lower().split())
    return normalized in {
        "questions worth asking the speaker",
        "key take-away",
        "key takeaways",
    }


def generate_preparation_section_titles(paper: dict) -> list[str]:
    user = _preparation_user_message(paper)
    sections_data = _chat_json(PREPARATION_SYSTEM_PROMPT, user, temperature=0.2)
    section_titles = []
    for item in sections_data.get("sections", []):
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if isinstance(title, str) and title.strip() and not _is_fixed_section(title):
            section_titles.append(title.strip())
    section_titles = section_titles[:5]
    if not section_titles:
        raise ValueError("LLM preparation response is missing section titles.")
    return section_titles


def generate_preparation_section_content(paper: dict, title: str,
                                         other_titles: list[str] | None = None) -> str:
    user = _preparation_user_message(paper)
    extra = f"\n\nWrite the content for this section: {title}"
    if other_titles:
        extra += ("\n\nThe other sections of this guide are: "
                  f"{', '.join(other_titles)}. You may reference them, but do not produce "
                  "content that belongs in those sections or repeat their material.")
    section_data = _chat_json(PREPARATION_SECTION_SYSTEM_PROMPT, user, temperature=0.2,
                              extra=extra)
    content = section_data.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM preparation response is missing section content.")
    return content.strip()


def generate_preparation_closing(paper: dict, other_titles: list[str] | None = None) -> dict:
    """Write the fixed sections and the executive briefing in one request.

    Returns ``questions``, ``take_away``, ``short_briefing`` and ``prerequisites``.
    """
    user = _preparation_user_message(paper)
    extra = ""
    if other_titles:
        extra = ("\n\nThe other sections of this guide are: "
                 f"{', '.join(other_titles)}. You may reference them, but do not repeat "
                 "their material.")
    data = _chat_json(PREPARATION_CLOSING_SYSTEM_PROMPT, user, temperature=0.2, extra=extra)
    result = {}
    for key in ("questions", "take_away", "short_briefing"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"LLM preparation response is missing {key}.")
        result[key] = value.strip()
    prerequisites = data.get("prerequisites")
    result["prerequisites"] = prerequisites.strip() if isinstance(prerequisites, str) else ""
    return result


def generate_preparation(paper: dict, on_section=None) -> dict:
    section_titles = generate_preparation_section_titles(paper)
    sections = []
    for title in section_titles:
        content = generate_preparation_section_content(paper, title)
        sections.append({"title": title, "content": content})
        if on_section is not None:
            on_section(sections)

    closing = generate_preparation_closing(paper, section_titles)
    for title, key in zip(FIXED_SECTION_TITLES, ("questions", "take_away")):
        sections.append({"title": title, "content": closing[key]})
        if on_section is not None:
            on_section(sections)

    return {
        "short_briefing": closing["short_briefing"],
        "prerequisites": closing["prerequisites"],
        "sections": sections,
    }
