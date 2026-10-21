import difflib
import html
import logging
import re
import time
import xml.etree.ElementTree as ET

import httpx

from app.config import get_settings
from app.document_parser import html_to_markdown, pdf_to_markdown

logger = logging.getLogger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
PDF_TEXT_LIMIT = 100_000

_REFERENCES_HEADING_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*{1,2})?(?:\d+\.?[ \t]+)?(?:references|bibliography)(?:\*{1,2})?[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_references(text: str) -> str:
    """Cut the reference list (and any appendix after it) so the model's input budget
    goes to the paper body. Only a heading in the latter part of the text counts, so a
    table-of-contents entry or an early mention is left alone."""
    cut = None
    for match in _REFERENCES_HEADING_RE.finditer(text):
        if match.start() >= len(text) * 0.4:
            cut = match.start()
    return text[:cut].rstrip() if cut is not None else text
_ARXIV_MAX_RETRIES = 4
_ARXIV_DEFAULT_BACKOFF_SECONDS = 1.0
_ARXIV_MAX_BACKOFF_SECONDS = 30.0
_ARXIV_API_HEADERS = {"User-Agent": "seminar-assistant/1.0"}
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_ARXIV_ID_RE = re.compile(
    r"^(?:https?://arxiv\.org/(?:abs|pdf)/)?"
    r"([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7})"
    r"(?:\.pdf)?$",
    re.IGNORECASE,
)

def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in element:
        if _local_tag(child.tag) == name:
            return child
    return None


def _findall(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [child for child in element if _local_tag(child.tag) == name]


def _normalize(value: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in value).split()).lower()


def _title_score(result_title: str, query_title: str) -> float:
    result = _normalize(result_title)
    query = _normalize(query_title)
    if not result or not query:
        return 0.0
    if result == query:
        return 100.0
    if result in query or query in result:
        return 70.0
    return difflib.SequenceMatcher(None, result, query).ratio() * 100


_ID_PATTERN = re.compile(
    r"^(?:[0-9]{4}\.[0-9]{4,5}|[a-z-]+(?:\.[A-Z]{2})?/[0-9]{7})(?:v\d+)?$",
    re.IGNORECASE,
)

_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_META_ATTR_RE = re.compile(
    r"""([A-Za-z][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
    re.IGNORECASE,
)
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}


def _meta_attrs(tag: str) -> dict[str, str]:
    attrs = {}
    for match in _META_ATTR_RE.finditer(tag):
        key = match.group(1).lower()
        value = html.unescape(
            (match.group(2) or match.group(3) or match.group(4) or "").strip()
        )
        if key not in attrs:
            attrs[key] = value
    return attrs


def _format_author_name(value: str) -> str:
    value = value.strip()
    if "," in value:
        parts = [part.strip() for part in value.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            return f"{parts[1]} {parts[0]}"
    return value


def _split_meta_list(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        value = value.strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _normalize_meta_date(value: str) -> str:
    value = value.strip()
    match = re.search(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", value)
    if match:
        year, month, day = match.group(1), int(match.group(2)), match.group(3)
        base = f"{year}-{month:02d}"
        return f"{base}-{int(day):02d}" if day else base
    match = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", value)
    if match:
        day, month_name, year = int(match.group(1)), match.group(2).lower(), match.group(3)
        month = _MONTH_NAMES.get(month_name)
        if month:
            return f"{year}-{month:02d}-{day:02d}"
    return value


def _extract_html_metadata(html: str) -> dict:
    """Pull bibliographic metadata from common publisher <meta> tags."""
    meta: dict[str, list[str]] = {}
    for tag in _META_TAG_RE.findall(html):
        attrs = _meta_attrs(tag)
        name = (attrs.get("name") or attrs.get("property") or "").lower()
        content = attrs.get("content", "").strip()
        if name and content:
            meta.setdefault(name, []).append(content)

    authors: list[str] = []
    for value in meta.get("citation_author", []):
        authors.append(_format_author_name(value))
    if not authors:
        for value in meta.get("dc.creator", []):
            for part in re.split(r";", value):
                part = part.strip()
                if part:
                    authors.append(_format_author_name(part))
    authors = _dedupe(authors)

    published = ""
    for key in ("citation_online_date", "citation_publication_date", "dc.date",
                "prism.publicationdate", "article:published_time"):
        values = meta.get(key)
        if values:
            published = _normalize_meta_date(values[0])
            if published:
                break

    categories: list[str] = []
    for key in ("dc.subject", "citation_keywords", "keywords"):
        for value in meta.get(key, []):
            categories.extend(_split_meta_list(value))
    categories = _dedupe(categories)

    return {
        "authors": authors,
        "published": published,
        "categories": categories,
    }


def _paper_id_from_query(query: str) -> str | None:
    match = _ARXIV_ID_RE.match(query)
    if match:
        return match.group(1)

    lowered = query.lower()
    if not (lowered.startswith("https://arxiv.org/pdf/")
            or lowered.startswith("http://arxiv.org/pdf/")):
        return None

    tail = query.split("/pdf/", 1)[1]
    tail = re.split(r"[?#]", tail, maxsplit=1)[0]
    tail = tail.rstrip("/")
    if tail.lower().endswith(".pdf"):
        tail = tail[:-4]
    if _ID_PATTERN.fullmatch(tail):
        return tail
    return None


def _retry_delay_seconds(response, attempt: int) -> float:
    """Delay before retrying a rate-limited request.

    Prefers the server's Retry-After header; otherwise backs off exponentially.
    """
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("Retry-After")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
                if seconds >= 0:
                    return min(seconds, _ARXIV_MAX_BACKOFF_SECONDS)
            except (TypeError, ValueError):
                pass
    return min(
        _ARXIV_DEFAULT_BACKOFF_SECONDS * (2 ** attempt),
        _ARXIV_MAX_BACKOFF_SECONDS,
    )


def _get_with_retry(client, url: str, **kwargs):
    """Perform an HTTP GET, retrying transient failures with backoff.

    Retries on HTTP 429 (rate limit), 5xx server errors, and transport-level
    failures (connection errors, read/connect timeouts) up to
    ``_ARXIV_MAX_RETRIES`` times.
    """
    response = None
    last_exc: Exception | None = None
    for attempt in range(_ARXIV_MAX_RETRIES + 1):
        try:
            response = client.get(url, **kwargs)
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt >= _ARXIV_MAX_RETRIES:
                raise
            logger.warning(
                "arXiv request attempt %d/%d failed: %s",
                attempt + 1, _ARXIV_MAX_RETRIES + 1, exc,
            )
            time.sleep(_retry_delay_seconds(None, attempt))
            continue

        status = response.status_code
        if (status == 429 or status >= 500) and attempt < _ARXIV_MAX_RETRIES:
            time.sleep(_retry_delay_seconds(response, attempt))
            continue
        return response

    if last_exc is not None:
        raise last_exc
    return response


def is_pdf_url(value: str) -> bool:
    return bool(re.search(r"\.pdf(?:[/?#].*)?$", (value or "").strip().lower()))


def is_article_url(value: str) -> bool:
    """True for external article URLs (HTML or PDF) that should be fetched directly.

    arXiv IDs / abs / pdf URLs are excluded because those go through the arXiv API.
    """
    value = (value or "").strip()
    if not re.match(r"^https?://", value, re.IGNORECASE):
        return False
    if "arxiv.org" in value.lower():
        return False
    return _paper_id_from_query(value) is None


def _parse_entry(entry: ET.Element) -> dict:
    title_node = _find(entry, "title")
    summary_node = _find(entry, "summary")
    id_node = _find(entry, "id")
    published_node = _find(entry, "published")

    title = (title_node.text or "").strip() if title_node is not None else ""
    abstract = (summary_node.text or "").strip() if summary_node is not None else ""
    entry_id = (id_node.text or "").strip() if id_node is not None else ""
    published = (published_node.text or "").strip() if published_node is not None else ""

    arxiv_id = ""
    if "/abs/" in entry_id:
        arxiv_id = entry_id.split("/abs/", 1)[1].split("?", 1)[0].strip()
    elif entry_id:
        arxiv_id = entry_id.rsplit("/", 1)[-1].split("?", 1)[0].strip()

    pdf_url = ""
    for link in _findall(entry, "link"):
        if (link.get("title") or "").lower() == "pdf":
            pdf_url = (link.get("href") or "").strip()
            break

    authors = []
    for author in _findall(entry, "author"):
        name_node = _find(author, "name")
        if name_node is not None and (name_node.text or "").strip():
            authors.append(name_node.text.strip())

    categories = [
        category.get("term", "").strip()
        for category in _findall(entry, "category")
        if category.get("term")
    ]

    return {
        "arxiv_id": arxiv_id,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        "pdf_url": pdf_url,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published": published[:10] if published else "",
        "categories": categories,
    }


def find_paper(query: str, title: str = "", presenter: str = "") -> dict | None:
    query = (query or "").strip()
    s = get_settings()
    params: dict[str, str | int] = {"max_results": 1}
    paper_id = _paper_id_from_query(query)
    is_identifier = paper_id is not None

    if is_identifier:
        params["id_list"] = paper_id
    else:
        if not title.strip():
            return None
        params["search_query"] = f'ti:"{title.strip()}"'
        params["start"] = 0
        params["max_results"] = 5

    try:
        with httpx.Client(timeout=s.arxiv_timeout_seconds,
                          headers=_ARXIV_API_HEADERS) as client:
            response = _get_with_retry(client, ARXIV_API_URL, params=params)
    except Exception as exc:
        logger.error("arXiv API request failed: %s", exc, exc_info=True)
        raise RuntimeError("Could not reach the arXiv API.") from exc

    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"arXiv request failed: HTTP {response.status_code}")

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        raise RuntimeError("Could not parse the arXiv response.") from None

    raw_root = root if _local_tag(root.tag) == "entry" else root
    entries = (
        [raw_root]
        if _local_tag(raw_root.tag) == "entry"
        else [entry for entry in raw_root if _local_tag(entry.tag) == "entry"]
    )

    candidates = []
    for entry in entries:
        paper = _parse_entry(entry)
        if is_identifier and paper.get("arxiv_id", "").split("v", 1)[0] == paper_id.split("v", 1)[0]:
            paper["score"] = 100.0
        else:
            paper["score"] = _title_score(paper.get("title", ""), title)
        normalized_presenter = _normalize(presenter)
        if normalized_presenter:
            author_names = " ".join(_normalize(a) for a in paper.get("authors", []))
            if normalized_presenter in author_names:
                paper["score"] += 10
        candidates.append(paper)

    if not candidates:
        return None

    best = max(candidates, key=lambda paper: paper["score"])
    if best["score"] < 65:
        return None

    best.pop("score", None)
    return best


def _titles_match(extracted: str, expected: str) -> bool:
    return bool(extracted and expected and _title_score(extracted, expected) >= 65)


def _looks_like_html(body: bytes, content_type: str) -> bool:
    if "text/html" in content_type:
        return True
    return body.lstrip().startswith(b"<")


def extract_paper(paper: dict) -> dict:
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        raise RuntimeError("arXiv paper is missing a PDF URL.")
    s = get_settings()
    try:
        with httpx.Client(timeout=s.arxiv_timeout_seconds,
                          headers=_BROWSER_HEADERS,
                          follow_redirects=True) as client:
            response = _get_with_retry(client, pdf_url)
    except Exception as exc:
        logger.error("arXiv PDF download failed: %s", exc, exc_info=True)
        raise RuntimeError("Could not download the arXiv PDF.") from exc

    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"arXiv PDF request failed: HTTP {response.status_code}")

    headers = getattr(response, "headers", {}) or {}
    content_type = (headers.get("content-type") or "").lower()
    html_metadata: dict | None = None
    try:
        parsed = pdf_to_markdown(response.content)
    except Exception as exc:
        if _looks_like_html(response.content, content_type):
            html = response.content.decode("utf-8", errors="replace")
            html_result = html_to_markdown(response.content, url=pdf_url)
            text = _strip_references(html_result["markdown"])
            if len(text) < 500:
                raise RuntimeError(
                    f"Downloaded file is not a readable PDF (content-type: {content_type or 'unknown'})."
                ) from None
            expected_title = (paper.get("title") or "").strip()
            html_title = html_result["title"]
            if expected_title and html_title and not _titles_match(html_title, expected_title):
                raise RuntimeError(
                    f"HTML article title \"{html_title}\" does not match the scheduled "
                    f"paper title \"{expected_title}\"."
                )
            html_metadata = _extract_html_metadata(html)
        else:
            raise RuntimeError(
                f"Downloaded file is not a readable PDF (content-type: {content_type or 'unknown'})."
            ) from exc
    else:
        text = _strip_references(parsed["markdown"])
        if len(text) < 500:
            raise RuntimeError("Could not extract enough text from the arXiv PDF.")
        if parsed["pages_needing_ocr"]:
            pages = ", ".join(map(str, parsed["pages_needing_ocr"]))
            text += f"\n\n[Note: text extraction is incomplete for PDF pages {pages}.]"

    paper_text = text[:PDF_TEXT_LIMIT]
    result = dict(paper)
    result["paper_text"] = paper_text
    result["paper_text_truncated"] = len(text) > PDF_TEXT_LIMIT
    if html_metadata:
        result["html_metadata"] = html_metadata
    return result