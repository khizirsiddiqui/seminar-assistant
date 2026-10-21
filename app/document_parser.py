"""Document parsing helpers.

This module mirrors the omp ``markit`` converter behaviour in Python:

* PDFs are converted to plain-text Markdown with ``<!-- Page N -->`` page
  markers, basic dehyphenation, and empty-page detection.
* HTML pages are converted to Markdown with trafilatura (Readability-style
  extraction).  If trafilatura is unavailable or returns nothing, a simple
  tag-stripping fallback is used.
"""
from __future__ import annotations

import re

try:
    import pymupdf as fitz  # PyMuPDF

    HAS_FITZ = True
except Exception:  # pragma: no cover - exercised only when deps are missing
    try:
        import fitz  # older PyMuPDF releases expose this name

        HAS_FITZ = True
    except Exception:  # pragma: no cover - exercised only when deps are missing
        HAS_FITZ = False

try:
    import trafilatura

    HAS_TRAFILATURA = True
except Exception:  # pragma: no cover - exercised only when deps are missing
    HAS_TRAFILATURA = False


def _clean_title(title: str) -> str:
    """Return a stripped metadata title, dropping filename-like values."""
    title = (title or "").strip()
    if title.lower().endswith(".pdf"):
        return ""
    return title


def _normalize_pdf_text(text: str) -> str:
    """Wrap page-level PDF text into continuous paragraphs.

    ``page.get_text("text")`` usually emits one line per visual line.  We
    dehyphenate line-wrapped words and join the remaining single newlines so
    LLM prompts receive flowing prose rather than ragged line breaks.
    """
    # Join a trailing hyphen with the next line before newlines are collapsed.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse single line breaks inside paragraphs, but keep blank-line
    # paragraph separators intact.
    text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)
    # Normalise accidental spacing left around the join.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _page_link_items(page) -> list[tuple[str, str]]:
    """Return ``(label, url)`` pairs for hyperlink annotations on a page.

    ``get_text("text")`` does not include link targets, so without this step
    URLs that only exist as clickable PDF annotations would never reach the
    LLM.  We group all link rectangles by URL and rebuild the visible label
    from the words they cover.
    """
    words = page.get_text("words")
    groups: dict[str, list] = {}
    for link in page.links() or []:
        uri = link.get("uri") or ""
        if link.get("kind") != 2 or not uri:
            continue
        if not (uri.startswith("http://") or uri.startswith("https://")):
            continue
        rect = fitz.Rect(link.get("from") or (0, 0, 0, 0))
        groups.setdefault(uri, []).append(rect)

    items = []
    for uri, rects in groups.items():
        covered = []
        for word in words:
            word_rect = fitz.Rect(word[:4])
            if any(word_rect.intersects(rect) for rect in rects):
                covered.append((word[1], word[0], word[4]))
        covered.sort(key=lambda item: (round(item[0], 1), item[1]))

        label = ""
        for _, _, word in covered:
            if label.endswith("-"):
                label += word
            elif label:
                label += " " + word
            else:
                label = word
        label = re.sub(r"\s+", " ", label).strip()
        items.append((label or uri, uri))
    return items


def _page_links_markdown(page) -> str:
    """Render a page's hyperlinks as a compact markdown list."""
    items = _page_link_items(page)
    if not items:
        return ""
    lines = ["Links:"]
    for label, uri in items:
        # Escape closing brackets in labels so malformed links don't break
        # the markdown passed to the LLM.
        safe_label = label.replace("]", r"\]").replace("[", r"\[")
        lines.append(f"- [{safe_label}]({uri})")
    return "\n".join(lines)


def pdf_to_markdown(data: bytes) -> dict:
    """Convert PDF bytes to omp-style markdown.

    Returns a dict with ``markdown``, ``title`` and ``pages_needing_ocr``.
    """
    if not HAS_FITZ:
        raise RuntimeError("PDF conversion failed: PyMuPDF is not available.")

    try:
        doc = fitz.open(stream=data, filetype="pdf")
        if not doc.is_pdf:
            # PyMuPDF can opportunistically open other formats even when
            # filetype="pdf" is requested.  Preserve the previous
            # behaviour of treating non-PDF downloads as conversion failures.
            raise RuntimeError("not a PDF document")
    except Exception as exc:
        # Ensure a document opened before the not-PDF check is closed.
        if "doc" in locals():
            doc.close()
        raise RuntimeError(f"PDF conversion failed: {exc}") from exc

    try:
        pages = []
        pages_needing_ocr = []

        for page_number, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            if not raw_text.strip():
                pages_needing_ocr.append(page_number)
            page_text = _normalize_pdf_text(raw_text)
            links_markdown = _page_links_markdown(page)
            if links_markdown:
                page_text = f"{page_text}\n\n{links_markdown}" if page_text else links_markdown
            pages.append(
                f"<!-- Page {page_number} -->\n\n"
                f"{page_text}"
            )

        markdown = "\n\n".join(pages).strip()
        title = _clean_title(doc.metadata.get("title", "")) if doc.metadata else ""
        return {
            "markdown": markdown,
            "title": title,
            "pages_needing_ocr": pages_needing_ocr,
        }
    finally:
        doc.close()


def _html_to_text(body: bytes) -> str:
    """Minimal HTML-to-text fallback used when trafilatura is unavailable."""
    html = body.decode("utf-8", errors="replace")
    html = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    html = re.sub(r"&#\d+;", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def _html_title(body: bytes) -> str:
    html = body.decode("utf-8", errors="replace")
    match = re.search(r"(?is)<title\b[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    title = re.sub(r"(?s)<[^>]+>", " ", match.group(1))
    title = re.sub(r"\s+", " ", title).strip()
    # Strip common site suffixes such as " | Nature" or " - Springer".
    title = re.split(r"\s+[|\-–—]+\s*", title, maxsplit=1)[0].strip()
    return title


def html_to_markdown(body: bytes, url: str = "") -> dict:
    """Convert HTML bytes to omp-style article Markdown.

    Returns a dict with ``markdown`` and ``title``.  If trafilatura cannot
    extract an article (or is not installed), the fallback is the previous
    simple tag-stripping implementation.
    """
    html = body.decode("utf-8", errors="replace")
    markdown = ""

    if HAS_TRAFILATURA:
        try:
            extracted = trafilatura.extract(
                html,
                output_format="markdown",
                include_links=False,
                include_tables=True,
                url=url,
                favor_recall=True,
            )
            if extracted:
                markdown = extracted.strip()
        except Exception:
            # Fall back to the simple extractor rather than failing the whole
            # document fetch because of a parsing library hiccup.
            pass

    if not markdown:
        markdown = _html_to_text(body)

    return {"markdown": markdown, "title": _html_title(body)}