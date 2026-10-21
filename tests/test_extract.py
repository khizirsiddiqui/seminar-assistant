import io

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older PyMuPDF releases
    import fitz

from app import arxiv
from app.document_parser import pdf_to_markdown
from app.extract import extract_text


def test_extract_pdf():
    with open("tests/fixtures/sample.pdf", "rb") as f:
        text = extract_text(f, "sample.pdf", "application/pdf")
    assert text.startswith("<!-- Page 1 -->")
    assert "Attention Is All You Need" in text


def test_extract_text_passthrough():
    assert extract_text(io.BytesIO("hello".encode()), "x.txt", "text/plain") == "hello"


def test_pdf_to_markdown_adds_page_markers_and_dehyphenates():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 720), "Hel-")
    page.insert_text((72, 700), "lo world")
    second = doc.new_page()
    second.insert_text((72, 720), "Second page")
    data = doc.tobytes()
    doc.close()

    parsed = pdf_to_markdown(data)

    assert parsed["markdown"].startswith("<!-- Page 1 -->")
    assert "<!-- Page 2 -->" in parsed["markdown"]
    assert "Hello world" in parsed["markdown"]
    assert parsed["pages_needing_ocr"] == []


def test_pdf_to_markdown_includes_hyperlink_annotations():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 720), "My Paper Title")
    page.insert_link({
        "kind": fitz.LINK_URI,
        "from": fitz.Rect(72, 700, 300, 730),
        "uri": "https://example.com/paper",
    })
    data = doc.tobytes()
    doc.close()

    parsed = pdf_to_markdown(data)

    assert "https://example.com/paper" in parsed["markdown"]
    assert "[My Paper Title]" in parsed["markdown"]


def test_pdf_to_markdown_reports_blank_page():
    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()

    parsed = pdf_to_markdown(data)

    assert parsed["pages_needing_ocr"] == [1]


def test_extract_paper_appends_ocr_notice_for_blank_pages(monkeypatch):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 750), "Attention is all you need. " * 20)
    doc.new_page()
    pdf_data = doc.tobytes()
    doc.close()

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        content = pdf_data

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(arxiv.httpx, "Client", FakeClient)

    result = arxiv.extract_paper({
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "title": "Attention Is All You Need",
        "arxiv_id": "",
        "url": "",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
    })

    assert "<!-- Page 1 -->" in result["paper_text"]
    assert "<!-- Page 2 -->" in result["paper_text"]
    assert "[Note: text extraction is incomplete for PDF pages 2.]" in result["paper_text"]
    assert result["paper_text_truncated"] is False


def test_extract_paper_uses_trafilatura_markdown_and_strips_noise(monkeypatch):
    body = (
        "<html><head><title>Attention Is All You Need | Nature</title></head><body>"
        "<nav class=\"nav\"><a href=\"/x\">Nav nonsense</a><span>More nav</span></nav>"
        "<article><h1>Attention Is All You Need</h1><p>"
        + "This is the article text. " * 60
        + "</p></article><footer>Footer nonsense</footer></body></html>"
    ).encode()

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=UTF-8"}
        content = body

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(arxiv.httpx, "Client", FakeClient)

    result = arxiv.extract_paper({
        "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4",
        "title": "Attention Is All You Need",
        "arxiv_id": "",
        "url": "",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
    })

    assert "Nav nonsense" not in result["paper_text"]
    assert "Footer nonsense" not in result["paper_text"]
    assert "# Attention Is All You Need" in result["paper_text"]
    assert "This is the article text." in result["paper_text"]
    assert len(result["paper_text"]) >= 500
    assert result["paper_text_truncated"] is False


def test_extract_paper_pulls_citation_meta_from_html(monkeypatch):
    body = (
        "<html><head><title>Attention Is All You Need | Nature</title>"
        '<meta name="citation_author" content="Hao, Qianyue">'
        '<meta name="citation_author" content="Xu, Fengli">'
        '<meta name="citation_author" content="Arlt, S&#246;ren">'
        '<meta name="citation_online_date" content="2026/01/14">'
        '<meta name="dc.subject" content="Machine Learning; Artificial Intelligence">'
        "</head><body><h1>Attention Is All You Need</h1><p>"
        + "This is the article text. " * 60
        + "</p></body></html>"
    ).encode()

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=UTF-8"}
        content = body

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResponse()

    monkeypatch.setattr(arxiv.httpx, "Client", FakeClient)

    result = arxiv.extract_paper({
        "pdf_url": "https://www.nature.com/articles/s41586-025-09142-4",
        "title": "Attention Is All You Need",
        "arxiv_id": "",
        "url": "",
        "authors": [],
        "published": "",
        "categories": [],
        "abstract": "",
    })

    assert result["html_metadata"]["authors"] == ["Qianyue Hao", "Fengli Xu", "Sören Arlt"]
    assert result["html_metadata"]["published"] == "2026-01-14"
    assert result["html_metadata"]["categories"] == ["Machine Learning", "Artificial Intelligence"]
