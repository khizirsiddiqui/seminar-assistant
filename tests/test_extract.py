import io

from app.extract import extract_text


def test_extract_pdf():
    with open("tests/fixtures/sample.pdf", "rb") as f:
        assert "Attention Is All You Need" in extract_text(f, "sample.pdf", "application/pdf")


def test_extract_text_passthrough():
    assert extract_text(io.BytesIO("hello".encode()), "x.txt", "text/plain") == "hello"
