import io

from pypdf import PdfReader


def extract_text(fileobj, filename: str, content_type: str) -> str:
    """Extract plain text from an uploaded file (PDF) or return raw text for csv/txt."""
    data = fileobj.read()
    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")
