from app.document_parser import pdf_to_markdown


def extract_text(fileobj, filename: str, content_type: str) -> str:
    """Extract omp-style Markdown from an uploaded PDF, or raw text for csv/txt."""
    data = fileobj.read()
    if filename.lower().endswith(".pdf") or content_type == "application/pdf":
        return pdf_to_markdown(data)["markdown"]
    return data.decode("utf-8", errors="replace")
