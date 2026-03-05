from __future__ import annotations

import csv
import io
import json
from pathlib import Path


def parse_document(file_name: str, content_bytes: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".txt", ".md", ".log"}:
        return content_bytes.decode("utf-8", errors="ignore")
    if suffix == ".csv":
        return _parse_csv(content_bytes)
    if suffix == ".json":
        return _parse_json(content_bytes)
    if suffix == ".pdf":
        return _parse_pdf(content_bytes)
    if suffix == ".docx":
        return _parse_docx(content_bytes)
    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_csv(content_bytes: bytes) -> str:
    text = content_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    lines = []
    for row in reader:
        if not row:
            continue
        lines.append(" | ".join(col.strip() for col in row))
    return "\n".join(lines)


def _parse_json(content_bytes: bytes) -> str:
    text = content_bytes.decode("utf-8", errors="ignore")
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return text


def _parse_pdf(content_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("pypdf is required for PDF parsing. Install with: pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(content_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def _parse_docx(content_bytes: bytes) -> str:
    try:
        import docx
    except Exception as exc:
        raise RuntimeError(
            "python-docx is required for DOCX parsing. Install with: pip install python-docx"
        ) from exc

    document = docx.Document(io.BytesIO(content_bytes))
    lines = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    return "\n".join(lines)

