"""Extract raw text from supported upload types (.pdf, .txt)."""
import io

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

SUPPORTED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
}


def _kind_from_upload(file: UploadFile) -> str:
    if file.content_type in SUPPORTED_CONTENT_TYPES:
        return SUPPORTED_CONTENT_TYPES[file.content_type]
    # Fall back to file extension since browsers/clients don't always set
    # content_type reliably for .txt files.
    if file.filename and file.filename.lower().endswith(".pdf"):
        return "pdf"
    if file.filename and file.filename.lower().endswith(".txt"):
        return "txt"
    raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")


def extract_text(raw_bytes: bytes, file: UploadFile) -> str:
    """Extract plain text from an uploaded PDF or TXT file.

    Raises HTTPException(400) for unsupported file types or unreadable PDFs.
    """
    kind = _kind_from_upload(file)

    if kind == "txt":
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1")

    # kind == "pdf"
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    pages_text = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages_text).strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF (it may be a scanned image without OCR).",
        )
    return text
