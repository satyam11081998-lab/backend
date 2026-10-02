"""Upload validation (spec §7, §92): type by magic bytes (never trust the extension or the
browser's MIME), size caps, zip-bomb and macro checks for DOCX, encrypted-PDF detection."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

from ..errors import Unprocessable

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
MAX_ZIP_UNCOMPRESSED = 60 * 1024 * 1024
MAX_ZIP_ENTRIES = 3000
MAX_ZIP_RATIO = 120

MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
}


@dataclass(frozen=True)
class DetectedFile:
    ext: str
    mime: str


def _ext(name: str) -> str:
    name = (name or "").lower().strip()
    return name.rsplit(".", 1)[-1] if "." in name else ""


def detect(data: bytes, filename: str, max_bytes: int) -> DetectedFile:
    if not data:
        raise Unprocessable("The file is empty. Please choose the file again.", code="empty_file")
    if len(data) > max_bytes:
        mb = max_bytes // (1024 * 1024)
        raise Unprocessable(f"The file is larger than {mb} MB. Please upload a smaller version.",
                            code="file_too_large")
    claimed = _ext(filename)
    if data.startswith(PDF_MAGIC):
        kind = "pdf"
    elif data.startswith(ZIP_MAGIC):
        kind = "docx"
    elif data.startswith(OLE_MAGIC):
        kind = "doc"
    else:
        raise Unprocessable("Unsupported file. Upload your document as PDF, DOC or DOCX.", code="unsupported_type")
    if claimed and claimed not in {"pdf", "doc", "docx"}:
        raise Unprocessable("Unsupported file. Upload your document as PDF, DOC or DOCX.", code="unsupported_type")
    if claimed and claimed != kind and not (claimed == "doc" and kind == "docx"):
        # A .doc that is really a .docx is common (renamed files) and harmless; anything else is not.
        raise Unprocessable("The file's contents don't match its extension. Re-save it as PDF or DOCX and try again.",
                            code="type_mismatch")
    if kind == "docx":
        _check_docx(data)
    if kind == "pdf" and b"/Encrypt" in data[:200000] and b"/Encrypt" in data:
        # Many CVs exported with "restrict editing" are encrypted with an empty user password and
        # still open; pdfium decides for real in the parser. This is only an early hint.
        pass
    return DetectedFile(ext=kind, mime=MIME[kind])


def _check_docx(data: bytes) -> None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise Unprocessable("This DOCX file appears to be damaged. Re-save it and try again.", code="corrupt_docx")
    infos = zf.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        raise Unprocessable("This document is too complex to process safely.", code="zip_entries")
    total = 0
    names = set()
    for i in infos:
        total += i.file_size
        names.add(i.filename)
        if i.compress_size and i.file_size / max(i.compress_size, 1) > MAX_ZIP_RATIO and i.file_size > 1_000_000:
            raise Unprocessable("This document could not be processed safely.", code="zip_bomb")
        low = i.filename.lower()
        if low.endswith("vbaproject.bin") or low.endswith(".bin") and "vba" in low:
            raise Unprocessable("Documents containing macros aren't accepted. Save it as a regular DOCX or PDF.",
                                code="macro_document")
    if total > MAX_ZIP_UNCOMPRESSED:
        raise Unprocessable("This document is too large to process safely.", code="zip_too_large")
    if "word/document.xml" not in names or "[Content_Types].xml" not in names:
        raise Unprocessable("This doesn't look like a Word document. Upload a PDF, DOC or DOCX.", code="not_word")


def validate_pasted_text(text: str, *, min_chars: int = 80, max_chars: int = 40000) -> str:
    t = (text or "").replace("\x00", "").strip()
    if len(t) < min_chars:
        raise Unprocessable("The job description is too short for a role-specific interview. Paste the full JD.",
                            code="jd_too_short")
    if len(t) > max_chars:
        raise Unprocessable(f"The job description is too long (over {max_chars:,} characters).", code="jd_too_long")
    return t
