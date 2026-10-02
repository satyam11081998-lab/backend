"""Text extraction for PDF / DOCX / DOC with quality measurement (spec §7, §54).

Heavy libraries are imported inside the functions so they only load when a document is
actually parsed (memory budget on a small instance)."""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from typing import List, Optional

MAX_CHARS = {"cv": 60000, "jd": 40000}
MIN_CHARS = {"cv": 250, "jd": 150}
MAX_PAGES = 40


@dataclass
class ParseResult:
    ok: bool
    text: str = ""
    page_count: Optional[int] = None
    quality: dict = field(default_factory=dict)
    error_code: str = ""
    user_message: str = ""
    warnings: List[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\x00", "").replace("­", "")
    t = re.sub(r"[​-‏﻿]", "", t)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)            # de-hyphenate line breaks
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    return t.strip()


def _quality(text: str, pages: Optional[int]) -> dict:
    n = len(text)
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    letters = sum(1 for ch in text if ch.isalpha())
    words = len(re.findall(r"[A-Za-zÀ-ɏऀ-ॿ]{2,}", text))
    return {
        "chars": n,
        "words": words,
        "pages": pages,
        "chars_per_page": round(n / pages, 1) if pages else None,
        "garbled_ratio": round(1 - printable / n, 4) if n else 0.0,
        "letter_ratio": round(letters / n, 3) if n else 0.0,
        "replacement_chars": text.count("�"),
    }


def _finish(kind: str, raw: str, pages: Optional[int], warnings: List[str]) -> ParseResult:
    text = normalize_text(raw)
    q = _quality(text, pages)
    if len(text) > MAX_CHARS[kind]:
        text = text[:MAX_CHARS[kind]]
        warnings.append(f"Document truncated to the first {MAX_CHARS[kind]:,} characters.")
        q["truncated"] = True
    q["warnings"] = warnings
    if len(text) < MIN_CHARS[kind] or q["words"] < (40 if kind == "cv" else 25):
        if pages and q["chars_per_page"] is not None and q["chars_per_page"] < 80:
            return ParseResult(False, text, pages, q, "image_only",
                               "This file looks like a scanned image, so its text can't be read reliably. "
                               "Upload a text-based PDF or a DOCX.", warnings)
        return ParseResult(False, text, pages, q, "too_little_text",
                           "We couldn't find enough readable text in this file. Please re-upload it as a PDF or DOCX.",
                           warnings)
    if q["garbled_ratio"] > 0.08 or q["letter_ratio"] < 0.45 or q["replacement_chars"] > 50:
        return ParseResult(False, text, pages, q, "garbled",
                           "We couldn't reliably read this file (the text came out garbled). "
                           "Re-save it as a PDF or DOCX and upload again.", warnings)
    return ParseResult(True, text, pages, q, warnings=warnings)


def parse_pdf(data: bytes, kind: str) -> ParseResult:
    import pypdfium2 as pdfium

    warnings: List[str] = []
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        msg = str(e).lower()
        if "password" in msg:
            return ParseResult(False, error_code="password_protected",
                               user_message="This PDF is password-protected. Remove the password and upload again.")
        return ParseResult(False, error_code="corrupt_pdf",
                           user_message="This PDF appears to be damaged. Re-export it and try again.")
    try:
        n = len(pdf)
        if n > MAX_PAGES:
            warnings.append(f"Only the first {MAX_PAGES} pages were read.")
        parts = []
        for i in range(min(n, MAX_PAGES)):
            page = pdf[i]
            tp = page.get_textpage()
            parts.append(tp.get_text_range())
            tp.close()
            page.close()
        return _finish(kind, "\n\n".join(parts), n, warnings)
    finally:
        pdf.close()


def _docx_ordered_text(data: bytes) -> str:
    import docx  # python-docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(data))
    out: List[str] = []
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            txt = Paragraph(child, d).text.strip()
            if txt:
                out.append(txt)
        elif tag == "tbl":
            table = Table(child, d)
            for row in table.rows:
                cells = []
                seen = set()
                for c in row.cells:
                    t = c.text.strip()
                    if t and id(c._tc) not in seen:
                        cells.append(t)
                        seen.add(id(c._tc))
                if cells:
                    out.append(" | ".join(cells))
    return "\n".join(out)


def _docx_all_runs(data: bytes) -> str:
    """Fallback for text boxes / shapes that python-docx's body walk misses."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    texts = re.findall(r"<w:t[^>]*>([^<]*)</w:t>|\n", xml)
    joined = "".join(t if t else "\n" for t in texts)
    import html
    return html.unescape(joined)


def parse_docx(data: bytes, kind: str) -> ParseResult:
    warnings: List[str] = []
    try:
        text = _docx_ordered_text(data)
    except Exception:
        text = ""
    if len(text) < MIN_CHARS[kind] * 2:
        try:
            alt = _docx_all_runs(data)
            if len(alt) > len(text) * 1.3:
                text = alt
                warnings.append("Some text was inside text boxes; reading order may be approximate.")
        except Exception:
            pass
    if not text:
        return ParseResult(False, error_code="corrupt_docx",
                           user_message="We couldn't read this DOCX file. Re-save it and try again.")
    return _finish(kind, text, None, warnings)


def parse_doc(data: bytes, kind: str) -> ParseResult:
    """Legacy binary .doc: best-effort text recovery from the WordDocument stream. When the
    result is not clearly readable we say so instead of passing garbage to the model."""
    import olefile

    try:
        ole = olefile.OleFileIO(io.BytesIO(data))
    except Exception:
        return ParseResult(False, error_code="corrupt_doc",
                           user_message="We couldn't read this .doc file. Save it as PDF or DOCX and upload again.")
    try:
        if not ole.exists("WordDocument"):
            return ParseResult(False, error_code="not_word",
                               user_message="This doesn't look like a Word document. Upload a PDF or DOCX.")
        stream = ole.openstream("WordDocument").read()
    finally:
        ole.close()
    candidates = []
    for enc in ("utf-16-le", "cp1252"):
        decoded = stream.decode(enc, errors="ignore")
        runs = re.findall(r"[\w\s,.;:()'\"&%/+\-@#€₹$]{4,}", decoded)
        txt = "\n".join(r.strip() for r in runs if sum(ch.isalpha() for ch in r) >= 3)
        candidates.append(txt)
    best = max(candidates, key=lambda t: sum(ch.isalpha() for ch in t))
    res = _finish(kind, best, None, ["Read from a legacy .doc file; formatting may be lost."])
    if res.ok and res.quality.get("words", 0) < 80:
        res.ok = False
        res.error_code = "doc_unreliable"
        res.user_message = ("We couldn't reliably read this .doc file. Save it as PDF or DOCX and upload again.")
    return res


def parse(data: bytes, ext: str, kind: str) -> ParseResult:
    if ext == "pdf":
        return parse_pdf(data, kind)
    if ext == "docx":
        return parse_docx(data, kind)
    if ext == "doc":
        return parse_doc(data, kind)
    return ParseResult(False, error_code="unsupported_type", user_message="Unsupported file type.")


def parse_plain(text: str, kind: str) -> ParseResult:
    return _finish(kind, text, None, [])
