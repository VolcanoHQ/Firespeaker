#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import io
import os
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from typing import Literal, Optional
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field

UPLOAD_ROOT = os.path.join("data", "uploads")
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
SUPPORTED_FORMATS = ("txt", "docx", "epub")
_PRINTABLE_SAMPLE = 20_000
_PRINTABLE_THRESHOLD = 0.9
_EPUB_NS = {
    "cn": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
}
_EPUB_SKIP_NAME = re.compile(
    r"cover|titlepage|title-page|halftitle|nav|toc|contents|copyright|colophon|"
    r"dedication|imprint|frontmatter|backmatter|acknowledg|about-the|advert",
    re.IGNORECASE,
)


class UploadRequest(BaseModel):
    filename: str
    raw_bytes: bytes | None = None
    text: str | None = None
    content_type: str | None = None
    tier: int = 1
    surface: Literal["express", "studio", "console", "fastapi", "unknown"] = "unknown"


class UploadValidation(BaseModel):
    ok: bool = True
    book: str
    filename: str
    format: Literal["txt", "docx", "epub"]
    bytes: int
    text: str | None = None
    raw_bytes: bytes | None = None
    content_type: str | None = None
    tier: int = 1
    surface: Literal["express", "studio", "console", "fastapi", "unknown"] = "unknown"
    next_action: Literal["analyze", "review_structure"] = "analyze"
    message: str = "Upload validated."


class UploadResult(BaseModel):
    ok: bool = True
    book: str
    filename: str
    source_file: str
    format: Literal["txt", "docx", "epub"]
    bytes: int
    status: Literal["uploaded", "validated"]
    next_action: Literal["analyze", "review_structure"]
    message: str


class UploadValidationError(BaseModel):
    ok: bool = False
    code: Literal[
        "MISSING_FILE",
        "UNSUPPORTED_FORMAT",
        "FILE_TOO_LARGE",
        "TEXT_DECODE_FAILED",
        "INVALID_DOCX",
        "INVALID_EPUB",
        "EMPTY_EPUB",
        "SPOOFED_EXTENSION",
    ]
    message: str
    details: str | None = None
    supported_formats: list[str] = Field(default_factory=lambda: list(SUPPORTED_FORMATS))


class UploadContractError(Exception):
    def __init__(self, error: UploadValidationError, *, status_code: int = 400):
        super().__init__(error.message)
        self.error = error
        self.status_code = status_code


class _HTMLTextProbe(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "head", "title"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "li", "blockquote", "section", "article", "br", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "head", "title"}:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in {"p", "div", "li", "blockquote", "section", "article", "br", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r" ?\n ?", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def request_from_json(body: dict, *, surface: str) -> UploadRequest:
    filename = str(body.get("filename") or "")
    tier = int(body.get("tier") or 1)
    content_type = body.get("content_type")
    text = body.get("text")
    raw_bytes = None

    data_value = body.get("data")
    if isinstance(data_value, str) and data_value.startswith("data:"):
        content_type, raw_bytes = _decode_data_url(data_value, fallback_content_type=content_type)
    elif isinstance(text, str) and text.startswith("data:"):
        content_type, raw_bytes = _decode_data_url(text, fallback_content_type=content_type)
        text = None
    elif filename and "." not in os.path.basename(filename) and isinstance(text, str):
        filename = f"{filename}.txt"

    return UploadRequest(
        filename=filename,
        raw_bytes=raw_bytes,
        text=text,
        content_type=content_type,
        tier=tier,
        surface=surface,
    )


async def request_from_fastapi_upload(file, *, surface: str, tier: int = 1) -> UploadRequest:
    raw_bytes = await file.read()
    return UploadRequest(
        filename=getattr(file, "filename", "") or "",
        raw_bytes=raw_bytes,
        content_type=getattr(file, "content_type", None),
        tier=tier,
        surface=surface,
    )


def validate_upload(request: UploadRequest) -> UploadValidation:
    if not request.filename:
        _raise_upload_error("MISSING_FILE", "Missing upload filename.")
    if request.raw_bytes is None and (request.text is None or not request.text.strip()):
        _raise_upload_error("MISSING_FILE", "Missing upload payload.")

    normalized_filename = _normalize_filename(request.filename)
    book = _book_stem(normalized_filename)
    upload_format = _detect_format(normalized_filename, request.content_type)

    if upload_format == "txt":
        text = _validate_txt(request)
        size_bytes = len(text.encode("utf-8"))
        _enforce_size_cap(size_bytes)
        return UploadValidation(
            book=book,
            filename=normalized_filename,
            format="txt",
            bytes=size_bytes,
            text=text,
            content_type=request.content_type,
            tier=request.tier,
            surface=request.surface,
        )

    if request.raw_bytes is None:
        _raise_upload_error("MISSING_FILE", f"Missing binary payload for .{upload_format} upload.")

    _enforce_size_cap(len(request.raw_bytes))
    if upload_format == "docx":
        _validate_docx(request.raw_bytes)
    elif upload_format == "epub":
        _validate_epub(request.raw_bytes)

    return UploadValidation(
        book=book,
        filename=normalized_filename,
        format=upload_format,
        bytes=len(request.raw_bytes),
        raw_bytes=request.raw_bytes,
        content_type=request.content_type,
        tier=request.tier,
        surface=request.surface,
    )


def process_upload(request: UploadRequest) -> UploadResult:
    validation = validate_upload(request)
    source_file = _save_upload(validation)
    return UploadResult(
        book=validation.book,
        filename=validation.filename,
        source_file=source_file,
        format=validation.format,
        bytes=validation.bytes,
        status="uploaded",
        next_action=validation.next_action,
        message=f"{validation.format.upper()} upload saved and ready for Tier 1 ingestion.",
    )


def ingest_upload(result: UploadResult, *, enable_llm_enrichment: bool = False):
    from src.tier_1_parser import ingest_manuscript_tier_1

    return ingest_manuscript_tier_1(result.source_file, enable_llm_enrichment=enable_llm_enrichment)


def error_response(error: UploadValidationError) -> dict:
    return {"ok": False, "error": error.model_dump()}


def success_response(result: UploadResult) -> dict:
    return result.model_dump()


def http_status_for_error(error: UploadValidationError) -> int:
    if error.code == "FILE_TOO_LARGE":
        return 413
    return 400


def _normalize_filename(filename: str) -> str:
    base = os.path.basename((filename or "").strip())
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._\- ]", "", stem).strip(" ._-") or "uploaded_book"
    safe_ext = re.sub(r"[^A-Za-z0-9.]", "", ext.lower())
    return f"{stem}{safe_ext}"


def _detect_format(filename: str, content_type: Optional[str]) -> Literal["txt", "docx", "epub"]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return "txt"
    if ext == ".docx":
        return "docx"
    if ext == ".epub":
        return "epub"
    hint = f" MIME hint: {content_type}." if content_type else ""
    _raise_upload_error(
        "UNSUPPORTED_FORMAT",
        "Unsupported upload format.",
        details=f"Supported formats are .txt, .docx, and .epub.{hint}",
    )


def _validate_txt(request: UploadRequest) -> str:
    if request.text is not None:
        text = request.text
    else:
        assert request.raw_bytes is not None
        if _is_zip_bytes(request.raw_bytes):
            _raise_upload_error(
                "SPOOFED_EXTENSION",
                "The .txt upload looks like a ZIP-based binary document.",
                details="Use the original .docx or .epub extension when uploading binary manuscript files.",
            )
        try:
            text = request.raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            _raise_upload_error(
                "TEXT_DECODE_FAILED",
                "The .txt upload is not valid UTF-8 text.",
                details=str(exc),
            )

    if not text.strip():
        _raise_upload_error("MISSING_FILE", "The uploaded manuscript is empty.")
    if _printable_ratio(text) < _PRINTABLE_THRESHOLD:
        _raise_upload_error(
            "SPOOFED_EXTENSION",
            "The .txt upload does not look like plain text.",
            details="Binary or heavily non-printable payloads must be uploaded with their real extension.",
        )
    return text


def _validate_docx(raw_bytes: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as docx:
            if "word/document.xml" not in docx.namelist():
                _raise_upload_error(
                    "INVALID_DOCX",
                    "The DOCX upload is missing word/document.xml.",
                )
    except UploadContractError:
        raise
    except zipfile.BadZipFile as exc:
        _raise_upload_error("INVALID_DOCX", "The DOCX upload is not a valid ZIP package.", details=str(exc))


def _validate_epub(raw_bytes: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as epub:
            if "META-INF/container.xml" not in epub.namelist():
                _raise_upload_error("INVALID_EPUB", "The EPUB is missing META-INF/container.xml.")
            try:
                container = ET.fromstring(epub.read("META-INF/container.xml"))
            except ET.ParseError as exc:
                _raise_upload_error("INVALID_EPUB", "The EPUB container.xml is not valid XML.", details=str(exc))
            rootfile = container.find(".//cn:rootfile", _EPUB_NS)
            if rootfile is None or not rootfile.get("full-path"):
                _raise_upload_error("INVALID_EPUB", "The EPUB container does not point to an OPF package.")
            opf_path = rootfile.get("full-path")
            if opf_path not in epub.namelist():
                _raise_upload_error("INVALID_EPUB", "The EPUB package OPF listed in container.xml is missing.")

            try:
                package = ET.fromstring(epub.read(opf_path))
            except ET.ParseError as exc:
                _raise_upload_error("INVALID_EPUB", "The EPUB OPF package is not valid XML.", details=str(exc))

            manifest = {
                item.get("id"): (
                    item.get("href"),
                    item.get("media-type", ""),
                    item.get("properties", ""),
                )
                for item in package.findall(".//opf:manifest/opf:item", _EPUB_NS)
                if item.get("id") and item.get("href")
            }
            spine = package.findall(".//opf:spine/opf:itemref", _EPUB_NS)
            if not manifest or not spine:
                _raise_upload_error("EMPTY_EPUB", "The EPUB does not contain a usable manifest/spine.")

            opf_dir = posixpath.dirname(opf_path)
            usable_docs = 0
            for itemref in spine:
                entry = manifest.get(itemref.get("idref"))
                if not entry:
                    continue
                href, media_type, props = entry
                media_lower = (media_type or "").lower()
                if "html" not in media_lower and not href.lower().endswith((".xhtml", ".html", ".htm")):
                    continue
                if "nav" in (props or "").lower():
                    continue
                archive_path = posixpath.join(opf_dir, href) if opf_dir else href
                name_probe = f"{itemref.get('idref', '')} {posixpath.basename(archive_path)}"
                if _EPUB_SKIP_NAME.search(name_probe):
                    continue
                try:
                    content = epub.read(archive_path).decode("utf-8", errors="replace")
                except KeyError:
                    continue
                text = _extract_html_text(content)
                if len(re.sub(r"\W+", "", text)) >= 20:
                    usable_docs += 1
            if usable_docs == 0:
                _raise_upload_error(
                    "EMPTY_EPUB",
                    "The EPUB spine does not contain any readable content documents.",
                )
    except UploadContractError:
        raise
    except zipfile.BadZipFile as exc:
        _raise_upload_error("INVALID_EPUB", "The EPUB upload is not a valid ZIP package.", details=str(exc))


def _save_upload(validation: UploadValidation) -> str:
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    source_file = os.path.join(UPLOAD_ROOT, validation.filename)
    temp_file = f"{source_file}.tmp"
    if validation.format == "txt":
        with open(temp_file, "w", encoding="utf-8", newline="") as handle:
            handle.write(validation.text or "")
    else:
        with open(temp_file, "wb") as handle:
            handle.write(validation.raw_bytes or b"")
    os.replace(temp_file, source_file)
    return source_file


def _book_stem(filename: str) -> str:
    stem = os.path.splitext(filename)[0].strip()
    return stem or "uploaded_book"


def _decode_data_url(value: str, *, fallback_content_type: Optional[str]) -> tuple[Optional[str], bytes]:
    if "," not in value:
        _raise_upload_error("MISSING_FILE", "Malformed data URL upload payload.")
    header, encoded = value.split(",", 1)
    match = re.match(r"^data:(?P<content_type>[^;,]+)?(?:;base64)?$", header)
    content_type = fallback_content_type
    if match and match.group("content_type"):
        content_type = match.group("content_type")
    try:
        return content_type, base64.b64decode(encoded)
    except Exception as exc:
        _raise_upload_error("MISSING_FILE", "Malformed data URL upload payload.", details=str(exc))


def _enforce_size_cap(size_bytes: int) -> None:
    if size_bytes > MAX_UPLOAD_BYTES:
        _raise_upload_error(
            "FILE_TOO_LARGE",
            f"File too large ({MAX_UPLOAD_BYTES // (1024 * 1024)}MB cap).",
        )


def _is_zip_bytes(raw_bytes: bytes) -> bool:
    if len(raw_bytes) < 4 or not raw_bytes.startswith(b"PK"):
        return False
    try:
        return zipfile.is_zipfile(io.BytesIO(raw_bytes))
    except Exception:
        return False


def _printable_ratio(text: str) -> float:
    sample = text[:_PRINTABLE_SAMPLE]
    if not sample:
        return 1.0
    printable = sum(1 for ch in sample if ch.isprintable() or ch in "\n\r\t")
    return printable / len(sample)


def _extract_html_text(content: str) -> str:
    parser = _HTMLTextProbe()
    parser.feed(content)
    return parser.text()


def _raise_upload_error(code: str, message: str, details: Optional[str] = None) -> None:
    error = UploadValidationError(code=code, message=message, details=details)
    raise UploadContractError(error, status_code=http_status_for_error(error))
