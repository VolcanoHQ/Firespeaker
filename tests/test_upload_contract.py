#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import base64
import io
import sys
import types
import zipfile

import pytest

from src import upload_contract as uc


class DummyUploadFile:
    def __init__(self, filename, content, content_type):
        self.filename = filename
        self.content_type = content_type
        self._content = content

    async def read(self):
        return self._content


def _make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buf.getvalue()


def _make_docx_bytes(text="Hello world.\n\nChapter One."):
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(f'<w:p><w:r><w:t>{part}</w:t></w:r></w:p>' for part in text.splitlines() if part)}
  </w:body>
</w:document>
"""
    return _make_zip({"word/document.xml": document_xml})


def _make_epub_bytes(spine_items, manifest_docs, *, include_container=True):
    entries = {"mimetype": "application/epub+zip"}
    if include_container:
        entries["META-INF/container.xml"] = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

    manifest_xml = []
    spine_xml = []
    for item_id, href, media_type, properties in spine_items:
        props_attr = f' properties="{properties}"' if properties else ""
        manifest_xml.append(
            f'<item id="{item_id}" href="{href}" media-type="{media_type}"{props_attr}/>'
        )
        spine_xml.append(f'<itemref idref="{item_id}"/>')

    entries["OEBPS/content.opf"] = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Frankenstein</dc:title>
  </metadata>
  <manifest>
    {''.join(manifest_xml)}
  </manifest>
  <spine>
    {''.join(spine_xml)}
  </spine>
</package>
"""
    for name, content in manifest_docs.items():
        entries[f"OEBPS/{name}"] = content
    return _make_zip(entries)


def _valid_epub_bytes():
    return _make_epub_bytes(
        [
            ("nav", "nav.xhtml", "application/xhtml+xml", "nav"),
            ("c1", "chapter1.xhtml", "application/xhtml+xml", ""),
        ],
        {
            "nav.xhtml": "<html><body><nav epub:type='toc'>TOC</nav></body></html>",
            "chapter1.xhtml": "<html><body><h1>CHAPTER I</h1><p>Hello there.</p><p>This is a valid manuscript chapter with enough readable words.</p></body></html>",
        },
    )


def _letters_epub_bytes():
    letter1_body = (
        "You will rejoice to hear that no disaster has accompanied the commencement of an enterprise "
        "which you have regarded with such evil forebodings. I arrived here yesterday, and my first "
        "task is to assure my dear sister of my welfare and increasing confidence in the success of "
        "my undertaking. The cold is not excessive, if you are wrapped in furs, and the sea remains "
        "still and solemn around us as if it guarded the secret of the pole. "
    ) * 2
    letter2_body = (
        "I am already far north of London, and as I walk in the streets of Petersburgh, I feel a cold "
        "northern breeze play upon my cheeks, which braces my nerves and fills me with delight. This "
        "breeze, which has travelled from the regions towards which I am advancing, gives me a foretaste "
        "of those icy climes. Inspired by this wind of promise, my daydreams become more fervent and vivid, "
        "and I trust that the compass of my purpose will not fail me as I continue northward. "
    ) * 2
    return _make_epub_bytes(
        [
            ("title", "title.xhtml", "application/xhtml+xml", ""),
            ("nav", "nav.xhtml", "application/xhtml+xml", "nav"),
            ("letter1", "letter1.xhtml", "application/xhtml+xml", ""),
            ("letter2", "letter2.xhtml", "application/xhtml+xml", ""),
        ],
        {
            "title.xhtml": "<html><body><h1>Frankenstein</h1><p>By Mary Shelley</p></body></html>",
            "nav.xhtml": "<html><body><nav epub:type='toc'>Letters</nav></body></html>",
            "letter1.xhtml": f"<html><body><h1>LETTER I.</h1><p>{letter1_body}</p></body></html>",
            "letter2.xhtml": f"<html><body><h1>LETTER II.</h1><p>{letter2_body}</p></body></html>",
        },
    )


def _expect_error(exc_info, code):
    assert exc_info.value.error.code == code
    payload = uc.error_response(exc_info.value.error)
    assert payload["ok"] is False
    assert payload["error"]["code"] == code


@pytest.fixture(autouse=True)
def _upload_root(tmp_path, monkeypatch):
    monkeypatch.setattr(uc, "UPLOAD_ROOT", str(tmp_path))


def test_txt_text_upload_saves_utf8():
    result = uc.process_upload(
        uc.UploadRequest(filename="letters.txt", text="LETTER I.\n\nHello world.", surface="studio")
    )
    assert result.format == "txt"
    assert result.filename == "letters.txt"
    assert open(result.source_file, encoding="utf-8").read() == "LETTER I.\n\nHello world."


def test_txt_binary_zip_rejected():
    with pytest.raises(uc.UploadContractError) as exc_info:
        uc.process_upload(
            uc.UploadRequest(filename="letters.txt", raw_bytes=_valid_epub_bytes(), surface="studio")
        )
    _expect_error(exc_info, "SPOOFED_EXTENSION")


def test_docx_valid_upload():
    raw = _make_docx_bytes()
    result = uc.process_upload(
        uc.UploadRequest(filename="story.docx", raw_bytes=raw, content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    )
    assert result.format == "docx"
    assert open(result.source_file, "rb").read() == raw


def test_docx_missing_document_xml():
    with pytest.raises(uc.UploadContractError) as exc_info:
        uc.process_upload(
            uc.UploadRequest(filename="story.docx", raw_bytes=_make_zip({"word/styles.xml": "<styles/>"}))
        )
    _expect_error(exc_info, "INVALID_DOCX")


def test_epub_valid_upload():
    raw = _valid_epub_bytes()
    result = uc.process_upload(
        uc.UploadRequest(filename="Frankenstein.epub", raw_bytes=raw, content_type="application/epub+zip")
    )
    assert result.format == "epub"
    assert result.filename.endswith(".epub")
    assert open(result.source_file, "rb").read() == raw


def test_epub_not_zip():
    with pytest.raises(uc.UploadContractError) as exc_info:
        uc.process_upload(uc.UploadRequest(filename="broken.epub", raw_bytes=b"not a zip"))
    _expect_error(exc_info, "INVALID_EPUB")


def test_epub_missing_container():
    raw = _make_epub_bytes(
        [("c1", "chapter1.xhtml", "application/xhtml+xml", "")],
        {"chapter1.xhtml": "<html><body><p>Hello there.</p></body></html>"},
        include_container=False,
    )
    with pytest.raises(uc.UploadContractError) as exc_info:
        uc.process_upload(uc.UploadRequest(filename="broken.epub", raw_bytes=raw))
    _expect_error(exc_info, "INVALID_EPUB")


def test_epub_empty_spine():
    raw = _make_epub_bytes(
        [("nav", "nav.xhtml", "application/xhtml+xml", "nav")],
        {"nav.xhtml": "<html><body><nav epub:type='toc'>TOC</nav></body></html>"},
    )
    with pytest.raises(uc.UploadContractError) as exc_info:
        uc.process_upload(uc.UploadRequest(filename="empty.epub", raw_bytes=raw))
    _expect_error(exc_info, "EMPTY_EPUB")


def test_unsupported_extension():
    with pytest.raises(uc.UploadContractError) as exc_info:
        uc.process_upload(uc.UploadRequest(filename="story.pdf", raw_bytes=b"%PDF-1.7"))
    _expect_error(exc_info, "UNSUPPORTED_FORMAT")


def test_response_contract_success_fields():
    result = uc.process_upload(uc.UploadRequest(filename="story.txt", text="CHAPTER I\n\nHello"))
    payload = uc.success_response(result)
    assert set(("ok", "book", "filename", "source_file", "format", "bytes", "status", "next_action", "message")).issubset(payload)


def test_error_contract_fields():
    with pytest.raises(uc.UploadContractError) as exc_info:
        uc.process_upload(uc.UploadRequest(filename="broken.epub", raw_bytes=b"bad"))
    payload = uc.error_response(exc_info.value.error)
    assert payload == {
        "ok": False,
        "error": {
            "ok": False,
            "code": "INVALID_EPUB",
            "message": "The EPUB upload is not a valid ZIP package.",
            "details": exc_info.value.error.details,
            "supported_formats": ["txt", "docx", "epub"],
        },
    }


def test_ingest_upload_calls_tier1_with_source_file(monkeypatch):
    calls = []

    def fake_ingest(path, enable_llm_enrichment=False):
        calls.append((path, enable_llm_enrichment))
        return {"ok": True}

    monkeypatch.setitem(sys.modules, "src.tier_1_parser", types.SimpleNamespace(ingest_manuscript_tier_1=fake_ingest))
    result = uc.UploadResult(
        book="story",
        filename="story.txt",
        source_file="data/uploads/story.txt",
        format="txt",
        bytes=42,
        status="uploaded",
        next_action="analyze",
        message="ready",
    )
    assert uc.ingest_upload(result, enable_llm_enrichment=True) == {"ok": True}
    assert calls == [("data/uploads/story.txt", True)]


def test_fastapi_upload_handling():
    request = asyncio.run(
        uc.request_from_fastapi_upload(
            DummyUploadFile("story.epub", _valid_epub_bytes(), "application/epub+zip"),
            surface="fastapi",
        )
    )
    result = uc.process_upload(request)
    assert result.format == "epub"
    assert result.filename == "story.epub"


def test_express_upload_handling():
    data_url = "data:application/epub+zip;base64," + base64.b64encode(_valid_epub_bytes()).decode("ascii")
    request = uc.request_from_json({"filename": "story.epub", "text": data_url, "tier": 1}, surface="express")
    result = uc.process_upload(request)
    assert result.format == "epub"
    assert result.book == "story"


def test_cross_surface_consistency():
    raw = _valid_epub_bytes()
    express_request = uc.request_from_json(
        {"filename": "Frankenstein.epub", "text": "data:application/epub+zip;base64," + base64.b64encode(raw).decode("ascii")},
        surface="express",
    )
    console_request = uc.request_from_json(
        {"filename": "Frankenstein.epub", "data": "data:application/epub+zip;base64," + base64.b64encode(raw).decode("ascii")},
        surface="console",
    )
    fastapi_request = asyncio.run(
        uc.request_from_fastapi_upload(
            DummyUploadFile("Frankenstein.epub", raw, "application/epub+zip"),
            surface="fastapi",
        )
    )

    results = [uc.process_upload(req) for req in (express_request, console_request, fastapi_request)]
    assert {(result.format, result.filename, result.book, result.next_action) for result in results} == {
        ("epub", "Frankenstein.epub", "Frankenstein", "analyze")
    }


def test_frankenstein_letters_front_matter_structure_detection():
    result = uc.process_upload(
        uc.UploadRequest(filename="Frankenstein.epub", raw_bytes=_letters_epub_bytes(), surface="studio")
    )
    manifest = uc.ingest_upload(result)
    titles = [chapter.title for part in manifest.parts for chapter in part.chapters]
    assert manifest.total_chapters == 2
    assert titles[0].upper().startswith("LETTER I")
    assert titles[1].upper().startswith("LETTER II")
