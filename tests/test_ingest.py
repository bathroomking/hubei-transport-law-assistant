import hashlib
import logging
from io import BytesIO
from types import SimpleNamespace

import pypdfium2
import pytest
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc import ContentLayer
from pdf_fixtures import pdf_bytes
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from transport_law import ingest
from transport_law.errors import AppError


def fake_conversion(monkeypatch, *, status=ConversionStatus.SUCCESS, text="Extracted text"):
    calls = []

    def export(**kwargs):
        calls.append(kwargs)
        return text

    result = SimpleNamespace(status=status, document=SimpleNamespace(export_to_markdown=export))
    converter = SimpleNamespace(convert=lambda stream: result)
    monkeypatch.setattr(ingest, "_converter", lambda: converter)
    return calls


@pytest.mark.parametrize(
    "filename,data,code",
    [
        ("example.docx", b"example", "unsupported_document"),
        ("bad.pdf", b"not a PDF", "pdf_read_failed"),
    ],
)
def test_rejects_unsupported_or_unreadable_uploads(filename, data, code):
    with pytest.raises(AppError) as error:
        ingest.load_document(data, filename=filename, sha256=hashlib.sha256(data).hexdigest())
    assert error.value.code == code


@pytest.mark.parametrize("status", [ConversionStatus.PARTIAL_SUCCESS, ConversionStatus.FAILURE])
def test_partial_conversion_never_enters_index(monkeypatch, status):
    calls = fake_conversion(monkeypatch, status=status)
    with pytest.raises(AppError) as error:
        ingest._read_pdf(pdf_bytes("Visible page"))
    assert error.value.code == "pdf_parse_incomplete"
    assert not calls


def test_page_exports_keep_headers_and_compact_tables(monkeypatch):
    calls = fake_conversion(monkeypatch)
    pages = ingest._read_pdf(pdf_bytes("First page", "Second page"))
    assert [page.pdf_page for page in pages] == [1, 2]
    assert [call["page_no"] for call in calls] == [1, 2]
    assert all(call["compact_tables"] for call in calls)
    assert all(
        call["included_content_layers"] == {ContentLayer.BODY, ContentLayer.FURNITURE}
        for call in calls
    )


def test_success_with_missing_nonblank_page_is_rejected(monkeypatch):
    fake_conversion(monkeypatch, text="")
    with pytest.raises(AppError) as error:
        ingest._read_pdf(pdf_bytes("Visible content"))
    assert error.value.code == "pdf_needs_ocr"


def test_blank_pdf_needs_no_models(monkeypatch):
    def unexpected():
        pytest.fail("Blank PDF must not initialize layout/OCR models")

    monkeypatch.setattr(ingest, "_converter", unexpected)
    pages = ingest._read_pdf(pdf_bytes("", "  "))
    assert [(p.pdf_page, p.is_blank, p.text) for p in pages] == [(1, True, ""), (2, True, "")]


def test_vector_only_page_is_not_treated_as_blank(monkeypatch):
    output = BytesIO()
    canvas = Canvas(output)
    canvas.rect(50, 50, 100, 100)
    canvas.save()
    fake_conversion(monkeypatch, text="")
    with pytest.raises(AppError) as error:
        ingest._read_pdf(output.getvalue())
    assert error.value.code == "pdf_needs_ocr"


def test_scan_and_same_page_mixed_content_use_real_ocr():
    # Rasterize a generated page so no hidden PDF text layer can make this pass.
    source = pypdfium2.PdfDocument(pdf_bytes("检查电源连接，保存检查记录。"))
    page = source[0]
    bitmap = page.render(scale=3)
    image = bitmap.to_pil().copy()
    bitmap.close()
    page.close()
    source.close()
    output = BytesIO()
    canvas = Canvas(output, pagesize=(612, 792), invariant=True)
    canvas.drawImage(ImageReader(image), 0, 0, width=612, height=792)
    canvas.showPage()
    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(50, 760, "Native heading")
    canvas.drawImage(ImageReader(image), 0, 0, width=612, height=720)
    canvas.save()
    data = output.getvalue()
    checksum = hashlib.sha256(data).hexdigest()
    loaded = ingest.load_document(data, filename="扫描和混排.PDF", sha256=checksum)
    assert loaded.file == "扫描和混排.PDF"
    assert loaded.sha256 == checksum
    assert [page.pdf_page for page in loaded.pages] == [1, 2]
    assert all("保存检查记录" in "".join(page.text.split()) for page in loaded.pages)
    assert "Native heading" in loaded.pages[1].text
    assert loaded.parser == ingest.PARSER_ID


def test_success_with_known_table_text_loss_is_rejected(monkeypatch):
    logger = logging.getLogger("MatchingPostProcessor")
    handlers = list(logger.handlers)
    exports = []

    def convert(stream):
        logger.warning(
            "2 of 18 pdf cells matched neither a row nor a column band of "
            "the 4x3 grid and were dropped from the table"
        )
        logger.warning(
            "1 of 12 pdf cells matched neither a row nor a column band of "
            "the 3x2 grid and were dropped from the table"
        )
        return SimpleNamespace(
            status=ConversionStatus.SUCCESS,
            document=SimpleNamespace(export_to_markdown=lambda **kwargs: exports.append(kwargs)),
        )

    monkeypatch.setattr(ingest, "_converter", lambda: SimpleNamespace(convert=convert))
    with pytest.raises(AppError) as error:
        ingest._read_pdf(pdf_bytes("Table content"))
    assert error.value.code == "pdf_parse_incomplete"
    assert "部分表格文字丢失" in error.value.message
    assert "3 个文本片段" in error.value.message
    assert exports == []
    assert logger.handlers == handlers


def test_unrelated_parser_warning_does_not_reject_success(monkeypatch):
    logger = logging.getLogger("MatchingPostProcessor")
    handlers = list(logger.handlers)

    def convert(stream):
        logger.warning("Using a fallback table grid; all text was recovered")
        return SimpleNamespace(
            status=ConversionStatus.SUCCESS,
            document=SimpleNamespace(export_to_markdown=lambda **kwargs: "Complete table text\n"),
        )

    monkeypatch.setattr(ingest, "_converter", lambda: SimpleNamespace(convert=convert))
    pages = ingest._read_pdf(pdf_bytes("Complete table text"))
    assert pages[0].text == "Complete table text\n"
    assert logger.handlers == handlers


def test_parser_exception_cleans_handler_and_does_not_leak_loss_to_next_task(monkeypatch):
    logger = logging.getLogger("MatchingPostProcessor")
    handlers = list(logger.handlers)
    calls = 0

    def convert(stream):
        nonlocal calls
        calls += 1
        if calls == 1:
            logger.warning(
                "1 of 5 pdf cells matched neither a row nor a column band of "
                "the 2x2 grid and were dropped from the table"
            )
            raise RuntimeError("Parser worker failed")
        return SimpleNamespace(
            status=ConversionStatus.SUCCESS,
            document=SimpleNamespace(export_to_markdown=lambda **kwargs: "Next task is complete"),
        )

    monkeypatch.setattr(ingest, "_converter", lambda: SimpleNamespace(convert=convert))
    data = pdf_bytes("Visible page")
    with pytest.raises(AppError) as error:
        ingest._read_pdf(data)
    assert error.value.code == "pdf_read_failed"
    assert logger.handlers == handlers
    assert ingest._read_pdf(data)[0].text == "Next task is complete"
    assert calls == 2
    assert logger.handlers == handlers


@pytest.mark.parametrize(
    "marker_value",
    [ingest.MODEL_VERSION, "docling-2.148.0-rapidocr-3.10.0-body-furniture-compact-v1"],
)
def test_parser_version_change_reuses_compatible_models(monkeypatch, tmp_path, marker_value):
    def unexpected_download(**kwargs):
        pytest.fail("Parser validation changes must not redownload unchanged model assets")

    (tmp_path / "ready.txt").write_text(marker_value)
    monkeypatch.setattr(ingest, "MODEL_PATH", tmp_path)
    monkeypatch.setattr("docling.utils.model_downloader.download_models", unexpected_download)
    assert ingest.prepare_pdf_models() == tmp_path
