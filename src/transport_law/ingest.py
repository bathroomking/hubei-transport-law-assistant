"""读取上传文档并保留来源位置，供自动分块流程使用。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import RLock
from typing import Literal

from transport_law.errors import AppError


@dataclass(frozen=True)
class DocumentPage:
    text: str
    pdf_page: int
    is_blank: bool = False


@dataclass(frozen=True)
class LoadedDocument:
    file: str
    sha256: str
    format: Literal["pdf"]
    pages: list[DocumentPage]
    parser: str = "plain-v1"


PARSER_ID = "docling-2.148.0-rapidocr-3.10.0-body-furniture-compact-v2"
MODEL_VERSION = "docling-2.148.0-rapidocr-3.10.0-onnxruntime-ch-v1"
_LEGACY_MODEL_MARKER = "docling-2.148.0-rapidocr-3.10.0-body-furniture-compact-v1"
MODEL_PATH = Path(__file__).resolve().parents[2] / "data/local/models/docling"
_PARSE_LOCK = RLock()
_TABLE_CONTENT_LOSS = re.compile(
    r"(?P<count>\d+) of \d+ pdf cells matched neither a row nor a column band of "
    r"the \d+x\d+ grid and were dropped from the table"
)


class _TableLossHandler(logging.Handler):
    """仅识别固定解析器版本明确报告的文字丢失，不将普通 warning 视为失败。"""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.dropped_cells = 0

    def emit(self, record: logging.LogRecord) -> None:
        match = _TABLE_CONTENT_LOSS.fullmatch(record.getMessage())
        if match:
            self.dropped_cells += int(match["count"])


def prepare_pdf_models() -> Path:
    """首次下载所需的版面、表格和中文 OCR 模型；后续使用本地模型。"""
    marker = MODEL_PATH / "ready.txt"
    with _PARSE_LOCK:
        # 解析/校验逻辑升级不改变模型；兼容早期使用解析器 ID 的就绪标记。
        if marker.exists() and marker.read_text() in {MODEL_VERSION, _LEGACY_MODEL_MARKER}:
            return MODEL_PATH
        try:
            from docling.utils.model_downloader import download_models

            download_models(
                output_dir=MODEL_PATH,
                with_code_formula=False,
                with_picture_classifier=False,
                rapidocr_models=["onnxruntime:ch"],
                progress=False,
            )
            marker.write_text(MODEL_VERSION)
        except ImportError as exc:
            raise AppError(
                "pdf_dependency_missing", '请安装 PDF 解析依赖：pip install -e ".[ingest]"', 500
            ) from exc
        except Exception as exc:
            raise AppError(
                "pdf_models_unavailable", "PDF 解析模型尚未就绪，请联网准备模型后重试。", 503
            ) from exc
    return MODEL_PATH


@lru_cache(maxsize=1)
def _converter():
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        artifacts_path=prepare_pdf_models(),
        do_ocr=True,
        do_table_structure=True,
        ocr_options=RapidOcrOptions(backend="onnxruntime"),
        accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU),
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
    )


def _is_blank(page) -> bool:
    """只把没有绘制内容的物理页当作空白；图像和矢量页必须进入解析/OCR。"""
    contents = page.get_contents()
    if contents is None:
        return True
    paint = {b"Do", b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"sh", b"INLINE IMAGE"}
    for operands, operator in contents.operations:
        if operator in paint:
            return False
        if operator in {b"Tj", b"TJ", b"'", b'"'}:
            values = operands[0] if operator == b"TJ" else operands
            if any(isinstance(value, (str, bytes)) and value.strip() for value in values):
                return False
    return True


def _read_pdf(data: bytes) -> list[DocumentPage]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise AppError(
            "pdf_dependency_missing", '请安装 PDF 依赖：pip install -e ".[ingest]"', 500
        ) from exc
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise AppError("pdf_encrypted", "暂不支持加密 PDF，请提供可读取的副本。", 400)
        blanks = [_is_blank(page) for page in reader.pages]
    except (OSError, ValueError, PdfReadError) as exc:
        raise AppError("pdf_read_failed", "PDF 读取失败，请检查文件。", 400) from exc
    if all(blanks):
        return [DocumentPage("", number, True) for number in range(1, len(blanks) + 1)]
    try:
        from docling.datamodel.base_models import ConversionStatus, DocumentStream
        from docling_core.types.doc import ContentLayer

        # 模型实例复用；原件使用已校验字节，不重新打开可变化的路径。
        with _PARSE_LOCK:
            converter = _converter()
            logger = logging.getLogger("MatchingPostProcessor")
            loss = _TableLossHandler()
            logger.addHandler(loss)
            try:
                result = converter.convert(DocumentStream(name="source.pdf", stream=BytesIO(data)))
            finally:
                logger.removeHandler(loss)
                loss.close()
            if loss.dropped_cells:
                raise AppError(
                    "pdf_parse_incomplete",
                    "PDF 部分表格文字丢失"
                    f"（{loss.dropped_cells} 个文本片段未能匹配），已停止入库。",
                    400,
                )
        if result.status != ConversionStatus.SUCCESS:
            raise AppError("pdf_parse_incomplete", "PDF 未完整解析，停止入库，请检查文件。", 400)
        pages = []
        for number, blank in enumerate(blanks, 1):
            text = result.document.export_to_markdown(
                page_no=number,
                compact_tables=True,
                included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
                image_placeholder="",
            )
            if not text.strip() and not blank:
                raise AppError(
                    "pdf_needs_ocr", f"PDF 第 {number} 页没有可靠提取结果，请核对扫描质量。", 400
                )
            pages.append(DocumentPage(text, number, blank))
        return pages
    except AppError:
        raise
    except ImportError as exc:
        raise AppError(
            "pdf_dependency_missing", '请安装 PDF 解析依赖：pip install -e ".[ingest]"', 500
        ) from exc
    except Exception as exc:
        raise AppError(
            "pdf_read_failed", "PDF 结构化解析失败，请检查文件与本地模型。", 400
        ) from exc


def load_document(data: bytes, *, filename: str, sha256: str) -> LoadedDocument:
    """解析调用方已核对 SHA-256 的原件字节；保存原文件名、物理页码和解析器版本。"""
    if Path(filename).suffix.lower() != ".pdf":
        raise AppError("unsupported_document", "目前只支持 PDF 文件。", 400)
    return LoadedDocument(filename, sha256, "pdf", _read_pdf(data), PARSER_ID)
