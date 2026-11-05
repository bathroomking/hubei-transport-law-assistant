"""PDF 只提取至待整理区，不把未核对的跨页表格自动发布到知识库。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from transport_law.errors import AppError

if TYPE_CHECKING:
    from langchain_core.documents import Document


@dataclass(frozen=True)
class DocumentPage:
    text: str
    pdf_page: int | None = None


@dataclass(frozen=True)
class LoadedDocument:
    file: str
    sha256: str
    format: str
    pages: list[DocumentPage]


def _read_pdf(data: bytes) -> list[DocumentPage]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise AppError(
            "pdf_dependency_missing", '请先安装 PDF 依赖：pip install -e ".[pdf]"', 500
        ) from exc
    try:
        # 解析和校验值使用同一份字节，避免文件变动后出现来源错配。
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise AppError("pdf_encrypted", "暂不支持加密 PDF，请提供可读取的副本。", 400)
        return [
            DocumentPage(page.extract_text() or "", number)
            for number, page in enumerate(reader.pages, 1)
        ]
    except (OSError, ValueError, PdfReadError) as exc:
        raise AppError("pdf_read_failed", "PDF 读取失败，请检查文件。", 400) from exc


def load_document(path: Path) -> LoadedDocument:
    """只在本机读取 PDF 或 UTF-8 文本；不调用模型、不执行 OCR。"""
    file_format = path.suffix.lower().lstrip(".")
    if file_format not in {"pdf", "txt", "md"}:
        raise AppError("unsupported_document", "目前支持 PDF、UTF-8 TXT 和 Markdown。", 400)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AppError("document_read_failed", "无法读取文档，请检查文件路径和权限。", 400) from exc
    if file_format == "pdf":
        pages = _read_pdf(data)
    else:
        try:
            pages = [DocumentPage(data.decode("utf-8-sig"))]
        except UnicodeError as exc:
            raise AppError("invalid_encoding", "请先把文本文件保存为 UTF-8 编码。", 400) from exc
    return LoadedDocument(path.name, hashlib.sha256(data).hexdigest(), file_format, pages)


def to_documents(
    document: LoadedDocument,
    *,
    document_id: str | None = None,
    title: str | None = None,
) -> list[Document]:
    """将提取结果转为 LangChain Document；每页从一开始就携带来源信息。"""
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise AppError(
            "chunking_dependency_missing", '请先安装分块依赖：pip install -e ".[ingest]"', 500
        ) from exc
    return [
        Document(
            page_content=page.text,
            metadata={
                "document_id": document_id or f"doc-{document.sha256[:20]}",
                "version_id": f"sha256-{document.sha256[:20]}",
                "title": title or Path(document.file).stem,
                "source": {
                    "file": document.file,
                    "sha256": document.sha256,
                    "format": document.format,
                },
                "pdf_page": page.pdf_page,
                "review_status": "needs_review",
            },
        )
        for page in document.pages
    ]


def load_documents(
    path: Path,
    *,
    document_id: str | None = None,
    title: str | None = None,
) -> list[Document]:
    """自动分块的加载入口；普通 extract-pdf 和服务启动不依赖 LangChain。"""
    return to_documents(load_document(path), document_id=document_id, title=title)


def check_new_output(path: Path, output: Path) -> None:
    if output.resolve() == path.resolve() or output.exists():
        raise AppError("output_exists", "输出文件已经存在，请选择新的路径。", 409)


def write_jsonl(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except FileExistsError as exc:
        raise AppError("output_exists", "输出文件已经存在，请选择新的路径。", 409) from exc


def extract_pdf(path: Path, output: Path) -> dict:
    check_new_output(path, output)
    if path.suffix.lower() != ".pdf":
        raise AppError("unsupported_document", "extract-pdf 只支持 PDF 文件。", 400)
    document = load_document(path)
    rows = [
        {
            "file": document.file,
            "sha256": document.sha256,
            "pdf_page": page.pdf_page,
            "text": page.text,
            "status": "needs_review" if page.text.strip() else "needs_ocr",
        }
        for page in document.pages
    ]
    write_jsonl(output, rows)
    return {
        "page_count": len(rows),
        "needs_ocr_count": sum(row["status"] == "needs_ocr" for row in rows),
        "output": str(output),
        "status": "staged_not_published",
    }
