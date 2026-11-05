"""读取上传文档并保留来源位置，供自动分块流程使用。"""

from __future__ import annotations

import hashlib
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
    """读取文档并转换为携带来源信息的 LangChain Document。"""
    return to_documents(load_document(path), document_id=document_id, title=title)
