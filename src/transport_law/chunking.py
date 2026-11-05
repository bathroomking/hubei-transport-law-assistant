"""LangChain 自动分块：保留原文和来源位置，供上传任务继续向量入库。"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from transport_law.errors import AppError
from transport_law.ingest import LoadedDocument, to_documents
from transport_law.schemas import Identifier, Model, Text

if TYPE_CHECKING:
    from langchain_core.documents import Document

# 行首的条、章、节和 Markdown 标题是候选结构边界；这不是法律语义解析器。
_HEADING = re.compile(
    r"(?m)^[ \t\u3000]*(?P<heading>"
    r"第[一二三四五六七八九十百千万零〇两0-9０-９]+[ \t]*[条章节]"
    r"(?:之[一二三四五六七八九十百千万零〇两0-9０-９]+)?"
    r"|#{1,6}[ \t]+[^\r\n]+)"
)


class ChunkingOptions(Model):
    strategy: Literal["legal_recursive_v1"] = "legal_recursive_v1"
    chunk_size: int = Field(default=800, ge=1, le=20000)
    chunk_overlap: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


class SourceLocation(Model):
    """字符位置从 0 开始，end_index 不包含在内；PDF 位置相对于各页提取文本。"""

    pdf_page: int | None = Field(default=None, ge=1)
    start_index: int = Field(ge=0)
    end_index: int = Field(gt=0)


class DraftSource(Model):
    file: Text
    sha256: str
    format: Literal["pdf", "txt", "md"]
    locations: list[SourceLocation] = Field(min_length=1)


class DraftChunk(Model):
    id: Identifier
    document_id: Identifier
    version_id: Identifier
    title: Text
    heading: str | None
    # 保留提取原文的空白和标点，便于根据位置回查；不能使用会 strip 的 Text 类型。
    text: str = Field(min_length=1)
    source: DraftSource
    chunking: ChunkingOptions
    review_status: Literal["needs_review"] = "needs_review"

    @classmethod
    def from_document(cls, document: Document) -> DraftChunk:
        """标准 Document 经过业务校验后，才成为可以保存的候选记录。"""
        return cls.model_validate(
            {**document.metadata, "id": document.id, "text": document.page_content}
        )

    def to_document(self) -> Document:
        """为后续检索/向量库适配提供转换入口；返回独立的 metadata 副本。"""
        from langchain_core.documents import Document

        return Document(
            id=self.id,
            page_content=self.text,
            metadata=self.model_dump(mode="json", exclude={"id", "text"}),
        )


def _can_join_pages(previous: Document, current: Document) -> bool:
    previous_meta, current_meta = dict(previous.metadata), dict(current.metadata)
    previous_page = previous_meta.pop("pdf_page")
    current_page = current_meta.pop("pdf_page")
    # 批量导入时，不拼接不同文档/版本，也不跨越被省略的页。
    return (
        previous_page is not None
        and current_page == previous_page + 1
        and previous_meta == current_meta
    )


def _page_groups(pages: list[Document]) -> list[list[Document]]:
    """遇到无文本页就断开，避免跨过可能缺失的扫描页拼成一条完整条文。"""
    groups = []
    current = []
    for page in pages:
        if page.page_content.strip():
            if current and not _can_join_pages(current[-1], page):
                groups.append(current)
                current = []
            current.append(page)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _sections(text: str):
    start, heading = 0, None
    for match in _HEADING.finditer(text):
        if match.start() > start:
            yield start, match.start(), heading
        start, heading = match.start(), match.group("heading")
    if start < len(text):
        yield start, len(text), heading


def _section_documents(pages: list[Document], options: ChunkingOptions) -> list[Document]:
    from langchain_core.documents import Document

    text_parts, spans = [], []
    offset = 0
    for page in pages:
        text_parts.append(page.page_content)
        spans.append(
            {
                "pdf_page": page.metadata["pdf_page"],
                "start_index": offset,
                "end_index": offset + len(page.page_content),
            }
        )
        offset += len(page.page_content) + 2
    text = "\n\n".join(text_parts)
    sections = []
    for start, end, heading in _sections(text):
        metadata = deepcopy(pages[0].metadata)
        metadata.pop("pdf_page")
        metadata.update(
            heading=heading,
            chunking=options.model_dump(),
            _section_start=start,
            _page_spans=[
                span
                for span in spans
                if max(start, span["start_index"]) < min(end, span["end_index"])
            ],
        )
        sections.append(Document(page_content=text[start:end], metadata=metadata))
    return sections


def split_legal_documents(documents: list[Document], options: ChunkingOptions) -> list[Document]:
    """页面 Document → 条文 Document → chunk Document，全程传递 metadata。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise AppError(
            "chunking_dependency_missing", '请先安装分块依赖：pip install -e ".[ingest]"', 500
        ) from exc
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=options.chunk_size,
        chunk_overlap=options.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "！", "？", ";", "!", "?", "，", "、", " ", ""],
        keep_separator="end",
        strip_whitespace=False,
        add_start_index=True,
    )
    chunks = []
    for pages in _page_groups(documents):
        for section in _section_documents(pages, options):
            # 先隔离条文，再递归分块；重叠不会把相邻法条混入同一 chunk。
            for part in splitter.split_documents([section]):
                content = part.page_content
                if not content.strip():
                    continue
                metadata = part.metadata
                local_start = metadata.pop("start_index")
                start = metadata.pop("_section_start") + local_start
                end = start + len(content)
                if (
                    local_start < 0
                    or section.page_content[local_start : local_start + len(content)] != content
                ):
                    raise AppError(
                        "chunk_location_failed", "分块位置无法回查，请检查提取文本。", 500
                    )
                locations = []
                for span in metadata.pop("_page_spans"):
                    left, right = span["start_index"], span["end_index"]
                    if max(start, left) < min(end, right):
                        locations.append(
                            {
                                "pdf_page": span["pdf_page"],
                                "start_index": max(start, left) - left,
                                "end_index": min(end, right) - left,
                            }
                        )
                metadata["source"]["locations"] = locations
                # 内容、版本、位置和切分参数决定 ID；改名或重复执行不会随机生成新 ID。
                identity = json.dumps(
                    [
                        metadata["document_id"],
                        metadata["source"]["sha256"],
                        metadata["chunking"],
                        locations,
                        content,
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                part.id = f"AUTO-{hashlib.sha256(identity.encode()).hexdigest()[:32]}"
                chunks.append(part)
    return chunks


def build_draft_chunks(
    document: LoadedDocument,
    options: ChunkingOptions,
    *,
    document_id: str | None = None,
    title: str | None = None,
) -> list[DraftChunk]:
    """将提取结果经 Document 分块流程转换为校验后的草稿。"""
    pages = to_documents(document, document_id=document_id, title=title)
    return [DraftChunk.from_document(chunk) for chunk in split_legal_documents(pages, options)]
