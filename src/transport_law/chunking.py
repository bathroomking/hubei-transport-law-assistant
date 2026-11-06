"""按通用 Markdown 结构分块；位置指向同版本解析器输出的页面文本。"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import Field, model_validator

from transport_law.errors import AppError
from transport_law.ingest import PARSER_ID, LoadedDocument, to_documents
from transport_law.schemas import Identifier, Model, Text

if TYPE_CHECKING:
    from langchain_core.documents import Document

_STRATEGY = "pdf_structure_v2"
PIPELINE_VERSION = f"{PARSER_ID}:{_STRATEGY}:500:60"
_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*#*\s*$")
_LIST = re.compile(r"^\s*(?:[-+*]|\d+[.)])[ \t]+")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_BOUNDARIES = (re.compile(r"[\n。！？；.!?;]"), re.compile(r"[，、,\s]"))
ContentType = Literal["text", "table", "list", "code"]


class ChunkingOptions(Model):
    # 旧值仅供历史缓存反序列化；新分块不再运行旧的法条正则策略。
    strategy: Literal["pdf_structure_v2", "pdf_legal_recursive_v1"] = _STRATEGY
    chunk_size: int = Field(default=500, ge=1, le=20000)
    chunk_overlap: int = Field(default=60, ge=0)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


class SourceLocation(Model):
    """字符偏移从 0 开始，end 不包含在内；多个切片按顺序拼接为 chunk 正文。"""

    pdf_page: int = Field(ge=1)
    start_index: int = Field(ge=0)
    end_index: int = Field(gt=0)

    @model_validator(mode="after")
    def check_range(self):
        if self.end_index <= self.start_index:
            raise ValueError("来源区间必须非空且 end_index 大于 start_index")
        return self


class DraftSource(Model):
    file: Text
    sha256: str
    format: Literal["pdf"]
    parser: str = "legacy-pypdf"
    locations: list[SourceLocation] = Field(min_length=1)


class DraftChunk(Model):
    id: Identifier
    document_id: Identifier
    version_id: Identifier
    title: Text
    heading: str | None
    parent_id: Identifier | None = None
    content_type: ContentType = "text"
    # 保留原始提取文本的空白和 Markdown，不能使用会 strip 的 Text 类型。
    text: str = Field(min_length=1)
    source: DraftSource
    chunking: ChunkingOptions
    review_status: Literal["needs_review"] = "needs_review"

    @classmethod
    def from_document(cls, document: Document) -> DraftChunk:
        return cls.model_validate(
            {**document.metadata, "id": document.id, "text": document.page_content}
        )

    def to_document(self) -> Document:
        from langchain_core.documents import Document

        return Document(
            id=self.id,
            page_content=self.text,
            metadata=self.model_dump(mode="json", exclude={"id", "text"}),
        )


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    kind: ContentType
    heading: str | None
    parent_start: int
    # 表头与分隔行作为真实来源切片重复；rows 是数据行的绝对区间。
    header_end: int | None = None
    rows: tuple[tuple[int, int], ...] = ()


def _cells(line: str) -> list[str]:
    return re.split(r"(?<!\\)\|", line.strip().strip("|"))


def _is_table(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines) or "|" not in lines[index]:
        return False
    headers, separators = _cells(lines[index]), _cells(lines[index + 1])
    return (
        len(headers) >= 2
        and len(headers) == len(separators)
        # Docling 的 compact_tables 输出单横线分隔单元格，仍保留完整表结构。
        and all(re.fullmatch(r"\s*:?-+:?\s*", cell) for cell in separators)
    )


def _blocks(text: str):
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    headings: list[tuple[int, str]] = []
    section_start = 0
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        start = index
        match = _HEADING.match(lines[index])
        if match:
            level, title = len(match[1]), match[2]
            headings = [(depth, name) for depth, name in headings if depth < level]
            headings.append((level, title))
            section_start = offsets[index]
            heading = " / ".join(name for _, name in headings)
            index += 1
            yield _Block(offsets[start], offsets[index], "text", heading, section_start)
            continue
        heading = " / ".join(name for _, name in headings) or None
        fence = _FENCE.match(lines[index])
        if fence:
            marker = fence[1]
            index += 1
            while index < len(lines):
                closing = lines[index].strip()
                index += 1
                if re.fullmatch(re.escape(marker[0]) + "{" + str(len(marker)) + ",}", closing):
                    break
            yield _Block(offsets[start], offsets[index], "code", heading, offsets[start])
            continue
        if _is_table(lines, index):
            index += 2
            header_end = offsets[index]
            rows = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                # 不跨越标题或下一张表，即使其标题本身含有竖线。
                if _HEADING.match(lines[index]) or _is_table(lines, index):
                    break
                rows.append((offsets[index], offsets[index + 1]))
                index += 1
            yield _Block(
                offsets[start],
                offsets[index],
                "table",
                heading,
                offsets[start],
                header_end,
                tuple(rows),
            )
            continue
        kind: ContentType = "list" if _LIST.match(lines[index]) else "text"
        parent_start = offsets[start] if kind == "list" else section_start
        index += 1
        while index < len(lines) and lines[index].strip():
            if (
                _HEADING.match(lines[index])
                or _FENCE.match(lines[index])
                or _is_table(lines, index)
            ):
                break
            if kind == "text" and _LIST.match(lines[index]):
                break
            # 普通段落从列表换回正文时必须隔离；缩进行仍归属于前一个列表项。
            if kind == "list" and not (_LIST.match(lines[index]) or lines[index][0].isspace()):
                break
            index += 1
        yield _Block(offsets[start], offsets[index], kind, heading, parent_start)


def _split_span(text: str, start: int, end: int, size: int, overlap: int = 0):
    """按句子、空白优先的有界切片；最终硬切保证所有语言都能前进。"""
    while start < end:
        right = min(start + size, end)
        if right < end:
            # 不为很靠前的句号生成大量碎块；候选边界需位于当前窗口后半段。
            floor = start + max(1, size // 2)
            for pattern in _BOUNDARIES:
                matches = list(pattern.finditer(text, floor, right))
                if matches:
                    right = matches[-1].end()
                    break
            # Markdown 单元格中的转义竖线不可跨片段，否则会被误读成列分隔符。
            if right - start > 1 and text[right - 1 : right + 1] == "\\|":
                right -= 1
        yield start, right
        if right == end:
            break
        start = max(start + 1, right - min(overlap, size - 1))


def _unit_fragments(text: str, block: _Block, options: ChunkingOptions):
    """保留短列表项/代码行；只对超长单元再做句子或字符拆分。"""
    if block.kind not in {"list", "code"}:
        for span in _split_span(
            text, block.start, block.end, options.chunk_size, options.chunk_overlap
        ):
            yield [span]
        return
    units = []
    offset = block.start
    for line in text[block.start : block.end].splitlines(keepends=True):
        if block.kind == "list" and units and not _LIST.match(line):
            units[-1] = (units[-1][0], offset + len(line))
        else:
            units.append((offset, offset + len(line)))
        offset += len(line)
    pending = None
    for start, end in units:
        if pending and end - pending[0] <= options.chunk_size:
            pending = (pending[0], end)
            continue
        if pending:
            yield [pending]
            pending = None
        if end - start > options.chunk_size:
            for span in _split_span(text, start, end, options.chunk_size, options.chunk_overlap):
                yield [span]
        else:
            pending = (start, end)
    if pending:
        yield [pending]


@dataclass(frozen=True)
class _TableFragment:
    spans: list[tuple[int, int]]
    row_start: int | None = None


@dataclass(frozen=True)
class _TableLine:
    cells: list[tuple[int, int]]
    suffix: tuple[int, int]


def _table_line(text: str, start: int, end: int) -> _TableLine:
    """每列连同它前面的原始竖线一起定位，投影列时无需生成字符。"""
    body_end = end
    while body_end > start and text[body_end - 1] in "\r\n":
        body_end -= 1
    pipes = [m.start() for m in re.finditer(r"(?<!\\)\|", text[start:body_end])]
    pipes = [start + pos for pos in pipes]
    if pipes and not text[start : pipes[0]].strip():
        pipes = pipes[1:]
    suffix_start = body_end
    if pipes and not text[pipes[-1] + 1 : body_end].strip():
        suffix_start = pipes.pop()
    starts, ends = [start, *pipes], [*pipes, suffix_start]
    return _TableLine(list(zip(starts, ends, strict=True)), (suffix_start, end))


def _projection(lines: list[_TableLine], columns: list[int]):
    return [
        span
        for line in lines
        for span in [*(line.cells[index] for index in columns), line.suffix]
        if span[0] < span[1]
    ]


def _span_size(spans: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in spans)


def _fallback_row(text: str, block: _Block, row: tuple[int, int], size: int):
    # 极小预算、畸形表格或单个行标识本身超长时无法构造完整表格。
    # 仍完整保留标签与数据，并使用逐行 parent_id 关联，避免丢弃任何内容。
    assert block.header_end is not None
    for source in [(block.start, block.header_end), row]:
        for span in _split_span(text, *source, size):
            yield _TableFragment([span], row[0])


def _row_fragments(text: str, block: _Block, row: tuple[int, int], size: int):
    assert block.header_end is not None
    first_end = text.find("\n", block.start, block.header_end) + 1
    lines = [
        _table_line(text, block.start, first_end),
        _table_line(text, first_end, block.header_end),
        _table_line(text, *row),
    ]
    count = len(lines[0].cells)
    if first_end <= block.start or count < 2 or any(len(line.cells) != count for line in lines):
        yield from _fallback_row(text, block, row, size)
        return
    # 先验证每列最小上下文能容纳，避免输出一半后再回退导致无谓重复。
    for index in range(1, count):
        context = _projection(lines[:2], [0, index])
        value_start, value_end = lines[2].cells[index]
        delimiter_end = value_start + int(text[value_start : value_start + 1] == "|")
        overhead = _span_size(context) + _span_size(
            [
                lines[2].cells[0],
                (value_start, delimiter_end),
                lines[2].suffix,
            ]
        )
        if size - overhead < min(16, max(1, size // 4)):
            yield from _fallback_row(text, block, row, size)
            return
    index = 1
    while index < count:
        columns = [0, index]
        spans = _projection(lines, columns)
        if _span_size(spans) <= size:
            # 先收集连续可容纳的单元格，再从右往左装箱，避免把尾部相邻
            # 短字段中的前一列塞进大字段后、只将最后一列孤零零地拆出。
            end_column = index + 1
            while end_column < count and _span_size(_projection(lines, [0, end_column])) <= size:
                end_column += 1
            groups = []
            right = end_column
            while right > index:
                left = right - 1
                spans = _projection(lines, [0, *range(left, right)])
                while left > index:
                    candidate = _projection(lines, [0, *range(left - 1, right)])
                    if _span_size(candidate) > size:
                        break
                    left -= 1
                    spans = candidate
                groups.append(spans)
                right = left
            for grouped in reversed(groups):
                yield _TableFragment(grouped, row[0])
            index = end_column
            continue
        # 单元格本身超长：保留对应的两列表头、行标识和原始分隔符。
        header = _projection(lines[:2], columns)
        value_start, value_end = lines[2].cells[index]
        delimiter_end = value_start + int(text[value_start : value_start + 1] == "|")
        prefix = [*header, lines[2].cells[0], (value_start, delimiter_end)]
        suffix = [lines[2].suffix]
        capacity = size - _span_size(prefix) - _span_size(suffix)
        for span in _split_span(text, delimiter_end, value_end, capacity):
            locations = [*prefix, span, *suffix]
            yield _TableFragment([loc for loc in locations if loc[0] < loc[1]], row[0])
        index += 1


def _table_fragments(text: str, block: _Block, options: ChunkingOptions):
    if block.end - block.start <= options.chunk_size:
        yield _TableFragment([(block.start, block.end)])
        return
    assert block.header_end is not None
    header = (block.start, block.header_end)
    header_size = block.header_end - block.start
    if not block.rows:
        for span in _split_span(text, *header, options.chunk_size):
            yield _TableFragment([span])
        return
    capacity = options.chunk_size - header_size
    pending = None
    for start, end in block.rows:
        if pending and end - pending[0] <= capacity:
            pending = (pending[0], end)
            continue
        if pending:
            yield _TableFragment([header, pending])
            pending = None
        if end - start > capacity:
            yield from _row_fragments(text, block, (start, end), options.chunk_size)
        else:
            pending = (start, end)
    if pending:
        yield _TableFragment([header, pending])


def _hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def _merge_text_blocks(blocks: list[_Block], size: int):
    pending = None
    for block in blocks:
        if (
            pending
            and pending.kind == block.kind == "text"
            and pending.parent_start == block.parent_start
            and pending.heading == block.heading
            and block.end - pending.start <= size
        ):
            pending = _Block(pending.start, block.end, "text", block.heading, block.parent_start)
        else:
            if pending:
                yield pending
            pending = block
    if pending:
        yield pending


def split_legal_documents(documents: list[Document], options: ChunkingOptions) -> list[Document]:
    """兼容原入口名称，执行领域无关的结构分块；不盲目跨物理页拼接。"""
    from langchain_core.documents import Document

    if options.strategy != _STRATEGY:
        raise AppError("chunking_strategy_retired", "旧分块策略仅可读取缓存，请重新解析。", 409)
    chunks = []
    for page in documents:
        text = page.page_content
        for block in _merge_text_blocks(list(_blocks(text)), options.chunk_size):
            metadata = deepcopy(page.metadata)
            pdf_page = metadata.pop("pdf_page")
            metadata["source"].setdefault("parser", "legacy-pypdf")
            identity = [
                metadata["document_id"],
                metadata["version_id"],
                metadata["source"]["sha256"],
                metadata["source"]["parser"],
                pdf_page,
            ]
            parent_id = "BLOCK-" + _hash([*identity, block.kind, block.parent_start])
            fragments = (
                _table_fragments(text, block, options)
                if block.kind == "table"
                else _unit_fragments(text, block, options)
            )
            for fragment in fragments:
                spans = fragment.spans if isinstance(fragment, _TableFragment) else fragment
                fragment_parent = parent_id
                if isinstance(fragment, _TableFragment) and fragment.row_start is not None:
                    fragment_parent = "ROW-" + _hash([*identity, block.start, fragment.row_start])
                content = "".join(text[start:end] for start, end in spans)
                if not content.strip():
                    continue
                if len(content) > options.chunk_size:
                    raise AppError("chunk_too_large", "分块超出配置上限，停止入库。", 500)
                locations = [
                    {"pdf_page": pdf_page, "start_index": start, "end_index": end}
                    for start, end in spans
                ]
                part_meta = deepcopy(metadata)
                part_meta.update(
                    heading=block.heading,
                    parent_id=fragment_parent,
                    content_type=block.kind,
                    chunking=options.model_dump(),
                )
                part_meta["source"]["locations"] = locations
                ident = "AUTO-" + _hash(
                    [
                        *identity,
                        options.model_dump(),
                        locations,
                        block.heading,
                        block.kind,
                        content,
                    ]
                )
                chunks.append(Document(id=ident, page_content=content, metadata=part_meta))
    return chunks


def build_draft_chunks(
    document: LoadedDocument,
    options: ChunkingOptions,
    *,
    document_id: str | None = None,
    title: str | None = None,
) -> list[DraftChunk]:
    pages = to_documents(document, document_id=document_id, title=title)
    return [DraftChunk.from_document(chunk) for chunk in split_legal_documents(pages, options)]
