"""法规按条、交通管理裁量基准按事项切分，并保留逐页原文位置。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from transport_law.ingest import PARSER_ID, LoadedDocument
from transport_law.schemas import Identifier, Model, Text

PIPELINE_VERSION = f"{PARSER_ID}:legal_units_v1"
ContentType = Literal["article", "item", "page"]
_NUM = r"[零〇一二三四五六七八九十百千两\d]+"
_ARTICLE = re.compile(rf"^(第\s*({_NUM})\s*条(?:之{_NUM})?)(.*)$")
# 紧跟条号的“所称”“规定”“第…款”等属于正文转引；有排版分隔的条题单独识别。
_REFERENCE = re.compile(rf"^(?:所|规定|[的中、，。；：]|第\s*{_NUM}\s*[款项目]|[至和与及]第)")
_ITEM = re.compile(rf"^交通管理册\s*第\s*({_NUM})\s*项$")
_SECTION = re.compile(rf"^第\s*{_NUM}\s*[编章节](?:\s|$|总则$|附则$)")
_APPENDIX = re.compile(r"^(?:附录|附件|附表)(?:$|\s|[一二三四五六七八九十\dA-Z：:])")
_TOC_ENTRY = re.compile(r"(?:[.…·]{2,}|\s{2,})\s*\d+\s*$")
_READER_HEADER = re.compile(r"\A[^\n]+\n\s*资料[ \t]*\d+[ \t]*\r?\n\s*")
_READER_FOOTER = re.compile(r"(?m)^阅读整理本\s*·\s*\d{4}-\d{2}-\d{2}\s*第\s*\d+\s*页\s*\Z")


class ChunkingOptions(Model):
    strategy: Literal["legal_units_v1"] = "legal_units_v1"


class SourceLocation(Model):
    """字符位置起点从 0 开始，终点不包含在内。"""

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
    heading: str
    parent_id: Identifier
    content_type: ContentType
    article_number: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1)
    source: DraftSource
    chunking: ChunkingOptions
    review_status: Literal["needs_review"] = "needs_review"


def legal_number(value: str) -> int:
    if value.isdecimal():
        return int(value)
    digits = dict(zip("零一二三四五六七八九", range(10), strict=True)) | {"〇": 0, "两": 2}
    units = {"十": 10, "百": 100, "千": 1000}
    total = pending = 0
    for char in value:
        if char in units:
            total += (pending or 1) * units[char]
            pending = 0
        else:
            pending = pending * 10 + (int(char) if char.isdecimal() else digits[char])
    return total + pending


@dataclass(frozen=True)
class _Line:
    label: str
    page: int
    start: int
    end: int
    reader_stamp: str | None


@dataclass
class _Unit:
    kind: ContentType
    heading: str
    number: int | None
    locations: list[SourceLocation]

    def append(self, line: _Line):
        if self.locations:
            last = self.locations[-1]
            if last.pdf_page == line.page and last.end_index == line.start:
                last.end_index = line.end
                return
        self.locations.append(
            SourceLocation(pdf_page=line.page, start_index=line.start, end_index=line.end)
        )


def _lines(document: LoadedDocument) -> list[_Line]:
    lines = []
    for page in document.pages:
        text = page.text
        start, end = 0, len(text)
        # 此整理本的成对印记可精确识别，不能把其他 PDF 的首末行一律删去。
        header, footer = _READER_HEADER.match(text), _READER_FOOTER.search(text)
        reader_stamp = None
        if header and footer and header.end() <= footer.start():
            reader_stamp = header.group().strip()
            start, end = header.end(), footer.start()
            body = text[start:end]
            if body.lstrip("# ").startswith("湖北交通法规资料 /") and "收录版本" in body:
                continue  # 本项目阅读整理本封面；普通资料即使排在第一页也须保留。
        for raw in text[start:end].splitlines(keepends=True):
            label = re.sub(r"^#{1,6}\s+", "", raw.strip()).strip()
            lines.append(_Line(label, page.pdf_page, start, start + len(raw), reader_stamp))
            start += len(raw)
    return lines


def _article(label: str) -> re.Match | None:
    match = _ARTICLE.fullmatch(label)
    if match and not _REFERENCE.match(match.group(3)):
        return match
    return None


def _body_after_toc(lines: list[_Line], index: int, article: re.Match | None) -> bool:
    """目录没有正文标记时，用条号后的正常段落区分目录行和正文起点。"""
    if not article or _TOC_ENTRY.search(lines[index].label):
        return False
    body = article.group(3).strip()
    if not body:
        body = next((lines[i].label for i in range(index + 1, len(lines)) if lines[i].label), "")
        if (
            _article(body)
            or _SECTION.match(body)
            or body.startswith("|")
            or _TOC_ENTRY.search(body)
        ):
            return False
    return bool(re.search(r"[。；：]", body))


def _legal_units(lines: list[_Line]) -> list[_Unit]:
    units = []
    current = None
    mode = "body"
    for i, line in enumerate(lines):
        label = line.label
        if i and line.reader_stamp != lines[i - 1].reader_stamp:
            mode, current = "body", None
        if label.replace(" ", "") == "目录":
            mode, current = "toc", None
            continue
        if line.reader_stamp and label == "版本与来源":
            mode, current = "sources", None
            continue
        if label in {"正文", "条文正文", "总则", "交通管理册"}:
            mode, current = "body", None
            continue
        if _APPENDIX.match(label):
            mode, current = "attachment", None
        if mode == "sources" or (
            line.reader_stamp and current is None and label.startswith("以下按原表序号整理")
        ):
            continue
        if _SECTION.match(label):
            current = None
            continue
        article, item = _article(label), _ITEM.fullmatch(label)
        if mode == "toc":
            if not _body_after_toc(lines, i, article):
                continue
            mode = "body"

        if article and (number := legal_number(article.group(2))):
            current = _Unit("article", re.sub(r"\s+", "", article.group(1)), number, [])
            units.append(current)
        elif item:
            current = _Unit("item", f"交通管理册第{item.group(1)}项", None, [])
            units.append(current)
        if current:
            current.append(line)
        else:
            # 普通文件中未归入条文的材料和实质性附件分别按页保留。每段都能独立定位。
            previous = units[-1] if units else None
            if (
                previous
                and previous.kind == "page"
                and previous.locations[-1].pdf_page == line.page
                and previous.locations[-1].end_index == line.start
            ):
                previous.append(line)
            elif label:
                page = _Unit("page", f"PDF 第 {line.page} 页", None, [])
                page.append(line)
                units.append(page)
    return units


def build_draft_chunks(
    document: LoadedDocument,
    options: ChunkingOptions,
    *,
    document_id: str | None = None,
    title: str | None = None,
) -> list[DraftChunk]:
    """每个输出的 text 均等于各来源区间在原快照中的切片依次拼接。"""
    pages = {page.pdf_page: page.text for page in document.pages}
    if list(pages) != list(range(1, len(document.pages) + 1)):
        raise ValueError("提取快照的页码不连续或重复")
    units = _legal_units(_lines(document))
    if not any(unit.kind != "page" for unit in units):
        units = [
            _Unit(
                "page",
                f"PDF 第 {page} 页",
                None,
                [SourceLocation(pdf_page=page, start_index=0, end_index=len(text))],
            )
            for page, text in pages.items()
            if text.strip()
        ]
    document_id = document_id or f"doc-{document.sha256[:20]}"
    version_id = f"sha256-{document.sha256[:20]}"
    chunks = []
    for unit in units:
        locations = unit.locations
        text = "".join(pages[loc.pdf_page][loc.start_index : loc.end_index] for loc in locations)
        identity = [
            document_id,
            document.sha256,
            document.parser,
            options.strategy,
            unit.kind,
            [loc.model_dump() for loc in locations],
            text,
        ]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:32]
        chunks.append(
            DraftChunk(
                id=f"AUTO-{digest}",
                document_id=document_id,
                version_id=version_id,
                title=title or Path(document.file).stem,
                heading=unit.heading,
                parent_id=f"UNIT-{digest}",
                content_type=unit.kind,
                article_number=unit.number,
                text=text,
                source=DraftSource(
                    file=document.file,
                    sha256=document.sha256,
                    format=document.format,
                    parser=document.parser,
                    locations=locations,
                ),
                chunking=options,
            )
        )
    return chunks
