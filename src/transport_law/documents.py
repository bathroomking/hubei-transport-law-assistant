"""Read a single collected document/version without treating excerpts as full laws."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from transport_law.config import PROJECT_ROOT
from transport_law.errors import AppError
from transport_law.repository import LawRepository
from transport_law.schemas import Identifier, LegalChunk, Model, Text

MANIFEST_PATH = PROJECT_ROOT / "data/raw/laws/manifest.json"


class LawDocument(Model):
    document_id: Identifier
    version_id: Identifier
    title: Text
    coverage: Literal["full_text", "selected_sections", "excerpts", "uploaded_document"]
    coverage_note: Text
    articles: list[LegalChunk] = Field(min_length=1)


def _manifest_entry(document_id: str, version_id: str) -> dict | None:
    # Only this fixed, project-owned catalog is read; request IDs never become paths.
    try:
        manifest = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("documents"), list):
        return None
    entries = [
        entry
        for entry in manifest["documents"]
        if isinstance(entry, dict)
        and entry.get("document_id") == document_id
        and entry.get("id") == version_id
    ]
    return entries[0] if len(entries) == 1 else None


def _matches_source(articles: list[LegalChunk], entry: dict) -> bool:
    return all(
        article.source.sha256 == entry.get("sha256")
        and article.source.file == entry.get("file")
        and article.source.document_title == entry.get("title")
        for article in articles
    )


def _coverage(articles: list[LegalChunk], entry: dict | None) -> tuple[str, str]:
    if any(article.kind == "document" for article in articles):
        return (
            "uploaded_document",
            "以下为用户上传文件已入库的文字片段，可能包含重叠内容；"
            "不表示已收录整部法规，文字提取完整性、法律效力及适用性尚待核验。",
        )
    excerpt_note = (
        "以下仅为该文档、该版本已收录的条文或节选，不能作为整部法规全文；"
        "请结合各条出处核对原文、文件效力及适用性。"
    )
    if entry is None or not _matches_source(articles, entry):
        return "excerpts", excerpt_note

    count = entry.get("article_count")
    if type(count) is not int or count < 1:
        return "excerpts", excerpt_note
    version_note = f"收录版本：{entry.get('version_label', entry['id'])}。"

    # The stored PDF is complete, but its other management booklets are not indexed.
    # Do not infer full-text coverage from the 22 numbered general provisions.
    if entry["id"] == "hb-police-discretion-2026":
        tables = entry.get("table_item_count")
        if type(tables) is not int or tables < 1:
            return "excerpts", excerpt_note
        expected_ids = {
            f"{entry['id']}-{prefix}{number:03d}"
            for prefix, total in (("G", count), ("T", tables))
            for number in range(1, total + 1)
        }
        if {article.id for article in articles} != expected_ids:
            return "excerpts", excerpt_note
        return (
            "selected_sections",
            version_note + f"仅收录总则{count}条和交通管理册{tables}项，未收录其他管理册，"
            "不能作为整本裁量基准。交通管理册仅细化行政拘留；"
            "文件效力与个案适用仍需核验。",
        )

    # Only catalogs generated from all numbered articles qualify, and the repository
    # must contain every number exactly once with the catalog's source identity.
    if (
        entry.get("snapshot_format") not in {"html", "web_text"}
        or entry.get("table_item_count", 0) != 0
        or len(articles) != count
        or {article.article_number for article in articles} != set(range(1, count + 1))
    ):
        return "excerpts", excerpt_note
    return (
        "full_text",
        version_note + f"已按来源清单收录第1至第{count}条的全部条文，保留各条各款；"
        "网页导航、目录和章节标题未纳入条文列表。此处为来源正文的整理文本，"
        "不是发布机关重新发布的正式文本；完整收录不代表现行效力或个案适用已核验。",
    )


def _reading_order(article: LegalChunk) -> tuple:
    if article.kind == "document":
        # Uploaded IDs are hashes. Read the first recorded source location instead.
        position = re.search(r"字符 \[(\d+),", article.source.locator or "")
        return (
            2,
            article.source.pdf_page or 0,
            int(position.group(1)) if position else float("inf"),
            article.id,
        )
    if article.article_number is not None:
        return (0, article.article_number, article.id)
    # G/T IDs in the selected police chapters use zero-padded numeric suffixes.
    if article.document_id == "hb-police-discretion":
        return (1, article.id)
    return (1, "")  # Preserve corpus order for the original table excerpts.


def load_law_document(
    repository: LawRepository, document_id: str, version_id: str | None = None
) -> LawDocument:
    """Return only a requested version, including archived source versions by ID."""
    candidates = [
        article for article in repository.all_chunks if article.document_id == document_id
    ]
    if not candidates:
        raise AppError("document_not_found", "未找到指定文档。", 404)
    versions = sorted({article.version_id for article in candidates})
    if version_id is None:
        if len(versions) > 1:
            raise AppError(
                "document_version_required",
                "该文档收录了多个版本，请通过 version_id 指定版本：" + "、".join(versions),
                409,
            )
        version_id = versions[0]
    articles = [article for article in candidates if article.version_id == version_id]
    if not articles:
        raise AppError("document_version_not_found", "未找到指定文档的该版本。", 404)
    articles.sort(key=_reading_order)
    entry = _manifest_entry(document_id, version_id)
    coverage, note = _coverage(articles, entry)
    return LawDocument(
        document_id=document_id,
        version_id=version_id,
        title=articles[0].source.document_title,
        coverage=coverage,
        coverage_note=note,
        articles=articles,
    )
