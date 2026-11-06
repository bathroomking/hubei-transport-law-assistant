"""按文档和版本读取已完成入库的上传片段。"""

import re
from typing import Literal

from pydantic import Field

from transport_law.errors import AppError
from transport_law.repository import ChunkRepository, KnowledgeRepository
from transport_law.schemas import Identifier, LegalChunk, Model, Text


class LawDocument(Model):
    document_id: Identifier
    version_id: Identifier
    title: Text
    coverage: Literal["uploaded_document"] = "uploaded_document"
    coverage_note: Text
    articles: list[LegalChunk] = Field(min_length=1)


def _reading_order(article: LegalChunk) -> tuple:
    # 表格续块会重复表头，首个字符位置可能相同；新记录用生成顺序，旧记录回退到字符位置。
    sequence = re.search(r"片段序号 (\d+)", article.source.locator or "")
    position = re.search(r"字符 \[(\d+),", article.source.locator or "")
    order = sequence or position
    return (
        article.source.pdf_page,
        int(order.group(1)) if order else float("inf"),
        article.id,
    )


def load_law_document(
    repository: ChunkRepository | KnowledgeRepository,
    document_id: str,
    version_id: str | None = None,
) -> LawDocument:
    candidates = [
        article for article in repository.all_chunks if article.document_id == document_id
    ]
    if not candidates:
        raise AppError("document_not_found", "未找到已入库的指定文档。", 404)
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
    return LawDocument(
        document_id=document_id,
        version_id=version_id,
        title=articles[0].source.document_title,
        coverage_note=(
            "以下为用户上传文件已入库的文字片段，可能包含重叠内容；"
            "不表示已收录整部法规，文字提取完整性、法律效力及适用性尚待核验。"
        ),
        articles=articles,
    )
