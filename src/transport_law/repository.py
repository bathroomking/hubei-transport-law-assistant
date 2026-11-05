"""上传知识库的动态读取及单次检索使用的内存快照。"""

from typing import TYPE_CHECKING

from transport_law.errors import AppError
from transport_law.schemas import LegalChunk

if TYPE_CHECKING:
    from transport_law.knowledge import KnowledgeStore


class ChunkRepository:
    def __init__(self, chunks: list[LegalChunk]):
        self._by_id: dict[str, LegalChunk] = {}
        for chunk in chunks:
            if chunk.id in self._by_id:
                raise AppError("invalid_data", f"条目 ID 重复：{chunk.id}", 500)
            self._by_id[chunk.id] = chunk

    @property
    def all_chunks(self) -> list[LegalChunk]:
        return list(self._by_id.values())

    @property
    def searchable(self) -> list[LegalChunk]:
        return self.all_chunks

    def get_article(self, chunk_id: str) -> LegalChunk:
        if chunk_id not in self._by_id:
            raise AppError("article_not_found", "未找到指定片段。", 404)
        return self._by_id[chunk_id]


class KnowledgeRepository:
    def __init__(self, knowledge: "KnowledgeStore"):
        self.knowledge = knowledge

    @property
    def all_chunks(self) -> list[LegalChunk]:
        return self.knowledge.ready_chunks()

    @property
    def searchable(self) -> list[LegalChunk]:
        return self.knowledge.ready_chunks()

    def get_article(self, chunk_id: str) -> LegalChunk:
        return self.knowledge.get_article(chunk_id)
