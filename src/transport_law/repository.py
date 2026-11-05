import json
from pathlib import Path

from pydantic import ValidationError

from transport_law.errors import AppError
from transport_law.schemas import LegalChunk


class LawRepository:
    def __init__(self, chunks: list[LegalChunk]):
        self._by_id: dict[str, LegalChunk] = {}
        for chunk in chunks:
            if chunk.id in self._by_id:
                raise AppError("invalid_data", f"条目 ID 重复：{chunk.id}", 500)
            self._by_id[chunk.id] = chunk

    @classmethod
    def from_jsonl(cls, path: Path) -> "LawRepository":
        chunks = []
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        chunks.append(LegalChunk.model_validate(json.loads(line)))
                    except (ValueError, ValidationError) as exc:
                        raise AppError(
                            "invalid_data", f"法规文件第 {line_number} 行格式不正确。", 500
                        ) from exc
        except (OSError, UnicodeError) as exc:
            raise AppError(
                "data_unavailable", "无法读取法规数据，请检查 LAW_DATA_PATH。", 503
            ) from exc
        if not chunks:
            raise AppError("invalid_data", "法规数据为空，请先整理至少一条资料。", 500)
        return cls(chunks)

    @property
    def all_chunks(self) -> list[LegalChunk]:
        """包括历史版本的正文，用于按来源和版本查看原文。"""
        return list(self._by_id.values())

    @property
    def searchable(self) -> list[LegalChunk]:
        return [
            chunk
            for chunk in self._by_id.values()
            if chunk.source.verification_status != "superseded"
        ]

    def get_article(self, chunk_id: str) -> LegalChunk:
        if chunk_id not in self._by_id:
            raise AppError("article_not_found", "未找到指定条目。", 404)
        return self._by_id[chunk_id]
