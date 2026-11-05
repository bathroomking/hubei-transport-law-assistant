"""对已入库的自动分块文本进行 BM25 关键词排序。"""

import math
import re
import unicodedata
from collections import Counter
from typing import Protocol

from transport_law.repository import ChunkRepository
from transport_law.schemas import LegalChunk, SearchHit


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    # 中文排版空白可以合并；英文单词间空白必须保留，否则 Road permit 变成 roadpermit。
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    parts = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized)
    tokens = []
    for part in parts:
        if re.fullmatch(r"[a-z0-9]+", part):
            tokens.append(part)
        else:
            for size in (2, 3):
                tokens.extend(part[i : i + size] for i in range(len(part) - size + 1))
    return tokens


class Retriever(Protocol):
    def search(self, question: str, top_k: int = 5) -> list[SearchHit]: ...


class BM25Retriever:
    def __init__(self, repository: ChunkRepository):
        all_chunks = repository.searchable
        self.by_id = {chunk.id: chunk for chunk in all_chunks}
        self.chunks = all_chunks
        self.terms = [Counter(tokenize(c.search_text + "\n" + c.text)) for c in self.chunks]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.document_frequency = Counter(term for terms in self.terms for term in terms)

    def explicit_chunks(self, question: str) -> list[LegalChunk]:
        normalized_question = normalize(question)
        return [
            chunk for ident, chunk in self.by_id.items() if normalize(ident) in normalized_question
        ]

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        explicit = self.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1000) for chunk in explicit[:top_k]]
        return self.rank_candidates(question, self.chunks, limit=top_k)

    def rank_candidates(
        self, question: str, chunks: list[LegalChunk], *, limit: int
    ) -> list[SearchHit]:
        """内部召回窗口可大于 API 的 top_k，供混合检索在截断前融合。"""
        if not 1 <= limit <= 200:
            raise ValueError("候选数量必须在 1 到 200 之间")
        candidates = {chunk.id for chunk in chunks}
        query_terms = set(tokenize(question))
        hits = []
        count = len(self.chunks)
        for chunk, terms, length in zip(self.chunks, self.terms, self.lengths, strict=True):
            if chunk.id not in candidates:
                continue
            score = 0.0
            for term in query_terms:
                frequency = terms[term]
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * frequency
                    * 2.5
                    / (frequency + 1.5 * (0.25 + 0.75 * length / self.average_length))
                )
            # 不用零匹配片段填满关键词召回窗口。
            if score == 0:
                continue
            hits.append(SearchHit(chunk=chunk, score=round(score, 6)))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]
