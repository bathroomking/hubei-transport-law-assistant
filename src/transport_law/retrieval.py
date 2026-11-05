"""BM25 关键词排序及法规、条号定位。"""

import math
import re
import unicodedata
from collections import Counter
from typing import Protocol

from transport_law.repository import LawRepository
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
    def __init__(self, repository: LawRepository):
        all_chunks = repository.searchable
        self.by_id = {chunk.id: chunk for chunk in all_chunks}
        excerpt_ids = {ident for chunk in all_chunks for ident in chunk.related_excerpt_ids}
        # 法规全文优先于裁量表重复引用；原节选仍可按 ID 读取，也仍供裁量规则追溯。
        self.chunks = [chunk for chunk in all_chunks if chunk.id not in excerpt_ids]
        self.terms = [Counter(tokenize(c.search_text + "\n" + c.text)) for c in self.chunks]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.document_frequency = Counter(term for terms in self.terms for term in terms)

    def explicit_chunks(self, question: str) -> list[LegalChunk]:
        normalized_question = normalize(question)
        return [
            chunk for ident, chunk in self.by_id.items() if normalize(ident) in normalized_question
        ]

    def candidate_chunks(self, question: str) -> list[LegalChunk]:
        """未指定法规时搜索全库；指定法规、条号或表项时保留定位约束。"""
        normalized_question = normalize(question)
        names = {
            (normalize(alias), chunk.document_id)
            for chunk in self.chunks
            for alias in chunk.document_aliases
        }
        mentions = {
            (match.start(), match.end(), doc)
            for alias, doc in names
            for match in re.finditer(re.escape(alias), normalized_question)
        }
        # “湖北省农村公路条例”不能同时当成国家“农村公路条例”。
        # 按出现位置消歧，仍支持问题中分别提到两部同名/近名法规。
        named_documents = {
            doc
            for start, end, doc in mentions
            if not any(a <= start and end <= b and b - a > end - start for a, b, _ in mentions)
        }
        article_refs = re.findall(r"第[0-9一二三四五六七八九十百零〇]+条", normalized_question)
        table_ref = re.search(r"交通管理册第[0-9]+项", normalized_question)
        candidates = []
        for chunk in self.chunks:
            if named_documents:
                if chunk.document_id not in named_documents:
                    continue
                if article_refs and not any(
                    ref == f"第{chunk.article_number}条" or ref in normalize(chunk.article)
                    for ref in article_refs
                ):
                    continue
                if table_ref and table_ref[0] not in normalize(chunk.article):
                    continue
            candidates.append(chunk)
        return candidates

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        explicit = self.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1000) for chunk in explicit[:top_k]]
        return self.rank_candidates(question, self.candidate_chunks(question), limit=top_k)

    def rank_candidates(
        self, question: str, chunks: list[LegalChunk], *, limit: int
    ) -> list[SearchHit]:
        """内部召回窗口可大于 API 的 top_k，供混合检索在截断前融合。"""
        if not 1 <= limit <= 200:
            raise ValueError("候选数量必须在 1 到 200 之间")
        normalized_question = normalize(question)
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
            # 整部法规不等于一个执法事项，不能用无词项匹配的任意条文填满 top_k。
            if score == 0 and chunk.item_id != "road_occupation_excavation":
                continue
            # 完整条号/条目别名匹配优先，避免“第44条”被含同一法律名称的其他条文挤掉。
            alias_bonus = (
                20 if any(normalize(alias) in normalized_question for alias in chunk.aliases) else 0
            )
            # 没有匹配口语词的同事项记录仍作为关联依据保留。
            hits.append(SearchHit(chunk=chunk, score=round(score + alias_bonus + 1, 6)))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]
