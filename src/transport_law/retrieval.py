"""小语料 BM25 检索；事项别名是可解释的召回入口，后续可替换为混合检索。"""

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
    parts = re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalize(text))
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
        self.terms = [Counter(tokenize(c.search_text + c.text)) for c in self.chunks]
        self.lengths = [sum(terms.values()) for terms in self.terms]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.document_frequency = Counter(term for terms in self.terms for term in terms)

    def _candidate_items(self, question: str) -> set[str]:
        # 通用词（公路、罚款、湖北）不能单独把范围外问题召回到本事项。
        candidate_items = {
            chunk.item_id
            for chunk in self.chunks
            if any(normalize(alias) in question for alias in chunk.aliases)
            or normalize(chunk.id) in question
        }
        # 第一事项的口语组合匹配；只产生候选事项，不认定道路属性或许可状态。
        roads = ("公路", "道路", "路边", "路旁", "路上", "路面", "国道", "省道", "县道", "乡道")
        actions = ("挖", "开沟", "开槽", "破路", "占用", "占了", "占着", "占道")
        if any(word in question for word in roads) and any(word in question for word in actions):
            candidate_items.add("road_occupation_excavation")
        return candidate_items

    def explicit_chunks(self, question: str) -> list[LegalChunk]:
        normalized_question = normalize(question)
        return [
            chunk for ident, chunk in self.by_id.items() if normalize(ident) in normalized_question
        ]

    def candidate_chunks(self, question: str) -> list[LegalChunk]:
        """复用法规名、条号及事项范围约束；向量检索只替换候选内的排序。"""
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
        candidate_items = self._candidate_items(normalized_question)
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
            elif chunk.item_id not in candidate_items:
                continue
            candidates.append(chunk)
        return candidates

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        explicit = self.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1000) for chunk in explicit[:top_k]]
        normalized_question = normalize(question)
        candidates = {chunk.id for chunk in self.candidate_chunks(question)}
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
            if normalize(chunk.id) in normalized_question:
                alias_bonus += 100
            # 没有匹配口语词的同事项记录仍作为关联依据保留。
            hits.append(SearchHit(chunk=chunk, score=round(score + alias_bonus + 1, 6)))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]
