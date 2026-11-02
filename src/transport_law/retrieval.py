"""小语料 BM25 检索；事项别名是可解释的召回入口，后续可替换为混合检索。"""

import math
import re
import unicodedata
from collections import Counter
from typing import Protocol

from transport_law.repository import LawRepository
from transport_law.schemas import SearchHit


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
        self.chunks = repository.searchable
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

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        normalized_question = normalize(question)
        candidate_items = self._candidate_items(normalized_question)
        query_terms = set(tokenize(question))
        hits = []
        count = len(self.chunks)
        for chunk, terms, length in zip(self.chunks, self.terms, self.lengths, strict=True):
            if chunk.item_id not in candidate_items:
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
            # 完整条号/条目别名匹配优先，避免“第44条”被含同一法律名称的其他条文挤掉。
            alias_bonus = (
                20 if any(normalize(alias) in normalized_question for alias in chunk.aliases) else 0
            )
            if normalize(chunk.id) in normalized_question:
                alias_bonus += 100
            # 没有匹配口语词的同事项记录仍作为关联依据保留。
            hits.append(SearchHit(chunk=chunk, score=round(score + alias_bonus + 1, 6)))
        return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:top_k]
