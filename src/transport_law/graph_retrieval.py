"""用微软 GraphRAG 的实体和有出处的关系排序原文片段。"""

from collections import Counter, defaultdict
from functools import lru_cache
from math import fsum, log, sqrt
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from transport_law.errors import AppError
from transport_law.retrieval import normalize, tokenize
from transport_law.schemas import LegalChunk, SearchHit


class _Entity(BaseModel):
    name: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)


class _Relationship(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    description: str
    chunk_ids: list[str] = Field(min_length=1)


class _Export(BaseModel):
    entities: list[_Entity] = Field(min_length=1)
    relationships: list[_Relationship] = Field(min_length=1)


def _strongest(scores: Counter, limit: int) -> dict:
    return dict(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit])


class EvidenceGraph:
    def __init__(self, artifact: _Export):
        self.mentions: dict[str, set[str]] = defaultdict(set)
        self.entities_in: dict[str, set[str]] = defaultdict(set)
        for entity in artifact.entities:
            self.mentions[entity.name].update(entity.chunk_ids)
            for chunk_id in entity.chunk_ids:
                self.entities_in[chunk_id].add(entity.name)
        self.labels = {name: normalize(name) for name in self.mentions}
        self.name_terms = {name: set(tokenize(name)) for name in self.mentions}
        frequencies = Counter(t for terms in self.name_terms.values() for t in terms)
        self.idf = {t: 1 + log((1 + len(self.mentions)) / (1 + n)) for t, n in frequencies.items()}

        self.edges = artifact.relationships
        self.adjacent: dict[str, list[tuple[int, str]]] = defaultdict(list)
        self.edge_terms = []
        self.chunk_ids = set(self.entities_in)
        for index, edge in enumerate(self.edges):
            if edge.source not in self.mentions or edge.target not in self.mentions:
                raise ValueError("关系指向未知实体")
            self.adjacent[edge.source].append((index, edge.target))
            if edge.target != edge.source:
                self.adjacent[edge.target].append((index, edge.source))
            self.edge_terms.append(set(tokenize(edge.description)))
            self.chunk_ids.update(edge.chunk_ids)

    def _seeds(self, question: str, dense_hits: list[SearchHit], count: int) -> dict[str, float]:
        # 题面实体可直接入图；口语问法还可借向量命中的原文定位其中提及的实体。
        terms = set(tokenize(question))
        normalized = normalize(question)
        lexical = Counter()
        for name, tokens in self.name_terms.items():
            shared = tokens & terms
            label = self.labels[name]
            if len(label) >= 2 and label in normalized:
                lexical[name] = 1 + fsum(self.idf[t] for t in tokens)
            elif shared:
                lexical[name] = fsum(self.idf[t] for t in shared) / sqrt(len(tokens) * len(terms))
        semantic = Counter()
        for i, hit in enumerate(dense_hits[:count], start=1):
            names = self.entities_in.get(hit.chunk.id, ())
            for name in names:
                semantic[name] += 1 / (i * len(names))
        seeds = Counter()
        for candidates in (_strongest(lexical, count), semantic):
            total = fsum(candidates.values())
            for name, value in candidates.items():
                seeds[name] += value / total
        return dict(seeds)

    def rank(
        self,
        question: str,
        dense_hits: list[SearchHit],
        by_id: dict[str, LegalChunk],
        *,
        limit: int,
        seed_count: int,
    ) -> list[SearchHit]:
        if not self.chunk_ids.issubset(by_id):
            raise AppError("graph_index_invalid", "知识图谱与可检索原文不一致。", 503)
        terms = set(tokenize(question))
        frontier = self._seeds(question, dense_hits, seed_count)
        visited = set(frontier)
        scores = Counter()
        # 最多两跳。按节点的边数分散权重，高频泛称不能独自拉起大量无关条文。
        for depth in range(3):
            following = Counter()
            for name, mass in frontier.items():
                mentions = self.mentions[name]
                for chunk_id in mentions:
                    scores[chunk_id] += mass / len(mentions)
                if depth == 2:
                    continue
                links = self.adjacent.get(name, ())
                weights = [
                    1 + len(terms & self.edge_terms[i]) / max(1, len(terms)) for i, _ in links
                ]
                total = sum(weights)
                for (i, target), weight in zip(links, weights, strict=True):
                    contribution = 0.5 * mass * weight / total
                    evidence = self.edges[i].chunk_ids
                    for chunk_id in evidence:
                        scores[chunk_id] += contribution / len(evidence)
                    if target not in visited:
                        following[target] += contribution
            frontier = _strongest(following, limit)
            visited.update(frontier)
        return [
            SearchHit(chunk=by_id[ident], score=score)
            for ident, score in _strongest(scores, limit).items()
        ]


@lru_cache(maxsize=2)
def _read_graph(path: Path, modified_ns: int, size: int) -> EvidenceGraph:
    return EvidenceGraph(_Export.model_validate_json(path.read_bytes()))


def load_graph(directory: Path) -> EvidenceGraph:
    path = directory / "graph.json"
    try:
        stat = path.stat()
        return _read_graph(path, stat.st_mtime_ns, stat.st_size)
    except (OSError, ValidationError, ValueError) as exc:
        raise AppError("graph_index_invalid", "知识图谱产物无法读取或校验失败。", 503) from exc
