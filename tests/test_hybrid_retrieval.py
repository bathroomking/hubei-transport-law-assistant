import threading

import pytest
from pydantic import ValidationError

from transport_law.cli import parser
from transport_law.config import Settings
from transport_law.errors import AppError
from transport_law.hybrid import HybridRetriever, reciprocal_rank_fusion
from transport_law.schemas import SearchHit


def test_rrf_uses_ranks_not_incomparable_scores_and_keeps_provenance(repository):
    a, b, c = repository.searchable[:3]
    sparse = [SearchHit(chunk=a, score=100000), SearchHit(chunk=b, score=1)]
    dense = [SearchHit(chunk=b, score=0.01), SearchHit(chunk=c, score=-0.9)]
    hits = reciprocal_rank_fusion(sparse, dense)
    assert [hit.chunk.id for hit in hits] == [b.id, a.id, c.id]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert hits[0].retrieval.bm25_rank == 2
    assert hits[0].retrieval.dense_rank == 1
    assert hits[0].chunk == repository.get_article(b.id)
    assert hits[1].retrieval.dense_rank is None
    assert hits[2].retrieval.bm25_rank is None
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_deduplicates_before_assigning_ranks_and_breaks_ties_by_id(repository):
    a, b = sorted(repository.searchable[:2], key=lambda chunk: chunk.id)
    first, second = SearchHit(chunk=a, score=99), SearchHit(chunk=b, score=1)
    hits = reciprocal_rank_fusion([first, first, second], [second, first], rrf_k=20)
    assert [hit.chunk.id for hit in hits] == [a.id, b.id]
    assert hits[0].score == hits[1].score == pytest.approx(1 / 21 + 1 / 22)
    assert hits[1].retrieval.bm25_rank == 2
    assert len(reciprocal_rank_fusion([first, second], [], top_k=1)) == 1


class StubDense:
    def __init__(self, repository, settings, embedder=None, *, scope=None):
        self.scope = scope
        self.calls = []

    def rank_candidates(self, question, chunks, *, limit):
        self.calls.append((question, [chunk.id for chunk in chunks], limit))
        return [SearchHit(chunk=chunk, score=0.5) for chunk in chunks[:limit]]


@pytest.fixture
def hybrid(repository, settings, monkeypatch):
    monkeypatch.setattr("transport_law.hybrid.ChromaRetriever", StubDense)
    return HybridRetriever(repository, settings)


@pytest.mark.parametrize(
    "question",
    [
        "武汉明天天气怎样？",
        "个人所得税专项附加扣除如何计算？",
        "道路交通安全法第999条",
        "湖北公安交管裁量交通管理册第99项",
    ],
)
def test_empty_scope_and_nonexistent_articles_never_call_embedding(hybrid, question):
    assert hybrid.search(question) == []
    assert hybrid.dense.calls == []


def test_explicit_excerpt_id_remains_available_without_embedding(hybrid):
    hits = hybrid.search("HB-ROAD-003-REG62", top_k=1)
    assert [hit.chunk.id for hit in hits] == ["HB-ROAD-003-REG62"]
    assert hits[0].retrieval is None  # ID 直接定位，不冒充 RRF 结果。
    assert hybrid.dense.calls == []


@pytest.mark.parametrize(
    "question,expected",
    [
        ("道路交通安全法第38条", "cn-traffic-safety-2021-A038"),
        ("道路交通安全法实施条例第38条", "cn-traffic-implementation-2017-A038"),
        ("湖北道交法实施办法第56条", "hb-traffic-2017-A056"),
        ("湖北公安交管裁量交通管理册第20项", "hb-police-discretion-2026-T020"),
    ],
)
def test_both_channels_respect_law_article_disambiguation(hybrid, question, expected):
    assert [hit.chunk.id for hit in hybrid.search(question)] == [expected]
    assert hybrid.dense.calls[0][1] == [expected]


def test_parallel_channels_share_scope_and_fuse_before_final_top_k(hybrid, monkeypatch):
    barrier = threading.Barrier(2)
    calls = {}
    candidates = hybrid.bm25.candidate_chunks("道路交通安全法")
    assert len(candidates) > 50

    def channel(name, ordering):
        def rank(question, chunks, *, limit):
            calls[name] = ([c.id for c in chunks], limit)
            barrier.wait(timeout=5)  # 如果两路改为串行，这里会失败。
            return [SearchHit(chunk=c, score=0.5) for c in ordering[:limit]]

        return rank

    # 首位法条仅在一路出现；第 20 位法条在两路出现，应经融合排到前面。
    sparse = candidates[:50]
    dense = [candidates[50], candidates[19]]
    monkeypatch.setattr(hybrid.bm25, "rank_candidates", channel("bm25", sparse))
    monkeypatch.setattr(hybrid.dense, "rank_candidates", channel("dense", dense))
    hit = hybrid.search("道路交通安全法", top_k=1)[0]
    assert hit.chunk.id == candidates[19].id
    assert hit.retrieval.bm25_rank == 20 and hit.retrieval.dense_rank == 2
    assert calls["bm25"] == calls["dense"] == ([c.id for c in candidates], 50)


def test_vector_failure_is_not_silently_reported_as_successful_fusion(hybrid, monkeypatch):
    def fail(*args, **kwargs):
        raise AppError("embedding_timeout", "模拟超时。", 503)

    monkeypatch.setattr(hybrid.dense, "rank_candidates", fail)
    with pytest.raises(AppError) as error:
        hybrid.search("公路挖沟")
    assert error.value.code == "embedding_timeout"


@pytest.mark.parametrize("top_k", [0, 11])
def test_hybrid_rejects_invalid_limit_before_query(hybrid, top_k):
    with pytest.raises(ValueError):
        hybrid.search("公路挖沟", top_k)
    assert not hybrid.dense.calls


@pytest.mark.parametrize(
    "update",
    [
        {"hybrid_candidate_k": 9},
        {"hybrid_candidate_k": 201},
        {"hybrid_rrf_k": 0},
        {"hybrid_rrf_k": 1001},
    ],
)
def test_hybrid_settings_validate_windows(settings, update):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **(settings.model_dump() | update))


@pytest.mark.parametrize("command", ["retrieve", "ask", "evaluate"])
def test_cli_accepts_hybrid_backend(command):
    args = [command] + ([] if command == "evaluate" else ["公路挖沟"])
    assert parser().parse_args(args + ["--backend", "hybrid"]).backend == "hybrid"
