import json
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from transport_law.cli import parser, run
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


class QueryEmbeddings:
    def __init__(self):
        self.queries = []

    def embed_query(self, question):
        self.queries.append(question)
        return [1.0, 0.0]


class SnapshotStore:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []
        self.reads = 0

    def ready_chunks(self):
        self.reads += 1
        return list(self.chunks)

    def rank_vectors(self, chunks, *, limit, query_vector):
        assert query_vector == [1.0, 0.0]
        self.calls.append(([c.id for c in chunks], limit))
        return [SearchHit(chunk=c, score=1 / (i + 1)) for i, c in enumerate(chunks[:limit])]


@pytest.fixture
def hybrid(repository, settings):
    return HybridRetriever(SnapshotStore(repository.all_chunks), settings, QueryEmbeddings())


def test_one_ready_snapshot_and_one_query_embedding_per_search(hybrid):
    result = hybrid.search("未知语义问题")
    assert result
    assert all(h.retrieval.dense_rank for h in result)
    assert hybrid.knowledge.reads == 1
    assert len(hybrid.knowledge.calls) == 1
    assert hybrid.embedder.queries == ["未知语义问题"]


def test_lane_penalty_can_recall_general_rule_without_matching_lane_words(
    repository, settings, monkeypatch
):
    base = repository.all_chunks[0]
    records = [
        base.model_copy(update={"id": ident, "text": text, "search_text": text})
        for ident, text in [
            ("lane", "非机动车应当在非机动车道内行驶。"),
            ("general", "非机动车驾驶人违反道路通行规定的，处警告或罚款。另有规定的从其规定。"),
        ]
    ]
    store, embedder = SnapshotStore(records), QueryEmbeddings()
    monkeypatch.setattr(store, "rank_vectors", lambda *args, **kwargs: [])
    retriever = HybridRetriever(store, settings, embedder)
    assert {hit.chunk.id for hit in retriever.search("骑单车走机动车道怎么罚？")} == {
        "general",
        "lane",
    }
    motorist = "小汽车占用非机动车道怎么罚？"
    retriever.search(motorist)
    assert len(embedder.queries) == 2 and embedder.queries[-1] == motorist


def test_exact_uploaded_id_does_not_call_embedding(hybrid):
    result = hybrid.search("UP-fixture-2", top_k=1)
    assert [h.chunk.id for h in result] == ["UP-fixture-2"]
    assert result[0].retrieval is None
    assert not hybrid.embedder.queries and not hybrid.knowledge.calls


def test_empty_store_never_calls_embedding_or_dense_index(settings):
    retriever = HybridRetriever(SnapshotStore([]), settings, QueryEmbeddings())
    assert retriever.search("施工许可") == []
    assert not retriever.embedder.queries and not retriever.knowledge.calls


def test_ready_changes_refresh_bm25_but_identical_snapshot_is_reused(hybrid, repository):
    hybrid.search("施工许可")
    before = hybrid._snapshot[1]
    hybrid.search("安全防护")
    assert hybrid._snapshot[1] is before
    added = repository.all_chunks[0].model_copy(
        update={"id": "UP-added", "text": "新增施工许可", "search_text": "新增施工许可"}
    )
    hybrid.knowledge.chunks.append(added)
    result = hybrid.search("施工许可")
    assert hybrid._snapshot[1] is not before
    assert "UP-added" in {h.chunk.id for h in result}
    hybrid.knowledge.chunks = []
    assert hybrid.search("施工许可") == []


def test_parallel_channels_fuse_before_final_top_k(hybrid, repository, monkeypatch):
    template = repository.all_chunks[0]
    candidates = [template.model_copy(update={"id": f"UP-test-{n:03}"}) for n in range(60)]
    hybrid.knowledge.chunks = candidates
    hybrid.search("初始化快照")
    barrier = threading.Barrier(2)
    calls = {}

    def sparse(question, chunks, *, limit):
        calls["sparse"] = ([c.id for c in chunks], limit)
        barrier.wait(timeout=5)
        return [SearchHit(chunk=c, score=1 / (i + 1)) for i, c in enumerate(chunks[:limit])]

    def dense(chunks, *, limit, query_vector):
        calls["dense"] = ([c.id for c in chunks], limit)
        barrier.wait(timeout=5)
        return [
            SearchHit(chunk=candidates[50], score=1),
            SearchHit(chunk=candidates[19], score=0.9),
        ]

    monkeypatch.setattr(hybrid._snapshot[1], "rank_candidates", sparse)
    monkeypatch.setattr(hybrid.knowledge, "rank_vectors", dense)
    hit = hybrid.search("施工许可", top_k=1)[0]
    assert hit.chunk.id == candidates[19].id
    assert hit.retrieval.bm25_rank == 20 and hit.retrieval.dense_rank == 2
    assert calls["sparse"] == calls["dense"] == ([c.id for c in candidates], 50)


def test_vector_failure_is_not_silently_reported_as_successful_fusion(hybrid, monkeypatch):
    def fail(*args, **kwargs):
        raise AppError("embedding_timeout", "模拟超时。", 503)

    monkeypatch.setattr(hybrid.knowledge, "rank_vectors", fail)
    with pytest.raises(AppError) as error:
        hybrid.search("施工许可")
    assert error.value.code == "embedding_timeout"


@pytest.mark.parametrize("top_k", [0, 11])
def test_hybrid_rejects_invalid_limit_before_query(hybrid, top_k):
    with pytest.raises(ValueError):
        hybrid.search("施工许可", top_k)
    assert not hybrid.knowledge.calls and not hybrid.embedder.queries


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


@pytest.mark.parametrize(
    "command",
    [
        "retrieve",
        "chat",
        "check-facts",
        "evaluate",
        "validate-data",
        "index-vectors",
        "vector-status",
        "chunk-document",
        "extract-pdf",
    ],
)
def test_cli_only_exposes_active_serve_and_ask_commands(command):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args([command])
    assert exc.value.code == 2


def test_cli_ask_uses_shared_runtime_and_preserves_question_and_limit(
    settings, monkeypatch, capsys
):
    seen = []

    class Service:
        async def ask(self, request):
            seen.append(request)
            return {"answer": "已根据资料说明。", "sources": []}

    def runtime(config):
        assert config is settings
        return SimpleNamespace(service=Service())

    monkeypatch.setattr("transport_law.cli.Settings", lambda: settings)
    monkeypatch.setattr("transport_law.cli.build_runtime", runtime)
    args = parser().parse_args(["ask", "  施工许可如何申请？  ", "--top-k", "3"])
    assert run(args) == 0
    assert seen[0].question == "施工许可如何申请？" and seen[0].top_k == 3
    assert json.loads(capsys.readouterr().out)["answer"] == "已根据资料说明。"
    with pytest.raises(SystemExit):
        parser().parse_args(["ask", "施工许可", "--backend", "bm25"])
