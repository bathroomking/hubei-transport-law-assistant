import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from transport_law.errors import AppError
from transport_law.graph_retrieval import load_graph
from transport_law.hybrid import HybridRetriever
from transport_law.main import create_app
from transport_law.schemas import SearchHit


@pytest.fixture
def graph_files(tmp_path, repository):
    template = repository.all_chunks[0]
    chunks = [
        template.model_copy(update={"id": f"UP-graph-{i}", "text": title, "search_text": title})
        for i, title in enumerate(["北港", "灯塔", "渡船", "记录一", "记录二", "独立", "远端"])
    ]
    names = ["北港", "灯塔", "渡船", "独立", "远端"]
    artifact = {
        "entities": [
            {"name": name, "chunk_ids": [chunks[i].id]}
            for name, i in zip(names, [0, 1, 2, 5, 6], strict=True)
        ],
        "relationships": [
            {"source": a, "target": b, "description": "相邻", "chunk_ids": [chunks[i].id]}
            for a, b, i in [("北港", "灯塔", 3), ("灯塔", "渡船", 4), ("独立", "远端", 5)]
        ],
    }
    (tmp_path / "graph.json").write_text(json.dumps(artifact), encoding="utf-8")
    return tmp_path, chunks


def test_two_hops_return_original_sources_and_exclude_disconnected_chunks(graph_files):
    directory, chunks = graph_files
    graph = load_graph(directory)
    hits = graph.rank("北港的记录", [], {c.id: c for c in chunks}, limit=50, seed_count=1)
    assert {h.chunk.id for h in hits} == {c.id for c in chunks[:5]}
    assert all(h.chunk is next(c for c in chunks if c.id == h.chunk.id) for h in hits)
    assert hits[-1].score > 0


def test_graph_rank_rejects_dangling_sources(graph_files):
    directory, chunks = graph_files
    with pytest.raises(AppError, match="不一致"):
        load_graph(directory).rank(
            "北港", [], {c.id: c for c in chunks[1:]}, limit=10, seed_count=2
        )
    (directory / "graph.json").write_text("{")
    with pytest.raises(AppError, match="无法读取"):
        load_graph(directory)


@pytest.mark.parametrize("mode", ["hybrid", "hybrid_graph", "dense_graph"])
def test_same_evidence_budget_and_only_enabled_rankers(mode, graph_files, settings, monkeypatch):
    directory, chunks = graph_files

    class Store:
        def ready_chunks(self):
            return chunks

        def rank_vectors(self, candidates, *, limit, query_vector):
            return [SearchHit(chunk=c, score=1) for c in candidates[:limit]]

        @contextmanager
        def graph_artifacts(self, candidates):
            assert mode != "hybrid" and candidates == chunks
            yield (directory,)

    class Embedder:
        def embed_query(self, query):
            return [1.0]

    if mode == "dense_graph":
        monkeypatch.setattr("transport_law.hybrid.BM25Retriever", lambda *args: pytest.fail("BM25"))
    retriever = HybridRetriever(
        Store(), settings.model_copy(update={"retrieval_mode": mode}), Embedder()
    )
    hits = retriever.search("北港", top_k=2)
    assert len(hits) == 2
    assert all(h.retrieval.dense_rank for h in hits)
    assert any(h.retrieval.graph_rank for h in hits) == (mode != "hybrid")
    assert any(h.retrieval.bm25_rank for h in hits) == (mode != "dense_graph")


@pytest.mark.parametrize("mode", ["hybrid_graph", "dense_graph"])
def test_ask_rejects_unpublished_graph_before_external_calls(
    mode, online_settings, seeded_repository, monkeypatch
):
    def unexpected(*args, **kwargs):
        pytest.fail("未发布图时不能通过普通检索或外部模型静默回答")

    monkeypatch.setattr("transport_law.hybrid.DashScopeEmbeddings.embed_query", unexpected)
    monkeypatch.setattr("transport_law.llm.DeepSeekClient.complete", unexpected)
    if mode == "dense_graph":
        monkeypatch.setattr("transport_law.hybrid.BM25Retriever", unexpected)
    with TestClient(create_app(online_settings.model_copy(update={"retrieval_mode": mode}))) as c:
        health = c.get("/health").json()
        assert health["chunk_count"] == len(seeded_repository.searchable)
        assert health["retrieval_backend"] == health["knowledge_retrieval_backend"] == mode
        assert c.get("/api/v1/knowledge/graph").json()["status"] == "disabled"
        response = c.post("/api/v1/ask", json={"question": "一般规定如何查阅？"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "graph_index_unavailable"
