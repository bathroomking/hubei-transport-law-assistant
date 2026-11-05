import json
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from test_vector_store import FakeEmbeddings
from test_vector_store import vector_fixture as vector_fixture

from transport_law.errors import AppError
from transport_law.hybrid import HybridRetriever
from transport_law.knowledge import KnowledgeStore
from transport_law.llm import Completion
from transport_law.main import create_app
from transport_law.retrieval import BM25Retriever
from transport_law.vector_store import build_vector_index

pytest.importorskip("langchain_text_splitters")


@pytest.fixture(autouse=True)
def forbid_external_http(monkeypatch):
    # TestClient 使用自己的 ASGI transport；真实 HTTP transport 在本模块一律禁止。
    def unexpected_request(*args, **kwargs):
        pytest.fail("统一问答测试不得发出真实 HTTP 请求")

    async def unexpected_async_request(*args, **kwargs):
        pytest.fail("统一问答测试不得发出真实 HTTP 请求")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", unexpected_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", unexpected_async_request)


@pytest.fixture
def online_setup(vector_fixture, monkeypatch):
    settings, repository, documents = vector_fixture
    settings = settings.model_copy(
        update={
            "deepseek_api_key": SecretStr("offline-test-key"),
            "dashscope_api_key": SecretStr("offline-test-key"),
        }
    )
    embedder = FakeEmbeddings()
    monkeypatch.setattr("transport_law.hybrid.DashScopeEmbeddings", lambda settings: embedder)
    monkeypatch.setattr("transport_law.knowledge.DashScopeEmbeddings", lambda settings: embedder)
    return settings, repository, documents, embedder


@pytest.fixture
def online_index(online_setup):
    settings, repository, documents, embedder = online_setup
    build_vector_index(settings, repository, documents, embedder=embedder)
    return settings, repository, embedder


class EvidenceLLM:
    def __init__(self):
        self.contexts = []

    async def complete(self, messages, *, json_mode):
        assert json_mode is True
        context_text = messages[1]["content"].split("Context:\n", 1)[1]
        context, _ = json.JSONDecoder().raw_decode(context_text)
        self.contexts.append(context)
        # 同时引用正式资料和上传资料；只有一类资料时引用当前这一类。
        formal = next((row for row in context if row["kind"] != "document"), None)
        uploaded = next((row for row in context if row["kind"] == "document"), None)
        citations = [row["id"] for row in (formal, uploaded) if row is not None]
        return Completion(
            content=json.dumps(
                {
                    "claims": [{"text": "请核对所附材料的原文。", "citation_ids": citations}],
                    "follow_up_questions": [],
                }
            ),
            model="offline-fixture",
        )


def test_unified_ask_merges_both_libraries_and_publishes_ready_sources_without_restart(
    online_index, monkeypatch
):
    settings, repository, embedder = online_index
    question = "那我接下来该咋办呀？"
    assert BM25Retriever(repository).candidate_chunks(question) == repository.searchable
    sparse_builds = []

    def build_sparse(repository):
        sparse_builds.append(len(repository.searchable))
        return BM25Retriever(repository)

    monkeypatch.setattr("transport_law.hybrid.BM25Retriever", build_sparse)
    llm = EvidenceLLM()
    with TestClient(create_app(settings, llm)) as client:
        runtime = client.app.state.runtime
        assert isinstance(runtime.service.retriever, HybridRetriever)
        assert runtime.repository.knowledge is runtime.knowledge
        first = client.post("/api/v1/ask", json={"question": question, "top_k": 3})
        assert first.status_code == 200, first.text
        assert len(first.json()["source_documents"]) == 2
        assert embedder.queries == [question]
        assert sparse_builds == [2]

        upload = client.post(
            "/api/v1/knowledge/documents",
            files={"file": ("施工说明.txt", "占用公路需许可，应核对许可范围。".encode())},
        )
        assert upload.status_code == 202, upload.text
        uploaded_id = upload.json()["id"]
        record = client.get(f"/api/v1/knowledge/documents/{uploaded_id}").json()
        assert record["status"] == "ready"
        assert record["chunk_count"] == record["embedded_count"] == 1
        embedded_batches = len(embedder.documents)

        hits = runtime.service.retriever.search(question, top_k=3)
        assert len(hits) == 3
        assert {hit.chunk.document_id for hit in hits} == {"test-road", uploaded_id}
        # 两库先形成同一路 dense 排名；上传库首位不能另外从 dense_rank=1 起算。
        assert {hit.retrieval.dense_rank for hit in hits} == {1, 2, 3}
        uploaded_hit = next(hit for hit in hits if hit.chunk.document_id == uploaded_id)
        assert uploaded_hit.retrieval.dense_rank == 2
        assert all(hit.retrieval.method == "rrf" for hit in hits)
        assert embedder.queries == [question, question]
        assert sparse_builds == [2, 3]

        second = client.post("/api/v1/ask", json={"question": question, "top_k": 3})
        assert second.status_code == 200, second.text
        answer = second.json()
        assert answer["source_documents"] == [hit.chunk.model_dump(mode="json") for hit in hits]
        assert len(answer["sources"]) == 2
        assert {source["document_id"] for source in answer["sources"]} == {
            "test-road",
            uploaded_id,
        }
        assert answer["applicability_status"] == "not_verified"
        assert len(llm.contexts) == 2
        assert [row["id"] for row in llm.contexts[-1]] == [hit.chunk.id for hit in hits]
        # 一次查询只有一个问题向量，两座 collection 不重复请求，也不重建文档向量。
        assert embedder.queries == [question, question, question]
        assert len(embedder.documents) == embedded_batches
        assert sparse_builds == [2, 3]  # ready 快照未变，不重复构建联合词频统计。

        for source in answer["sources"]:
            article = client.get(f"/api/v1/articles/{source['id']}")
            assert article.status_code == 200
            assert article.json() == source
            document = client.get(
                f"/api/v1/documents/{source['document_id']}",
                params={"version_id": source["version_id"]},
            )
            assert document.status_code == 200, document.text
            body = document.json()
            assert body["version_id"] == source["version_id"]
            assert source in body["articles"]
            if source["document_id"] == uploaded_id:
                assert body["coverage"] == "uploaded_document"
                assert body["articles"] == [source]
            else:
                assert body["articles"] == [
                    chunk.model_dump(mode="json") for chunk in repository.all_chunks
                ]


def test_sparse_and_dense_ranks_are_global_across_libraries(online_index):
    settings, repository, embedder = online_index
    knowledge = KnowledgeStore(settings)
    uploaded_id = knowledge.receive("说明.txt", BytesIO("占用公路需许可。".encode()))["id"]
    knowledge.process(uploaded_id)
    assert knowledge.get_document(uploaded_id)["status"] == "ready"
    retriever = HybridRetriever(repository, settings, embedder, knowledge=knowledge)
    hits = retriever.search("占用公路需许可", top_k=3)
    assert {hit.chunk.document_id for hit in hits} == {"test-road", uploaded_id}
    # 正式库和上传库共同竞争排名，两边各自第一名不能都标成同一路 rank=1。
    assert {hit.retrieval.bm25_rank for hit in hits} == {1, 2, 3}
    assert {hit.retrieval.dense_rank for hit in hits} == {1, 2, 3}
    assert embedder.queries == ["占用公路需许可"]


@pytest.mark.parametrize(
    "question",
    ["喝了点酒开车会怎么样？", "车停路边被贴条怎么办？", "开车时接了一个电话会怎么处理？"],
)
def test_everyday_language_reaches_dense_retrieval_without_aliases(online_index, question):
    settings, repository, embedder = online_index
    assert BM25Retriever(repository).candidate_chunks(question) == repository.searchable
    retriever = HybridRetriever(repository, settings, embedder, knowledge=KnowledgeStore(settings))
    hits = retriever.search(question)
    assert {hit.chunk.id for hit in hits} == {chunk.id for chunk in repository.searchable}
    assert all(hit.retrieval.dense_rank for hit in hits)
    assert embedder.queries == [question]


def test_explicit_missing_article_does_not_broaden_or_call_embedding(online_setup):
    settings, repository, _, embedder = online_setup
    retriever = HybridRetriever(repository, settings, embedder, knowledge=KnowledgeStore(settings))
    assert retriever.search("测试公路法第999条") == []
    assert embedder.queries == []
    assert not settings.chroma_path.exists()


def test_dense_failure_is_not_replaced_with_sparse_answer(online_index, monkeypatch):
    settings, repository, embedder = online_index
    question = "公路挖沟有哪些依据？"
    assert BM25Retriever(repository).search(question)

    def unavailable(question):
        raise AppError("embedding_unavailable", "模拟在线向量服务失败。", 503)

    monkeypatch.setattr(embedder, "embed_query", unavailable)
    llm = EvidenceLLM()
    with TestClient(create_app(settings, llm)) as client:
        response = client.post("/api/v1/ask", json={"question": question})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "embedding_unavailable"
        assert llm.contexts == []


def test_missing_index_allows_startup_but_ask_reports_local_error(online_setup):
    settings, _, _, embedder = online_setup
    llm = EvidenceLLM()
    with TestClient(create_app(settings, llm)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["online_only"] is True
        assert client.get("/").status_code == 200
        assert embedder.queries == []
        response = client.post("/api/v1/ask", json={"question": "公路挖沟有哪些依据？"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "vector_index_missing"
        assert llm.contexts == []
        assert embedder.queries == []
        assert not settings.chroma_path.exists()
