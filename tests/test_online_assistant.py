import json
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from test_knowledge import upload_setup as upload_setup

from transport_law.cli import parser, run
from transport_law.errors import AppError
from transport_law.hybrid import HybridRetriever
from transport_law.knowledge import KnowledgeStore
from transport_law.llm import Completion
from transport_law.main import create_app
from transport_law.retrieval import BM25Retriever


@pytest.fixture(autouse=True)
def forbid_external_http_and_legacy_corpus_reads(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("端到端离线测试不得调用真实 HTTP 服务")

    async def unexpected_async_request(*args, **kwargs):
        pytest.fail("端到端离线测试不得调用真实 HTTP 服务")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", unexpected_request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", unexpected_async_request)
    original = Path.open

    def guarded_open(path, *args, **kwargs):
        if "/data/raw/" in str(path) or "/data/processed/" in str(path):
            pytest.fail(f"问答不得读取旧资料目录：{path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)


@pytest.fixture
def online_setup(upload_setup, monkeypatch):
    settings, embedder = upload_setup
    monkeypatch.setattr("transport_law.hybrid.DashScopeEmbeddings", lambda settings: embedder)
    return settings, embedder


class EvidenceLLM:
    def __init__(self):
        self.contexts = []

    async def complete(self, messages, *, json_mode):
        assert json_mode is True
        content = messages[1]["content"].split("Context:\n", 1)[1]
        context, _ = json.JSONDecoder().raw_decode(content)
        self.contexts.append(context)
        return Completion(
            content=json.dumps(
                {
                    "claims": [
                        {
                            "text": "请核对上传资料原文。",
                            "citation_ids": [c["id"] for c in context[:2]],
                        }
                    ],
                    "follow_up_questions": [],
                }
            ),
            model="offline-fixture",
        )


def test_empty_store_starts_and_cli_ask_needs_no_handwritten_corpus(
    online_setup, monkeypatch, capsys
):
    settings, embedder = online_setup
    llm = EvidenceLLM()
    with TestClient(create_app(settings, llm)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["chunk_count"] == 0
        result = client.post("/api/v1/ask", json={"question": "施工许可如何申请？"})
        assert result.status_code == 200
        assert result.json()["status"] == "insufficient_evidence"
        assert not result.json()["source_documents"]
    monkeypatch.setattr("transport_law.cli.Settings", lambda: settings)
    monkeypatch.setattr("transport_law.runtime.DeepSeekClient", lambda config: llm)
    assert run(parser().parse_args(["ask", "施工许可如何申请？"])) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient_evidence"
    assert not embedder.calls and not embedder.queries and not llm.contexts
    assert not settings.chroma_path.exists()


def test_upload_is_visible_to_same_runtime_and_citations_without_restart(online_setup, monkeypatch):
    settings, embedder = online_setup
    builds = []

    def build_sparse(repository):
        builds.append(len(repository.searchable))
        return BM25Retriever(repository)

    monkeypatch.setattr("transport_law.hybrid.BM25Retriever", build_sparse)
    llm = EvidenceLLM()
    with TestClient(create_app(settings, llm)) as client:
        runtime = client.app.state.runtime
        assert isinstance(runtime.service.retriever, HybridRetriever)
        assert runtime.repository.knowledge is runtime.knowledge
        assert (
            client.post("/api/v1/ask", json={"question": "施工许可"}).json()["status"]
            == "insufficient_evidence"
        )
        assert builds == [0]
        response = client.post(
            "/api/v1/knowledge/documents",
            files={
                "file": (
                    "施工说明.txt",
                    "第一条 施工许可需要审批。\n第二条 施工许可应留存记录。".encode(),
                )
            },
        )
        assert response.status_code == 202, response.text
        ident = response.json()["id"]
        record = client.get(f"/api/v1/knowledge/documents/{ident}").json()
        assert record["status"] == "ready" and record["chunk_count"] == 2
        question = "施工许可"
        answer = client.post("/api/v1/ask", json={"question": question}).json()
        assert answer["status"] == "answered"
        assert len(answer["source_documents"]) == len(answer["sources"]) == 2
        assert {c["document_id"] for c in answer["sources"]} == {ident}
        assert embedder.queries == [question]
        assert builds == [0, 2]
        batches = len(embedder.calls)
        again = client.post("/api/v1/ask", json={"question": question}).json()
        assert again["sources"] == answer["sources"]
        assert builds == [0, 2] and len(embedder.calls) == batches
        assert embedder.queries == [question, question]
        for source in answer["sources"]:
            assert client.get(f"/api/v1/articles/{source['id']}").json() == source
            document = client.get(
                f"/api/v1/documents/{ident}", params={"version_id": source["version_id"]}
            )
            assert document.status_code == 200
            assert document.json()["coverage"] == "uploaded_document"
            assert source in document.json()["articles"]
        assert answer["applicability_status"] == "not_verified"
        hits = runtime.service.retriever.search(question)
        assert all(h.retrieval.bm25_rank and h.retrieval.dense_rank for h in hits)


def test_cli_reads_existing_uploaded_library_and_never_reembeds_documents(
    online_setup, monkeypatch, capsys
):
    settings, embedder = online_setup
    store = KnowledgeStore(settings)
    ident = store.receive("说明.txt", BytesIO("施工许可需要审批。".encode()))["id"]
    store.process(ident)
    expected = store.ready_chunks()[0]
    llm = EvidenceLLM()
    before = len(embedder.calls)
    monkeypatch.setattr("transport_law.cli.Settings", lambda: settings)
    monkeypatch.setattr("transport_law.runtime.DeepSeekClient", lambda config: llm)
    assert run(parser().parse_args(["ask", "施工许可"])) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["sources"] == [expected.model_dump(mode="json")]
    assert embedder.queries == ["施工许可"] and len(embedder.calls) == before


def test_queued_failed_chunks_and_partial_vectors_are_excluded_from_answer(online_setup):
    settings, embedder = online_setup
    store = KnowledgeStore(settings)
    queued = store.receive("queued.txt", BytesIO(b"Unprocessed permit."))["id"]
    embedder.fail_on_call = 2
    body = "\n".join(f"# Part {i}\nFailed permit {i}." for i in range(6))
    failed = store.receive("failed.md", BytesIO(body.encode()))["id"]
    store.process(failed)
    assert store.get_document(failed)["status"] == "failed"
    assert store.get_document(failed)["embedded_count"] == 2
    llm = EvidenceLLM()
    with TestClient(create_app(settings, llm)) as client:
        answer = client.post("/api/v1/ask", json={"question": "permit"}).json()
        assert answer["status"] == "insufficient_evidence"
        assert not embedder.queries and not llm.contexts
        assert client.get(f"/api/v1/documents/{failed}").status_code == 404
        assert client.get(f"/api/v1/documents/{queued}").status_code == 404
        embedder.fail_on_call = None
        added = client.post(
            "/api/v1/knowledge/documents", files={"file": ("ready.txt", b"Ready permit.")}
        )
        ready = added.json()["id"]
        answer = client.post("/api/v1/ask", json={"question": "permit"}).json()
        assert {c["document_id"] for c in answer["source_documents"]} == {ready}
        assert embedder.queries == ["permit"]


def test_dense_failure_does_not_fall_back_to_sparse_answer(online_setup, monkeypatch):
    settings, embedder = online_setup
    store = KnowledgeStore(settings)
    ident = store.receive("说明.txt", BytesIO("施工许可需要审批。".encode()))["id"]
    store.process(ident)

    def fail(question):
        raise AppError("embedding_unavailable", "模拟在线向量服务失败。", 503)

    monkeypatch.setattr(embedder, "embed_query", fail)
    llm = EvidenceLLM()
    with TestClient(create_app(settings, llm)) as client:
        response = client.post("/api/v1/ask", json={"question": "施工许可"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "embedding_unavailable"
        assert not llm.contexts
