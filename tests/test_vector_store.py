import hashlib
import json
import subprocess
import sys

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from transport_law.cli import parser, run
from transport_law.embeddings import DashScopeEmbeddings
from transport_law.errors import AppError
from transport_law.hybrid import HybridRetriever
from transport_law.llm import Completion
from transport_law.main import create_app
from transport_law.repository import LawRepository
from transport_law.vector_store import (
    ChromaRetriever,
    build_vector_index,
    load_vector_documents,
    open_vector_index,
)

pytest.importorskip("chromadb")


class FakeEmbeddings:
    def __init__(self, fail_call=None):
        self.documents = []
        self.queries = []
        self.fail_call = fail_call

    def embed_documents(self, texts):
        self.documents.append(texts)
        if len(self.documents) == self.fail_call:
            raise AppError("offline_failure", "模拟上游中断。")
        return [[1.0, 0.0, 0.0] if "第一条" in text else [0.8, 0.6, 0.0] for text in texts]

    def embed_query(self, question):
        self.queries.append(question)
        return [1.0, 0.0, 0.0]


@pytest.fixture
def vector_fixture(tmp_path, settings, repository):
    base = repository.get_article("cn-highway-2017-A044")
    parents = []
    rows = []
    for number, label in [(1, "一"), (2, "二")]:
        # 第一条拆成九块，验证扩大候选和父条文去重，不让 top_k 被同一法条占满。
        text = "占用公路需许可。" * (9 if number == 1 else 1)
        parent = base.model_copy(
            update={
                "id": f"test-A00{number}",
                "document_id": "test-road",
                "article": f"测试公路法第{label}条",
                "article_number": number,
                "text": text,
                "document_aliases": ["测试公路法"],
                "related_excerpt_ids": [],
                "source": base.source.model_copy(update={"document_title": "测试公路法"}),
            }
        )
        parents.append(parent)
        for start in range(0, len(text), 8):
            end = min(start + 8, len(text))
            rows.append(
                {
                    "id": f"child-{number}-{start}",
                    "page_content": text[start:end],
                    "metadata": {
                        "document_id": parent.document_id,
                        "version_id": parent.version_id,
                        "title": parent.source.document_title,
                        "heading": parent.article,
                        "source": {
                            "sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "locations": [
                                {"start_index": start, "end_index": end, "pdf_page": None}
                            ],
                        },
                        "provenance": parent.model_dump(mode="json", exclude={"text"}),
                    },
                }
            )
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(parent.model_dump_json() for parent in parents))
    configured = settings.model_copy(
        update={
            "law_data_path": corpus,
            "chroma_path": tmp_path / "chroma",
            "embedding_dimensions": 3,
            "embedding_batch_size": 4,
        }
    )
    return configured, LawRepository(parents), path


def test_preflight_does_not_write_and_validates_complete_provenance(vector_fixture):
    settings, repository, path = vector_fixture
    report = build_vector_index(settings, repository, path, dry_run=True)
    assert report["status"] == "validated" and report["parents"] == 2
    assert not settings.chroma_path.exists()


def test_dashscope_http_response_flows_through_chroma_index_and_query(vector_fixture):
    settings, repository, path = vector_fixture
    settings = settings.model_copy(
        update={
            "dashscope_api_key": SecretStr("offline-key"),
            "embedding_dimensions": 64,
            "embedding_batch_size": 10,
        }
    )
    calls = []

    def handler(request):
        body = json.loads(request.content)
        kind = body["parameters"]["text_type"]
        texts = body["input"]["texts"]
        assert body["parameters"]["dimension"] == 64
        calls.append((kind, len(texts)))
        rows = [
            {
                "text_index": index,
                "embedding": ([1.0, 0.0] if kind == "query" or "第一条" in text else [0.8, 0.6])
                + [0.0] * 62,
            }
            for index, text in enumerate(texts)
        ]
        return httpx.Response(200, json={"output": {"embeddings": list(reversed(rows))}})

    embedder = DashScopeEmbeddings(settings, httpx.MockTransport(handler))
    report = build_vector_index(settings, repository, path, embedder=embedder)
    assert report["embedded"] == 10
    assert calls == [("document", 10)]
    hits = ChromaRetriever(repository, settings, embedder).search("公路挖沟", top_k=2)
    assert [hit.chunk.id for hit in hits] == ["test-A001", "test-A002"]
    assert calls[-1] == ("query", 1)


@pytest.mark.parametrize("change", ["text", "provenance", "duplicate", "incomplete", "offset"])
def test_rejects_corrupt_or_stale_batch_before_writing(vector_fixture, change):
    settings, repository, path = vector_fixture
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "text":
        rows[0]["page_content"] = "伪造的法条"
    elif change == "provenance":
        rows[0]["metadata"]["provenance"]["source"]["verification_status"] = "verified"
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "incomplete":
        rows.pop()
    else:
        rows[0]["metadata"]["source"]["locations"][0]["start_index"] = True
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(AppError) as error:
        build_vector_index(settings, repository, path, embedder=FakeEmbeddings())
    assert error.value.code == "invalid_vector_documents"
    assert not settings.chroma_path.exists()


def test_index_persists_reuses_vectors_and_returns_full_parent_sources(vector_fixture):
    settings, repository, path = vector_fixture
    embedder = FakeEmbeddings()
    report = build_vector_index(settings, repository, path, embedder=embedder)
    count = len(load_vector_documents(path, repository))
    assert report["embedded"] == count and report["reused"] == 0
    before = len(embedder.documents)
    report = build_vector_index(settings, repository, path, embedder=embedder)
    assert report["embedded"] == 0 and report["reused"] == count
    assert len(embedder.documents) == before
    # 新进程读取落盘数据，避免只验证当前进程内共享的 Chroma client。
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import chromadb, sys; from chromadb.config import Settings; "
                "client = chromadb.PersistentClient(path=sys.argv[1], "
                "settings=Settings(anonymized_telemetry=False)); "
                "print(client.get_collection(sys.argv[2], embedding_function=None).count())"
            ),
            str(settings.chroma_path),
            settings.chroma_collection,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(result.stdout.strip()) == count
    retriever = ChromaRetriever(repository, settings, embedder)
    hits = retriever.search("公路挖沟有哪些依据？", top_k=2)
    assert [hit.chunk.id for hit in hits] == ["test-A001", "test-A002"]
    assert hits[0].chunk == repository.get_article("test-A001")
    assert hits[0].score == pytest.approx(1)
    stored = open_vector_index(settings, repository).get(include=["metadatas", "documents"])
    assert all("provenance_json" in meta for meta in stored["metadatas"])
    assert all(
        text in repository.get_article(meta["parent_id"]).text
        for text, meta in zip(stored["documents"], stored["metadatas"], strict=True)
    )
    assert retriever.search("测试公路法第999条") == []
    assert retriever.search("test-A001")[0].chunk.id == "test-A001"
    assert len(embedder.queries) == 1


def test_interrupted_index_is_not_served_and_resumes_only_missing_batches(vector_fixture):
    settings, repository, path = vector_fixture
    with pytest.raises(AppError, match="模拟上游中断"):
        build_vector_index(settings, repository, path, embedder=FakeEmbeddings(fail_call=2))
    with pytest.raises(AppError) as error:
        ChromaRetriever(repository, settings, FakeEmbeddings())
    assert error.value.code == "vector_index_incomplete"
    fresh = FakeEmbeddings()
    report = build_vector_index(settings, repository, path, embedder=fresh)
    assert report["reused"] == settings.embedding_batch_size
    assert sum(map(len, fresh.documents)) == report["chunks"] - settings.embedding_batch_size
    assert open_vector_index(settings, repository).count() == report["chunks"]


@pytest.mark.parametrize(
    "update",
    [
        {"embedding_model": "another-model"},
        {"embedding_dimensions": 2},
        {"embedding_query_instruction": "another instruction"},
    ],
)
def test_collection_cannot_mix_models_or_dimensions(vector_fixture, update):
    settings, repository, path = vector_fixture
    build_vector_index(settings, repository, path, embedder=FakeEmbeddings())
    changed = settings.model_copy(update=update)
    with pytest.raises(AppError) as error:
        build_vector_index(changed, repository, path, embedder=FakeEmbeddings())
    assert error.value.code == "vector_index_mismatch"
    with pytest.raises(AppError):
        ChromaRetriever(repository, changed, FakeEmbeddings())


def test_changed_corpus_and_changed_chunk_ids_require_new_collection(vector_fixture):
    settings, repository, path = vector_fixture
    build_vector_index(settings, repository, path, embedder=FakeEmbeddings())
    changed = LawRepository(
        [
            parent.model_copy(update={"text": parent.text + "新增文字"})
            for parent in repository.searchable
        ]
    )
    with pytest.raises(AppError) as error:
        ChromaRetriever(changed, settings, FakeEmbeddings())
    assert error.value.code == "vector_index_mismatch"
    path.write_text(path.read_text().replace("child-1-0", "new-child-1-0"))
    with pytest.raises(AppError) as error:
        build_vector_index(settings, repository, path, embedder=FakeEmbeddings())
    assert error.value.code == "vector_index_mismatch"


def test_api_uses_vectors_and_keeps_citation_validation(vector_fixture):
    settings, repository, path = vector_fixture
    settings = settings.model_copy(
        update={
            "deepseek_api_key": SecretStr("offline-test-key"),
            "dashscope_api_key": SecretStr("offline-test-key"),
        }
    )
    embedder = FakeEmbeddings()
    build_vector_index(settings, repository, path, embedder=embedder)
    retriever = HybridRetriever(repository, settings, embedder)

    class FakeLLM:
        citation = "test-A001"

        async def complete(self, messages, *, json_mode):
            context = messages[1]["content"].split("Context:\n", 1)[1]
            evidence, _ = json.JSONDecoder().raw_decode(context)
            assert {row["id"] for row in evidence} == {"test-A001", "test-A002"}
            return Completion(
                content=json.dumps(
                    {
                        "claims": [{"text": "测试解释", "citation_ids": [self.citation]}],
                        "follow_up_questions": [],
                    }
                ),
                model="offline-fixture",
            )

    llm = FakeLLM()
    with TestClient(create_app(settings, llm, retriever=retriever)) as c:
        assert c.get("/health").json()["retrieval_backend"] == "hybrid"
        assert c.get("/health").json()["online_only"] is True
        request = {"question": "公路挖沟有哪些依据？", "top_k": 2}
        hits = retriever.search(request["question"], request["top_k"])
        assert hits[0].chunk.id == "test-A001"
        assert hits[0].retrieval.method == "rrf"
        assert hits[0].retrieval.dense_rank == 1
        response = c.post("/api/v1/ask", json=request)
        assert response.status_code == 200, response.text
        answer = response.json()
        assert [row["id"] for row in answer["source_documents"]] == [hit.chunk.id for hit in hits]
        assert answer["sources"][0] == repository.get_article("test-A001").model_dump(mode="json")
        assert answer["applicability_status"] == "not_verified"
        llm.citation = "invented-source"
        invalid = c.post("/api/v1/ask", json=request)
        assert invalid.json()["error"]["code"] == "invalid_citation"


def test_hybrid_reuses_index_and_deduplicates_long_parent_before_rrf(vector_fixture):
    settings, repository, path = vector_fixture
    embedder = FakeEmbeddings()
    build_vector_index(settings, repository, path, embedder=embedder)
    embedded_batches = len(embedder.documents)
    for _ in range(2):
        retriever = HybridRetriever(repository, settings, embedder)
        hits = retriever.search("公路挖沟", top_k=2)
        assert {hit.chunk.id for hit in hits} == {"test-A001", "test-A002"}
        assert {hit.retrieval.dense_rank for hit in hits} == {1, 2}
        assert all(hit.chunk == repository.get_article(hit.chunk.id) for hit in hits)
    assert len(embedder.documents) == embedded_batches  # 重启只读已有向量。
    assert embedder.queries == ["公路挖沟", "公路挖沟"]


def test_cli_preflight_works_without_key_or_chroma_runtime(vector_fixture, monkeypatch, capsys):
    settings, _, path = vector_fixture
    monkeypatch.setattr(
        "transport_law.cli.Settings",
        lambda: settings,
    )
    assert run(parser().parse_args(["index-vectors", "--documents", str(path), "--dry-run"])) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert not settings.chroma_path.exists()
    assert run(parser().parse_args(["validate-data"])) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_hybrid_keeps_dense_matches_when_bm25_has_no_keyword_match(vector_fixture):
    settings, repository, path = vector_fixture
    embedder = FakeEmbeddings()
    build_vector_index(settings, repository, path, embedder=embedder)
    embedded_batches = len(embedder.documents)
    hits = HybridRetriever(repository, settings, embedder).search("zxqjv987654", top_k=2)
    assert [hit.chunk.id for hit in hits] == ["test-A001", "test-A002"]
    assert all(hit.retrieval.bm25_rank is None for hit in hits)
    assert [hit.retrieval.dense_rank for hit in hits] == [1, 2]
    assert embedder.queries == ["zxqjv987654"]
    assert len(embedder.documents) == embedded_batches
