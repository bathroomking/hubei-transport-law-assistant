import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier, Event

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import pdf_bytes
from pydantic import SecretStr

from transport_law.errors import AppError
from transport_law.hybrid import HybridRetriever
from transport_law.llm import Completion
from transport_law.main import create_app

pytest.importorskip("chromadb")
pytest.importorskip("docling")

BASE = "/api/v1/knowledge"


def uploaded_retriever(client):
    store = client.app.state.runtime.knowledge
    return HybridRetriever(store, store.settings, store.embedder)


def uploaded_hits(client, question, top_k=5):
    return [
        hit.model_dump(mode="json") for hit in uploaded_retriever(client).search(question, top_k)
    ]


class CountingEmbeddings:
    def __init__(self):
        self.calls = []
        self.successful_texts = []
        self.queries = []
        self.fail_on_call = None

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on_call:
            raise AppError("embedding_unavailable", "模拟向量服务暂时不可用。", 503)
        self.successful_texts.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, question):
        self.queries.append(question)
        return [1.0, 0.0]


@pytest.fixture
def upload_setup(settings, tmp_path, monkeypatch):
    from transport_law import knowledge

    configured = settings.model_copy(
        update={
            "knowledge_path": tmp_path / "knowledge",
            "chroma_path": tmp_path / "chroma",
            "knowledge_collection": "test_uploaded_documents",
            "embedding_dimensions": 2,
            "embedding_batch_size": 2,
            "dashscope_api_key": SecretStr("offline-test-key"),
            "deepseek_api_key": SecretStr("offline-test-key"),
        }
    )
    embedder = CountingEmbeddings()
    monkeypatch.setattr(knowledge, "DashScopeEmbeddings", lambda settings: embedder)
    return configured, embedder


@pytest.mark.parametrize("extension", ["pdf", "PDF"])
def test_pdf_upload_becomes_searchable_with_real_source(upload_setup, extension):
    settings, embedder = upload_setup
    filename = f"知识文档.{extension}"
    content = pdf_bytes("第一条 施工前应办理许可。")
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": (filename, content)})
        assert response.status_code == 202, response.text
        document_id = response.json()["id"]
        record = client.get(f"{BASE}/documents/{document_id}").json()
        assert record["status"] == "ready", record
        assert record["chunk_count"] == record["embedded_count"] > 0
        assert client.get(f"{BASE}/documents").json() == [record]
        hits = uploaded_hits(client, "施工许可")
        assert hits
        chunk = hits[0]["chunk"]
        assert chunk["kind"] == "document"
        assert chunk["source"]["file"] == filename
        assert chunk["source"]["sha256"] == hashlib.sha256(content).hexdigest()
        assert chunk["source"]["verification_status"] == "pending"
        assert chunk["source"]["official_url"] is None
        assert chunk["source"]["locator"]
        assert chunk["source"]["pdf_page"] == 1
        assert chunk["source"]["source_type"] == "pdf"
        assert client.get(f"/api/v1/articles/{chunk['id']}").json() == chunk
        from transport_law.vector_store import chroma_client

        assert [row.name for row in chroma_client(settings).list_collections()] == [
            settings.knowledge_collection
        ]
    assert len(embedder.queries) == 1


def test_duplicate_name_is_rejected_without_reprocessing_even_after_restart(upload_setup):
    settings, embedder = upload_setup
    for run in range(2):
        with TestClient(create_app(settings)) as client:
            if run == 0:
                response = client.post(
                    f"{BASE}/documents",
                    files={"file": ("Road.pdf", pdf_bytes("Road permit required."))},
                )
                assert response.status_code == 202
            before = len(embedder.calls)
            # 同名判断忽略大小写与 Unicode 兼容字符差异，不以正文变化绕过。
            response = client.post(
                f"{BASE}/documents", files={"file": ("ＲＯＡＤ.PDF", pdf_bytes("Changed body."))}
            )
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "duplicate_filename"
            assert len(embedder.calls) == before == 1
            assert len(client.get(f"{BASE}/documents").json()) == 1


def test_new_document_only_embeds_new_chunks_and_is_visible_without_restart(upload_setup):
    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["retrieval_backend"] == "hybrid"
        assert uploaded_hits(client, "document") == []
        for name, text in [("one.pdf", "First document."), ("two.pdf", "Second document.")]:
            response = client.post(f"{BASE}/documents", files={"file": (name, pdf_bytes(text))})
            assert response.status_code == 202
            # 查询后再追加，BM25 不得一直使用第一次请求的过期快照。
            hits = uploaded_hits(client, "document")
            assert any(hit["chunk"]["source"]["file"] == name for hit in hits)
            assert all(hit["retrieval"]["bm25_rank"] for hit in hits)
            assert all(hit["retrieval"]["dense_rank"] for hit in hits)
        assert len(embedder.calls) == 2
        assert len(embedder.successful_texts) == 2
        assert {hit["chunk"]["source"]["file"] for hit in uploaded_hits(client, "document")} == {
            "one.pdf",
            "two.pdf",
        }
        assert len(embedder.queries) == 3


def test_names_have_no_implicit_reserved_corpus_files(upload_setup):
    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents",
            files={"file": ("legal_chunks.pdf", pdf_bytes("New uploaded document."))},
        )
        assert response.status_code == 202
        record = client.get(f"{BASE}/documents/{response.json()['id']}").json()
        assert record["status"] == "ready"
    assert len(embedder.calls) == 1


def test_concurrent_same_name_has_one_durable_winner(upload_setup):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    barrier = Barrier(4)

    def receive():
        store = KnowledgeStore(settings)
        barrier.wait(timeout=10)
        try:
            return store.receive("same.pdf", BytesIO(pdf_bytes("Same document.")))["id"]
        except AppError as exc:
            assert exc.code == "duplicate_filename"
            assert exc.status_code == 409
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: receive(), range(4)))
    assert sum(ident is not None for ident in results) == 1
    assert len(KnowledgeStore(settings).list_documents()) == 1
    assert not embedder.calls


@pytest.mark.parametrize("filename", ["../bad.pdf", "folder/bad.pdf", "C:\\bad.pdf", "bad\n.pdf"])
def test_unsafe_filename_never_creates_document(upload_setup, filename):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings)
    with pytest.raises(AppError) as error:
        store.receive(filename, BytesIO(b"Body"))
    assert error.value.status_code == 400
    assert store.list_documents() == []
    assert not embedder.calls


@pytest.mark.parametrize("extension", ["txt", "md", "markdown", "docx"])
def test_non_pdf_upload_is_rejected_before_queueing(upload_setup, extension):
    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents", files={"file": (f"document.{extension}", pdf_bytes())}
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_document"
        assert client.get(f"{BASE}/documents").json() == []
    assert not embedder.calls


def test_empty_upload_is_rejected_before_queueing(upload_setup):
    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": ("empty.pdf", b"")})
        assert response.status_code == 400
        assert client.get(f"{BASE}/documents").json() == []
    assert not embedder.calls


def test_size_limit_does_not_reserve_filename(upload_setup):
    settings, embedder = upload_setup
    content = pdf_bytes("Valid small PDF.")
    settings = settings.model_copy(update={"upload_max_bytes": len(content)})
    with TestClient(create_app(settings)) as client:
        # The frontend must display and validate the same configured limit as receive().
        assert client.get("/health").json()["upload_max_bytes"] == len(content)
        too_large = client.post(f"{BASE}/documents", files={"file": ("small.pdf", content + b"a")})
        assert too_large.status_code == 413
        assert client.get(f"{BASE}/documents").json() == []
        valid = client.post(f"{BASE}/documents", files={"file": ("small.pdf", content)})
        assert valid.status_code == 202
        record = client.get(f"{BASE}/documents/{valid.json()['id']}").json()
        assert record["status"] == "ready"
    assert len(embedder.calls) == 1


@pytest.mark.parametrize(
    "filename,content,error_code",
    [
        ("corrupt.pdf", b"not a pdf", "pdf_read_failed"),
        ("blank.pdf", "blank", "no_extractable_text"),
        ("whitespace.pdf", "whitespace", "no_extractable_text"),
        ("encrypted.pdf", "encrypted", "pdf_encrypted"),
    ],
)
def test_bad_document_fails_explicitly_without_publishing(
    upload_setup, filename, content, error_code
):
    settings, embedder = upload_setup
    if content == "blank":
        content = pdf_bytes("")
    elif content == "whitespace":
        content = pdf_bytes(" \n ")
    elif content == "encrypted":
        content = pdf_bytes(password="fixture-password")
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": (filename, content)})
        assert response.status_code == 202
        record = client.get(f"{BASE}/documents/{response.json()['id']}").json()
        assert record["status"] == "failed"
        assert record["error_code"] == error_code
        assert record["error_message"]
        assert uploaded_hits(client, "任何内容") == []
    assert not embedder.calls
    assert not embedder.queries


def test_intentional_blank_page_does_not_block_other_page_text(upload_setup):
    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents",
            files={"file": ("blank-separator.pdf", pdf_bytes("", "Road permit required.", ""))},
        )
        record = client.get(f"{BASE}/documents/{response.json()['id']}").json()
        assert record["status"] == "ready", record
        hits = uploaded_hits(client, "Road permit")
        assert hits and all(hit["chunk"]["source"]["pdf_page"] == 2 for hit in hits)
    assert embedder.calls


def test_nonblank_page_without_ocr_text_cannot_be_silently_skipped(upload_setup, monkeypatch):
    from transport_law import ingest

    settings, embedder = upload_setup
    data = pdf_bytes("Digital page.", "")

    def incompletely_parsed(path):
        return ingest.LoadedDocument(
            path.name,
            hashlib.sha256(data).hexdigest(),
            "pdf",
            [
                ingest.DocumentPage("Digital page.", 1),
                ingest.DocumentPage("", 2, is_blank=False),
            ],
            parser="test-unread-image",
        )

    monkeypatch.setattr(ingest, "load_document", incompletely_parsed)
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": ("missing-ocr.pdf", data)})
        record = client.get(f"{BASE}/documents/{response.json()['id']}").json()
        assert record["status"] == "failed" and record["error_code"] == "pdf_needs_ocr"
        assert uploaded_hits(client, "Digital page") == []
    assert not embedder.calls


def test_retry_reuses_chunks_and_successful_batches(upload_setup, monkeypatch):
    from transport_law import chunking, knowledge

    settings, embedder = upload_setup
    embedder.fail_on_call = 2
    pages = [f"Road permit clause {number}." for number in range(7)]
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents", files={"file": ("retry.pdf", pdf_bytes(*pages))}
        )
        document_id = response.json()["id"]
        before = client.get(f"{BASE}/documents/{document_id}").json()
        assert before["status"] == "failed"
        assert before["embedded_count"] == 2
        assert before["chunk_count"] > 2
        assert uploaded_hits(client, "Road permit") == []

        def unexpected_chunking(*args, **kwargs):
            pytest.fail("重试应读取已保存的分块，不能再次切分")

        monkeypatch.setattr(chunking, "build_draft_chunks", unexpected_chunking)
        if hasattr(knowledge, "build_draft_chunks"):
            monkeypatch.setattr(knowledge, "build_draft_chunks", unexpected_chunking)
        embedder.fail_on_call = None
        retry = client.post(f"{BASE}/documents/{document_id}/retry")
        assert retry.status_code == 202, retry.text
        after = client.get(f"{BASE}/documents/{document_id}").json()
        assert after["status"] == "ready", after
        assert after["embedded_count"] == after["chunk_count"] == before["chunk_count"]
        assert len(embedder.successful_texts) == after["chunk_count"]
        assert len(set(embedder.successful_texts)) == after["chunk_count"]
        assert client.post(f"{BASE}/documents/{document_id}/retry").status_code == 409


def test_restart_reads_ready_documents_without_chunking_or_embedding(upload_setup, monkeypatch):
    from transport_law import chunking

    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents", files={"file": ("persist.pdf", pdf_bytes("Persistent text."))}
        )
        document_id = response.json()["id"]
        expected = client.get(f"{BASE}/documents/{document_id}").json()

    def unexpected_work(*args, **kwargs):
        pytest.fail("新实例读取已有文档不应触发切分或 embedding")

    monkeypatch.setattr(chunking, "build_draft_chunks", unexpected_work)
    monkeypatch.setattr(embedder, "embed_documents", unexpected_work)
    monkeypatch.setattr(embedder, "embed_query", unexpected_work)
    with TestClient(create_app(settings)) as client:
        assert client.get(f"{BASE}/documents/{document_id}").json() == expected
        assert client.get(f"{BASE}/documents").json() == [expected]
        assert client.get("/health").status_code == 200


def test_partial_failed_vectors_do_not_hide_ready_document(upload_setup):
    settings, embedder = upload_setup
    embedder.fail_on_call = 2
    pages = [f"Unfinished permit {number}." for number in range(7)]
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents", files={"file": ("failed.pdf", pdf_bytes(*pages))}
        )
        document_id = response.json()["id"]
        assert client.get(f"{BASE}/documents/{document_id}").json()["status"] == "failed"
        embedder.fail_on_call = None
        assert (
            client.post(
                f"{BASE}/documents",
                files={"file": ("ready.pdf", pdf_bytes("Completed permit document."))},
            ).status_code
            == 202
        )
        hits = uploaded_hits(client, "permit", top_k=1)
        assert len(hits) == 1
        assert hits[0]["chunk"]["source"]["file"] == "ready.pdf"


def test_same_job_cannot_run_or_retry_concurrently(upload_setup, monkeypatch):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings)
    document_id = store.receive("running.pdf", BytesIO(pdf_bytes("One running task.")))["id"]
    started, release = Event(), Event()
    original = embedder.embed_documents

    def wait_for_release(texts):
        started.set()
        assert release.wait(timeout=10)
        return original(texts)

    monkeypatch.setattr(embedder, "embed_documents", wait_for_release)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(store.process, document_id)
        try:
            assert started.wait(timeout=10)
            other_worker = KnowledgeStore(settings)
            assert other_worker.get_document(document_id)["status"] == "processing"
            other_worker.process(document_id)
            with pytest.raises(AppError) as error:
                other_worker.retry(document_id)
            assert error.value.code == "upload_busy"
            assert error.value.status_code == 409
            with pytest.raises(AppError) as error:
                other_worker.reprocess(document_id)
            assert error.value.code == "upload_busy"
            assert other_worker.ready_chunks() == []
        finally:
            release.set()
        running.result(timeout=10)
    assert store.get_document(document_id)["status"] == "ready"
    assert len(embedder.calls) == 1


def test_interrupted_task_requires_explicit_retry_and_reuses_saved_chunks(
    upload_setup, monkeypatch
):
    from transport_law import chunking
    from transport_law.knowledge import KnowledgeStore

    class ProcessExited(BaseException):
        pass

    settings, embedder = upload_setup
    store = KnowledgeStore(settings)
    document_id = store.receive(
        "interrupted.pdf", BytesIO(pdf_bytes("Recover after interruption."))
    )["id"]
    original = embedder.embed_documents

    def interrupt_before_vector_saved(texts):
        raise ProcessExited()

    monkeypatch.setattr(embedder, "embed_documents", interrupt_before_vector_saved)
    with pytest.raises(ProcessExited):
        store.process(document_id)

    def unexpected_chunking(*args, **kwargs):
        pytest.fail("已成功保存分块的中断任务应复用分块")

    monkeypatch.setattr(chunking, "build_draft_chunks", unexpected_chunking)
    restarted = KnowledgeStore(settings)
    record = restarted.get_document(document_id)
    assert record["status"] == "failed"
    assert record["error_code"] == "job_interrupted"
    assert not embedder.calls
    assert restarted.ready_chunks() == []
    monkeypatch.setattr(embedder, "embed_documents", original)
    assert restarted.retry(document_id)["status"] == "queued"
    restarted.process(document_id)
    assert restarted.get_document(document_id)["status"] == "ready"
    assert len(embedder.calls) == 1


def test_uploaded_qa_uses_stored_sources_and_rejects_invented_citations(upload_setup):
    settings, _ = upload_setup

    class FakeLLM:
        forged = False

        async def complete(self, messages, *, json_mode):
            context = messages[1]["content"].split("Context:\n", 1)[1]
            evidence, _ = json.JSONDecoder().raw_decode(context)
            assert json_mode and len(evidence) == 1
            citation = "not-retrieved" if self.forged else evidence[0]["id"]
            return Completion(
                content=json.dumps(
                    {
                        "claims": [{"text": "文档测试解释。", "citation_ids": [citation]}],
                        "follow_up_questions": [],
                    }
                ),
                model="offline-test",
            )

    llm = FakeLLM()
    with TestClient(create_app(settings, llm)) as client:
        client.app.state.runtime.service.retriever = uploaded_retriever(client)
        assert client.app.state.runtime.repository.knowledge is client.app.state.runtime.knowledge
        assert (
            client.post(
                f"{BASE}/documents", files={"file": ("source.pdf", pdf_bytes("施工前需办理许可。"))}
            ).status_code
            == 202
        )
        answer = client.post("/api/v1/ask", json={"question": "施工许可"})
        assert answer.status_code == 200, answer.text
        source = answer.json()["sources"][0]
        assert answer.json()["source_documents"] == [source]
        assert source == client.get(f"/api/v1/articles/{source['id']}").json()
        assert answer.json()["applicability_status"] == "not_verified"
        llm.forged = True
        error = client.post("/api/v1/ask", json={"question": "施工许可"})
        assert error.json()["error"]["code"] == "invalid_citation"


def test_unknown_upload_and_chunk_return_404(upload_setup):
    settings, _ = upload_setup
    with TestClient(create_app(settings)) as client:
        assert client.get(f"{BASE}/documents/unknown").status_code == 404
        assert client.post(f"{BASE}/documents/unknown/retry").status_code == 404
        assert client.post(f"{BASE}/documents/unknown/reprocess").status_code == 404
        assert client.get("/api/v1/articles/unknown").status_code == 404


@pytest.mark.parametrize("corruption", ["missing_document_chunks", "shifted_counts"])
def test_ready_snapshot_checks_each_document_without_reprocessing(upload_setup, corruption):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings)
    identifiers = []
    for name in ("one.pdf", "two.pdf"):
        ident = store.receive(name, BytesIO(pdf_bytes("Stored document text.")))["id"]
        store.process(ident)
        assert store.get_document(ident)["status"] == "ready"
        identifiers.append(ident)
    assert len(store.ready_chunks()) == 2
    before = len(embedder.calls)
    with store._connect() as connection:
        if corruption == "missing_document_chunks":
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (identifiers[0],))
        else:
            # 总数相同也不能掩盖某份文档丢失/多出正文。
            connection.execute(
                "UPDATE documents SET chunk_count = 0 WHERE id = ?", (identifiers[0],)
            )
            connection.execute(
                "UPDATE documents SET chunk_count = 2 WHERE id = ?", (identifiers[1],)
            )
    with pytest.raises(AppError) as error:
        store.ready_chunks()
    assert error.value.code == "vector_index_incomplete"
    assert len(embedder.calls) == before and not embedder.queries


@pytest.mark.parametrize(
    "update", [{"embedding_dimensions": 3}, {"embedding_query_instruction": "different"}]
)
def test_uploaded_index_rejects_incompatible_vector_configuration(upload_setup, update):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("contract.pdf", BytesIO(pdf_bytes("Stored configuration.")))["id"]
    store.process(ident)
    other = KnowledgeStore(settings.model_copy(update=update), embedder)
    with pytest.raises(AppError) as exc:
        other.rank_vectors(other.ready_chunks(), limit=5, query_vector=[1.0, 0.0])
    assert exc.value.code == "vector_index_mismatch"
    assert not embedder.queries


def test_retired_chunking_contract_cannot_process_or_retry_cached_upload(upload_setup):
    from transport_law.knowledge import KnowledgeStore
    from transport_law.vector_store import digest

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("old-contract.pdf", BytesIO(pdf_bytes()))["id"]
    old_contract = {**store.contract, "chunking": "legal_recursive_v1:800:120"}
    with store._connect() as connection:
        connection.execute(
            "UPDATE documents SET contract_hash = ? WHERE id = ?", (digest(old_contract), ident)
        )
    store.process(ident)
    record = store.get_document(ident)
    assert record["status"] == "failed" and record["error_code"] == "vector_index_mismatch"
    with pytest.raises(AppError) as exc:
        store.retry(ident)
    assert exc.value.code == "vector_index_mismatch"
    assert store.ready_chunks() == []
    assert not embedder.calls


def test_missing_uploaded_vectors_are_not_silently_ignored(upload_setup):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("lost.pdf", BytesIO(pdf_bytes("Stored document.")))["id"]
    store.process(ident)
    chunks = store.ready_chunks()
    store._collection().delete(ids=[chunks[0].id])
    with pytest.raises(AppError) as exc:
        store.rank_vectors(chunks, limit=5, query_vector=[1.0, 0.0])
    assert exc.value.code == "vector_index_incomplete"


def test_ready_upload_is_readable_in_a_fresh_process_without_model_calls(upload_setup):
    import subprocess
    import sys

    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("persist.pdf", BytesIO(pdf_bytes("Persistent permit text.")))["id"]
    store.process(ident)
    expected = [c.id for c in store.ready_chunks()]
    script = """
import json, sys
from transport_law.config import Settings
from transport_law.knowledge import KnowledgeStore
class NoModels:
    def embed_documents(self, *args): raise AssertionError("document embedding on read")
    def embed_query(self, *args): raise AssertionError("query embedding on supplied vector")
s = Settings(_env_file=None, knowledge_path=sys.argv[1], chroma_path=sys.argv[2],
             knowledge_collection=sys.argv[3]).model_copy(update={"embedding_dimensions": 2})
store = KnowledgeStore(s, NoModels())
hits = store.rank_vectors(store.ready_chunks(), limit=5, query_vector=[1.0, 0.0])
print(json.dumps([h.chunk.id for h in hits]))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(settings.knowledge_path),
            str(settings.chroma_path),
            settings.knowledge_collection,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == expected


def test_reprocess_rebuilds_vectors_and_preserves_source_and_extraction(upload_setup):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    original_pdf = pdf_bytes("Reprocess this source without uploading it again.")
    with TestClient(create_app(settings)) as client:
        ident = client.post(
            f"{BASE}/documents", files={"file": ("reprocess.pdf", original_pdf)}
        ).json()["id"]
        store = client.app.state.runtime.knowledge
        before = store._row(ident)
        folder = settings.knowledge_path / "documents" / ident
        assert (folder / f"chunks-{before['cache_key']}.jsonl").exists()
        first_snapshot = folder / f"extraction-{before['cache_key']}.json"
        assert json.loads(first_snapshot.read_text())["sha256"] == before["sha256"]
        before_calls = len(embedder.calls)
        response = client.post(f"{BASE}/documents/{ident}/reprocess")
        assert response.status_code == 202, response.text
        after = store._row(ident)
        assert after["status"] == "ready"
        assert after["cache_key"] != before["cache_key"]
        assert after["rebuild_pending"] == 0
        assert len(embedder.calls) == before_calls + 1
        assert (folder / "source.pdf").read_bytes() == original_pdf
        assert first_snapshot.exists()  # 旧引用的提取来源仍可审核。
        assert not (folder / f"chunks-{before['cache_key']}.jsonl").exists()
        assert (folder / f"chunks-{after['cache_key']}.jsonl").exists()
        assert len(store.ready_chunks()) == after["chunk_count"]
        assert (
            len(store._collection().get(where={"document_id": ident})["ids"])
            == after["chunk_count"]
        )
        assert uploaded_hits(client, "Reprocess")
        assert KnowledgeStore(settings).get_document(ident)["status"] == "ready"


def test_failed_reparse_preserves_previous_data_until_new_chunks_are_valid(
    upload_setup, monkeypatch
):
    from transport_law import ingest
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("parse-failure.pdf", BytesIO(pdf_bytes("Original source.")))["id"]
    store.process(ident)
    old_ids = store._collection().get(where={"document_id": ident})["ids"]
    with store._connect() as conn:
        old_json = conn.execute(
            "SELECT chunk_json FROM chunks WHERE document_id = ?", (ident,)
        ).fetchall()
    load = ingest.load_document

    def fail_parse(*args, **kwargs):
        raise AppError("pdf_read_failed", "模拟新解析器失败", 400)

    monkeypatch.setattr(ingest, "load_document", fail_parse)
    assert store.reprocess(ident)["status"] == "queued"
    assert store.ready_chunks() == []
    store.process(ident)
    assert store.get_document(ident)["status"] == "failed"
    assert store._row(ident)["rebuild_pending"] == 1
    assert store._collection().get(where={"document_id": ident})["ids"] == old_ids
    with store._connect() as conn:
        assert (
            conn.execute("SELECT chunk_json FROM chunks WHERE document_id = ?", (ident,)).fetchall()
            == old_json
        )
    monkeypatch.setattr(ingest, "load_document", load)
    store.retry(ident)
    store.process(ident)
    assert store.get_document(ident)["status"] == "ready"


def test_reprocess_failure_resumes_new_cached_chunks_without_reparsing(upload_setup, monkeypatch):
    from transport_law import chunking
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("resume-reparse.pdf", BytesIO(pdf_bytes("Retain new cached chunks.")))[
        "id"
    ]
    store.process(ident)
    store.reprocess(ident)
    embedder.fail_on_call = len(embedder.calls) + 1
    store.process(ident)
    assert store.get_document(ident)["status"] == "failed"
    assert store._row(ident)["rebuild_pending"] == 0

    def unexpected(*args, **kwargs):
        pytest.fail("重新解析后的向量失败应复用新版本缓存")

    monkeypatch.setattr(chunking, "build_draft_chunks", unexpected)
    embedder.fail_on_call = None
    store.retry(ident)
    store.process(ident)
    assert store.get_document(ident)["status"] == "ready"


def test_old_ready_vectors_remain_readable_and_explicit_reprocess_upgrades(upload_setup):
    from transport_law.knowledge import KnowledgeStore
    from transport_law.vector_store import digest

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("legacy-ready.pdf", BytesIO(pdf_bytes("Legacy stored text.")))["id"]
    store.process(ident)
    legacy_pipeline = "pdf_legal_recursive_v1:800:120"
    legacy_contract = {**store.contract, "chunking": legacy_pipeline}
    store._collection().modify(metadata=legacy_contract)
    with store._connect() as conn:
        conn.execute(
            "UPDATE documents SET contract_hash = ? WHERE id = ?",
            (digest(legacy_contract), ident),
        )
        # 模拟真正的旧数据库，而非仅将新 schema 的版本字段改旧。
        for column in ("pipeline_version", "cache_key", "rebuild_pending"):
            conn.execute(f"ALTER TABLE documents DROP COLUMN {column}")
    before_calls = len(embedder.calls)
    assert store.rank_vectors(store.ready_chunks(), limit=5, query_vector=[1.0, 0.0])
    assert store.get_document(ident)["pipeline_version"] == legacy_pipeline
    assert len(embedder.calls) == before_calls
    store.reprocess(ident)
    store.process(ident)
    assert store.get_document(ident)["status"] == "ready"
    assert store.get_document(ident)["pipeline_version"] != legacy_pipeline
    assert len(embedder.calls) == before_calls + 1


def test_old_pipeline_retry_requires_explicit_reprocess(upload_setup):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("legacy-pending.pdf", BytesIO(pdf_bytes("Unprocessed legacy source.")))[
        "id"
    ]
    store._update(ident, pipeline_version="retired-parser", status="failed")
    with pytest.raises(AppError) as error:
        store.retry(ident)
    assert error.value.code == "processing_version_mismatch"
    assert not embedder.calls
    store.reprocess(ident)
    store.process(ident)
    assert store.get_document(ident)["status"] == "ready"


@pytest.mark.parametrize("missing", [False, True])
def test_reprocess_refuses_changed_source_without_touching_ready_data(upload_setup, missing):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("source-integrity.pdf", BytesIO(pdf_bytes("Untampered source.")))["id"]
    store.process(ident)
    source = settings.knowledge_path / "documents" / ident / "source.pdf"
    if missing:
        source.unlink()
    else:
        source.write_bytes(pdf_bytes("Changed original."))
    with pytest.raises(AppError) as error:
        store.reprocess(ident)
    assert error.value.code == ("upload_source_missing" if missing else "upload_source_changed")
    assert store.get_document(ident)["status"] == "ready"
    assert len(embedder.calls) == 1


def test_interrupted_rebuild_after_vector_removal_can_resume(upload_setup, monkeypatch):
    from transport_law import chunking
    from transport_law.knowledge import KnowledgeStore

    class ProcessExited(BaseException):
        pass

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("reset-interrupted.pdf", BytesIO(pdf_bytes("Reset interruption.")))["id"]
    store.process(ident)
    store.reprocess(ident)
    save = store._save_chunks

    def interrupt(*args, **kwargs):
        raise ProcessExited()

    monkeypatch.setattr(store, "_save_chunks", interrupt)
    with pytest.raises(ProcessExited):
        store.process(ident)
    assert store._row(ident)["rebuild_pending"] == 1
    assert store._collection().get(where={"document_id": ident})["ids"] == []

    def unexpected(*args, **kwargs):
        pytest.fail("准备好的新分块应在中断后复用")

    monkeypatch.setattr(chunking, "build_draft_chunks", unexpected)
    monkeypatch.setattr(store, "_save_chunks", save)
    assert store.get_document(ident)["error_code"] == "job_interrupted"
    store.retry(ident)
    store.process(ident)
    assert store.get_document(ident)["status"] == "ready"
    assert store._row(ident)["rebuild_pending"] == 0
    assert store.ready_chunks()


@pytest.mark.parametrize("corruption", ["missing_snapshot", "altered_chunk", "invalid_range"])
def test_retry_rejects_untraceable_cache_before_embedding(upload_setup, corruption):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    embedder.fail_on_call = 1
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("cache-integrity.pdf", BytesIO(pdf_bytes("Traceable source.")))["id"]
    store.process(ident)
    assert store.get_document(ident)["status"] == "failed"
    row = store._row(ident)
    folder = settings.knowledge_path / "documents" / ident
    if corruption == "missing_snapshot":
        (folder / f"extraction-{row['cache_key']}.json").unlink()
    else:
        cache = folder / f"chunks-{row['cache_key']}.jsonl"
        chunks = [json.loads(line) for line in cache.read_text().splitlines()]
        if corruption == "altered_chunk":
            chunks[0]["text"] += "Not from source."
        else:
            chunks[0]["source"]["locations"][0]["end_index"] = 999999
        cache.write_text("\n".join(json.dumps(chunk) for chunk in chunks) + "\n")
    embedder.fail_on_call = None
    store.retry(ident)
    store.process(ident)
    assert store.get_document(ident)["error_code"] == "upload_cache_mismatch"
    assert len(embedder.calls) == 1
    assert (folder / "source.pdf").exists()


def test_uploaded_text_preserves_exact_snapshot_slices_through_storage_and_api(
    upload_setup, monkeypatch
):
    from pydantic import ValidationError

    from transport_law import ingest
    from transport_law.schemas import LegalChunk

    settings, _ = upload_setup
    content = pdf_bytes("The parser output below exercises exact provenance preservation.")
    canonical = (
        "  Permit evidence retains leading indentation and trailing spaces.  \n\n"
        "| Item | Description | Amount |\n"
        "| --- | --- | --- |\n"
        + "".join(
            f"| permit-{number} | " + "retain source whitespace " * 3 + f" | {number}00 |\n"
            for number in range(1, 11)
        )
    )

    def parsed_with_whitespace(path):
        return ingest.LoadedDocument(
            path.name,
            hashlib.sha256(content).hexdigest(),
            "pdf",
            [ingest.DocumentPage(canonical, 1)],
            parser="test-layout-snapshot",
        )

    monkeypatch.setattr(ingest, "load_document", parsed_with_whitespace)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents", files={"file": ("whitespace-evidence.pdf", content)}
        )
        assert response.status_code == 202, response.text
        ident = response.json()["id"]
        store = client.app.state.runtime.knowledge
        row = store._row(ident)
        assert row["status"] == "ready", row
        folder = settings.knowledge_path / "documents" / ident
        snapshot = json.loads((folder / f"extraction-{row['cache_key']}.json").read_text())
        assert snapshot["pages"][0]["text"] == canonical
        drafts = [
            json.loads(line)
            for line in (folder / f"chunks-{row['cache_key']}.jsonl").read_text().splitlines()
        ]
        assert any(draft["text"].startswith("  Permit") for draft in drafts)
        assert any(draft["content_type"] == "table" for draft in drafts)
        assert all(draft["text"].endswith("\n") for draft in drafts)
        assert 1 < len(drafts) <= 10
        with store._connect() as conn:
            persisted = {
                record["id"]: json.loads(record["chunk_json"])
                for record in conn.execute(
                    "SELECT id, chunk_json FROM chunks WHERE document_id = ?", (ident,)
                )
            }
        retrieved = {
            hit["chunk"]["id"]: hit["chunk"] for hit in uploaded_hits(client, "permit", 10)
        }
        document = client.get(f"/api/v1/documents/{ident}").json()
        document_chunks = {chunk["id"]: chunk for chunk in document["articles"]}
        assert set(persisted) == set(retrieved) == set(document_chunks)
        for draft in drafts:
            chunk_id = "UP-" + draft["id"].removeprefix("AUTO-")
            expected = "".join(
                snapshot["pages"][loc["pdf_page"] - 1]["text"][
                    loc["start_index"] : loc["end_index"]
                ]
                for loc in draft["source"]["locations"]
            )
            article = client.get(f"/api/v1/articles/{chunk_id}")
            assert article.status_code == 200
            assert (
                expected
                == draft["text"]
                == persisted[chunk_id]["text"]
                == retrieved[chunk_id]["text"]
                == document_chunks[chunk_id]["text"]
                == article.json()["text"]
            )
        with pytest.raises(ValidationError):
            LegalChunk.model_validate({**next(iter(persisted.values())), "text": " \t\n"})
