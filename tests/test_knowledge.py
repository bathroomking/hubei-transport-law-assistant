import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier, Event

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from transport_law.errors import AppError
from transport_law.hybrid import HybridRetriever
from transport_law.llm import Completion
from transport_law.main import create_app

pytest.importorskip("chromadb")
pytest.importorskip("langchain_text_splitters")

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


def pdf_bytes(*, text=True, blank_second_page=False):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 72 720 Td (Road construction requires permission.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
    if blank_second_page:
        writer.add_blank_page(width=612, height=792)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.mark.parametrize("extension", ["txt", "md", "markdown", "pdf"])
def test_upload_all_formats_becomes_searchable_with_real_source(upload_setup, extension):
    settings, embedder = upload_setup
    filename = f"知识文档.{extension}"
    content = pdf_bytes() if extension == "pdf" else "# 道路施工\n施工前应办理许可。".encode()
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
        assert chunk["source"]["pdf_page"] == (1 if extension == "pdf" else None)
        assert chunk["source"]["source_type"] == ("md" if extension == "markdown" else extension)
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
                    f"{BASE}/documents", files={"file": ("Road.txt", b"Road permit required.")}
                )
                assert response.status_code == 202
            before = len(embedder.calls)
            # 同名判断忽略大小写与 Unicode 兼容字符差异，不以正文变化绕过。
            response = client.post(
                f"{BASE}/documents", files={"file": ("ＲＯＡＤ.TXT", b"Changed body.")}
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
        for name, text in [("one.txt", b"First document."), ("two.md", b"Second document.")]:
            response = client.post(f"{BASE}/documents", files={"file": (name, text)})
            assert response.status_code == 202
            # 查询后再追加，BM25 不得一直使用第一次请求的过期快照。
            hits = uploaded_hits(client, "document")
            assert any(hit["chunk"]["source"]["file"] == name for hit in hits)
            assert all(hit["retrieval"]["bm25_rank"] for hit in hits)
            assert all(hit["retrieval"]["dense_rank"] for hit in hits)
        assert len(embedder.calls) == 2
        assert len(embedder.successful_texts) == 2
        assert {hit["chunk"]["source"]["file"] for hit in uploaded_hits(client, "document")} == {
            "one.txt",
            "two.md",
        }
        assert len(embedder.queries) == 3


def test_names_have_no_implicit_reserved_corpus_files(upload_setup):
    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(
            f"{BASE}/documents", files={"file": ("legal_chunks.txt", b"New uploaded document.")}
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
            return store.receive("same.txt", BytesIO(b"Same document."))["id"]
        except AppError as exc:
            assert exc.code == "duplicate_filename"
            assert exc.status_code == 409
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: receive(), range(4)))
    assert sum(ident is not None for ident in results) == 1
    assert len(KnowledgeStore(settings).list_documents()) == 1
    assert not embedder.calls


@pytest.mark.parametrize("filename", ["../bad.txt", "folder/bad.md", "C:\\bad.txt", "bad\n.txt"])
def test_unsafe_filename_never_creates_document(upload_setup, filename):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings)
    with pytest.raises(AppError) as error:
        store.receive(filename, BytesIO(b"Body"))
    assert error.value.status_code == 400
    assert store.list_documents() == []
    assert not embedder.calls


@pytest.mark.parametrize("filename,content", [("bad.docx", b"text"), ("empty.txt", b"")])
def test_invalid_upload_is_rejected_before_queueing(upload_setup, filename, content):
    settings, embedder = upload_setup
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": (filename, content)})
        assert response.status_code in (400, 415)
        assert client.get(f"{BASE}/documents").json() == []
    assert not embedder.calls


def test_size_limit_does_not_reserve_filename(upload_setup):
    settings, embedder = upload_setup
    settings = settings.model_copy(update={"upload_max_bytes": 8})
    with TestClient(create_app(settings)) as client:
        # The frontend must display and validate the same configured limit as receive().
        assert client.get("/health").json()["upload_max_bytes"] == 8
        too_large = client.post(f"{BASE}/documents", files={"file": ("small.txt", b"a" * 9)})
        assert too_large.status_code == 413
        assert client.get(f"{BASE}/documents").json() == []
        valid = client.post(f"{BASE}/documents", files={"file": ("small.txt", b"hello")})
        assert valid.status_code == 202
    assert len(embedder.calls) == 1


@pytest.mark.parametrize(
    "filename,content",
    [
        ("encoding.txt", b"\xff\xfe\x00"),
        ("whitespace.md", b" \n\t"),
        ("corrupt.pdf", b"not a pdf"),
        ("scan.pdf", None),
        ("partial-scan.pdf", "partial"),
    ],
)
def test_bad_document_fails_explicitly_without_publishing(upload_setup, filename, content):
    settings, embedder = upload_setup
    if content is None:
        content = pdf_bytes(text=False)
    elif content == "partial":
        content = pdf_bytes(blank_second_page=True)
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": (filename, content)})
        assert response.status_code == 202
        record = client.get(f"{BASE}/documents/{response.json()['id']}").json()
        assert record["status"] == "failed"
        assert record["error_code"]
        assert record["error_message"]
        assert uploaded_hits(client, "任何内容") == []
    assert not embedder.calls
    assert not embedder.queries


def test_retry_reuses_chunks_and_successful_batches(upload_setup, monkeypatch):
    from transport_law import chunking, knowledge

    settings, embedder = upload_setup
    embedder.fail_on_call = 2
    text = "\n".join(f"# Section {number}\nRoad permit clause {number}." for number in range(7))
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": ("retry.md", text.encode())})
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
            f"{BASE}/documents", files={"file": ("persist.txt", b"Persistent text.")}
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
    text = "\n".join(f"# Section {number}\nUnfinished permit {number}." for number in range(7))
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{BASE}/documents", files={"file": ("failed.md", text.encode())})
        document_id = response.json()["id"]
        assert client.get(f"{BASE}/documents/{document_id}").json()["status"] == "failed"
        embedder.fail_on_call = None
        assert (
            client.post(
                f"{BASE}/documents", files={"file": ("ready.txt", b"Completed permit document.")}
            ).status_code
            == 202
        )
        hits = uploaded_hits(client, "permit", top_k=1)
        assert len(hits) == 1
        assert hits[0]["chunk"]["source"]["file"] == "ready.txt"


def test_same_job_cannot_run_or_retry_concurrently(upload_setup, monkeypatch):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings)
    document_id = store.receive("running.txt", BytesIO(b"One running task."))["id"]
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
    document_id = store.receive("interrupted.txt", BytesIO(b"Recover after interruption."))["id"]
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
                f"{BASE}/documents", files={"file": ("source.txt", "施工前需办理许可。".encode())}
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
        assert client.get("/api/v1/articles/unknown").status_code == 404


@pytest.mark.parametrize("corruption", ["missing_document_chunks", "shifted_counts"])
def test_ready_snapshot_checks_each_document_without_reprocessing(upload_setup, corruption):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings)
    identifiers = []
    for name in ("one.txt", "two.txt"):
        ident = store.receive(name, BytesIO(b"Stored document text."))["id"]
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
    ident = store.receive("contract.txt", BytesIO(b"Stored configuration."))["id"]
    store.process(ident)
    other = KnowledgeStore(settings.model_copy(update=update), embedder)
    with pytest.raises(AppError) as exc:
        other.rank_vectors(other.ready_chunks(), limit=5, query_vector=[1.0, 0.0])
    assert exc.value.code == "vector_index_mismatch"
    assert not embedder.queries


def test_missing_uploaded_vectors_are_not_silently_ignored(upload_setup):
    from transport_law.knowledge import KnowledgeStore

    settings, embedder = upload_setup
    store = KnowledgeStore(settings, embedder)
    ident = store.receive("lost.txt", BytesIO(b"Stored document."))["id"]
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
    ident = store.receive("persist.txt", BytesIO(b"Persistent permit text."))["id"]
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
