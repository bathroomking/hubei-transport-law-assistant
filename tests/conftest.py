import pytest
from pydantic import SecretStr

from transport_law.config import Settings
from transport_law.repository import ChunkRepository
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import LegalChunk, Source


@pytest.fixture
def settings(monkeypatch, tmp_path):
    # 本地 .env 和真实 Key 永远不进入离线测试。
    for key in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_RETRIES",
        "DEEPSEEK_MAX_TOKENS",
        "LAW_DATA_PATH",
        "RETRIEVAL_TOP_K",
        "HYBRID_CANDIDATE_K",
        "HYBRID_RRF_K",
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_TIMEOUT_SECONDS",
        "EMBEDDING_MAX_RETRIES",
        "EMBEDDING_QUERY_INSTRUCTION",
        "CHROMA_PATH",
        "CHROMA_COLLECTION",
        "KNOWLEDGE_PATH",
        "KNOWLEDGE_COLLECTION",
        "UPLOAD_MAX_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings(
        _env_file=None, knowledge_path=tmp_path / "knowledge", chroma_path=tmp_path / "chroma"
    )


@pytest.fixture
def online_settings(settings):
    return settings.model_copy(
        update={
            "deepseek_api_key": SecretStr("test-deepseek-key"),
            "dashscope_api_key": SecretStr("test-dashscope-key"),
            "retrieval_top_k": 3,
        }
    )


@pytest.fixture
def repository():
    # 运行测试只使用明确的合成上传片段，不依赖仓库附带的手工法规或来源文件。
    texts = [
        "第一条 施工许可：开展道路施工前应申请施工许可。",
        "第二条 安全防护：施工区域应设置安全防护设施。",
        "第三条 记录保存：检查记录应当妥善保存。",
    ]
    return ChunkRepository(
        [
            LegalChunk(
                id=f"UP-fixture-{number}",
                document_id="upload-fixture",
                version_id="sha256-fixture",
                item_id="upload-fixture",
                title="施工说明",
                article=f"片段 {number}",
                kind="document",
                region="未标注",
                domain="上传资料",
                text=text,
                search_text=text,
                source=Source(
                    document_title="施工说明",
                    file="施工说明.pdf",
                    sha256="a" * 64,
                    source_type="pdf",
                    pdf_page=1,
                    locator=f"字符 [{number * 100}, {number * 100 + len(text)})",
                    verification_status="pending",
                    verification_note="合成测试资料，待核验。",
                ),
            )
            for number, text in enumerate(texts, 1)
        ]
    )


@pytest.fixture
def seeded_repository(settings, repository):
    # 原文接口/重启测试直接播种持久化记录，检索与向量端到端测试另走真实上传流程。
    from transport_law.knowledge import KnowledgeStore

    store = KnowledgeStore(settings)
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO documents (id, filename, name_key, suffix, status, contract_hash, "
            "created_at, updated_at, chunk_count, embedded_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "upload-fixture",
                "施工说明.pdf",
                "施工说明.pdf",
                ".pdf",
                "ready",
                "fixture",
                "2026-01-01",
                "2026-01-01",
                3,
                3,
            ),
        )
        connection.executemany(
            "INSERT INTO chunks (id, document_id, chunk_json) VALUES (?, ?, ?)",
            [(c.id, c.document_id, c.model_dump_json()) for c in repository.all_chunks],
        )
    return repository


@pytest.fixture
def retriever(repository):
    return BM25Retriever(repository)
