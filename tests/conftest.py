import pytest

from transport_law.config import Settings
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever


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
        "DISCRETION_DATA_PATH",
        "RETRIEVAL_TOP_K",
        "RETRIEVAL_BACKEND",
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
        "KNOWLEDGE_RETRIEVAL_BACKEND",
        "UPLOAD_MAX_BYTES",
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None, knowledge_path=tmp_path / "knowledge")


@pytest.fixture
def repository(settings):
    return LawRepository.from_jsonl(settings.law_data_path)


@pytest.fixture
def retriever(repository):
    return BM25Retriever(repository)
