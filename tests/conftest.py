import pytest

from transport_law.config import Settings
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever


@pytest.fixture
def settings(monkeypatch):
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
    ):
        monkeypatch.delenv(key, raising=False)
    return Settings(_env_file=None)


@pytest.fixture
def repository(settings):
    return LawRepository.from_jsonl(settings.law_data_path)


@pytest.fixture
def retriever(repository):
    return BM25Retriever(repository)
