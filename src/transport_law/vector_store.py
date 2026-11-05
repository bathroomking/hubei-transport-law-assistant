"""上传知识库共用的 Chroma 客户端与向量配置校验。"""

import hashlib
import json
from contextlib import contextmanager

from transport_law.config import Settings
from transport_law.errors import AppError


def digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@contextmanager
def chroma_errors():
    try:
        from chromadb.errors import ChromaError
    except ImportError as exc:
        raise AppError(
            "chroma_not_installed", '请安装向量依赖：pip install -e ".[vectors]"。', 503
        ) from exc
    try:
        yield
    except (ChromaError, ValueError, TypeError, OSError) as exc:
        raise AppError(
            "chroma_unavailable", "Chroma 索引操作失败，请检查本地路径和 collection 配置。", 503
        ) from exc


def chroma_client(settings: Settings):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    return chromadb.PersistentClient(
        path=str(settings.chroma_path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def check_contract(collection, expected: dict) -> None:
    metadata = collection.metadata or {}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise AppError(
            "vector_index_mismatch",
            "索引与当前向量配置不一致，请使用匹配的模型、维度和知识库配置。",
            409,
        )
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise AppError("vector_index_mismatch", "索引距离设置不一致，请新建 cosine 索引。", 409)
