"""已整理资料的向量索引：可续传、持久化，检索仍回填正式父法条。"""

import hashlib
import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from transport_law.config import Settings
from transport_law.embeddings import DashScopeEmbeddings, Embeddings
from transport_law.errors import AppError
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import LegalChunk, SearchHit


def digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class VectorDocument:
    id: str
    text: str
    embedding_text: str
    metadata: dict


def load_vector_documents(path: Path, repository: LawRepository) -> list[VectorDocument]:
    """只接收关联现有 LegalChunk 的批次，不把普通 DraftChunk 自动发布入库。"""
    parents = {chunk.id: chunk for chunk in repository.searchable}
    coverage = {ident: bytearray(len(parent.text)) for ident, parent in parents.items()}
    documents, seen = [], set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ident, text, metadata = row["id"], row["page_content"], row["metadata"]
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ValueError("invalid or duplicate id")
            seen.add(ident)
            provenance = metadata["provenance"]
            parent = repository.get_article(provenance["id"])
            if provenance != parent.model_dump(mode="json", exclude={"text"}):
                raise ValueError("stale provenance")
            if (
                metadata["document_id"] != parent.document_id
                or metadata["version_id"] != parent.version_id
                or metadata["title"] != parent.source.document_title
                or metadata["heading"] != parent.article
            ):
                raise ValueError("incorrect document identity")
            [location] = metadata["source"]["locations"]
            start, end = location["start_index"], location["end_index"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(parent.text)
                or location["pdf_page"] is not None
                or not isinstance(text, str)
                or not text.strip()
                or parent.text[start:end] != text
                or metadata["source"]["sha256"] != hashlib.sha256(parent.text.encode()).hexdigest()
            ):
                raise ValueError("incorrect text or location")
            if parent.id not in parents:  # 不索引 superseded 条目。
                continue
            coverage[parent.id][start:end] = b"\1" * (end - start)
            embedding_text = f"{parent.source.document_title}\n{parent.article}\n{text}"
            flat_metadata = {
                "parent_id": parent.id,
                "document_id": parent.document_id,
                "version_id": parent.version_id,
                "document_title": parent.source.document_title,
                "article": parent.article,
                "region": parent.region,
                "domain": parent.domain,
                "kind": parent.kind,
                "verification_status": parent.source.verification_status,
                "content_sha256": digest(embedding_text),
                # 嵌套出处序列化保存；过滤字段单独展开，不虚构 PDF 页码。
                "provenance_json": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
            }
            if parent.source.official_url:
                flat_metadata["official_url"] = str(parent.source.official_url)
            if parent.source.pdf_page is not None:
                flat_metadata["pdf_page"] = parent.source.pdf_page
            documents.append(VectorDocument(ident, text, embedding_text, flat_metadata))
        if not documents or any(
            not marks[index] and not char.isspace()
            for ident, marks in coverage.items()
            for index, char in enumerate(parents[ident].text)
        ):
            raise ValueError("incomplete corpus coverage")
    except (OSError, ValueError, KeyError, TypeError, AppError) as exc:
        raise AppError(
            "invalid_vector_documents",
            "分块文件缺失、重复、不完整或与当前法条不一致；请重新生成完整 Document 批次。",
            400,
        ) from exc
    return documents


def index_contract(settings: Settings, repository: LawRepository) -> dict:
    return {
        "schema_version": 2,
        "embedding_provider": "dashscope",
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "embedding_base_url": settings.dashscope_base_url,
        "query_template": "dashscope-query-instruct-v1",
        "query_instruction": settings.embedding_query_instruction,
        "document_template": "title-article-text-v1",
        "normalization": "l2",
        "distance": "cosine",
        "corpus_sha256": digest(
            [c.model_dump(mode="json") for c in sorted(repository.searchable, key=lambda c: c.id)]
        ),
    }


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
            "索引与当前模型、维度、语料或分块批次不一致；请更换 CHROMA_COLLECTION 后重新入库。",
            409,
        )
    if collection.configuration["hnsw"]["space"] != "cosine":
        raise AppError("vector_index_mismatch", "索引距离设置不一致，请新建 cosine 索引。", 409)


def build_vector_index(
    settings: Settings,
    repository: LawRepository,
    documents_path: Path,
    *,
    dry_run: bool = False,
    embedder: Embeddings | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    documents = load_vector_documents(documents_path, repository)
    contract = index_contract(settings, repository)
    contract.update(
        documents_sha256=digest(
            [[doc.id, doc.metadata] for doc in sorted(documents, key=lambda d: d.id)]
        ),
        expected_count=len(documents),
    )
    report = {
        "status": "validated" if dry_run else "ready",
        "collection": settings.chroma_collection,
        "path": str(settings.chroma_path),
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "chunks": len(documents),
        "parents": len({doc.metadata["parent_id"] for doc in documents}),
    }
    if dry_run:
        return report | {"note": "仅校验分块和出处；没有调用模型或写入数据库。"}
    embedder = embedder or DashScopeEmbeddings(settings)
    with chroma_errors():
        from chromadb.errors import NotFoundError

        client = chroma_client(settings)
        try:
            collection = client.get_collection(settings.chroma_collection, embedding_function=None)
        except NotFoundError:
            # 在创建空索引前发现缺 Key，避免把没有开始的任务误认为已入库。
            if isinstance(embedder, DashScopeEmbeddings) and not settings.embedding_configured:
                raise AppError(
                    "embedding_not_configured", "请在本地 .env 配置 DASHSCOPE_API_KEY。", 503
                ) from None
            collection = client.create_collection(
                settings.chroma_collection,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata=contract | {"state": "building"},
            )
        check_contract(collection, contract)
        existing = collection.get(include=["metadatas"])
        expected = {doc.id: doc for doc in documents}
        for ident, metadata in zip(existing["ids"], existing["metadatas"], strict=True):
            if ident not in expected or metadata != expected[ident].metadata:
                raise AppError("vector_index_mismatch", "已存记录与当前批次不一致。", 409)
        existing_ids = set(existing["ids"])
        pending = [doc for doc in documents if doc.id not in existing_ids]
        if pending:
            collection.modify(metadata=contract | {"state": "building"})
        size = min(settings.embedding_batch_size, client.get_max_batch_size())
        completed = len(existing_ids)
        for offset in range(0, len(pending), size):
            batch = pending[offset : offset + size]
            vectors = embedder.embed_documents([doc.embedding_text for doc in batch])
            collection.upsert(
                ids=[doc.id for doc in batch],
                embeddings=vectors,
                documents=[doc.text for doc in batch],
                metadatas=[doc.metadata for doc in batch],
            )
            completed += len(batch)
            if progress:
                progress(completed, len(documents))
        if collection.count() != len(documents):
            raise AppError("vector_index_incomplete", "入库数量不一致，请重试同一批次。", 409)
        collection.modify(metadata=contract | {"state": "ready"})
    return report | {"embedded": len(pending), "reused": len(existing_ids)}


def open_vector_index(settings: Settings, repository: LawRepository):
    if not (settings.chroma_path / "chroma.sqlite3").is_file():
        raise AppError("vector_index_missing", "向量索引尚未创建，请先运行 index-vectors。", 503)
    with chroma_errors():
        from chromadb.errors import NotFoundError

        try:
            collection = chroma_client(settings).get_collection(
                settings.chroma_collection, embedding_function=None
            )
        except NotFoundError:
            raise AppError(
                "vector_index_missing", "指定 collection 不存在，请先运行 index-vectors。", 503
            ) from None
        check_contract(collection, index_contract(settings, repository))
        if collection.metadata.get(
            "state"
        ) != "ready" or collection.count() != collection.metadata.get("expected_count"):
            raise AppError(
                "vector_index_incomplete", "向量索引尚未完成，请重跑同一批次以继续入库。", 503
            )
        return collection


class ChromaRetriever:
    def __init__(
        self,
        repository: LawRepository,
        settings: Settings,
        embedder: Embeddings | None = None,
        *,
        scope: BM25Retriever | None = None,
    ):
        self.scope = scope or BM25Retriever(repository)
        self.collection = open_vector_index(settings, repository)
        self.embedder = embedder or DashScopeEmbeddings(settings)

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        explicit = self.scope.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1.0) for chunk in explicit[:top_k]]
        return self.rank_candidates(question, self.scope.candidate_chunks(question), limit=top_k)

    def rank_candidates(
        self,
        question: str,
        chunks: list[LegalChunk],
        *,
        limit: int,
        query_vector: list[float] | None = None,
    ) -> list[SearchHit]:
        """先聚合子块为父法条，再按父法条数量截取召回窗口。"""
        if not 1 <= limit <= 200:
            raise ValueError("候选数量必须在 1 到 200 之间")
        parents = {chunk.id: chunk for chunk in chunks}
        if not parents:
            return []
        vector = query_vector if query_vector is not None else self.embedder.embed_query(question)
        with chroma_errors():
            where = {"parent_id": {"$in": list(parents)}}
            count = len(self.collection.get(where=where, include=[])["ids"])
            child_limit = min(count, limit * 4)
            if not child_limit:
                raise AppError("vector_index_incomplete", "候选法条缺少向量记录。", 503)
            while True:
                result = self.collection.query(
                    query_embeddings=[vector],
                    n_results=child_limit,
                    where=where,
                    include=["metadatas", "distances"],
                )
                hits = {}
                for metadata, distance in zip(
                    result["metadatas"][0], result["distances"][0], strict=True
                ):
                    ident = metadata["parent_id"]
                    if ident in parents and ident not in hits:
                        hits[ident] = SearchHit(
                            chunk=parents[ident], score=round(1.0 - distance, 6)
                        )
                if len(hits) >= limit or child_limit >= count:
                    return sorted(hits.values(), key=lambda h: (-h.score, h.chunk.id))[:limit]
                # 一个长法条可能拆成很多子块，扩大候选直到得到足够的不同父法条。
                child_limit = min(count, child_limit * 2)
