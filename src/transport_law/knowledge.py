"""上传资料的持久化导入任务；同名拒绝、分块缓存、分批向量续传。"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import unicodedata
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from transport_law.config import Settings
from transport_law.embeddings import DashScopeEmbeddings, Embeddings
from transport_law.errors import AppError
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import LegalChunk, SearchHit, Source
from transport_law.vector_store import check_contract, chroma_client, chroma_errors, digest


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _filename(value: str) -> tuple[str, str, str]:
    name = unicodedata.normalize("NFKC", value or "").strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(unicodedata.category(char).startswith("C") for char in name)
        or len(name.encode("utf-8")) > 240
    ):
        raise AppError("invalid_filename", "文件名无效；请使用不包含路径的普通文件名。", 400)
    suffix = Path(name).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".markdown"}:
        raise AppError("unsupported_document", "上传只支持 PDF、TXT、MD 和 Markdown 文件。", 400)
    return name, name.casefold(), ".md" if suffix == ".markdown" else suffix


class KnowledgeStore:
    def __init__(
        self,
        settings: Settings,
        embedder: Embeddings | None = None,
        reserved_filenames: Iterable[str] = (),
    ):
        self.settings = settings
        self.path = settings.knowledge_path
        self.database = self.path / "uploads.sqlite3"
        self.embedder = embedder or DashScopeEmbeddings(settings)
        self.reserved_names = {
            unicodedata.normalize("NFKC", name.replace("\\", "/").rsplit("/", 1)[-1])
            .strip()
            .casefold()
            for name in reserved_filenames
        }

    @property
    def contract(self) -> dict:
        return {
            "schema_version": 1,
            "embedding_provider": "dashscope",
            "embedding_model": self.settings.embedding_model,
            "embedding_dimensions": self.settings.embedding_dimensions,
            "embedding_base_url": self.settings.dashscope_base_url,
            "query_instruction": self.settings.embedding_query_instruction,
            "document_template": "upload-title-heading-text-v1",
            "chunking": "legal_recursive_v1:800:120",
            "normalization": "l2",
            "distance": "cosine",
        }

    @contextmanager
    def _connect(self):
        self.path.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    name_key TEXT NOT NULL UNIQUE,
                    suffix TEXT NOT NULL,
                    sha256 TEXT,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    embedded_count INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    contract_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                """
            )
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _lock(self, name: str, *, blocking: bool = False):
        try:
            import fcntl
        except ImportError as exc:
            raise AppError(
                "upload_lock_unavailable", "上传任务当前需要 macOS 或 Linux 文件锁支持。", 503
            ) from exc
        folder = self.path / "locks"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / (name + ".lock")).open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise AppError("upload_busy", "该文档正在处理中，请稍后查看进度。", 409) from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _row(self, ident: str) -> dict:
        if not self.database.exists():
            raise AppError("document_not_found", "未找到上传文档。", 404)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (ident,)).fetchone()
        if row is None:
            raise AppError("document_not_found", "未找到上传文档。", 404)
        return dict(row)

    @staticmethod
    def _public(row: dict) -> dict:
        return {
            key: value
            for key, value in row.items()
            if key not in {"name_key", "contract_hash", "suffix"}
        }

    def _update(self, ident: str, **values) -> None:
        values["updated_at"] = _now()
        with self._connect() as connection:
            connection.execute(
                "UPDATE documents SET "
                + ", ".join(key + " = ?" for key in values)
                + " WHERE id = ?",
                (*values.values(), ident),
            )

    def receive(self, filename: str, stream: BinaryIO) -> dict:
        name, name_key, suffix = _filename(filename)
        if name_key in self.reserved_names:
            raise AppError("duplicate_filename", "正式知识库已存在同名文件，禁止重复上传。", 409)
        try:
            with self._lock("filename-" + digest(name_key)):
                return self._receive(name, name_key, suffix, stream)
        except AppError as exc:
            if exc.code == "upload_busy":
                raise AppError(
                    "duplicate_filename", "同名文件正在上传，禁止重复上传。", 409
                ) from exc
            raise

    def _receive(self, name: str, name_key: str, suffix: str, stream: BinaryIO) -> dict:
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM documents WHERE name_key = ?", (name_key,)
            ).fetchone():
                raise AppError("duplicate_filename", "已存在同名文件，禁止重复上传。", 409)
        ident = "upload-" + uuid.uuid4().hex
        folder = self.path / "documents" / ident
        with self._lock(ident):
            try:
                folder.mkdir(parents=True)
                temporary = folder / "receiving.part"
                size, checksum = 0, hashlib.sha256()
                with temporary.open("xb") as output:
                    while data := stream.read(64 * 1024):
                        size += len(data)
                        if size > self.settings.upload_max_bytes:
                            raise AppError("upload_too_large", "文件超过允许的上传大小。", 413)
                        checksum.update(data)
                        output.write(data)
                    if not size:
                        raise AppError("empty_upload", "不能上传空文件。", 400)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(folder / ("source" + suffix))
                # 完整原件落盘后才登记 queued，上传过程中崩溃不会留下永远占名的半文件任务。
                try:
                    with self._connect() as connection:
                        connection.execute(
                            """INSERT INTO documents
                            (id, filename, name_key, suffix, status, contract_hash,
                             sha256, size_bytes, created_at, updated_at)
                            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)""",
                            (
                                ident,
                                name,
                                name_key,
                                suffix,
                                digest(self.contract),
                                checksum.hexdigest(),
                                size,
                                _now(),
                                _now(),
                            ),
                        )
                except sqlite3.IntegrityError as exc:
                    raise AppError(
                        "duplicate_filename", "已存在同名文件，禁止重复上传。", 409
                    ) from exc
            except BaseException:
                for path in folder.glob("*"):
                    path.unlink()
                if folder.exists():
                    folder.rmdir()
                raise
        return self.get_document(ident)

    def get_document(self, ident: str) -> dict:
        row = self._row(ident)
        if row["status"] == "processing":
            try:
                with self._lock(row["id"]):
                    # 锁随进程退出释放；只有没有活跃工作进程的遗留任务才标记中断。
                    row = self._row(ident)
                    if row["status"] == "processing":
                        self._update(
                            ident,
                            status="failed",
                            error_code="job_interrupted",
                            error_message="导入进程已中断，请调用重试接口继续。",
                        )
                        row = self._row(ident)
            except AppError as exc:
                if exc.code != "upload_busy":
                    raise
        return self._public(row)

    def list_documents(self) -> list[dict]:
        if not self.database.exists():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self.get_document(row["id"]) for row in rows]

    def retry(self, ident: str) -> dict:
        row = self._row(ident)
        with self._lock(row["id"]):
            row = self._row(ident)
            if row["status"] == "ready":
                raise AppError("document_already_ready", "文档已完成入库，不需要重试。", 409)
            if row["contract_hash"] != digest(self.contract):
                raise AppError(
                    "vector_index_mismatch", "导入配置已变化，请恢复原模型和切分配置。", 409
                )
            self._update(ident, status="queued", error_code=None, error_message=None)
        return self.get_document(ident)

    def _collection(self, *, create: bool = False):
        if self.settings.knowledge_collection == self.settings.chroma_collection:
            raise AppError("knowledge_collection_conflict", "上传库须使用独立的 collection。", 409)
        with chroma_errors():
            from chromadb.errors import NotFoundError

            client = chroma_client(self.settings)
            try:
                collection = client.get_collection(
                    self.settings.knowledge_collection, embedding_function=None
                )
            except NotFoundError:
                if not create:
                    raise AppError(
                        "vector_index_missing", "上传资料的向量库不存在。", 503
                    ) from None
                collection = client.get_or_create_collection(
                    self.settings.knowledge_collection,
                    embedding_function=None,
                    configuration={"hnsw": {"space": "cosine"}},
                    metadata=self.contract,
                )
            check_contract(collection, self.contract)
            return collection

    def _chunks(self, row: dict) -> list[LegalChunk]:
        # 这些依赖只在导入任务中加载，启动和读取已完成记录均不执行提取或切分。
        from transport_law.chunking import ChunkingOptions, DraftChunk, build_draft_chunks
        from transport_law.ingest import load_document

        folder = self.path / "documents" / row["id"]
        raw = folder / ("source" + row["suffix"])
        if hashlib.sha256(raw.read_bytes()).hexdigest() != row["sha256"]:
            raise AppError("upload_source_changed", "保存的原文件已改变，停止导入。", 409)
        cache = folder / "chunks.jsonl"
        if cache.exists():
            drafts = [
                DraftChunk.model_validate_json(line) for line in cache.read_text().splitlines()
            ]
        else:
            loaded = replace(load_document(raw), file=row["filename"])
            if loaded.format == "pdf" and any(not page.text.strip() for page in loaded.pages):
                raise AppError(
                    "pdf_needs_ocr", "PDF 含无法提取文字的页面，请先做 OCR 后再导入。", 400
                )
            drafts = build_draft_chunks(
                loaded, ChunkingOptions(), document_id=row["id"], title=Path(row["filename"]).stem
            )
            if not drafts:
                raise AppError("no_extractable_text", "文件中没有可切分的文字。", 400)
            temporary = folder / "chunks.part"
            with temporary.open("w", encoding="utf-8") as output:
                for draft in drafts:
                    output.write(draft.model_dump_json() + "\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(cache)
        chunks = []
        for number, draft in enumerate(drafts, 1):
            if draft.document_id != row["id"] or draft.source.sha256 != row["sha256"]:
                raise AppError("upload_cache_mismatch", "已存分块与原文件不一致。", 409)
            locator = "；".join(
                (f"PDF 第 {loc.pdf_page} 页，" if loc.pdf_page is not None else "")
                + f"字符 [{loc.start_index}, {loc.end_index})"
                for loc in draft.source.locations
            )
            chunks.append(
                LegalChunk(
                    id="UP-" + draft.id.removeprefix("AUTO-"),
                    document_id=row["id"],
                    version_id=draft.version_id,
                    item_id=row["id"],
                    title=draft.title,
                    article=draft.heading or f"片段 {number}",
                    kind="document",
                    region="未标注",
                    domain="上传资料",
                    text=draft.text,
                    search_text=draft.title + "\n" + draft.text,
                    source=Source(
                        document_title=draft.title,
                        file=row["filename"],
                        sha256=row["sha256"],
                        source_type=draft.source.format,
                        pdf_page=draft.source.locations[0].pdf_page,
                        locator=locator,
                        verification_status="pending",
                        verification_note="用户上传资料，自动提取与分块；文字、表格关系、法律效力及适用性待核验。",
                    ),
                )
            )
        if not chunks:
            raise AppError("no_extractable_text", "文件中没有可切分的文字。", 400)
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO chunks (id, document_id, chunk_json) VALUES (?, ?, ?)",
                [(chunk.id, row["id"], chunk.model_dump_json()) for chunk in chunks],
            )
        return chunks

    def process(self, ident: str) -> None:
        row = self._row(ident)
        try:
            with self._lock(row["id"]):
                row = self._row(ident)
                if row["status"] != "queued":
                    return
                self._update(ident, status="processing", error_code=None, error_message=None)
                try:
                    if row["contract_hash"] != digest(self.contract):
                        raise AppError("vector_index_mismatch", "导入配置与已保存任务不一致。", 409)
                    chunks = self._chunks(row)
                    self._update(ident, chunk_count=len(chunks))
                    # 串行写同一集合；失败后已保存的批次保持不变，不锁住 SQLite 网络事务。
                    with self._lock(
                        "collection-" + digest(str(self.settings.chroma_path)), blocking=True
                    ):
                        self._embed_chunks(ident, chunks)
                    self._update(ident, status="ready", embedded_count=len(chunks))
                except AppError as exc:
                    self._update(
                        ident, status="failed", error_code=exc.code, error_message=exc.message
                    )
                except Exception:
                    self._update(
                        ident,
                        status="failed",
                        error_code="upload_processing_failed",
                        error_message="文档导入失败，请检查文件与本地存储后重试。",
                    )
        except AppError as exc:
            if exc.code != "upload_busy":
                raise

    def _embed_chunks(self, ident: str, chunks: list[LegalChunk]) -> None:
        with chroma_errors():
            collection = self._collection(create=True)
            entries = {}
            for chunk in chunks:
                text = f"{chunk.title}\n{chunk.article}\n{chunk.text}"
                entries[chunk.id] = (
                    chunk,
                    text,
                    {
                        "document_id": ident,
                        "content_sha256": digest(text),
                        "chunk_json": chunk.model_dump_json(),
                    },
                )
            existing = collection.get(where={"document_id": ident}, include=["metadatas"])
            for key, metadata in zip(existing["ids"], existing["metadatas"], strict=True):
                if key not in entries or metadata != entries[key][2]:
                    raise AppError("upload_cache_mismatch", "已存向量与分块不一致，停止覆盖。", 409)
            stored = set(existing["ids"])
            pending = [value for key, value in entries.items() if key not in stored]
            done = len(stored)
            self._update(ident, embedded_count=done)
            size = min(10, self.settings.embedding_batch_size)
            for offset in range(0, len(pending), size):
                batch = pending[offset : offset + size]
                vectors = self.embedder.embed_documents([text for _, text, _ in batch])
                collection.upsert(
                    ids=[chunk.id for chunk, _, _ in batch],
                    embeddings=vectors,
                    documents=[chunk.text for chunk, _, _ in batch],
                    metadatas=[metadata for _, _, metadata in batch],
                )
                done += len(batch)
                self._update(ident, embedded_count=done)
            if len(collection.get(where={"document_id": ident}, include=[])["ids"]) != len(chunks):
                raise AppError("vector_index_incomplete", "文档向量数量不完整，请重试。", 503)

    def _ready(self) -> list[dict]:
        if not self.database.exists():
            return []
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT id, chunk_count FROM documents WHERE status = 'ready'"
                ).fetchall()
            ]

    @property
    def searchable(self) -> list[LegalChunk]:
        if not self.database.exists():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT c.chunk_json FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.status = 'ready' ORDER BY c.id"
            ).fetchall()
        return [LegalChunk.model_validate_json(row["chunk_json"]) for row in rows]

    def get_article(self, chunk_id: str) -> LegalChunk:
        if self.database.exists():
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT c.chunk_json FROM chunks c JOIN documents d ON d.id = c.document_id "
                    "WHERE c.id = ? AND d.status = 'ready'",
                    (chunk_id,),
                ).fetchone()
            if row:
                return LegalChunk.model_validate_json(row["chunk_json"])
        raise AppError("article_not_found", "未找到已完成入库的上传片段。", 404)

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        ready = self._ready()
        if not ready:
            return []
        document_ids = {row["id"] for row in ready}
        # 固定本次查询的 ready 快照；下一次请求即能看到刚完成的新文档。
        chunks = [chunk for chunk in self.searchable if chunk.document_id in document_ids]
        count = sum(row["chunk_count"] for row in ready)
        if len(chunks) != count:
            raise AppError("vector_index_incomplete", "上传资料正文缺失，请检查本地索引。", 503)
        bm25 = BM25Retriever(LawRepository(chunks))
        explicit = bm25.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1000) for chunk in explicit[:top_k]]
        backend = self.settings.knowledge_retrieval_backend
        if backend == "bm25":
            return bm25.rank_candidates(question, chunks, limit=top_k)
        limit = self.settings.hybrid_candidate_k if backend == "hybrid" else top_k

        def dense():
            return self._rank_vectors(question, chunks, document_ids, limit=limit)

        if backend == "chroma":
            return dense()
        from transport_law.hybrid import fuse_rankers

        # 通用上传资料尚无可靠的法规/事项标签，因此两路都搜索全部 ready 片段。
        return fuse_rankers(
            lambda: bm25.rank_candidates(question, chunks, limit=limit),
            dense,
            top_k=top_k,
            rrf_k=self.settings.hybrid_rrf_k,
        )

    def _rank_vectors(
        self, question: str, chunks: list[LegalChunk], document_ids: set[str], *, limit: int
    ) -> list[SearchHit]:
        by_id = {chunk.id: chunk for chunk in chunks}
        with chroma_errors():
            collection = self._collection()
            where = {"document_id": {"$in": sorted(document_ids)}}
            if set(collection.get(where=where, include=[])["ids"]) != set(by_id):
                raise AppError("vector_index_incomplete", "上传资料向量缺失，请检查本地索引。", 503)
            vector = self.embedder.embed_query(question)
            result = collection.query(
                query_embeddings=[vector],
                n_results=min(limit, len(chunks)),
                where=where,
                include=["metadatas", "distances"],
            )
            hits = [
                SearchHit(chunk=by_id[ident], score=round(1.0 - distance, 6))
                for ident, distance in zip(result["ids"][0], result["distances"][0], strict=True)
            ]
            return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))
