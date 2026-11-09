"""上传资料的持久化导入任务；同名拒绝、分块缓存、分批向量续传。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import unicodedata
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from transport_law.config import Settings
from transport_law.embeddings import DashScopeEmbeddings, Embeddings
from transport_law.errors import AppError
from transport_law.schemas import LegalChunk, SearchHit, Source
from transport_law.vector_store import check_contract, chroma_client, chroma_errors, digest

if TYPE_CHECKING:
    from transport_law.chunking import DraftSource
    from transport_law.ingest import LoadedDocument


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _pipeline_version() -> str:
    from transport_law.chunking import PIPELINE_VERSION

    return PIPELINE_VERSION


def _cache_digest(snapshot: bytes, drafts: bytes) -> str:
    return hashlib.sha256(hashlib.sha256(snapshot).digest() + drafts).hexdigest()


def _locator(number: int, source: DraftSource, key: str) -> str:
    ranges = "；".join(
        f"PDF 第 {loc.pdf_page} 页，字符 [{loc.start_index}, {loc.end_index})"
        for loc in source.locations
    )
    return f"片段序号 {number}；{ranges}；解析器 {source.parser}；提取快照 {key}"


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
    if suffix != ".pdf":
        raise AppError("unsupported_document", "上传只支持 PDF 文件。", 400)
    return name, name.casefold(), suffix


class KnowledgeStore:
    def __init__(
        self,
        settings: Settings,
        embedder: Embeddings | None = None,
    ):
        self.settings = settings
        self.path = settings.knowledge_path
        self.database = self.path / "uploads.sqlite3"
        self.embedder = embedder or DashScopeEmbeddings(settings)

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
            # 分块版本属于文档处理记录，不属于向量空间。旧集合仍可读取，旧任务须显式重解析。
            columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
            for name, declaration in {
                "pipeline_version": "TEXT NOT NULL DEFAULT 'pdf_legal_recursive_v1:800:120'",
                "cache_key": "TEXT NOT NULL DEFAULT ''",
                "rebuild_pending": "INTEGER NOT NULL DEFAULT 0",
                "reuse_cache_key": "TEXT NOT NULL DEFAULT ''",
                "cache_digest": "TEXT NOT NULL DEFAULT ''",
            }.items():
                if name not in columns:
                    try:
                        connection.execute(f"ALTER TABLE documents ADD COLUMN {name} {declaration}")
                    except sqlite3.OperationalError as exc:
                        # 另一进程可能刚刚完成同一迁移。
                        if "duplicate column name" not in str(exc):
                            raise
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
            if key
            not in {
                "name_key",
                "contract_hash",
                "suffix",
                "cache_key",
                "cache_digest",
                "reuse_cache_key",
                "rebuild_pending",
            }
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
                             sha256, size_bytes, created_at, updated_at,
                             pipeline_version, cache_key)
                            VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)""",
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
                                _pipeline_version(),
                                uuid.uuid4().hex,
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
                    "vector_index_mismatch", "导入配置已变化，请核对模型配置后重新解析并入库。", 409
                )
            if row["pipeline_version"] != _pipeline_version():
                raise AppError(
                    "processing_version_mismatch", "解析方式已更新，请使用重新解析并入库。", 409
                )
            self._update(ident, status="queued", error_code=None, error_message=None)
        return self.get_document(ident)

    def reprocess(self, ident: str) -> dict:
        """升级分块时可复用已核验的提取快照，同版本重新解析仍从原 PDF 提取。"""
        row = self._row(ident)
        with self._lock(row["id"]):
            row = self._row(ident)
            self._verified_source(row)
            pipeline = _pipeline_version()
            upgrade_ready = row["status"] == "ready" and row["pipeline_version"] != pipeline
            self._update(
                ident,
                status="queued",
                contract_hash=digest(self.contract),
                pipeline_version=pipeline,
                cache_key=uuid.uuid4().hex,
                reuse_cache_key=row["cache_key"] if upgrade_ready else "",
                cache_digest="",
                rebuild_pending=1,
                chunk_count=0,
                embedded_count=0,
                error_code=None,
                error_message=None,
            )
        return self.get_document(ident)

    def _verified_source(self, row: dict) -> bytes:
        raw = self.path / "documents" / row["id"] / ("source" + row["suffix"])
        try:
            data = raw.read_bytes()
        except OSError as exc:
            raise AppError(
                "upload_source_missing", "保存的原文件无法读取，请恢复原件后重试。", 409
            ) from exc
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise AppError("upload_source_changed", "保存的原文件已改变，停止导入。", 409)
        return data

    def _collection(self, *, create: bool = False):
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

    def _reusable_extraction(self, row: dict, folder: Path) -> LoadedDocument | None:
        """核对旧快照每段文字及页内位置；不能证明来源的旧缓存交给当前 PDF 解析器。"""
        key = row["reuse_cache_key"]
        if not key:
            return None
        from pydantic import TypeAdapter

        from transport_law.chunking import DraftSource
        from transport_law.ingest import PARSER_ID, LoadedDocument

        try:
            snapshot = TypeAdapter(LoadedDocument).validate_json(
                (folder / f"extraction-{key}.json").read_bytes(), strict=True
            )
            if (
                snapshot.parser != PARSER_ID
                or snapshot.file != row["filename"]
                or snapshot.sha256 != row["sha256"]
                or not snapshot.pages
                or any(p.pdf_page != i for i, p in enumerate(snapshot.pages, 1))
            ):
                return None
            with self._connect() as connection:
                records = connection.execute(
                    "SELECT id, chunk_json FROM chunks WHERE document_id = ?", (row["id"],)
                ).fetchall()
            published = {r[0]: LegalChunk.model_validate_json(r[1]) for r in records}
            if not published:
                return None
            covered = [bytearray(len(p.text)) for p in snapshot.pages]
            lines = (folder / f"chunks-{key}.jsonl").read_text().splitlines()
            for number, line in enumerate(lines, 1):
                draft = json.loads(line)
                source = DraftSource.model_validate(draft["source"])
                chunk = published.pop("UP-" + draft["id"].removeprefix("AUTO-"))
                if (
                    source.parser != snapshot.parser
                    or source.sha256 != snapshot.sha256
                    or chunk.source.sha256 != snapshot.sha256
                    or chunk.source.locator != _locator(number, source, key)
                ):
                    return None
                parts = []
                for loc in source.locations:
                    page = snapshot.pages[loc.pdf_page - 1]
                    if loc.end_index > len(page.text):
                        return None
                    parts.append(page.text[loc.start_index : loc.end_index])
                    covered[loc.pdf_page - 1][loc.start_index : loc.end_index] = b"\x01" * (
                        loc.end_index - loc.start_index
                    )
                if "".join(parts) != chunk.text or draft["text"] != chunk.text:
                    return None
            # 旧通用分块允许跳过空白，不可把后来插入的非空内容当作已核验文字。
            if published or any(
                not marked and not char.isspace()
                for page, mask in zip(snapshot.pages, covered, strict=True)
                for char, marked in zip(page.text, mask, strict=True)
            ):
                return None
            return snapshot
        except (ValueError, OSError, KeyError, IndexError, TypeError, AttributeError):
            return None

    def _chunks(self, row: dict) -> list[LegalChunk]:
        # 这些依赖只在导入任务中加载，启动和读取已完成记录均不执行提取或切分。
        from transport_law.chunking import ChunkingOptions, DraftChunk, build_draft_chunks
        from transport_law.ingest import load_document

        folder = self.path / "documents" / row["id"]
        raw = self._verified_source(row)
        if row["pipeline_version"] != _pipeline_version() or not row["cache_key"]:
            raise AppError("processing_version_mismatch", "分块缓存版本已更新，请重新解析。", 409)
        cache = folder / f"chunks-{row['cache_key']}.jsonl"
        extraction = folder / f"extraction-{row['cache_key']}.json"
        if row["cache_digest"]:
            try:
                snapshot_bytes = extraction.read_bytes()
                draft_bytes = cache.read_bytes()
            except OSError as exc:
                raise AppError(
                    "upload_cache_mismatch", "已保存的分块或提取快照缺失。", 409
                ) from exc
            if _cache_digest(snapshot_bytes, draft_bytes) != row["cache_digest"]:
                raise AppError("upload_cache_mismatch", "分块或提取快照已改变，请重新解析。", 409)
            drafts = [DraftChunk.model_validate_json(line) for line in draft_bytes.splitlines()]
        else:
            loaded = self._reusable_extraction(row, folder) or load_document(
                raw, filename=row["filename"], sha256=row["sha256"]
            )
            if any(not page.text.strip() and not page.is_blank for page in loaded.pages):
                raise AppError(
                    "pdf_needs_ocr", "PDF 有非空白页面未能提取文字，请检查原文件或 OCR 结果。", 400
                )
            drafts = build_draft_chunks(
                loaded, ChunkingOptions(), document_id=row["id"], title=Path(row["filename"]).stem
            )
            if not drafts:
                raise AppError("no_extractable_text", "文件中没有可切分的文字。", 400)
            snapshot_bytes = json.dumps(asdict(loaded), ensure_ascii=False).encode()
            draft_bytes = "".join(draft.model_dump_json() + "\n" for draft in drafts).encode()
            for target, content in ((extraction, snapshot_bytes), (cache, draft_bytes)):
                temporary = target.with_suffix(".part")
                with temporary.open("wb") as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(target)
            # 两个文件均完整落盘后才记录指纹。中途退出时重试从原件/旧快照重建。
            self._update(row["id"], cache_digest=_cache_digest(snapshot_bytes, draft_bytes))
        try:
            snapshot = json.loads(snapshot_bytes)
            page_text = {page["pdf_page"]: page["text"] for page in snapshot["pages"]}
            if snapshot["sha256"] != row["sha256"]:
                raise ValueError("source snapshot mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AppError(
                "upload_cache_mismatch", "分块来源快照缺失或不一致，请重新解析。", 409
            ) from exc
        chunks = []
        for number, draft in enumerate(drafts, 1):
            if draft.document_id != row["id"] or draft.source.sha256 != row["sha256"]:
                raise AppError("upload_cache_mismatch", "已存分块与原文件不一致。", 409)
            if draft.chunking != ChunkingOptions():
                raise AppError(
                    "processing_version_mismatch", "缓存切分参数已变化，请重新解析。", 409
                )
            try:
                if any(
                    not 0 <= loc.start_index < loc.end_index <= len(page_text[loc.pdf_page])
                    for loc in draft.source.locations
                ):
                    raise ValueError("invalid source range")
                source_text = "".join(
                    page_text[loc.pdf_page][loc.start_index : loc.end_index]
                    for loc in draft.source.locations
                )
                if source_text != draft.text or draft.source.parser != snapshot["parser"]:
                    raise ValueError("chunk source mismatch")
            except (KeyError, TypeError, ValueError) as exc:
                raise AppError(
                    "upload_cache_mismatch", "分块不能回查提取原文，请重新解析。", 409
                ) from exc
            first_page = draft.source.locations[0].pdf_page
            last_page = draft.source.locations[-1].pdf_page
            chunks.append(
                LegalChunk(
                    id="UP-" + draft.id.removeprefix("AUTO-"),
                    document_id=row["id"],
                    version_id=draft.version_id,
                    item_id=getattr(draft, "parent_id", None) or row["id"],
                    title=draft.title,
                    article=draft.heading or f"片段 {number}",
                    article_number=draft.article_number,
                    kind="document",
                    region="未标注",
                    domain="上传资料",
                    text=draft.text,
                    search_text="\n".join(filter(None, [draft.title, draft.heading, draft.text])),
                    source=Source(
                        document_title=draft.title,
                        file=row["filename"],
                        sha256=row["sha256"],
                        source_type=draft.source.format,
                        pdf_page=first_page,
                        pdf_page_end=last_page if last_page != first_page else None,
                        locator=_locator(number, draft.source, row["cache_key"]),
                        verification_status="pending",
                        verification_note="用户上传资料，自动提取与分块；文字、表格关系、法律效力及适用性待核验。",
                    ),
                )
            )
        if not chunks:
            raise AppError("no_extractable_text", "文件中没有可切分的文字。", 400)
        return chunks

    def _save_chunks(self, ident: str, chunks: list[LegalChunk], *, rebuild: bool) -> None:
        with self._connect() as connection:
            if rebuild:
                connection.execute("DELETE FROM chunks WHERE document_id = ?", (ident,))
            connection.executemany(
                "INSERT OR REPLACE INTO chunks (id, document_id, chunk_json) VALUES (?, ?, ?)",
                [(chunk.id, ident, chunk.model_dump_json()) for chunk in chunks],
            )
            connection.execute(
                "UPDATE documents SET rebuild_pending = 0, reuse_cache_key = '', "
                "chunk_count = ? WHERE id = ?",
                (len(chunks), ident),
            )

    def _clean_superseded_caches(self, row: dict) -> None:
        folder = self.path / "documents" / row["id"]
        keep = {f"chunks-{row['cache_key']}.jsonl", f"extraction-{row['cache_key']}.json"}
        # 原件及来源提取快照不在清理范围内；仅清理不再使用的分块缓存与未完成临时文件。
        for pattern in ("chunks*.jsonl", "chunks*.part", "extraction-*.part"):
            for path in folder.glob(pattern):
                if path.name not in keep:
                    try:
                        path.unlink()
                    except OSError:
                        pass  # 清理失败不能把已完整入库的文档改成 failed。

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
                    if row["pipeline_version"] != _pipeline_version():
                        raise AppError(
                            "processing_version_mismatch", "解析方式已更新，请重新解析并入库。", 409
                        )
                    chunks = self._chunks(row)
                    # 串行写同一集合；失败后已保存的批次保持不变，不锁住 SQLite 网络事务。
                    with self._lock(
                        "collection-" + digest(str(self.settings.chroma_path)), blocking=True
                    ):
                        rebuild = bool(row["rebuild_pending"])
                        if rebuild:
                            # 新分块及来源验证完毕才移除旧向量；pending 标记使中断后可幂等重做。
                            with chroma_errors():
                                self._collection(create=True).delete(where={"document_id": ident})
                        self._save_chunks(ident, chunks, rebuild=rebuild)
                        self._embed_chunks(ident, chunks)
                    self._update(ident, status="ready", embedded_count=len(chunks))
                    self._clean_superseded_caches(row)
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

    def ready_chunks(self) -> list[LegalChunk]:
        """一次读取已完成文档及分块，保证查询使用同一个数据库快照。"""
        if not self.database.exists():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT d.id, d.chunk_count, c.chunk_json FROM documents d "
                "LEFT JOIN chunks c ON c.document_id = d.id "
                "WHERE d.status = 'ready' ORDER BY c.id"
            ).fetchall()
        chunks = [
            LegalChunk.model_validate_json(row["chunk_json"])
            for row in rows
            if row["chunk_json"] is not None
        ]
        expected = Counter({row["id"]: row["chunk_count"] for row in rows})
        if Counter(chunk.document_id for chunk in chunks) != expected:
            raise AppError("vector_index_incomplete", "上传资料正文缺失，请检查本地索引。", 503)
        return chunks

    def rank_vectors(
        self,
        chunks: list[LegalChunk],
        *,
        limit: int,
        query_vector: list[float],
    ) -> list[SearchHit]:
        by_id = {chunk.id: chunk for chunk in chunks}
        if not by_id:
            return []
        with chroma_errors():
            collection = self._collection()
            where = {"document_id": {"$in": sorted({chunk.document_id for chunk in chunks})}}
            if set(collection.get(where=where, include=[])["ids"]) != set(by_id):
                raise AppError("vector_index_incomplete", "上传资料向量缺失，请检查本地索引。", 503)
            result = collection.query(
                query_embeddings=[query_vector],
                n_results=min(limit, len(chunks)),
                where=where,
                include=["metadatas", "distances"],
            )
            hits = [
                SearchHit(chunk=by_id[ident], score=round(1.0 - distance, 6))
                for ident, distance in zip(result["ids"][0], result["distances"][0], strict=True)
            ]
            return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))
