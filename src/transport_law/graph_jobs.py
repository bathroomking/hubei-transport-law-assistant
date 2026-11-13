"""Publish complete Standard GraphRAG indexes and refresh them after uploads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from transport_law.config import PROJECT_ROOT
from transport_law.errors import AppError

if TYPE_CHECKING:
    from transport_law.knowledge import KnowledgeStore
    from transport_law.schemas import LegalChunk


_WORKFLOWS = {
    "configure",
    "index",
    "export",
    "load_input_documents",
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "finalize_graph",
    "extract_covariates",
    "create_communities",
    "create_final_text_units",
    "create_community_reports",
    "generate_text_embeddings",
}
_ERROR_TYPES = {
    "authentication",
    "rate_limit",
    "timeout",
    "network",
    "validation",
    "dependency",
    "storage",
    "upstream",
    "unknown",
}


def _failure(destination: Path) -> tuple[str, str]:
    try:
        report = json.loads((destination / "failure.json").read_text(encoding="utf-8"))
        stage, category = report["stage"], report["category"]
    except (OSError, ValueError, KeyError, TypeError):
        return "index", "unknown"
    if not isinstance(stage, str) or stage not in _WORKFLOWS:
        stage = "index"
    if not isinstance(category, str) or category not in _ERROR_TYPES:
        category = "unknown"
    return stage, category


def _inputs(chunks: list[LegalChunk]):
    for chunk in sorted(chunks, key=lambda c: c.id):
        row = chunk.model_dump(
            include={"id", "document_id", "version_id", "title", "article", "text"}
        )
        yield (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()


class GraphJobs:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.root = store.path / "graphrag"
        self.marker = self.root / "enabled"
        self._guard = threading.Lock()
        self._running = False
        self._requested = False
        self._retry = False

    def _snapshot(self, chunks):
        cfg = self.store.settings
        # 输入格式或模型发生变化时，须生成新索引。
        h = hashlib.sha256(
            json.dumps(
                ["standard-v1", cfg.deepseek_model, cfg.embedding_model, cfg.embedding_dimensions]
            ).encode()
        )
        for line in _inputs(chunks):
            h.update(line)
        return h.hexdigest()

    def _read(self) -> dict:
        try:
            return json.loads((self.root / "state.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}

    def _save(self, state: dict) -> None:
        path = self.root / "state.tmp"
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        path.replace(self.root / "state.json")

    def _published(self, state, fingerprint) -> Path | None:
        name = state.get("generation", "")
        if state.get("published_snapshot") != fingerprint or not re.fullmatch("[0-9a-f]{32}", name):
            return None
        path = self.root / "generations" / name
        if (path / "manifest.json").is_file() and (path / "graph.json").is_file():
            return path
        return None

    @contextmanager
    def artifacts(self, chunks: list[LegalChunk]):
        if self.marker.exists():
            with self.store._lock("graph-publish", blocking=True, shared=True):
                path = self._published(self._read(), self._snapshot(chunks))
                if path:
                    yield (path,)
                    return
        raise AppError("graph_index_unavailable", "当前知识库快照的完整图索引尚不可用。", 503)

    def status(self) -> dict:
        if not self.marker.exists():
            return {"enabled": False, "status": "disabled"}
        chunks = self.store.ready_chunks()
        fingerprint = self._snapshot(chunks)
        state = self._read()
        if not chunks:
            phase = "empty"
        elif self._published(state, fingerprint):
            phase = "ready"
        elif state.get("attempt") == fingerprint and state.get("status") == "failed":
            phase = "failed"
        elif state.get("status") == "building":
            try:
                with self.store._lock("graph-build"):
                    phase = "queued"  # Interrupted process; the next worker resumes from the cache.
            except AppError as exc:
                if exc.code != "upload_busy":
                    raise
                phase = "building"
        else:
            phase = "queued"
        result = {
            "enabled": True,
            "status": phase,
            "documents": len({c.document_id for c in chunks}),
            "chunks": len(chunks),
            "snapshot": fingerprint,
        }
        if state.get("attempt") == fingerprint:
            result.update(started_at=state.get("started_at"), finished_at=state.get("finished_at"))
        if phase == "ready":
            result["generation"] = state["generation"]
        if phase == "failed":
            result.update(
                error_code=state.get("error_code"), error_message=state.get("error_message")
            )
        return result

    def _build(self, source: Path, destination: Path, fd: int) -> None:
        cfg = self.store.settings
        if not cfg.llm_configured or not cfg.embedding_configured:
            raise AppError(
                "graph_keys_missing", "图索引需要已配置的 DeepSeek 和百炼 API Key。", 503
            )
        if (
            cfg.deepseek_base_url not in {"https://api.deepseek.com", "https://api.deepseek.com/v1"}
            or cfg.dashscope_base_url != "https://dashscope.aliyuncs.com/api/v1"
        ):
            raise AppError(
                "graph_endpoint_unsupported",
                "此建图程序仅支持项目使用的 DeepSeek 与百炼官方服务地址。",
                503,
            )
        worker = PROJECT_ROOT / "scripts" / "graph_index.py"
        if not cfg.graphrag_python.is_file() or not worker.is_file():
            raise AppError(
                "graph_worker_missing", "请安装标准 GraphRAG 专用环境并核对 GRAPHRAG_PYTHON。", 503
            )
        env = dict(os.environ)
        env.update(
            DEEPSEEK_API_KEY=cfg.deepseek_api_key.get_secret_value(),
            DASHSCOPE_API_KEY=cfg.dashscope_api_key.get_secret_value(),
            DEEPSEEK_MODEL=cfg.deepseek_model,
            EMBEDDING_MODEL=cfg.embedding_model,
            EMBEDDING_DIMENSIONS=str(cfg.embedding_dimensions),
        )
        # Keep the corpus lock in the child if the supervising API process is killed mid-build.
        result = subprocess.run(
            [
                cfg.graphrag_python,
                worker,
                "--input",
                source,
                "--output",
                destination,
                "--cache",
                self.root / "cache",
            ],
            env=env,
            pass_fds=(fd,),
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode:
            stage, category = _failure(destination)
            raise AppError(
                f"graph_worker_{category}",
                f"建图阶段 {stage} 未完成（退出码 {result.returncode}，错误类别 {category}）。",
            )

    def _run(self, retry: bool) -> None:
        with self.store._lock("graph-build", blocking=True) as fd:
            state = self._read()
            while True:
                chunks = self.store.ready_chunks()
                snapshot = self._snapshot(chunks)
                if not chunks or self._published(state, snapshot):
                    return
                if (
                    not retry
                    and state.get("attempt") == snapshot
                    and state.get("status") == "failed"
                ):
                    return
                state.update(
                    attempt=snapshot,
                    status="building",
                    error_code=None,
                    error_message=None,
                    started_at=datetime.now(UTC).isoformat(),
                    finished_at=None,
                )
                self._save(state)
                try:
                    # 子进程继承互斥锁；持锁后才能清理之前未发布的工作区。
                    stage = self.root / "staging"
                    if stage.exists():
                        shutil.rmtree(stage)
                    stage.mkdir()
                    with tempfile.TemporaryDirectory(dir=stage) as temp:
                        source = Path(temp) / "chunks.jsonl"
                        with source.open("wb") as file:
                            file.writelines(_inputs(chunks))
                        dest = Path(temp) / "index"
                        self._build(source, dest, fd)
                        if not all((dest / f).is_file() for f in ("manifest.json", "graph.json")):
                            raise AppError(
                                "graph_output_incomplete", "建图程序未生成完整的发布产物。"
                            )
                        if self._snapshot(self.store.ready_chunks()) != snapshot:
                            continue  # 构建期间有新文档就绪，复用缓存追到最新快照。
                        self._publish(dest, state, snapshot)
                except Exception as exc:
                    code = exc.code if isinstance(exc, AppError) else "graph_build_failed"
                    message = (
                        exc.message
                        if isinstance(exc, AppError)
                        else "图索引构建失败，请检查本地存储后重试。"
                    )
                    self._save(
                        {
                            **state,
                            "status": "failed",
                            "error_code": code,
                            "error_message": message,
                            "finished_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    return
                return

    def _publish(self, dest: Path, state: dict, snapshot: str) -> None:
        generations = self.root / "generations"
        generations.mkdir(exist_ok=True)
        name = uuid.uuid4().hex
        with self.store._lock("graph-publish", blocking=True):
            dest.rename(generations / name)
            state.update(
                generation=name,
                published_snapshot=snapshot,
                status="ready",
                finished_at=datetime.now(UTC).isoformat(),
            )
            self._save(state)
            for old in generations.iterdir():
                if old.name != name and re.fullmatch("[0-9a-f]{32}", old.name):
                    shutil.rmtree(old, ignore_errors=True)

    def build(self) -> dict:
        """Explicit bootstrap/retry. A durable marker enrolls subsequent uploads and restarts."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.marker.touch(exist_ok=True)
        self._run(retry=True)
        return self.status()

    def wake(self, *, retry: bool = False) -> None:
        if not self.marker.exists():
            if retry:
                raise AppError("graph_not_enabled", "请先运行 graph-build 启用图索引。", 409)
            return
        with self._guard:
            self._requested = True
            self._retry = retry or self._retry
            if not self._running:
                self._running = True
                threading.Thread(target=self._drain, daemon=True, name="graphrag-index").start()

    def _drain(self):
        while True:
            with self._guard:
                if not self._requested:
                    self._running = False
                    return
                retry, self._retry = self._retry, False
                self._requested = False
            try:
                self._run(retry)
            except Exception:
                with self._guard:
                    self._running = False
                raise
