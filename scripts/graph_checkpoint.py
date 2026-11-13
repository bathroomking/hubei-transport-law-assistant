"""Persist validated pre-report Parquet tables without publishing an unfinished graph."""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

TABLES = ("documents", "text_units", "entities", "relationships", "communities")
RESUME_WORKFLOWS = ["create_community_reports", "generate_text_embeddings"]


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seal(directory):
    hashes = {name: file_hash(directory / f"{name}.parquet") for name in TABLES}
    (directory / "checkpoint.json").write_text(json.dumps(hashes), encoding="utf-8")


def has_extraction_error(output):
    with (output / "logs" / "indexing-engine.log").open(encoding="utf-8") as log:
        return any(" - ERROR - graphrag." in line for line in log)


def snapshot_key(path, config):
    # Match the lifecycle's immutable digest of settings and ordered source records.
    header = [
        "standard-v1",
        config.completion_models["default_completion_model"].model,
        config.embedding_models["default_embedding_model"].model,
        config.vector_store.vector_size,
    ]
    h = hashlib.sha256(json.dumps(header).encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class BuildObserver(NoopWorkflowCallbacks):
    def __init__(self, output: Path, checkpoint: Path):
        self.output = output
        self.checkpoint = checkpoint
        self.stage = "configure"
        self.task = asyncio.current_task()

    def workflow_start(self, name, instance):
        self.stage = name

    def workflow_end(self, name, instance):
        if name != "create_final_text_units" or self.checkpoint.is_dir():
            return
        if has_extraction_error(self.output):
            raise RuntimeError("抽取步骤有上游错误，不能保存检查点")
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.checkpoint.parent) as tmp:
            stage = Path(tmp) / "tables"
            stage.mkdir()
            for name in TABLES:
                shutil.copyfile(self.output / f"{name}.parquet", stage / f"{name}.parquet")
            seal(stage)
            os.replace(stage, self.checkpoint)

    def restore(self):
        marker = self.checkpoint / "checkpoint.json"
        if not marker.exists():
            return False
        hashes = json.loads(marker.read_text(encoding="utf-8"))
        for name in TABLES:
            source = self.checkpoint / f"{name}.parquet"
            if file_hash(source) != hashes.get(name):
                raise ValueError("图谱检查点校验失败")
        for name in TABLES:
            shutil.copyfile(self.checkpoint / f"{name}.parquet", self.output / f"{name}.parquet")
        return True
