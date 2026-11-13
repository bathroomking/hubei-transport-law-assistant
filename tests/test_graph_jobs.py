import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from transport_law.errors import AppError
from transport_law.graph_jobs import GraphJobs
from transport_law.knowledge import KnowledgeStore
from transport_law.main import create_app


def write_outputs(source, dest):
    rows = [json.loads(row) for row in source.read_text().splitlines()]
    dest.mkdir()
    (dest / "manifest.json").write_text(json.dumps({"chunks": len(rows)}))
    (dest / "graph.json").write_text(json.dumps([row["id"] for row in rows]))


def wait_for(store, phase):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = store.graph.status()
        if status["status"] == phase:
            return status
        time.sleep(0.02)
    pytest.fail(f"expected {phase}, found {status}")


def test_upload_auto_build_failure_restart_and_retry(settings, repository, monkeypatch):
    uploads = "/api/v1/knowledge/documents"
    graph = "/api/v1/knowledge/graph"
    attempts = []
    base = repository.all_chunks[0]

    def chunks(self, row):
        return [
            base.model_copy(update={"document_id": row["id"], "text": f"正文 {row['cache_key']}"})
        ]

    def worker(self, source, dest, _fd):
        attempts.append(source.read_bytes())
        if len(attempts) == 1:
            raise AppError("graph_provider_error", "测试服务故障")
        write_outputs(source, dest)

    monkeypatch.setattr(KnowledgeStore, "_chunks", chunks)
    monkeypatch.setattr(KnowledgeStore, "_embed_chunks", lambda *_: None)
    monkeypatch.setattr(GraphJobs, "_build", worker)

    # Reprocess only needs a test double for Chroma's delete call.
    class Collection:
        def delete(self, **_kwargs):
            pass

    monkeypatch.setattr(KnowledgeStore, "_collection", lambda *_args, **_kwargs: Collection())
    with TestClient(create_app(settings)) as client:
        store = client.app.state.runtime.knowledge
        assert client.get(graph).json()["status"] == "disabled"
        assert client.post(graph + "/retry").status_code == 409
        assert store.graph.build()["status"] == "empty"
        doc = client.post(uploads, files={"file": ("测试.pdf", b"%PDF-test")}).json()
        assert wait_for(store, "failed")["error_code"] == "graph_provider_error"
        assert client.get(f"{uploads}/{doc['id']}").json()["status"] == "ready"
        assert store.ready_chunks()

    with TestClient(create_app(settings)) as client:
        store = client.app.state.runtime.knowledge
        assert client.get(graph).json()["status"] == "failed"
        assert client.post(graph + "/retry").status_code == 202
        first = wait_for(store, "ready")
        previous = store.ready_chunks()
        assert client.post(f"{uploads}/{doc['id']}/reprocess").status_code == 202
        second = wait_for(store, "ready")
        assert second["generation"] != first["generation"]
        assert attempts[0] == attempts[1] != attempts[2]
        with pytest.raises(AppError, match="快照"):
            with store.graph_artifacts(previous):
                pytest.fail("stale chunks matched new graph")
        with store.graph_artifacts(store.ready_chunks()) as (path,):
            assert json.loads((path / "graph.json").read_text()) == [base.id]
            assert not (path.parent / first["generation"]).exists()
        assert store.graph.build()["generation"] == second["generation"]
        assert len(attempts) == 3


def test_unpublished_graph_never_visible_and_changes_coalesce(
    settings, seeded_repository, monkeypatch
):
    store = KnowledgeStore(settings)
    started = threading.Event()
    finish = threading.Event()
    calls = []

    def worker(source, dest, _fd):
        calls.append(source.read_bytes())
        write_outputs(source, dest)
        if len(calls) == 1:
            started.set()
            assert finish.wait(5)

    monkeypatch.setattr(store.graph, "_build", worker)
    thread = threading.Thread(target=store.graph.build)
    thread.start()
    try:
        assert started.wait(5)
        assert store.graph.status()["status"] == "building"
        original = store.ready_chunks()
        with pytest.raises(AppError):
            with store.graph_artifacts(original):
                pytest.fail("published staging graph")
        chunk = original[0].model_copy(update={"text": "changed provision"})
        with store._connect() as con:
            con.execute(
                "UPDATE chunks SET chunk_json = ? WHERE id = ?", (chunk.model_dump_json(), chunk.id)
            )
        store.graph.wake()
    finally:
        finish.set()
        thread.join(6)
    assert not thread.is_alive()
    wait_for(store, "ready")
    assert len(calls) == 2
    assert len(list((store.graph.root / "generations").iterdir())) == 1


def test_missing_keys_are_reported_without_running_worker(settings, seeded_repository):
    store = KnowledgeStore(settings)
    result = store.graph.build()
    assert result["status"] == "failed"
    assert result["error_code"] == "graph_keys_missing"
    assert not (store.graph.root / "generations").exists()


def test_worker_diagnostics_exclude_response_bodies(settings, seeded_repository, monkeypatch):
    from pydantic import SecretStr

    cfg = settings.model_copy(
        update={
            "graphrag_python": Path(sys.executable),
            "deepseek_api_key": SecretStr("sensitive-test-value"),
            "dashscope_api_key": SecretStr("offline-test-value"),
        }
    )
    store = KnowledgeStore(cfg)

    def fail(argv, **kwargs):
        dest = Path(argv[argv.index("--output") + 1])
        dest.mkdir()
        (dest / "failure.json").write_text(
            json.dumps(
                {
                    "stage": "extract_graph",
                    "category": "rate_limit",
                    "message": "sensitive-test-value",
                    "body": "private document",
                }
            )
        )
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(subprocess, "run", fail)
    result = store.graph.build()
    assert result["error_code"] == "graph_worker_rate_limit"
    assert "extract_graph" in result["error_message"]
    state = (store.graph.root / "state.json").read_text()
    assert "private document" not in state and "sensitive-test-value" not in state
    assert list((store.graph.root / "staging").iterdir()) == []
    assert result["finished_at"] >= result["started_at"]
