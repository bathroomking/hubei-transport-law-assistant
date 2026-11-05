import builtins

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from transport_law.main import create_app
from transport_law.retrieval import BM25Retriever


def test_frontend_and_assets_are_served_without_exposing_project_files(settings):
    with TestClient(create_app(settings)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert 'lang="zh-CN"' in page.text
        for filename, content_type in (
            ("styles.css", "text/css"),
            ("app.js", "javascript"),
            ("favicon.svg", "image/svg+xml"),
        ):
            asset = client.get(f"/static/{filename}")
            assert asset.status_code == 200
            assert content_type in asset.headers["content-type"]
        assert client.get("/static/.env").status_code == 404
        assert client.get("/static/%2e%2e/config.py").status_code == 404
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/docs").status_code == 200


def test_api_smoke_validation_and_missing_key(settings, seeded_repository):
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["chunk_count"] == 3
        assert health["pending_verification_count"] == 3
        assert health["llm_configured"] is False
        assert health["retrieval_backend"] == "hybrid"
        assert health["online_only"] is True
        assert client.get("/docs").status_code == 200
        hits = BM25Retriever(client.app.state.runtime.repository).search("施工许可")
        assert hits[0].chunk.id == "UP-fixture-1"
        article = client.get("/api/v1/articles/" + hits[0].chunk.id)
        assert article.status_code == 200
        assert client.get("/api/v1/articles/unknown").status_code == 404
        assert client.post("/api/v1/ask", json={"question": " "}).status_code == 422
        for question in ("明天武汉天气怎么样？", "公路挖沟如何处理？"):
            missing_key = client.post("/api/v1/ask", json={"question": question})
            assert missing_key.status_code == 503
            assert missing_key.json()["error"]["code"] == "llm_not_configured"
        assert client.get("/api/v1/discretion/unknown").status_code == 404
        for path in (
            "/api/v1/retrieve",
            "/api/v1/discretion/check",
            "/api/v1/knowledge/ask",
            "/api/v1/knowledge/retrieve",
        ):
            assert client.post(path, json={"question": "公路挖沟如何处理？"}).status_code == 404
        assert client.get("/api/v1/knowledge/chunks/unknown").status_code == 404


def test_online_ask_requires_embedding_key_before_retrieval(settings):
    class NeverRetrieve:
        def search(self, *args, **kwargs):
            pytest.fail("在线配置不完整时不得开始检索")

    configured = settings.model_copy(update={"deepseek_api_key": SecretStr("offline-test-key")})
    with TestClient(create_app(configured, retriever=NeverRetrieve())) as client:
        response = client.post("/api/v1/ask", json={"question": "公路挖沟如何处理？"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "embedding_not_configured"


def test_service_restart_and_retrieval_do_not_load_chunking_dependencies(
    settings, seeded_repository, monkeypatch
):
    original_import = builtins.__import__

    def import_without_ingestion(name, *args, **kwargs):
        if name.startswith(("langchain", "transport_law.chunking", "transport_law.ingest")):
            pytest.fail("服务启动和检索不应加载文档提取或分块模块")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_ingestion)
    results = []
    # 两次完整 lifespan 对应两次启动，直接加载持久化资料即可检索。
    for _ in range(2):
        with TestClient(create_app(settings)) as client:
            assert client.get("/health").json()["status"] == "ok"
            hits = BM25Retriever(client.app.state.runtime.repository).search("施工许可")
            assert hits
            response = client.get("/api/v1/articles/" + hits[0].chunk.id)
            assert response.status_code == 200
            assert response.json() == hits[0].chunk.model_dump(mode="json")
            results.append(hits)
    assert results[0] and results[0] == results[1]
