import builtins
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from transport_law.discretion import DiscretionService
from transport_law.main import create_app
from transport_law.schemas import CaseFacts


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


def test_api_smoke_validation_and_missing_key(settings):
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["chunk_count"] == 1336
        assert health["pending_verification_count"] == 1336
        assert health["llm_configured"] is False
        assert client.get("/docs").status_code == 200
        hits = client.post("/api/v1/retrieve", json={"question": "路边挖沟有哪些依据？"}).json()[
            "hits"
        ]
        assert len(hits) == 5
        article = client.get("/api/v1/articles/" + hits[0]["chunk"]["id"])
        assert article.status_code == 200
        assert client.get("/api/v1/articles/unknown").status_code == 404
        assert client.post("/api/v1/retrieve", json={"question": " "}).status_code == 422
        answer = client.post("/api/v1/ask", json={"question": "明天武汉天气怎么样？"}).json()
        assert answer["status"] == "insufficient_evidence"
        missing_key = client.post("/api/v1/ask", json={"question": "公路挖沟如何处理？"})
        assert missing_key.status_code == 503
        assert missing_key.json()["error"]["code"] == "llm_not_configured"
        assert client.get("/api/v1/discretion/unknown").status_code == 404
        check = client.post("/api/v1/discretion/check", json={"region": "湖北省"}).json()
        assert check["status"] == "needs_facts"


def test_service_restart_and_retrieval_do_not_load_chunking_dependencies(settings, monkeypatch):
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
            response = client.post("/api/v1/retrieve", json={"question": "公路法第四十四条"})
            assert response.status_code == 200
            results.append(response.json()["hits"])
    assert results[0] and results[0] == results[1]


@pytest.mark.parametrize("area", ["19.9999", "20", "20.0001", "39.9999", "40", "40.0001"])
def test_draft_rules_never_produce_case_penalty(settings, repository, area):
    service = DiscretionService(settings.discretion_data_path, repository)
    result = service.evaluate_rules(
        CaseFacts(
            region="湖北省",
            location="测试县测试路段",
            occurred_on="2026-10-01",
            road_type="ordinary_highway",
            behavior="excavation",
            has_permission=False,
            area_m2=area,
            facts_verified=True,
        )
    )
    assert result.status == "needs_review"
    assert result.missing_fields == []
    assert result.applicability_status == "not_verified"
    assert "fine_amount" not in result.model_dump()
    boundary_note = any("恰好" in note for note in result.review_notes)
    assert boundary_note == (Decimal(area) in (Decimal(20), Decimal(40)))


def test_false_permission_is_present_not_missing(settings, repository):
    result = DiscretionService(settings.discretion_data_path, repository).evaluate_rules(
        CaseFacts(has_permission=False)
    )
    assert "has_permission" not in result.missing_fields
    assert "road_type" in result.missing_fields


@pytest.mark.parametrize(
    "data",
    [
        {"region": "湖南省"},
        {"road_type": "expressway"},
        {"road_type": "urban_road"},
        {"item_id": "other"},
    ],
)
def test_scope_is_checked(settings, repository, data):
    result = DiscretionService(settings.discretion_data_path, repository).evaluate_rules(
        CaseFacts(**data)
    )
    assert result.status == "out_of_scope"
    assert result.sources == []


@pytest.mark.parametrize(
    "data",
    [
        {"area_m2": 0},
        {"area_m2": -1},
        {"area_m2": "NaN"},
        {"area_m2": "Infinity"},
        {"has_permission": "false"},
    ],
)
def test_fact_validation(data):
    with pytest.raises(ValidationError):
        CaseFacts.model_validate(data)
