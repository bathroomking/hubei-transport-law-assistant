import hashlib
import json
import runpy
from collections import Counter

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from transport_law.config import PROJECT_ROOT
from transport_law.main import create_app
from transport_law.schemas import Source


def test_official_snapshots_and_generated_documents_are_reproducible(repository):
    build = runpy.run_path(str(PROJECT_ROOT / "scripts/build_legal_corpus.py"))["build"]
    assert build(check=True) == {
        "documents": 17,
        "official_articles": 1302,
        "discretion_items": 29,
        "total_chunks": 1336,
    }
    counts = Counter(c.document_id for c in repository.searchable)
    manifest = json.loads((PROJECT_ROOT / "data/raw/laws/manifest.json").read_text())
    for spec in manifest["documents"]:
        raw = PROJECT_ROOT / spec["file"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == spec["sha256"]
        assert counts[spec["document_id"]] == spec["article_count"] + spec.get(
            "table_item_count", 0
        )


def test_full_articles_include_context_missing_from_discretion_table(repository, retriever):
    article = retriever.search("公路法第44条", 1)[0].chunk
    assert "公安机关" in article.text and "经济补偿" in article.text
    assert article.source.pdf_page is None
    assert article.source.locator == "第四十四条"
    assert article.related_excerpt_ids == ["HB-ROAD-003-LAW44"]
    # 查原 ID 仍然返回真实的旧 PDF 引用，不能把它的出处悄悄换成官网。
    original = retriever.search("HB-ROAD-003-LAW44", 1)[0].chunk
    assert original == repository.get_article("HB-ROAD-003-LAW44")
    assert original.source.pdf_page == 2
    assert "经济补偿" not in original.text
    permit = repository.get_article("cn-highway-protection-2011-A027")
    assert "（七）" in permit.text and "公路建筑控制区" in permit.text


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("湖北省公路路政管理条例第十一条", "hb-road-2017-A011"),
        ("湖北省农村公路条例第二条", "hb-rural-2007-A002"),
        ("农村公路条例第二条", "cn-rural-2025-A002"),
        ("行政处罚法第33条", "cn-penalty-2021-A033"),
        ("首违不罚", "cn-penalty-2021-A033"),
        ("交通运输行政执法程序规定第82条", "cn-enforcement-2021-A082"),
        ("湖北省水路交通条例第70条", "hb-water-2026-A070"),
        ("无证经营道路运输会罚多少？", "cn-transport-2026-A063"),
    ],
)
def test_new_laws_are_retrievable(retriever, query, expected):
    assert retriever.search(query, 1)[0].chunk.id == expected


def test_missing_article_and_distinct_national_local_laws(retriever):
    assert retriever.search("行政处罚法第999条") == []
    assert retriever.search("首违不罚", 1)[0].chunk.id == "cn-penalty-2021-A033"
    hits = retriever.search("比较《湖北省农村公路条例》与《农村公路条例》的第二条")
    assert {h.chunk.id for h in hits} == {"hb-rural-2007-A002", "cn-rural-2025-A002"}


def test_2026_versions_are_kept_separate_from_old_laws(repository):
    for number in (9, 22):
        article = repository.get_article(f"cn-transport-2026-A{number:03d}")
        assert "国务院交通运输主管部门规定的年龄要求" in article.text
        assert "60周岁" not in article.text
        assert str(article.source.effective_on) == "2026-03-20"
    water = repository.get_article("hb-water-2026-A073")
    assert "2026年8月1日" in water.text
    assert str(water.source.effective_on) == "2026-08-01"


def test_web_sources_reject_fabricated_pdf_page(repository):
    raw = repository.get_article("cn-penalty-2021-A033").source.model_dump()
    raw["pdf_page"] = 1
    with pytest.raises(ValidationError, match="不得填写 PDF 页码"):
        Source.model_validate(raw)


def test_official_article_api_has_real_web_provenance(settings, retriever):
    selected = retriever.search("行政处罚法第33条", 1)[0].chunk
    with TestClient(create_app(settings)) as client:
        result = client.get(f"/api/v1/articles/{selected.id}")
        assert result.status_code == 200
        assert result.json() == selected.model_dump(mode="json")
        source = result.json()["source"]
        assert source["source_type"] == "web" and source["pdf_page"] is None
        assert source["official_url"].startswith("https://www.miit.gov.cn/")
        document = client.get(
            f"/api/v1/documents/{selected.document_id}",
            params={"version_id": selected.version_id},
        )
        assert document.status_code == 200
        body = document.json()
        assert body["coverage"] == "full_text"
        assert body["version_id"] == selected.version_id
        assert [article["article_number"] for article in body["articles"]] == list(range(1, 87))
        assert body["articles"][32] == result.json()
