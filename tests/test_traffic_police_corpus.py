import json
import re

from fastapi.testclient import TestClient

from transport_law.config import PROJECT_ROOT
from transport_law.main import create_app


def compact(text):
    return re.sub(r"\s+", "", text)


def test_traffic_articles_retain_conditions_and_amended_text(repository):
    signal = repository.get_article("cn-traffic-implementation-2017-A038")
    assert "红灯亮时，禁止车辆通行" in signal.text
    assert "右转弯的车辆在不妨碍被放行的车辆、行人通行的情况下，可以通行" in signal.text
    lane = repository.get_article("cn-traffic-implementation-2017-A044")
    assert "变更车道的机动车不得影响相关车道内行驶的机动车的正常行驶" in lane.text
    law = repository.get_article("cn-traffic-safety-2021-A020")
    assert "备案管理" in law.text  # 2021 年修改的内容，不误收旧版许可规定。
    procedure = repository.get_article("cn-violation-procedure-2020-A035")
    assert "涉嫌饮酒驾驶车辆发生交通事故" in procedure.text
    assert "拒绝配合酒精呼气测试" in procedure.text
    assert str(procedure.source.effective_on) == "2020-05-01"
    accident = repository.get_article("cn-accident-procedure-2017-A061")
    assert "可以适当减轻责任" in accident.text
    assert "不低于同等" not in accident.text  # 不把 2026 年征求意见稿混入正文。


def test_hubei_compilation_keeps_both_official_sources(repository):
    article = repository.get_article("hb-traffic-2017-A056")
    assert "的计量认证管理" in article.text
    assert "资格管理和计量认证管理" not in article.text
    assert "2013" in article.source.file
    assert "合并整理" in article.source.verification_note
    assert article.source.supporting_sources[0].locator.endswith("第五十六条第五款")
    assert "2017" in article.source.supporting_sources[0].file
    assert article.source.effective_on is None
    unchanged = repository.get_article("hb-traffic-2017-A074")
    assert unchanged.source.supporting_sources == []
    assert unchanged.text.startswith("第七十四条")
    assert "第七十四条" not in repository.get_article("hb-traffic-2017-A073").text


def test_police_table_preserves_all_bands_and_actual_pages(repository):
    extracted = json.loads(
        (PROJECT_ROOT / "data/raw/laws/hb-police-discretion-2026.extract.json").read_text()
    )
    tables = [e for e in extracted["entries"] if e["kind"] == "discretion"]
    assert len(tables) == 29 and sum(len(e["bands"]) for e in tables) == 79
    for entry in tables:
        chunk = repository.get_article(f"hb-police-discretion-2026-{entry['id_suffix']}")
        assert chunk.source.source_type == "pdf"
        assert chunk.source.pdf_page == entry["pages"][0]
        assert chunk.source.printed_page == str(entry["pages"][0] + 2)
        for band in entry["bands"]:
            assert band["circumstances"] in chunk.text and band["detention"] in chunk.text
    scope = repository.get_article("hb-police-discretion-2026-G020")
    assert "仅对行政拘留裁量基准予以细化" in compact(scope.text)
    assert scope.source.pdf_page == 12 and scope.source.printed_page == "14"
    repeated = repository.get_article("hb-police-discretion-2026-T014")
    assert repeated.text.count("违法情节：") == 4
    assert "不认为是犯罪的" in repeated.text
    contradiction = repository.get_article("hb-police-discretion-2026-T006")
    assert "表述不一致" in contradiction.source.verification_note
    long_article = repository.get_article("hb-police-discretion-2026-G019")
    assert long_article.source.pdf_page == 10
    assert "10—12" in long_article.source.locator
    assert "（十一）" in long_article.text


def test_named_traffic_law_and_table_locations_do_not_overlap(retriever):
    for query, expected in [
        ("道路交通安全法第38条", "cn-traffic-safety-2021-A038"),
        ("道路交通安全法实施条例第38条", "cn-traffic-implementation-2017-A038"),
        ("湖北省实施《中华人民共和国道路交通安全法》办法第38条", "hb-traffic-2017-A038"),
        ("湖北公安裁量基准第20条", "hb-police-discretion-2026-G020"),
        ("湖北公安交管裁量交通管理册第20项", "hb-police-discretion-2026-T020"),
    ]:
        assert [hit.chunk.id for hit in retriever.search(query)] == [expected]


def test_police_sources_are_available_through_api(settings, retriever):
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/articles/hb-traffic-2017-A056")
        assert response.status_code == 200
        article = response.json()
        assert article["source"]["supporting_sources"][0]["official_url"].startswith(
            "https://www.hppc.gov.cn/"
        )
        selected = retriever.search("变更车道发生剐蹭", 1)[0].chunk
        assert selected.id == "cn-traffic-implementation-2017-A044"
        response = client.get(f"/api/v1/articles/{selected.id}")
        assert response.status_code == 200
        assert response.json() == selected.model_dump(mode="json")

        compiled = client.get(
            "/api/v1/documents/hb-traffic", params={"version_id": "hb-traffic-2017"}
        )
        assert compiled.status_code == 200
        assert compiled.json()["coverage"] == "full_text"
        assert "非官方重发全文" in compiled.json()["coverage_note"]
        assert compiled.json()["articles"][55] == article

        police = client.get(
            "/api/v1/documents/hb-police-discretion",
            params={"version_id": "hb-police-discretion-2026"},
        )
        assert police.status_code == 200
        body = police.json()
        assert body["coverage"] == "selected_sections"
        assert "未收录其他管理册" in body["coverage_note"]
        assert len(body["articles"]) == 51
        assert body["articles"][19]["id"] == "hb-police-discretion-2026-G020"
        assert body["articles"][19]["source"]["pdf_page"] == 12
        assert body["articles"][-1]["id"] == "hb-police-discretion-2026-T029"
        # 问答产品不开放独立裁量接口。
        assert client.get("/api/v1/discretion/hb-police-discretion").status_code == 404
