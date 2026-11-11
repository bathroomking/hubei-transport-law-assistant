import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from transport_law.knowledge import KnowledgeStore
from transport_law.llm import Completion
from transport_law.main import create_app
from transport_law.prompts import build_qa_messages
from transport_law.relations import (
    agency_search_terms,
    build_article_relations,
    expand_article_hits,
)
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import SearchHit


@pytest.fixture
def chunk(repository):
    template = repository.all_chunks[0]

    def make(ident, title, heading, number, text):
        return template.model_copy(
            update={
                "id": ident,
                "document_id": title.split("_")[0],
                "version_id": "v1",
                "title": title,
                "article": heading,
                "article_number": number,
                "text": text,
                "search_text": text,
                "source": template.source.model_copy(),
            }
        )

    return make


ROAD = "07_中华人民共和国公路法_2017"
TRAFFIC = "05_中华人民共和国道路交通安全法_2021"
LOCAL = "01_湖北省实施道路交通安全法办法_2017合并整理"
MANUAL = "17_湖北省公安机关行政处罚裁量基准_总则及交通管理册_2026"


@pytest.fixture
def highway(chunk):
    return [
        chunk("rule", ROAD, "第四十五条", 45, "在公路用地范围内埋设管线应事先经同意。"),
        chunk(
            "penalty",
            ROAD,
            "第七十六条",
            76,
            "由交通主管部门责令停止违法行为，可以处罚款：\n（二）违反本法第四十五条规定，埋设管线的；",
        ),
        *[chunk(f"other-{n}", "misc_手册", f"第{n}条", n, "申请材料。") for n in range(1, 5)],
    ]


@pytest.fixture
def discretion(chunk):
    return [
        chunk(
            "item",
            MANUAL,
            "交通管理册第5项",
            None,
            "违法行为：伪造、变造机动车保险标志的\n违反条款：《中华人民共和国道路交通安全法》第十六条第三项"
            "\n处罚依据：《中华人民共和国道路交通安全法》第九十六条第二款、"
            "《湖北省实施&lt;中华人民共和国道 路交通安全法&gt;办法》第七十六条第二款"
            "\n违法情节：轻微。",
        ),
        chunk("duty", TRAFFIC, "第十六条", 16, "禁止伪造机动车保险标志。"),
        chunk("national", TRAFFIC, "第九十六条", 96, "伪造机动车保险标志的处罚。"),
        chunk("local", LOCAL, "第七十六条", 76, "地方处罚规定。"),
    ]


def expanded(question, chunks, ranked, k=5):
    index = {c.id: c for c in chunks}
    hits = [SearchHit(chunk=index[i], score=1) for i in ranked[:k]]
    result = expand_article_hits(question, hits, index, build_article_relations(chunks), k)
    return [hit.chunk.id for hit in result]


def test_explicit_references_keep_the_original_clause_and_edition(chunk, highway, discretion):
    extra = [
        chunk("alt", "20_公路法_2004", "第四十五条", 45, "旧版。"),
        chunk("extra", ROAD, "第四十五条之一", 45, "增设条文。"),
        chunk("compilation", "mixed_交通法规汇编", "第九十六条", 96, "同号条文。"),
        chunk("two", TRAFFIC, "第四十条", 40, "交通管制。"),
        chunk("one", TRAFFIC, "第三十九条", 39, "限制通行。"),
        chunk(
            "control",
            MANUAL,
            "交通管理册第21项",
            None,
            "违反条款：《中华人民共和国道路交通安全法》第三十九条、第四十条\n"
            "处罚依据：《未入库的规定》第三十九条\n违法情节：一般。",
        ),
    ]
    chunks = highway + discretion + extra
    relations = build_article_relations(chunks)
    assert {(r.source_id, r.target_id, r.kind) for r in relations} == {
        ("penalty", "rule", "highway_penalty"),
        ("item", "duty", "violation_basis"),
        ("item", "national", "penalty_basis"),
        ("item", "local", "penalty_basis"),
        ("control", "one", "violation_basis"),
        ("control", "two", "violation_basis"),
    }
    by_id = {c.id: c for c in chunks}
    assert all(r.evidence in by_id[r.source_id].text for r in relations)
    # 两份同名法规的版本无法从裁量表的引用字段区分；同法内的指向仍限于原版。
    old = chunk("old", "21_中华人民共和国道路交通安全法_2003", "第九十六条", 96, "旧条文。")
    assert {r.target_id for r in build_article_relations([*chunks, old])} == {"rule", "local"}
    old.source.verification_status = "superseded"
    assert build_article_relations([*chunks, old]) == relations


def test_ambiguous_headings_and_legacy_fragments_are_skipped(chunk, highway):
    rule, penalty = highway[:2]
    repeated = rule.model_copy(update={"id": "duplicate"})
    assert not build_article_relations([rule, penalty, repeated])
    assert not build_article_relations([rule.model_copy(update={"article_number": None}), penalty])
    extra = chunk("sub", ROAD, "第四十五条之一", 45, "条之一。")
    penalty = penalty.model_copy(update={"text": "违反本法第45条之一规定的，可以处罚款。"})
    assert [r.target_id for r in build_article_relations([rule, extra, penalty])] == ["sub"]


def test_fixed_budget_keeps_seed_and_skips_irrelevant_intents(highway):
    ranked = ["rule", "other-1", "other-2", "other-3", "other-4"]
    question = "公路用地范围内埋设管线怎么处理？"
    assert expanded(question, highway, ranked) == [*ranked[:2], "penalty", *ranked[2:4]]
    assert expanded(question, highway, ranked, k=1) == ["rule"]
    assert expanded(question, highway, ranked, k=2) == ["rule", "penalty"]
    included = ["rule", "other-1", "penalty", "other-2", "other-3"]
    assert expanded(question, highway, included) == included
    for other in (
        "埋设管线要提交哪些材料？",
        "公路上酒驾怎么罚？",
        "公路上停车吗？\nA. 埋设管线怎么罚",
    ):
        assert expanded(other, highway, ranked) == ranked


def test_discretion_requires_matching_conduct_and_local_scope(discretion):
    ranked = ["item"]
    assert expanded("未放置机动车保险标志怎么罚？", discretion, ranked) == ranked
    assert expanded("使用伪造机动车保险标志怎么罚？", discretion, ranked) == ranked
    assert expanded("伪造机动车保险标志会被怎么处罚？", discretion, ranked) == ["item", "national"]
    assert expanded("在湖北，伪造机动车保险标志怎么罚？", discretion, ranked) == ["item", "local"]
    assert expanded("交通管理册第五项违反哪条？", discretion, ranked) == ["item", "duty"]
    assert expanded("在湖北，交通管理册第6项处罚依据是什么？", discretion, ranked) == ranked
    assert expanded("在湖北，交通管理册第5项怎么罚？", discretion, ["local", "item"]) == [
        "local",
        "item",
        "national",
    ]


def test_seizure_link_requires_both_texts_in_one_edition_and_matching_question(chunk):
    a = chunk("final", TRAFFIC, "第九十一条", 91, "醉酒驾驶机动车的，吊销机动车驾驶证。")
    b = chunk(
        "procedure",
        TRAFFIC,
        "第一百一十条",
        110,
        "认为应当给予暂扣或者吊销机动车驾驶证处罚的，可\n以先予扣留机动车驾驶证。",
    )
    assert expanded("醉驾能否先扣证？", [a, b], ["final"]) == ["final", "procedure"]
    assert expanded("醉驾，交管部门可以先把驾照扣下吗？", [a, b], ["final"]) == [
        "final",
        "procedure",
    ]
    assert expanded("醉驾最终如何处罚？", [a, b], ["final"]) == ["final"]
    assert expanded("无证驾驶能扣留驾驶证吗？", [a, b], ["final"]) == ["final"]
    assert not build_article_relations([a, b.model_copy(update={"version_id": "v2"})])
    assert not build_article_relations([a, b.model_copy(update={"text": "其他规定。"})])


def test_agency_terms_only_for_subject_and_action_in_the_question():
    question = "醉驾时交管部门能不能先把驾照扣下？"
    assert agency_search_terms(question).startswith(question + " ")
    assert "行政强制措施" in agency_search_terms(question)
    for other in (
        "醉驾会吊销驾驶证吗？",
        "交管部门能否扣留机动车？",
        "谁管醉驾？\nA. 交管部门扣留驾驶证",
    ):
        assert agency_search_terms(other) == other


def test_relation_hint_requires_both_visible_sources_and_an_exact_excerpt(highway):
    (edge,) = build_article_relations(highway)
    for docs, relation, shown in (
        (highway, edge, True),
        (highway[:1], edge, False),
        (highway[1:], edge, False),
        (highway, edge._replace(evidence="不在原文中的引用。"), False),
    ):
        prompt = build_qa_messages("该如何处理？", docs, article_relation=relation)[1]["content"]
        assert ("本轮关联原文" in prompt) == shown


def seed(store, chunks):
    with store._connect() as db:
        for doc in {c.document_id for c in chunks}:
            count = sum(c.document_id == doc for c in chunks)
            db.execute(
                "INSERT INTO documents (id,filename,name_key,suffix,status,contract_hash,"
                "created_at,updated_at,chunk_count) VALUES (?,?,?,'.pdf','ready','test','a','a',?)",
                (doc, doc + ".pdf", doc, count),
            )
        db.executemany(
            "INSERT INTO chunks VALUES (?,?,?)",
            [(c.id, c.document_id, c.model_dump_json()) for c in chunks],
        )


def test_existing_db_backfills_once_and_ready_transitions_refresh(
    settings, discretion, monkeypatch
):
    store = KnowledgeStore(settings)
    seed(store, discretion)
    calls = []

    def build(chunks):
        calls.append(len(chunks))
        return build_article_relations(chunks)

    monkeypatch.setattr("transport_law.knowledge.build_article_relations", build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        snapshots = list(pool.map(lambda _: store.ready_evidence(), range(2)))
    assert calls == [4] and snapshots[0] == snapshots[1]
    assert len(snapshots[0][1]) == 3
    other = KnowledgeStore(settings)
    assert other.ready_evidence() == snapshots[0] and calls == [4]
    store._update("05", status="queued", cache_key="reprocessing")
    chunks, relations = other.ready_evidence()
    assert {c.id for c in chunks} == {"item", "local"}
    assert [r.target_id for r in relations] == ["local"]
    store._update("05", status="ready")
    assert other.ready_evidence() == snapshots[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_ask_and_clickable_sources_use_same_budget_and_original_text(
    online_settings, highway, monkeypatch, enabled
):
    settings = online_settings.model_copy(update={"article_relations_enabled": enabled})
    seed(KnowledgeStore(settings), highway)
    order = ["rule", "other-1", "other-2", "other-3", "other-4"]

    def dense(self, chunks, **kwargs):
        by_id = {c.id: c for c in chunks}
        return [SearchHit(chunk=by_id[i], score=1) for i in order]

    monkeypatch.setattr(KnowledgeStore, "rank_vectors", dense)
    monkeypatch.setattr("transport_law.embeddings.DashScopeEmbeddings.embed_query", lambda *_: [1])
    monkeypatch.setattr(BM25Retriever, "rank_candidates", lambda *args, **kwargs: [])

    class Model:
        async def complete(self, messages, *, json_mode):
            ident = "penalty" if enabled else "rule"
            assert f'"id": "{ident}"' in messages[-1]["content"]
            assert ("本轮关联原文" in messages[-1]["content"]) == enabled
            return Completion(
                json.dumps({"claims": [{"text": "依据所引原文。", "citation_ids": [ident]}]}),
                "fixture",
            )

    with TestClient(create_app(settings, Model())) as client:
        result = client.post(
            "/api/v1/ask", json={"question": "未经同意埋设管线怎么罚？", "top_k": 5}
        )
        assert result.status_code == 200
        answer = result.json()
        expected = ["rule", "other-1", "penalty", "other-2", "other-3"] if enabled else order
        assert [c["id"] for c in answer["source_documents"]] == expected
        (source,) = answer["sources"]
        assert source == client.get("/api/v1/articles/" + source["id"]).json()
