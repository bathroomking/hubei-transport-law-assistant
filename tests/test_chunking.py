import hashlib
from dataclasses import replace

import pytest
from pydantic import ValidationError

from transport_law.chunking import ChunkingOptions, SourceLocation, build_draft_chunks
from transport_law.ingest import DocumentPage, LoadedDocument


def document(*pages):
    return LoadedDocument(
        file="交通资料.pdf",
        sha256=hashlib.sha256("\f".join(pages).encode()).hexdigest(),
        format="pdf",
        pages=[DocumentPage(text=text, pdf_page=i) for i, text in enumerate(pages, 1)],
    )


def split(doc, **kwargs):
    chunks = build_draft_chunks(doc, ChunkingOptions(), **kwargs)
    for chunk in chunks:
        assert chunk.text == "".join(
            doc.pages[loc.pdf_page - 1].text[loc.start_index : loc.end_index]
            for loc in chunk.source.locations
        )
    return chunks


def reading_page(page, body):
    return f"湖北道路交通资料\n\n资料 01\n\n{body}\n\n阅读整理本 · 2026-11-06\n\n第 {page} 页"


def test_article_keeps_intro_list_and_refusal_clause_across_pages_without_size_cut():
    long_list = "（一）某项违法行为及其适用条件。\n" * 35
    first = "## 第六十六条\n\n处警告或者50元罚款：\n" + long_list
    second = "（六）驾驶非机动车在机动车之间穿行的；\n拒绝接受罚款处罚的，可以扣留该非机动车。"
    doc = document(
        reading_page(1, "## 条文正文\n\n" + first),
        reading_page(2, second + "\n\n## 第六十七条\n另一条文。"),
        reading_page(3, "## 版本与来源\n\n第八条 原始版本的说明。"),
    )
    law, following = split(doc)
    assert (law.content_type, law.heading, law.article_number) == ("article", "第六十六条", 66)
    assert len(law.text) > 500
    assert law.text == first + "\n\n" + second + "\n\n"
    assert [loc.pdf_page for loc in law.source.locations] == [1, 2]
    assert following.article_number == 67
    assert "版本与来源" not in following.text


def test_discretion_items_keep_conditions_punishment_and_original_locator():
    body = (
        "## 总则\n## 第二十二条\n说明。\n## 交通管理册\n"
        "以下按原表序号整理。\n## 交通管理册第 1 项\n违法行为：甲。\n"
        "违反条款：某法第十条。\n处罚依据：某法第九十六条。\n"
    )
    doc = document(
        reading_page(1, body),
        reading_page(
            2,
            "违法情节：乙；拘留期限：丙。\n原件位置：物理第200页。\n"
            "## 交通管理册第 2 项\n违法行为：丁。",
        ),
    )
    law, first, second = split(doc)
    assert [(c.heading, c.content_type) for c in (law, first, second)] == [
        ("第二十二条", "article"),
        ("交通管理册第1项", "item"),
        ("交通管理册第2项", "item"),
    ]
    assert first.article_number is None
    assert "处罚依据" in first.text and "原件位置" in first.text
    assert [p.pdf_page for p in first.source.locations] == [1, 2]


def test_toc_chapter_and_reference_in_text_do_not_start_or_extend_articles():
    doc = document(
        "目 录\n第一条 适用范围 …… 1\n第二条 规定 …… 2\n"
        "第一章 总则\n第一条\n本办法适用。\n依照本办法第二条规定办理。\n"
        "第二章 附则\n第二条 本办法施行。\n"
    )
    first, second = split(doc)
    assert [first.article_number, second.article_number] == [1, 2]
    assert first.text == "第一条\n本办法适用。\n依照本办法第二条规定办理。\n"
    assert second.text == "第二条 本办法施行。\n"


def test_same_line_plain_heading_supports_numeric_and_inserted_articles():
    doc = document("第 101 条　第一段。\n第一百零一条之一 本条新增。\n第102条\n另一条。")
    chunks = split(doc)
    assert [c.article_number for c in chunks] == [101, 101, 102]
    assert [c.heading for c in chunks] == ["第101条", "第一百零一条之一", "第102条"]
    assert len({c.id for c in chunks}) == 3


def test_general_document_keeps_unassigned_text_and_substantive_annex():
    doc = document(
        "现场办理说明\n本页包含重要的调查事项。\n第一条 规定了处理要求。\n"
        "附件1：处罚裁量表\n| 违法行为 | 处罚范围 |\n| 甲 | 乙 |\n",
        "附表二\n| 其他行为 | 其他范围 |\n"
        "第一条 附件中的配套规定。\n交通管理册第 1 项\n违法情节：丙。\n",
    )
    intro, law, table, next_table, attached_law, item = split(doc)
    assert (intro.content_type, intro.text) == ("page", "现场办理说明\n本页包含重要的调查事项。\n")
    assert law.text == "第一条 规定了处理要求。\n"
    assert table.content_type == next_table.content_type == "page"
    assert "| 甲 | 乙 |" in table.text and "| 其他行为 | 其他范围 |" in next_table.text
    assert attached_law.article_number == 1 and item.content_type == "item"


def test_reading_copy_excludes_versions_but_preserves_substantive_annex():
    doc = document(
        reading_page(1, "条文正文\n第一条 内容。\n附表：计算标准\n| 项目 | 数量 |"),
        reading_page(2, "| 罚款 | 50 |\n版本与来源\n第八条 仅为沿革说明。"),
    )
    law, table, continued = split(doc)
    assert law.text == "第一条 内容。\n"
    assert table.text.startswith("附表：") and "| 罚款 | 50 |" in continued.text
    assert "沿革" not in continued.text


def test_unstructured_pages_keep_every_character_including_table_and_ocr():
    doc = document(
        "  资料说明\r\n\r\n| a | b |\r\n| - | - |\r\n| 1 | 2 |\r\n",
        "",
        "  扫描页OCR的文字。  \n",
    )
    chunks = split(doc)
    assert [c.content_type for c in chunks] == ["page", "page"]
    assert [c.text for c in chunks] == [doc.pages[0].text, doc.pages[2].text]
    assert [c.source.locations[0].pdf_page for c in chunks] == [1, 3]
    assert split(document("", " \n")) == []


def test_ids_are_stable_and_change_with_document_version_or_snapshot_parser():
    doc = document("第一条 旧文。\n")
    first = split(doc, document_id="my-law", title="展示名")[0]
    same = split(doc, document_id="my-law", title="新展示名")[0]
    revised = split(document("第一条 新文。"), document_id="my-law")[0]
    parsed = split(replace(doc, parser="changed"), document_id="my-law")[0]
    assert first.id == same.id and first.document_id == revised.document_id
    assert first.version_id != revised.version_id
    assert len({first.id, revised.id, parsed.id, split(doc)[0].id}) == 4


def test_invalid_page_order_and_source_ranges_rejected():
    doc = document("第一条 标题", "疑似续页")
    doc.pages[1] = replace(doc.pages[1], pdf_page=3)
    with pytest.raises(ValueError, match="页码"):
        split(doc)
    with pytest.raises(ValidationError):
        SourceLocation(pdf_page=1, start_index=5, end_index=5)


def test_toc_recovers_article_whose_definition_begins_with_another_article_reference():
    doc = document(
        "目录\n第一条 适用范围 …… 1\n第二条 行为要求 …… 2\n"
        "第一条\n第九条所称车辆，包括以下车辆。\n"
        "第二条驾驶人应当减速。\n第九条规定的例外情形另行处理。\n"
        "第三条（罚则）处警告。\n"
    )
    first, second, third = split(doc)
    assert [chunk.article_number for chunk in (first, second, third)] == [1, 2, 3]
    assert "第九条所称车辆" in first.text
    assert "第九条规定的例外情形" in second.text
    assert third.text == "第三条（罚则）处警告。\n"


def test_mixed_pdf_keeps_ordinary_supplementary_pages_separate_from_reader_copy():
    doc = document(
        "独立办理说明\n该说明后附有法规原文。\n",
        reading_page(2, "## 条文正文\n\n第一条 法规正文。"),
        "独立补充材料\n| 项目 | 内容 |\n",
        reading_page(4, "## 版本与来源\n第九条 是版本说明。"),
        "后附说明\n实际勘验记录。",
    )
    intro, law, table, note = split(doc)
    assert [x.content_type for x in (intro, law, table, note)] == [
        "page",
        "article",
        "page",
        "page",
    ]
    assert [x.text for x in (intro, table, note)] == [doc.pages[i].text for i in (0, 2, 4)]
    assert law.text.strip() == "第一条 法规正文。"
