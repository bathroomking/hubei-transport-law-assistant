import hashlib
from dataclasses import replace

import pytest
from pdf_fixtures import pdf_bytes
from pydantic import ValidationError

from transport_law.chunking import (
    PIPELINE_VERSION,
    ChunkingOptions,
    DraftChunk,
    build_draft_chunks,
    split_legal_documents,
)
from transport_law.errors import AppError
from transport_law.ingest import (
    PARSER_ID,
    DocumentPage,
    LoadedDocument,
    load_document,
    to_documents,
)


def document(*pages):
    return LoadedDocument(
        "sample.pdf",
        hashlib.sha256("".join(pages).encode()).hexdigest(),
        "pdf",
        [DocumentPage(text, number) for number, text in enumerate(pages, 1)],
    )


def assert_source_coverage(doc, chunks):
    covered = {page.pdf_page: set() for page in doc.pages}
    pages = {page.pdf_page: page.text for page in doc.pages}
    for chunk in chunks:
        assert 0 < len(chunk.text) <= chunk.chunking.chunk_size
        source_parts = []
        for location in chunk.source.locations:
            page = pages[location.pdf_page]
            assert 0 <= location.start_index < location.end_index <= len(page)
            source_parts.append(page[location.start_index : location.end_index])
            covered[location.pdf_page].update(range(location.start_index, location.end_index))
        assert "".join(source_parts) == chunk.text
    for page in doc.pages:
        non_whitespace = {i for i, char in enumerate(page.text) if not char.isspace()}
        assert non_whitespace <= covered[page.pdf_page]


def test_default_contract_matches_configuration():
    options = ChunkingOptions()
    assert PIPELINE_VERSION == (
        f"{PARSER_ID}:{options.strategy}:{options.chunk_size}:{options.chunk_overlap}"
    )


def test_nested_generic_headings_keep_context_and_do_not_mix_siblings():
    doc = document("# 总览\n概述。\n\n## 设备\n设备正文。\n\n### 维护\n保养。\n## 人员\n培训。")
    chunks = build_draft_chunks(doc, ChunkingOptions())
    assert [c.heading for c in chunks] == [
        "总览",
        "总览 / 设备",
        "总览 / 设备 / 维护",
        "总览 / 人员",
    ]
    assert not any("保养" in c.text and "培训" in c.text for c in chunks)
    assert len({c.parent_id for c in chunks}) == 4
    assert_source_coverage(doc, chunks)


def test_plain_legal_references_are_not_artificial_heading_boundaries():
    doc = document("第一条 与第二条都是引用文字。\n第二条 不代表这个 PDF 的章节标题。")
    [chunk] = build_draft_chunks(doc, ChunkingOptions())
    assert chunk.heading is None
    assert chunk.text == doc.pages[0].text


def test_paragraphs_with_same_heading_merge_without_changing_source():
    doc = document("# 中文报告\n\n首段。\n\n次段。")
    [chunk] = build_draft_chunks(doc, ChunkingOptions())
    assert chunk.text == doc.pages[0].text
    assert_source_coverage(doc, [chunk])


def test_long_paragraph_splits_at_chinese_sentences_and_keeps_overlap():
    doc = document("# 主题\n" + "甲乙丙丁戊己庚辛。" * 30)
    options = ChunkingOptions(chunk_size=60, chunk_overlap=10)
    chunks = build_draft_chunks(doc, options)
    assert len(chunks) > 3
    spans = [c.source.locations[0] for c in chunks]
    assert any(b.start_index < a.end_index for a, b in zip(spans, spans[1:], strict=False))
    assert all(c.heading == "主题" for c in chunks)
    assert all(c.text.endswith("。") for c in chunks[1:-1])
    assert_source_coverage(doc, chunks)


def test_pages_are_not_blindly_joined_and_empty_page_is_not_skipped():
    doc = document("# 设备\n前半段，", "  ", "后半段。")
    chunks = build_draft_chunks(doc, ChunkingOptions())
    assert [c.source.locations[0].pdf_page for c in chunks] == [1, 3]
    assert all(len(c.source.locations) == 1 for c in chunks)
    assert chunks[1].heading is None
    assert_source_coverage(doc, chunks)


def test_repeated_text_has_stable_unique_ids_and_coverage():
    doc = document("重复文字" * 150)
    options = ChunkingOptions(chunk_size=60, chunk_overlap=12)
    chunks = build_draft_chunks(doc, options)
    assert chunks == build_draft_chunks(doc, options)
    assert len({c.id for c in chunks}) == len(chunks)
    assert len({c.parent_id for c in chunks}) == 1
    assert_source_coverage(doc, chunks)


def test_ids_track_document_version_parser_and_configuration():
    before = document("# Report\nOriginal.")
    revised = document("# Report\nChanged.")
    first = build_draft_chunks(before, ChunkingOptions(), document_id="example")[0]
    changed = build_draft_chunks(revised, ChunkingOptions(), document_id="example")[0]
    resized = build_draft_chunks(before, ChunkingOptions(chunk_size=900), document_id="example")[0]
    pages = to_documents(before, document_id="example")
    pages[0].metadata["source"]["parser"] = "different-parser-v1"
    reparsed = DraftChunk.from_document(split_legal_documents(pages, ChunkingOptions())[0])
    assert first.document_id == changed.document_id == "example"
    assert first.version_id != changed.version_id
    assert len({c.id for c in [first, changed, resized, reparsed]}) == 4
    assert first.parent_id != reparsed.parent_id
    renamed = build_draft_chunks(
        replace(before, file="renamed.pdf"), ChunkingOptions(), document_id="example"
    )[0]
    assert renamed.id == first.id


def test_table_rows_remain_together_headers_repeat_and_context_survives():
    header = "| 产品 | 价格 |\n| --- | --- |\n"
    rows = [f"| 产品{i} | {i}00元 |\n" for i in range(8)]
    doc = document("# 商品\n\n" + header + "".join(rows) + "\n结束。")
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=65, chunk_overlap=5))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) > 1
    assert all(c.text.startswith(header) and c.heading == "商品" for c in tables)
    assert len({c.parent_id for c in tables}) == 1
    assert all(sum(row in c.text for c in tables) == 1 for row in rows)
    assert all("结束" not in c.text for c in tables)
    assert any(len(c.source.locations) == 2 for c in tables)
    assert_source_coverage(doc, chunks)


def test_table_without_outer_pipes_and_inline_line_breaks():
    text = "Name | Value\n:--- | ---:\nA<br>B | 10\nC | 20\n"
    doc = document(text)
    [chunk] = build_draft_chunks(doc, ChunkingOptions())
    assert chunk.content_type == "table"
    assert chunk.text == text
    assert_source_coverage(doc, [chunk])


@pytest.mark.parametrize(
    "text", ["A | B\nnot a table", "A | B\n--- | text\n", "A | B\n--- | --- | ---\n"]
)
def test_pipe_text_without_valid_separator_is_not_a_table(text):
    chunks = build_draft_chunks(document(text), ChunkingOptions())
    assert all(c.content_type == "text" for c in chunks)


def test_escaped_pipes_in_table_cells_do_not_change_header_count():
    doc = document("| A \\| B | C |\n| --- | --- |\n| one | two |\n")
    [chunk] = build_draft_chunks(doc, ChunkingOptions())
    assert chunk.content_type == "table"
    assert_source_coverage(doc, [chunk])


def test_compact_parser_tables_keep_single_dash_separator_and_repeat_header():
    header = "| 编号 | 说明 | 结果 |\n| - | - | - |\n"
    rows = [f"| 记录{i} | 说明{i} | 结果{i} |\n" for i in range(8)]
    doc = document("## 结构\n\n" + header + "".join(rows))
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=100, chunk_overlap=10))
    tables = [c for c in chunks if c.content_type == "table"]
    assert len(tables) > 1
    assert all(c.text.startswith(header) for c in tables)
    assert all(sum(row in c.text for c in tables) == 1 for row in rows)
    assert_source_coverage(doc, chunks)


def test_oversized_table_row_repeats_header_without_exceeding_cap():
    header = "| Field | Value |\n| --- | --- |\n"
    doc = document(header + "| A | " + "长文本。" * 60 + " |\n| B | Done |\n")
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=80, chunk_overlap=10))
    assert len(chunks) > 3
    assert all(c.content_type == "table" and c.text.startswith(header) for c in chunks)
    long_row = [c for c in chunks if "| A |" in c.text]
    assert len(long_row) > 1
    assert len({c.parent_id for c in long_row}) == 1
    assert long_row[0].parent_id != chunks[-1].parent_id
    assert all(len(c.text.splitlines()) == 3 for c in long_row)
    assert_source_coverage(doc, chunks)


def test_oversized_table_header_is_kept_as_linked_bounded_fragments():
    header = "| " + "超长字段" * 30 + " | Value |\n| --- | --- |\n"
    doc = document(header + "| A | 100 |\n| B | 200 |\n")
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=45, chunk_overlap=5))
    assert len(chunks) > 2
    assert all(c.content_type == "table" for c in chunks)
    # 两行各自关联完整的表头碎片，不把不同行的片段误认为同一行。
    assert len({c.parent_id for c in chunks}) == 2
    assert_source_coverage(doc, chunks)


def test_wide_rows_project_columns_and_preserve_short_adjacent_field_pairs():
    header = "| Key | Name | Details | Policy | Condition | Outcome |\n"
    separator = "| --- | --- | --- | --- | --- | --- |\n"
    rows = [
        f"| {key} | Widget | "
        + "详细说明。" * 140
        + " | "
        + "操作要求。" * 120
        + f" | condition-{key} | outcome-{key} |\n"
        for key in ("item-A", "item-B")
    ]
    doc = document("# Inventory\n\n" + header + separator + "".join(rows))
    chunks = build_draft_chunks(doc, ChunkingOptions())
    tables = [c for c in chunks if c.content_type == "table"]
    parents = []
    for key in ("item-A", "item-B"):
        fragments = [c for c in tables if f"| {key} |" in c.text]
        assert len(fragments) > 3
        assert all(c.heading == "Inventory" for c in fragments)
        assert len({c.parent_id for c in fragments}) == 1
        parents.append(fragments[0].parent_id)
        assert any(
            c.text
            == (
                "| Key | Condition | Outcome |\n| --- | --- | --- |\n"
                f"| {key} | condition-{key} | outcome-{key} |\n"
            )
            for c in fragments
        )
        details = [c for c in fragments if "| Details |" in c.text]
        assert len(details) > 1
        for chunk in fragments:
            lines = chunk.text.splitlines()
            assert len(lines) == 3
            assert lines[0].count("|") == lines[1].count("|") == lines[2].count("|")
    assert parents[0] != parents[1]
    assert len(tables) == sum("| item-A |" in c.text or "| item-B |" in c.text for c in tables)
    assert_source_coverage(doc, chunks)


def test_oversized_rows_without_outer_pipes_and_missing_final_newline():
    text = "Key | Description | Result\n--- | --- | ---\nID-A | " + "detail " * 80 + "| Passed"
    doc = document(text)
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=130, chunk_overlap=10))
    assert len(chunks) > 3
    assert all("ID-A |" in c.text for c in chunks)
    assert any("Result\n" in c.text and "Passed" in c.text for c in chunks)
    assert all(len(c.text.splitlines()) == 3 for c in chunks)
    assert_source_coverage(doc, chunks)


def test_short_tail_columns_stay_together_when_previous_long_cell_almost_fills_chunk():
    text = (
        "| Key | Description | Setting | Result |\n| - | - | - | - |\n"
        "| A | " + "x" * 112 + " | enabled | accepted |\n"
    )
    doc = document(text)
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=180, chunk_overlap=10))
    assert len(chunks) == 2
    assert any(
        c.text == "| Key | Setting | Result |\n| - | - | - |\n| A | enabled | accepted |\n"
        for c in chunks
    )
    assert len({c.parent_id for c in chunks}) == 1
    assert_source_coverage(doc, chunks)


def test_malformed_oversized_table_rows_fallback_without_losing_cells():
    text = "| Key | Value |\n| - | - |\n| item | " + "data" * 100 + " | extra |\n"
    doc = document(text)
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=100, chunk_overlap=0))
    assert len(chunks) > 2
    assert len({c.parent_id for c in chunks}) == 1
    assert_source_coverage(doc, chunks)


@pytest.mark.parametrize("size", [75, 80, 90, 120])
def test_long_cell_fragments_do_not_turn_escaped_pipes_into_new_columns(size):
    text = "| Key | Value |\n| - | - |\n| A | " + "abc\\|def" * 60 + " |\n"
    doc = document(text)
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=size, chunk_overlap=0))
    for chunk in chunks:
        lines = chunk.text.splitlines()
        assert len(lines) == 3
        row = lines[-1]
        # 去掉源文本中的转义符号后，只应剩下两列的三个真实分隔符。
        assert row.replace("\\|", "").count("|") == 3
        assert "| A |" in row
    assert_source_coverage(doc, chunks)


def test_header_only_table_and_tiny_budget_do_not_drop_or_loop():
    doc = document("| Name | Value |\n| --- | --- |\n")
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=1, chunk_overlap=0))
    assert all(len(c.text) == 1 for c in chunks)
    assert_source_coverage(doc, chunks)


def test_header_near_cap_does_not_multiply_data_into_single_character_chunks():
    header = "| " + "column" * 8 + " | value |\n| --- | --- |\n"
    text = header + "| A | " + "payload" * 100 + " |\n"
    size = len(header) + 1
    doc = document(text)
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=size, chunk_overlap=0))
    assert len(chunks) < 20
    assert len({c.parent_id for c in chunks}) == 1
    assert_source_coverage(doc, chunks)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 31, 64, 100, 500])
def test_mixed_structure_size_extremes_preserve_content_and_bounds(size):
    doc = document(
        "# 总览\n\n中英混排 Mixed text; αβγ，大小限制。\n\n"
        "- first item\n  continuation\n- second item\n\n"
        "| 列 A | 列 B |\n| --- | --- |\n| 单元格甲 | " + "内容" * 90 + " |\n\n"
        "```python\n# a comment\nprint('hello')\n```\n\n## 小节\n末段。"
    )
    chunks = build_draft_chunks(
        doc, ChunkingOptions(chunk_size=size, chunk_overlap=min(size - 1, 5))
    )
    assert len({c.id for c in chunks}) == len(chunks)
    assert_source_coverage(doc, chunks)


def test_adjacent_tables_have_distinct_parents():
    doc = document("| A | B |\n| --- | --- |\n| 1 | 2 |\n| C | D |\n| --- | --- |\n| 3 | 4 |\n")
    chunks = build_draft_chunks(doc, ChunkingOptions())
    assert len(chunks) == 2
    assert chunks[0].parent_id != chunks[1].parent_id
    assert_source_coverage(doc, chunks)


def test_lists_preserve_item_boundaries_and_continuations():
    items = [f"{i}. 检查设备{i}及记录。\n  补充说明{i}。\n" for i in range(1, 6)]
    doc = document("# 操作\n\n" + "".join(items) + "\n结束正文。")
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=70, chunk_overlap=5))
    lists = [c for c in chunks if c.content_type == "list"]
    assert len(lists) >= 2
    assert all(sum(item in c.text for c in lists) == 1 for item in items)
    assert all(c.heading == "操作" for c in lists)
    assert all("结束正文" not in c.text for c in lists)
    assert_source_coverage(doc, chunks)


def test_fenced_code_does_not_parse_internal_headings_or_tables():
    code = "```python\n# Not a heading\n| A | B |\n| --- | --- |\nprint('ok')\n```\n"
    doc = document("# Code example\n\n" + code + "\n## Next\nDone")
    chunks = build_draft_chunks(doc, ChunkingOptions())
    code_chunks = [c for c in chunks if c.content_type == "code"]
    assert len(code_chunks) == 1 and code_chunks[0].text == code
    assert code_chunks[0].heading == "Code example"
    assert not any(c.content_type == "table" for c in chunks)
    assert_source_coverage(doc, chunks)


def test_oversized_code_keeps_lines_when_possible_and_unclosed_fence_keeps_text():
    doc = document("~~~text\n" + "完整的一行文本。\n" * 15 + "x" * 100)
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=40, chunk_overlap=5))
    assert len(chunks) > 3
    assert all(c.content_type == "code" for c in chunks)
    assert len({c.parent_id for c in chunks}) == 1
    assert_source_coverage(doc, chunks)


def test_document_adapter_roundtrip_and_metadata_independence():
    pages = to_documents(
        document("# 示例\n" + "测试文字，" * 60), document_id="sample", title="样本"
    )
    original = [p.model_dump() for p in pages]
    parts = split_legal_documents(pages, ChunkingOptions(chunk_size=80, chunk_overlap=12))
    assert len(parts) > 1
    for part in parts:
        loaded = DraftChunk.from_document(part)
        assert loaded.to_document() == part
        assert loaded.document_id == "sample" and loaded.heading == "示例"
    parts[0].metadata["source"]["file"] = "changed.pdf"
    parts[0].metadata["source"]["locations"][0]["pdf_page"] = 99
    assert parts[1].metadata["source"]["file"] == "sample.pdf"
    assert [p.model_dump() for p in pages] == original


@pytest.mark.parametrize("boundary", ["another_document", "another_version", "missing_page"])
def test_batch_documents_do_not_join_different_sources(boundary):
    pages = to_documents(document("前半段", "后半段"))
    if boundary == "another_document":
        pages[1].metadata["document_id"] = "other"
    elif boundary == "another_version":
        pages[1].metadata["version_id"] = "other"
    else:
        pages[1].metadata["pdf_page"] = 3
    chunks = split_legal_documents(pages, ChunkingOptions())
    assert [c.page_content for c in chunks] == ["前半段", "后半段"]
    assert len({c.metadata["parent_id"] for c in chunks}) == 2


def test_legacy_cache_is_readable_but_retired_strategy_cannot_generate():
    [chunk] = build_draft_chunks(document("原始文字"), ChunkingOptions())
    data = chunk.model_dump()
    data.pop("parent_id")
    data.pop("content_type")
    data["source"].pop("parser")
    data["chunking"]["strategy"] = "pdf_legal_recursive_v1"
    loaded = DraftChunk.model_validate(data)
    assert loaded.source.parser == "legacy-pypdf" and loaded.parent_id is None
    with pytest.raises(AppError, match="旧分块策略"):
        build_draft_chunks(document("原始文字"), loaded.chunking)


@pytest.mark.parametrize("size, overlap", [(0, 0), (20001, 0), (100, -1), (100, 100), (100, 101)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValidationError):
        ChunkingOptions(chunk_size=size, chunk_overlap=overlap)


@pytest.mark.parametrize(
    "page,start,end", [(None, 0, 1), (0, 0, 1), (-1, 0, 1), (1, 1, 1), (1, 2, 1)]
)
def test_source_locations_require_valid_page_and_nonempty_ranges(page, start, end):
    [chunk] = build_draft_chunks(document("原始文字"), ChunkingOptions())
    payload = chunk.model_dump()
    payload["source"]["locations"][0] = {"pdf_page": page, "start_index": start, "end_index": end}
    with pytest.raises(ValidationError):
        DraftChunk.model_validate(payload)


@pytest.mark.parametrize(
    "name,content,code",
    [
        ("example.txt", b"example", "unsupported_document"),
        ("example.md", b"example", "unsupported_document"),
        ("example.docx", b"example", "unsupported_document"),
        ("bad.pdf", b"not a PDF", "pdf_read_failed"),
    ],
)
def test_unreadable_documents_fail(tmp_path, name, content, code):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(AppError) as error:
        load_document(path)
    assert error.value.code == code


def test_blank_documents_have_no_chunks():
    assert build_draft_chunks(document("", "  "), ChunkingOptions()) == []


def test_real_pdf_extraction_and_chunking_keep_page_source_and_original_bytes(tmp_path):
    path = tmp_path / "multi.PDF"
    path.write_bytes(pdf_bytes("设备检查需要记录。", "施工结束需要检查。"))
    original = path.read_bytes()
    loaded = load_document(path)
    chunks = build_draft_chunks(loaded, ChunkingOptions())
    assert {loc.pdf_page for c in chunks for loc in c.source.locations} == {1, 2}
    assert chunks[0].source.sha256 == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original
    assert_source_coverage(loaded, chunks)
