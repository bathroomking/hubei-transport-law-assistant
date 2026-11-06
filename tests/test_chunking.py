import hashlib

import pytest
from pdf_fixtures import pdf_bytes
from pydantic import ValidationError

from transport_law.chunking import (
    ChunkingOptions,
    DraftChunk,
    build_draft_chunks,
    split_legal_documents,
)
from transport_law.errors import AppError
from transport_law.ingest import (
    DocumentPage,
    LoadedDocument,
    load_document,
    load_documents,
    to_documents,
)

pytest.importorskip("langchain_text_splitters")


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
        # 跨页只加入两个换行；分块在页间换行处起止时允许额外空白。
        assert "\n\n".join(source_parts).strip() == chunk.text.strip()
    for page in doc.pages:
        non_whitespace = {i for i, char in enumerate(page.text) if not char.isspace()}
        assert non_whitespace <= covered[page.pdf_page]


def test_articles_are_separate_and_short_articles_remain_intact():
    first = "第一条　这是用于分块测试的示例文字。\n"
    second = "第二条　本条中引用第一条的字样不应再分出一块。"
    doc = document(first + second)
    chunks = build_draft_chunks(doc, ChunkingOptions())
    assert [chunk.text for chunk in chunks] == [first, second]
    assert [chunk.heading for chunk in chunks] == ["第一条", "第二条"]
    assert_source_coverage(doc, chunks)


def test_long_article_has_overlap_without_mixing_the_next_article():
    first = "第一条　" + "甲乙丙丁戊己庚辛壬癸" * 30 + "。\n"
    doc = document(first + "第二条　后一个条文保持独立。")
    chunks = build_draft_chunks(doc, ChunkingOptions(chunk_size=80, chunk_overlap=15))
    first_chunks = [chunk for chunk in chunks if chunk.heading == "第一条"]
    assert len(first_chunks) > 1
    assert "第二条" not in "".join(chunk.text for chunk in first_chunks)
    spans = [chunk.source.locations[0] for chunk in first_chunks]
    assert any(
        right.start_index < left.end_index for left, right in zip(spans, spans[1:], strict=False)
    )
    assert_source_coverage(doc, chunks)


def test_cross_page_article_keeps_every_physical_page():
    doc = document("第一条　示例条文的前半段，", "这是本条的后半段。\n第二条　另一个示例。")
    chunks = build_draft_chunks(doc, ChunkingOptions())
    assert len(chunks) == 2
    assert [location.pdf_page for location in chunks[0].source.locations] == [1, 2]
    assert chunks[1].source.locations[0].pdf_page == 2
    assert_source_coverage(doc, chunks)


def test_missing_pdf_text_breaks_cross_page_join():
    doc = document("第一条　前页文字。", "   ", "缺失页之后的文字。")
    chunks = build_draft_chunks(doc, ChunkingOptions())
    assert len(chunks) == 2
    assert [chunk.source.locations[0].pdf_page for chunk in chunks] == [1, 3]
    assert_source_coverage(doc, chunks)


def test_repeated_text_has_stable_unique_ids_and_complete_coverage():
    doc = document("重复文字" * 150)
    options = ChunkingOptions(chunk_size=60, chunk_overlap=12)
    chunks = build_draft_chunks(doc, options)
    again = build_draft_chunks(doc, options)
    assert chunks == again
    assert len({chunk.id for chunk in chunks}) == len(chunks)
    assert_source_coverage(doc, chunks)


def test_ids_track_document_version_and_chunking_configuration():
    before = document("第一条　示例内容。")
    after = document("第一条　已修改的示例内容。")
    first = build_draft_chunks(before, ChunkingOptions(), document_id="example-law")[0]
    revised = build_draft_chunks(after, ChunkingOptions(), document_id="example-law")[0]
    resized = build_draft_chunks(
        before, ChunkingOptions(chunk_size=900), document_id="example-law"
    )[0]
    assert first.document_id == revised.document_id == "example-law"
    assert first.version_id != revised.version_id
    assert len({first.id, revised.id, resized.id}) == 3


def test_pdf_preserves_chinese_chapters_sections_articles_and_punctuation(tmp_path):
    text = "第一章 示例章\n引言。\n第一节 示例节\n第一条 示例一。\n第二条 示例二。"
    path = tmp_path / "sample.PDF"
    path.write_bytes(pdf_bytes(text))
    doc = load_document(path)
    chunks = build_draft_chunks(doc, ChunkingOptions())
    assert [chunk.heading for chunk in chunks] == ["第一章", "第一节", "第一条", "第二条"]
    assert chunks[0].source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert doc.format == "pdf"
    assert all(chunk.source.locations[0].pdf_page == 1 for chunk in chunks)
    assert_source_coverage(doc, chunks)


def test_markdown_markers_do_not_split_pdf_content():
    doc = document("# 文件中的普通文字\n正文。\n## 另一个普通段落\n后文。")
    [chunk] = build_draft_chunks(doc, ChunkingOptions())
    assert chunk.heading is None
    assert chunk.text == doc.pages[0].text


def test_document_metadata_survives_loading_splitting_and_storage(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(pdf_bytes("第一条　" + "测试文本，" * 60))
    pages = load_documents(path, document_id="sample-law", title="示例法规")
    assert pages[0].metadata["pdf_page"] == 1
    assert pages[0].metadata["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    parts = split_legal_documents(pages, ChunkingOptions(chunk_size=80, chunk_overlap=12))
    assert len(parts) > 1
    assert len({part.id for part in parts}) == len(parts)
    for part in parts:
        assert part.metadata["document_id"] == "sample-law"
        assert part.metadata["version_id"] == pages[0].metadata["version_id"]
        assert part.metadata["title"] == "示例法规"
        assert part.metadata["heading"] == "第一条"
        assert part.metadata["review_status"] == "needs_review"
        assert not any(key.startswith("_") for key in part.metadata)
        stored = DraftChunk.from_document(part)
        loaded = DraftChunk.model_validate_json(stored.model_dump_json()).to_document()
        assert loaded == part


def test_document_metadata_is_not_shared_or_modified_by_splitting():
    pages = to_documents(document("第一条　" + "示例文字" * 100))
    original = [page.model_dump() for page in pages]
    parts = split_legal_documents(pages, ChunkingOptions(chunk_size=80, chunk_overlap=12))
    assert [page.model_dump() for page in pages] == original
    parts[0].metadata["source"]["file"] = "changed.pdf"
    parts[0].metadata["source"]["locations"][0]["pdf_page"] = 99
    assert parts[1].metadata["source"]["file"] == "sample.pdf"
    assert parts[1].metadata["source"]["locations"][0]["pdf_page"] == 1
    assert [page.model_dump() for page in pages] == original


@pytest.mark.parametrize("boundary", ["another_document", "another_version", "missing_page"])
def test_batch_documents_do_not_join_unrelated_or_missing_pages(boundary):
    pages = to_documents(document("第一条　前半段", "后半段。"))
    if boundary == "another_document":
        pages[1].metadata["document_id"] = "another-document"
    elif boundary == "another_version":
        pages[1].metadata["version_id"] = "another-version"
    else:
        pages[1].metadata["pdf_page"] = 3
    chunks = split_legal_documents(pages, ChunkingOptions())
    assert [chunk.page_content for chunk in chunks] == ["第一条　前半段", "后半段。"]
    assert all(len(chunk.metadata["source"]["locations"]) == 1 for chunk in chunks)


def test_document_adapter_still_rejects_invalid_business_metadata():
    [part] = split_legal_documents(to_documents(document("示例文字")), ChunkingOptions())
    part.metadata["source"]["locations"][0]["pdf_page"] = 0
    with pytest.raises(ValidationError):
        DraftChunk.from_document(part)


@pytest.mark.parametrize("size, overlap", [(0, 0), (20001, 0), (100, -1), (100, 100), (100, 101)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValidationError):
        ChunkingOptions(chunk_size=size, chunk_overlap=overlap)


@pytest.mark.parametrize(
    "name, content, code",
    [
        ("example.txt", b"example", "unsupported_document"),
        ("example.md", b"example", "unsupported_document"),
        ("example.markdown", b"example", "unsupported_document"),
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


def test_loading_blank_pdf_keeps_physical_page(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(pdf_bytes(""))
    before = path.read_bytes()
    loaded = load_document(path)
    assert len(loaded.pages) == 1
    assert loaded.pages[0].pdf_page == 1 and not loaded.pages[0].text.strip()
    assert path.read_bytes() == before


def test_loading_pdf_and_cross_page_chunking_preserve_physical_source_locations(tmp_path):
    path = tmp_path / "multi.pdf"
    path.write_bytes(pdf_bytes("第一条 示例条文前半段，", "这是后半段。\n第二条 另一个条文。"))
    loaded = load_document(path)
    chunks = build_draft_chunks(loaded, ChunkingOptions())
    assert len(chunks) == 2
    assert [location.pdf_page for location in chunks[0].source.locations] == [1, 2]
    assert [location.pdf_page for location in chunks[1].source.locations] == [2]
    assert_source_coverage(loaded, chunks)


@pytest.mark.parametrize("page", [None, 0, -1])
def test_draft_source_requires_positive_pdf_page(page):
    [chunk] = build_draft_chunks(document("第一条 示例内容。"), ChunkingOptions())
    payload = chunk.model_dump()
    payload["source"]["locations"][0]["pdf_page"] = page
    with pytest.raises(ValidationError):
        DraftChunk.model_validate(payload)
