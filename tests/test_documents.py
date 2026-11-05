import pytest

from transport_law.documents import LawDocument, load_law_document
from transport_law.errors import AppError
from transport_law.repository import ChunkRepository


def test_uploaded_document_contains_all_chunks_in_source_order(repository):
    document = load_law_document(
        ChunkRepository(list(reversed(repository.all_chunks))), "upload-fixture"
    )
    assert isinstance(document, LawDocument)
    assert document.version_id == "sha256-fixture"
    assert document.title == "施工说明"
    assert document.coverage == "uploaded_document"
    assert [c.id for c in document.articles] == ["UP-fixture-1", "UP-fixture-2", "UP-fixture-3"]
    assert "不表示已收录整部法规" in document.coverage_note


def test_uploaded_pdf_sorting_uses_page_and_character_position(repository):
    base = repository.all_chunks[0]
    chunks = [
        base.model_copy(
            update={
                "id": ident,
                "source": base.source.model_copy(
                    update={
                        "source_type": "pdf",
                        "pdf_page": page,
                        "locator": f"字符 [{offset}, {offset + 50})",
                    }
                ),
            }
        )
        for ident, page, offset in [("UP-a", 2, 0), ("UP-c", 1, 200), ("UP-z", 1, 0)]
    ]
    result = load_law_document(ChunkRepository(chunks), "upload-fixture")
    assert [c.id for c in result.articles] == ["UP-z", "UP-c", "UP-a"]


def test_versions_are_selected_and_never_merged(repository):
    current = repository.all_chunks[0]
    old = current.model_copy(update={"id": "UP-old", "version_id": "old-version", "text": "旧文本"})
    repo = ChunkRepository([current, old])
    with pytest.raises(AppError) as exc:
        load_law_document(repo, "upload-fixture")
    assert exc.value.code == "document_version_required"
    assert exc.value.status_code == 409
    assert load_law_document(repo, "upload-fixture", "old-version").articles == [old]
    assert load_law_document(repo, "upload-fixture", current.version_id).articles == [current]


@pytest.mark.parametrize(
    "document_id, version, code",
    [
        ("missing", None, "document_not_found"),
        ("../../.env", None, "document_not_found"),
        ("upload-fixture", "missing", "document_version_not_found"),
        ("upload-fixture", "../../.env", "document_version_not_found"),
    ],
)
def test_unknown_document_or_version_returns_404(repository, document_id, version, code):
    with pytest.raises(AppError) as exc:
        load_law_document(repository, document_id, version)
    assert exc.value.code == code and exc.value.status_code == 404


def test_document_reading_does_not_access_a_source_manifest(repository, monkeypatch):
    from pathlib import Path

    def unexpected_read(*args, **kwargs):
        pytest.fail("原文展示只能使用已入库片段，不得读取手工清单")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    result = load_law_document(repository, "upload-fixture")
    assert len(result.articles) == 3 and result.coverage == "uploaded_document"
