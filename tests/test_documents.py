import json

import pytest

from transport_law.config import PROJECT_ROOT
from transport_law.documents import LawDocument, load_law_document
from transport_law.errors import AppError
from transport_law.repository import LawRepository


def test_full_document_reads_all_articles_in_numeric_order(repository):
    # The returned law must not be limited to a search top_k or insertion order.
    reversed_repository = LawRepository(list(reversed(repository.all_chunks)))
    document = load_law_document(reversed_repository, "cn-traffic-safety")
    assert isinstance(document, LawDocument)
    assert document.version_id == "cn-traffic-safety-2021"
    assert document.title == "中华人民共和国道路交通安全法"
    assert document.coverage == "full_text"
    assert [article.article_number for article in document.articles] == list(range(1, 125))
    assert "现行效力或个案适用已核验" in document.coverage_note
    assert all(article.source.official_url for article in document.articles)


def test_all_cataloged_laws_report_actual_collection_scope(repository):
    catalog = json.loads((PROJECT_ROOT / "data/raw/laws/manifest.json").read_text())
    for entry in catalog["documents"]:
        document = load_law_document(repository, entry["document_id"], entry["id"])
        expected = "selected_sections" if entry.get("table_item_count") else "full_text"
        assert document.coverage == expected
        assert len(document.articles) == entry["article_count"] + entry.get("table_item_count", 0)
        assert entry["version_label"] in document.coverage_note


def test_police_discretion_keeps_general_articles_before_tables(repository):
    document = load_law_document(
        LawRepository(list(reversed(repository.all_chunks))), "hb-police-discretion"
    )
    assert document.coverage == "selected_sections"
    assert "未收录其他管理册" in document.coverage_note
    assert "仅细化行政拘留" in document.coverage_note
    assert [article.article_number for article in document.articles[:22]] == list(range(1, 23))
    assert [article.id for article in document.articles[22:]] == [
        f"hb-police-discretion-2026-T{number:03d}" for number in range(1, 30)
    ]


def test_original_five_excerpts_use_the_source_document_title(repository):
    document = load_law_document(repository, "hubei-discretion-upload")
    assert document.coverage == "excerpts"
    assert len(document.articles) == 5
    assert document.title == document.articles[0].source.document_title
    assert document.title != document.articles[0].title
    assert "不能作为整部法规全文" in document.coverage_note
    assert document.articles[0].id == "HB-ROAD-003-LAW44"


@pytest.mark.parametrize("document_id", ["cn-highway", "hb-police-discretion"])
def test_incomplete_repository_does_not_claim_catalog_completeness(repository, document_id):
    articles = [c for c in repository.all_chunks if c.document_id == document_id]
    document = load_law_document(LawRepository(articles[1:]), document_id)
    assert document.coverage == "excerpts"


def test_duplicate_article_number_is_not_full_text(repository):
    articles = [c for c in repository.all_chunks if c.document_id == "cn-highway"]
    articles[-1] = articles[-1].model_copy(update={"article_number": 1})
    assert load_law_document(LawRepository(articles), "cn-highway").coverage == "excerpts"


def test_other_source_with_catalog_id_cannot_claim_full_text(repository):
    articles = [c for c in repository.all_chunks if c.document_id == "cn-highway"]
    articles[0] = articles[0].model_copy(
        update={"source": articles[0].source.model_copy(update={"sha256": "a" * 64})}
    )
    assert load_law_document(LawRepository(articles), "cn-highway").coverage == "excerpts"


def test_versions_must_be_selected_and_never_merged(repository):
    current = repository.get_article("cn-highway-2017-A001")
    archived = current.model_copy(
        update={
            "id": "cn-highway-historical-A001",
            "version_id": "historical",
            "text": "另一版本的条文。",
            "source": current.source.model_copy(update={"verification_status": "superseded"}),
        }
    )
    mixed = LawRepository([current, archived])
    with pytest.raises(AppError) as exc:
        load_law_document(mixed, "cn-highway")
    assert exc.value.status_code == 409
    assert exc.value.code == "document_version_required"
    assert "historical" in exc.value.message
    old = load_law_document(mixed, "cn-highway", "historical")
    assert [c.id for c in old.articles] == [archived.id]
    assert old.articles[0].source.verification_status == "superseded"
    assert old.coverage == "excerpts"
    assert load_law_document(mixed, "cn-highway", current.version_id).articles == [current]


@pytest.mark.parametrize(
    ("document_id", "version_id", "code"),
    [
        ("missing", None, "document_not_found"),
        ("../../.env", None, "document_not_found"),
        ("cn-highway", "missing", "document_version_not_found"),
        ("cn-highway", "../../.env", "document_version_not_found"),
    ],
)
def test_unknown_documents_and_versions_return_404(repository, document_id, version_id, code):
    with pytest.raises(AppError) as exc:
        load_law_document(repository, document_id, version_id)
    assert exc.value.status_code == 404
    assert exc.value.code == code


@pytest.mark.parametrize("manifest", [None, "bad json", "[]", '{"documents": "invalid"}'])
def test_missing_or_invalid_catalog_falls_back_to_excerpts(
    repository, monkeypatch, tmp_path, manifest
):
    path = tmp_path / "manifest.json"
    if manifest is not None:
        path.write_text(manifest)
    monkeypatch.setattr("transport_law.documents.MANIFEST_PATH", path)
    document = load_law_document(repository, "cn-highway")
    assert document.coverage == "excerpts"
    assert len(document.articles) == 87


def test_uploaded_documents_are_sorted_by_source_location_not_hash(repository):
    original = repository.get_article("HB-ROAD-003-LAW44")
    articles = []
    for identifier, page, offset in [("UP-a", 2, 0), ("UP-c", 1, 200), ("UP-z", 1, 0)]:
        articles.append(
            original.model_copy(
                update={
                    "id": identifier,
                    "document_id": "upload-1",
                    "kind": "document",
                    "source": original.source.model_copy(
                        update={"pdf_page": page, "locator": f"字符 [{offset}, {offset + 100})"}
                    ),
                }
            )
        )
    document = load_law_document(LawRepository(articles), "upload-1")
    assert document.coverage == "uploaded_document"
    assert "不表示已收录整部法规" in document.coverage_note
    assert [c.id for c in document.articles] == ["UP-z", "UP-c", "UP-a"]


def test_uploaded_kind_cannot_be_mislabeled_as_catalog_full_text(repository):
    articles = [c for c in repository.all_chunks if c.document_id == "cn-highway"]
    articles = [c.model_copy(update={"kind": "document"}) for c in articles]
    assert load_law_document(LawRepository(articles), "cn-highway").coverage == "uploaded_document"
