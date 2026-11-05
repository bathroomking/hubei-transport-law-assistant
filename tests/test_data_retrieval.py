import pytest
from pydantic import ValidationError

from transport_law.config import PROJECT_ROOT, Settings
from transport_law.errors import AppError
from transport_law.repository import ChunkRepository
from transport_law.retrieval import BM25Retriever, tokenize
from transport_law.schemas import SearchRequest


def test_keyword_ranking_uses_uploaded_text_and_exact_ids(retriever):
    hits = retriever.search("施工许可", top_k=1)
    assert [hit.chunk.id for hit in hits] == ["UP-fixture-1"]
    assert retriever.search("请查看 UP-fixture-2")[0].chunk.id == "UP-fixture-2"
    assert retriever.search("zxq987654") == []


def test_legacy_classification_aliases_and_excerpt_links_do_not_change_bm25(repository):
    expected = BM25Retriever(repository).search("施工许可")
    changed = ChunkRepository(
        [
            c.model_copy(
                update={
                    "item_id": "old-item",
                    "aliases": ["施工许可"],
                    "document_aliases": ["另一名称"],
                    "related_excerpt_ids": ["UP-fixture-1"],
                }
            )
            for c in repository.all_chunks
        ]
    )
    actual = BM25Retriever(changed).search("施工许可")
    assert [(h.chunk.id, h.score) for h in actual] == [(h.chunk.id, h.score) for h in expected]


def test_no_keyword_match_is_not_padded(repository):
    base = repository.all_chunks[0]
    data = ChunkRepository(
        [
            base.model_copy(update={"id": ident, "text": text, "search_text": text})
            for ident, text in [
                ("sample-a", "alpha"),
                ("sample-b", "alpha alpha"),
                ("other", "omega"),
            ]
        ]
    )
    hits = BM25Retriever(data).search("alpha")
    assert {h.chunk.id for h in hits} == {"sample-a", "sample-b"}
    assert all(h.score > 0 for h in hits)


def test_tokenization_preserves_english_word_boundaries_and_chinese_layout():
    assert tokenize("Road permit\nrequired") == ["road", "permit", "required"]
    assert tokenize("公 路\n许可") == tokenize("公路许可")


@pytest.mark.parametrize("top_k", [0, 11])
def test_bm25_rejects_invalid_top_k(retriever, top_k):
    with pytest.raises(ValueError):
        retriever.search("施工许可", top_k)


def test_duplicate_id_rejected(repository):
    chunk = repository.all_chunks[0]
    with pytest.raises(AppError, match="重复"):
        ChunkRepository([chunk, chunk])


def test_unknown_chunk_is_explicit_404(repository):
    with pytest.raises(AppError) as exc:
        repository.get_article("missing")
    assert exc.value.code == "article_not_found" and exc.value.status_code == 404


@pytest.mark.parametrize(
    "data",
    [
        {"question": "   "},
        {"question": "公路", "top_k": 0},
        {"question": "公路", "top_k": 11},
        {"question": "字" * 2001},
    ],
)
def test_request_boundaries(data):
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(data)


def test_secret_mask_and_relative_storage_paths(settings):
    configured = Settings(
        _env_file=None,
        deepseek_api_key="test-private-key",
        knowledge_path="sample-knowledge",
        chroma_path="sample-chroma",
    )
    assert "test-private-key" not in repr(configured)
    assert configured.knowledge_path == PROJECT_ROOT / "sample-knowledge"
    assert configured.chroma_path == PROJECT_ROOT / "sample-chroma"
    assert "law_data_path" not in Settings.model_fields
    assert "chroma_collection" not in Settings.model_fields
    assert not settings.llm_configured
