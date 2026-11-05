import hashlib
import json
import re

import pytest
from pydantic import ValidationError

from transport_law.config import PROJECT_ROOT, Settings
from transport_law.errors import AppError
from transport_law.evaluation import evaluate_retrieval
from transport_law.ingest import extract_pdf
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import SearchRequest


def test_regression_dataset(retriever):
    report = evaluate_retrieval(PROJECT_ROOT / "evals/retrieval.jsonl", retriever)
    assert report["total"] == 20
    assert report["passed"] == report["total"], report["results"]
    assert report["recall_at_k"] == 1


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("公路法第四十四条第一款", "cn-highway-2017-A044"),
        ("公路法第44条在资料里是怎么写的？", "cn-highway-2017-A044"),
        ("公路法第76条", "cn-highway-2017-A076"),
        ("公路安全保护条例第27条", "cn-highway-protection-2011-A027"),
        ("HB-ROAD-003-REG62", "HB-ROAD-003-REG62"),
    ],
)
def test_top_k_and_exact_article_rank(retriever, question, expected):
    hits = retriever.search(question, 1)
    assert len(hits) == 1
    assert hits[0].chunk.id == expected


def test_superseded_entries_are_not_retrieved(repository):
    chunks = [chunk.model_copy(deep=True) for chunk in repository.searchable]
    for chunk in chunks:
        chunk.source.verification_status = "superseded"
    assert BM25Retriever(LawRepository(chunks)).search("挖掘公路") == []


@pytest.mark.parametrize("raw", ["", "{}\n", "not-json\n"])
def test_bad_data_fails_at_load(tmp_path, raw):
    path = tmp_path / "bad.jsonl"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(AppError, match="为空|格式不正确"):
        LawRepository.from_jsonl(path)


def test_duplicate_id_rejected(repository):
    chunk = repository.searchable[0]
    with pytest.raises(AppError, match="重复"):
        LawRepository([chunk, chunk])


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


def test_secret_mask_and_relative_data_path(settings):
    configured = Settings(
        _env_file=None, deepseek_api_key="test-private-key", law_data_path="x.jsonl"
    )
    assert "test-private-key" not in repr(configured)
    assert configured.law_data_path == PROJECT_ROOT / "x.jsonl"
    assert not settings.llm_configured


def test_seed_provenance_matches_actual_pdf(repository):
    pypdf = pytest.importorskip("pypdf")
    path = PROJECT_ROOT / "data/raw/P020250918567517504506.pdf"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = pypdf.PdfReader(path)
    assert len(reader.pages) == 215
    text = re.sub(r"\s+", "", reader.pages[1].extract_text())
    for chunk in repository.searchable:
        if chunk.document_id != "hubei-discretion-upload":
            continue
        assert chunk.source.sha256 == digest
        assert chunk.source.pdf_page == 2
        assert chunk.source.verification_status == "pending"
        assert chunk.source.effective_on is None
        # 表格多栏文本提取顺序不稳定，独立核对关键原文，不将全文串接当成版式核验。
    assert "任何单位和个人不得擅自占用、挖掘公路" in text
    assert "可处1万元以上2万元以下罚款" in text


def test_pdf_extraction_preserves_pages_and_does_not_overwrite(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    path = tmp_path / "empty.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=400)
    with path.open("wb") as stream:
        writer.write(stream)
    output = tmp_path / "pages.jsonl"
    result = extract_pdf(path, output)
    assert result["page_count"] == result["needs_ocr_count"] == 1
    row = json.loads(output.read_text())
    assert row["pdf_page"] == 1 and row["status"] == "needs_ocr"
    original = output.read_bytes()
    with pytest.raises(AppError, match="已经存在"):
        extract_pdf(path, output)
    assert output.read_bytes() == original
