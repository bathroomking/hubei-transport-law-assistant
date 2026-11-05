import hashlib
import json
import runpy

import pytest

from transport_law.chunking import DraftChunk
from transport_law.config import PROJECT_ROOT
from transport_law.errors import AppError

pytest.importorskip("langchain_text_splitters")


@pytest.fixture
def process_corpus():
    return runpy.run_path(str(PROJECT_ROOT / "scripts/chunk_legal_corpus.py"))["process_corpus"]


@pytest.fixture
def example_corpus(tmp_path, repository):
    # 两个不同父条目可以有完全相同的正文；它们的来源和身份不可合并丢失。
    parent = repository.get_article("cn-traffic-safety-2021-A001").model_dump(mode="json")
    source = tmp_path / "original.html"
    source.write_text("测试来源快照", encoding="utf-8")
    parent["source"].update(
        file=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    amendment = tmp_path / "amendment.html"
    amendment.write_text("测试修改决定", encoding="utf-8")
    parent["source"]["supporting_sources"] = [
        {
            "title": "测试修改决定",
            "file": amendment.name,
            "sha256": hashlib.sha256(amendment.read_bytes()).hexdigest(),
            "official_url": "https://example.com/amendment",
            "locator": "第一项",
        }
    ]
    parent["text"] = "第一条　" + "用于检查完整覆盖和重叠片段的示例正文。" * 20
    other = {**parent, "id": "another-parent"}
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in [parent, other]) + "\n",
        encoding="utf-8",
    )
    return corpus, [parent, other]


def test_batch_preserves_provenance_and_exact_positions(tmp_path, process_corpus, example_corpus):
    corpus, parents = example_corpus
    before = corpus.read_bytes()
    output = tmp_path / "batch"
    report = process_corpus(corpus, output, chunk_size=80, chunk_overlap=12, source_root=tmp_path)
    assert corpus.read_bytes() == before
    assert report["input_record_count"] == 2
    assert report["chunk_count"] > 2
    assert report["status"] == "staged_not_published"
    assert len(report["verified_source_files"]) == 2
    parents = {parent["id"]: parent for parent in parents}
    rows = [
        json.loads(line) for line in (output / "langchain_documents.jsonl").read_text().splitlines()
    ]
    assert len({row["id"] for row in rows}) == len(rows)
    assert {row["metadata"]["provenance"]["id"] for row in rows} == set(parents)
    for row in rows:
        metadata = row["metadata"]
        parent = parents[metadata["provenance"]["id"]]
        assert metadata["provenance"]["source"] == parent["source"]
        assert metadata["version_id"] == parent["version_id"]
        text = (output / metadata["source"]["file"]).read_text()
        [location] = metadata["source"]["locations"]
        assert location["pdf_page"] is None
        assert row["page_content"] == text[location["start_index"] : location["end_index"]]
    for line in (output / "draft_chunks.jsonl").read_text().splitlines():
        DraftChunk.model_validate_json(line)
    assert len(list((output / "documents").glob("*.jsonl"))) == 1


def test_batch_is_reproducible_and_never_overwrites(tmp_path, process_corpus, example_corpus):
    corpus, _ = example_corpus
    first, second = tmp_path / "first", tmp_path / "second"
    process_corpus(corpus, first, source_root=tmp_path)
    process_corpus(corpus, second, source_root=tmp_path)
    for name in ["draft_chunks.jsonl", "langchain_documents.jsonl", "manifest.json"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with pytest.raises(AppError, match="已经存在"):
        process_corpus(corpus, first, source_root=tmp_path)


@pytest.mark.parametrize("filename", ["original.html", "amendment.html"])
def test_batch_rejects_changed_original_or_supporting_source(
    tmp_path, process_corpus, example_corpus, filename
):
    corpus, _ = example_corpus
    (tmp_path / filename).write_text("已变化的文件", encoding="utf-8")
    output = tmp_path / "batch"
    with pytest.raises(AppError, match="校验值不符"):
        process_corpus(corpus, output, source_root=tmp_path)
    assert not output.exists()
