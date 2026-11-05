"""将已整理的全部法条/表项交给 Document 分块流程，输出可追溯的候选批次。"""

import argparse
import hashlib
import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from transport_law.chunking import ChunkingOptions, DraftChunk, split_legal_documents
from transport_law.config import PROJECT_ROOT
from transport_law.errors import AppError
from transport_law.ingest import load_documents, write_jsonl
from transport_law.repository import LawRepository
from transport_law.schemas import LegalChunk


def filename(identifier: str) -> str:
    """保留可读编号，加哈希避免冒号转换后发生文件名冲突。"""
    suffix = hashlib.sha256(identifier.encode()).hexdigest()[:10]
    return f"{identifier.replace(':', '_')[:160]}-{suffix}"


def verify_sources(parents: list[LegalChunk], source_root: Path) -> list[dict]:
    checked = {}
    for parent in parents:
        for source in [parent.source, *parent.source.supporting_sources]:
            key = (source.file, source.sha256)
            if key in checked:
                continue
            paths = [source_root / source.file, source_root / "data/raw" / source.file]
            path = next((candidate for candidate in paths if candidate.is_file()), None)
            if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != source.sha256:
                raise AppError("source_mismatch", f"来源文件缺失或校验值不符：{source.file}", 400)
            checked[key] = {"file": source.file, "sha256": source.sha256}
    return list(checked.values())


def process_corpus(
    corpus: Path,
    output: Path,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    source_root: Path = PROJECT_ROOT,
) -> dict:
    if output.exists():
        raise AppError("output_exists", "批次目录已经存在，请选择新的输出目录。", 409)
    options = ChunkingOptions(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    corpus_bytes = corpus.read_bytes()
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    parents = [
        LegalChunk.model_validate_json(line)
        for line in corpus_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not parents:
        raise AppError("invalid_data", "没有可处理的已整理条目。", 400)
    LawRepository(parents)  # 检查重复 ID；保留输入记录各自的核验状态。
    sources = verify_sources(parents, source_root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    draft_rows, document_rows = [], []
    groups = defaultdict(list)
    summaries = {}
    seen_ids = set()
    split_parent_ids = []
    for parent in parents:
        # 固化真正送入 splitter 的正文。位置属于这份 TXT，不能冒充原 PDF 页内位置。
        input_file = Path("inputs") / f"{filename(parent.id)}.txt"
        (output / input_file).write_text(parent.text, encoding="utf-8")
        pages = load_documents(
            output / input_file,
            document_id=parent.document_id,
            title=parent.source.document_title,
        )
        for page in pages:
            page.metadata["source"]["file"] = input_file.as_posix()
            page.metadata["version_id"] = parent.version_id
        parts = split_legal_documents(pages, options)
        if len(parts) > 1:
            split_parent_ids.append(parent.id)
        covered = set()
        key = (parent.document_id, parent.version_id)
        if key not in summaries:
            summaries[key] = {
                "document_id": parent.document_id,
                "version_id": parent.version_id,
                "title": parent.source.document_title,
                "input_records": 0,
                "chunk_count": 0,
            }
        summaries[key]["input_records"] += 1
        for part in parts:
            # 同一文件中的相同文字也可能属于不同法条，子块 ID 需限定在父记录内。
            identity = json.dumps([parent.id, part.id], ensure_ascii=False)
            part.id = f"AUTO-{hashlib.sha256(identity.encode()).hexdigest()[:32]}"
            part.metadata["heading"] = parent.article
            draft = DraftChunk.from_document(part)
            if draft.id in seen_ids:
                raise AppError("duplicate_chunk", "批次中出现重复分块 ID。", 500)
            seen_ids.add(draft.id)
            if len(draft.source.locations) != 1 or len(draft.text) > options.chunk_size:
                raise AppError("invalid_chunk", "候选片段的位置或长度不符合约定。", 500)
            location = draft.source.locations[0]
            if (
                location.pdf_page is not None
                or parent.text[location.start_index : location.end_index] != draft.text
            ):
                raise AppError("chunk_location_failed", "片段无法回查到已整理原文。", 500)
            covered.update(range(location.start_index, location.end_index))
            draft_rows.append(draft.model_dump(mode="json"))
            result = draft.to_document()
            # 官方链接、真实 PDF 页码、条号、修改依据和业务分类都完整保留。
            result.metadata["provenance"] = parent.model_dump(mode="json", exclude={"text"})
            row = result.model_dump(mode="json")
            document_rows.append(row)
            groups[key].append(row)
            summaries[key]["chunk_count"] += 1
        required = {index for index, char in enumerate(parent.text) if not char.isspace()}
        if not required <= covered:
            raise AppError("incomplete_chunking", f"分块未覆盖完整正文：{parent.id}", 500)
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != corpus_sha256:
        raise AppError("corpus_changed", "处理期间知识库发生变化，请基于新快照重新运行。", 409)
    write_jsonl(output / "draft_chunks.jsonl", draft_rows)
    write_jsonl(output / "langchain_documents.jsonl", document_rows)
    for key, rows in groups.items():
        group_file = Path("documents") / f"{filename(':'.join(key))}.jsonl"
        write_jsonl(output / group_file, rows)
        summaries[key]["output"] = group_file.as_posix()
    report = {
        "status": "staged_not_published",
        "scope": "全部已整理的法条和裁量表项；不扩展原 PDF 尚未收录的内容。",
        "input_corpus": str(corpus.resolve()),
        "input_corpus_sha256": corpus_sha256,
        "document_count": len({parent.document_id for parent in parents}),
        "document_version_count": len(groups),
        "input_record_count": len(parents),
        "chunk_count": len(draft_rows),
        "max_chunk_characters": max(len(row["text"]) for row in draft_rows),
        "split_parent_ids": split_parent_ids,
        "chunking": options.model_dump(),
        "id_strategy": "sha256(parent_id, segment_id)",
        "dependencies": {
            name: version(name) for name in ["langchain-core", "langchain-text-splitters"]
        },
        "validation": {
            "all_input_records_processed": True,
            "all_non_whitespace_covered": True,
            "all_chunk_locations_checked": True,
            "unique_chunk_ids": True,
            "input_corpus_unchanged": True,
        },
        "verified_source_files": sources,
        "documents": list(summaries.values()),
        "outputs": {
            name: {"file": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest()}
            for name in ["draft_chunks.jsonl", "langchain_documents.jsonl"]
        },
        "metadata_note": (
            "source.locations 是 inputs/ 中固化正文的字符位置；"
            "metadata.provenance.source 保留真正法源的网页/PDF 定位和修改依据。"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT_ROOT / "data/processed/legal_chunks.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()
    report = process_corpus(
        args.corpus,
        args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ["status", "document_count", "input_record_count", "chunk_count"]
            }
            | {"manifest": str(args.output_dir / "manifest.json")},
            ensure_ascii=False,
            indent=2,
        )
    )
