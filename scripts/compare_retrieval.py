"""三路检索对照；公开回归题每题只生成一次向量，不调用 DeepSeek。"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from transport_law.config import PROJECT_ROOT, Settings
from transport_law.embeddings import DashScopeEmbeddings
from transport_law.errors import AppError
from transport_law.evaluation import EvalCase, evaluate_retrieval
from transport_law.hybrid import HybridRetriever
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.vector_store import ChromaRetriever, index_contract


class QueryCache:
    """仅在一次评测内存中共享查询向量，不保存用户问题或向量缓存。"""

    def __init__(self, settings):
        self.client = DashScopeEmbeddings(settings)
        self.vectors = {}
        self.lock = threading.Lock()

    def embed_query(self, question):
        with self.lock:
            cached = self.vectors.get(question)
        if cached is not None:
            return cached
        vector = self.client.embed_query(question)
        with self.lock:
            self.vectors[question] = vector
        return vector

    def embed_documents(self, texts):
        raise AssertionError("检索评测不得重建文档向量")


class CachedResults:
    def __init__(self, results, backend):
        self.results, self.backend = results, backend

    def search(self, question, top_k=5):
        return self.results[question][self.backend][:top_k]


def compare(output: Path, datasets: list[Path], workers: int = 3) -> dict:
    if output.exists():
        raise ValueError("报告已存在，请指定新的输出文件")
    cases = {
        str(path): [
            EvalCase.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        for path in datasets
    }
    questions = list(dict.fromkeys(case.question for rows in cases.values() for case in rows))
    settings = Settings()
    repository = LawRepository.from_jsonl(settings.law_data_path)
    embedder = QueryCache(settings)
    retrievers = {
        "bm25": BM25Retriever(repository),
        "chroma": ChromaRetriever(repository, settings, embedder),
        "hybrid": HybridRetriever(repository, settings, embedder),
    }
    results, durations, errors = {}, {}, {}
    started = time.monotonic()

    def search(question):
        hits, seconds = {}, {}
        for backend, retriever in retrievers.items():
            start = time.monotonic()
            hits[backend] = retriever.search(question, top_k=5)
            seconds[backend] = round(time.monotonic() - start, 4)
        return hits, seconds

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(search, question): question for question in questions}
        for future in as_completed(futures):
            question = futures[future]
            try:
                results[question], durations[question] = future.result()
            except AppError as exc:
                errors[question] = {"code": exc.code, "message": exc.message}
            completed = len(results) + len(errors)
            if completed % 10 == 0 or completed == len(questions):
                print(f"检索对照：{completed}/{len(questions)}，错误：{len(errors)}", flush=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "index": index_contract(settings, repository),
        "vector_count": retrievers["chroma"].collection.count(),
        "candidate_k": settings.hybrid_candidate_k,
        "rrf_k": settings.hybrid_rrf_k,
        "query_embeddings": len(embedder.vectors),
        "unique_queries": len(questions),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "query_seconds": durations,
        "timing_note": "Chroma 和 hybrid 共用一次问题向量，耗时不能用于比较在线延迟。",
        "errors": errors,
        "datasets": {},
        "hybrid_hits": {
            question: [
                {"id": hit.chunk.id, "score": hit.score, "retrieval": hit.retrieval.model_dump()}
                if hit.retrieval
                else {"id": hit.chunk.id, "score": hit.score, "retrieval": None}
                for hit in hits["hybrid"]
            ]
            for question, hits in results.items()
        },
    }
    for path in datasets:
        if any(case.question in errors for case in cases[str(path)]):
            continue
        comparison = {
            backend: evaluate_retrieval(path, CachedResults(results, backend))
            for backend in retrievers
        }
        report["datasets"][str(path)] = comparison
        print(
            json.dumps(
                {
                    "dataset": path.name,
                    **{
                        name: {k: metrics[k] for k in ("total", "passed", "recall_at_k")}
                        for name, metrics in comparison.items()
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = parser.parse_args()
    datasets = args.dataset or [
        PROJECT_ROOT / "evals" / f"{name}.jsonl"
        for name in ("retrieval", "official_retrieval", "traffic_police_retrieval")
    ]
    try:
        report = compare(args.output, datasets, args.workers)
    except AppError as exc:
        print(json.dumps({"error": exc.code, "message": exc.message}, ensure_ascii=False))
        raise SystemExit(1) from None
    print(f"报告：{args.output}")
    raise SystemExit(1 if report["errors"] else 0)
