from pathlib import Path

from pydantic import Field

from transport_law.errors import AppError
from transport_law.retrieval import Retriever
from transport_law.schemas import Model, SearchRequest, Text


class EvalCase(Model):
    id: Text
    question: Text
    expected_ids: list[Text] = Field(default_factory=list)
    expect_empty: bool = False


def evaluate_retrieval(path: Path, retriever: Retriever, top_k: int = 5) -> dict:
    try:
        cases = [
            EvalCase.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not cases or len({case.id for case in cases}) != len(cases):
            raise ValueError("empty or duplicate cases")
        if any(case.expect_empty == bool(case.expected_ids) for case in cases):
            raise ValueError("each case needs either expected_ids or expect_empty")
    except (OSError, ValueError) as exc:
        raise AppError("invalid_eval_data", "评测集格式错误或为空。", 400) from exc
    results = []
    recalls = []
    for case in cases:
        request = SearchRequest(question=case.question, top_k=top_k)
        ids = [hit.chunk.id for hit in retriever.search(request.question, top_k)]
        if case.expect_empty:
            passed = not ids
        else:
            recall = len(set(ids) & set(case.expected_ids)) / len(set(case.expected_ids))
            recalls.append(recall)
            passed = recall == 1
        results.append({"id": case.id, "passed": passed, "retrieved_ids": ids})
    return {
        "total": len(results),
        "passed": sum(row["passed"] for row in results),
        "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "top_k": top_k,
        "results": results,
        "note": "开发回归集，只评估指定语料上的检索；不代表实际业务或模型回答准确率。",
    }
