"""在相同法规范围内并行召回 BM25 与 dense，再以父法条 ID 做 RRF。"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from transport_law.config import Settings
from transport_law.embeddings import Embeddings
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import HybridScore, SearchHit
from transport_law.vector_store import ChromaRetriever


def reciprocal_rank_fusion(
    bm25_hits: list[SearchHit],
    dense_hits: list[SearchHit],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[SearchHit]:
    if not 1 <= top_k <= 10 or rrf_k < 1:
        raise ValueError("top_k 必须在 1 到 10 之间，rrf_k 必须为正整数")
    chunks, scores, details = {}, {}, {}
    for name, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
        seen = set()
        for hit in hits:
            ident = hit.chunk.id
            if ident in seen:
                continue
            seen.add(ident)
            rank = len(seen)  # 重复子块不重复计分，也不占据父法条排名。
            chunks[ident] = hit.chunk
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (rrf_k + rank)
            detail = details.setdefault(ident, {"rrf_k": rrf_k})
            detail[f"{name}_rank"] = rank
            detail[f"{name}_score"] = hit.score
    # 分数相同用稳定 ID 排序；保留完整精度，不把小额分差提前舍入掉。
    return [
        SearchHit(chunk=chunks[ident], score=scores[ident], retrieval=HybridScore(**details[ident]))
        for ident in sorted(scores, key=lambda ident: (-scores[ident], ident))[:top_k]
    ]


def fuse_rankers(
    sparse: Callable[[], list[SearchHit]],
    dense: Callable[[], list[SearchHit]],
    *,
    top_k: int,
    rrf_k: int,
) -> list[SearchHit]:
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as pool:
        sparse_result = pool.submit(sparse)
        dense_result = pool.submit(dense)
        # 任一路失败向调用方报错，不把单路结果伪装成已完成的融合。
        return reciprocal_rank_fusion(
            sparse_result.result(), dense_result.result(), top_k=top_k, rrf_k=rrf_k
        )


class HybridRetriever:
    def __init__(
        self, repository: LawRepository, settings: Settings, embedder: Embeddings | None = None
    ):
        self.bm25 = BM25Retriever(repository)
        self.dense = ChromaRetriever(repository, settings, embedder, scope=self.bm25)
        self.candidate_k = settings.hybrid_candidate_k
        self.rrf_k = settings.hybrid_rrf_k

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        explicit = self.bm25.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1000) for chunk in explicit[:top_k]]
        candidates = self.bm25.candidate_chunks(question)
        if not candidates:
            return []
        return fuse_rankers(
            lambda: self.bm25.rank_candidates(question, candidates, limit=self.candidate_k),
            lambda: self.dense.rank_candidates(question, candidates, limit=self.candidate_k),
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
