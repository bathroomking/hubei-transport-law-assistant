"""正式法规及可选上传资料共享 BM25、向量排名和 RRF 融合。"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from transport_law.config import Settings
from transport_law.embeddings import DashScopeEmbeddings, Embeddings
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import HybridScore, SearchHit
from transport_law.vector_store import ChromaRetriever

if TYPE_CHECKING:
    from transport_law.knowledge import KnowledgeStore


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
        self,
        repository: LawRepository,
        settings: Settings,
        embedder: Embeddings | None = None,
        *,
        knowledge: "KnowledgeStore | None" = None,
    ):
        self.repository = repository
        self.settings = settings
        self.embedder = embedder or DashScopeEmbeddings(settings)
        self.knowledge = knowledge
        self.bm25 = BM25Retriever(repository)
        self._snapshot = ([], self.bm25)

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        # 每次请求读取 ready 快照；只有上传内容变化时重建联合词频统计。
        uploaded = self.knowledge.ready_chunks() if self.knowledge is not None else []
        previous, bm25 = self._snapshot
        if uploaded != previous:
            bm25 = (
                BM25Retriever(LawRepository(self.repository.searchable + uploaded))
                if uploaded
                else self.bm25
            )
            self._snapshot = (uploaded, bm25)
        explicit = bm25.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1000) for chunk in explicit[:top_k]]
        candidates = bm25.candidate_chunks(question)
        if not candidates:
            return []
        uploaded_ids = {chunk.id for chunk in uploaded}
        formal = [chunk for chunk in candidates if chunk.id not in uploaded_ids]
        extra = [chunk for chunk in candidates if chunk.id in uploaded_ids]
        limit = self.settings.hybrid_candidate_k

        def dense():
            # 在请求问题向量前检查正式索引；缺索引不影响页面启动或精确 ID 读取。
            retriever = (
                ChromaRetriever(self.repository, self.settings, self.embedder, scope=bm25)
                if formal
                else None
            )
            vector = self.embedder.embed_query(question)
            hits = []
            if retriever is not None:
                hits.extend(
                    retriever.rank_candidates(question, formal, limit=limit, query_vector=vector)
                )
            if extra:
                hits.extend(self.knowledge._rank_vectors(extra, limit=limit, query_vector=vector))
            # 两座库共用一次 embedding；相同 cosine 度量合并成全局 dense 排名。
            return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]

        return fuse_rankers(
            lambda: bm25.rank_candidates(question, candidates, limit=limit),
            dense,
            top_k=top_k,
            rrf_k=self.settings.hybrid_rrf_k,
        )
