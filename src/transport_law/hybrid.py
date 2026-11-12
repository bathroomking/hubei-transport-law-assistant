"""对已完成入库的上传片段执行 BM25、向量排名和 RRF 融合。"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from transport_law.config import Settings
from transport_law.embeddings import DashScopeEmbeddings, Embeddings
from transport_law.repository import ChunkRepository
from transport_law.retrieval import BM25Retriever, normalize
from transport_law.schemas import HybridScore, SearchHit

if TYPE_CHECKING:
    from transport_law.knowledge import KnowledgeStore


def _search_terms(question: str) -> str:
    """车道处罚提问兼查一般通行罚则；新增词只用于召回，不作为用户陈述。"""
    q = normalize(question)
    # “汽车占用非机动车道”里的车道名称不能用来推断驾驶人所骑的车辆。
    subject = q.replace("非机动车道", "").replace("自行车道", "")
    cycling = any(word in subject for word in ("非机动车", "自行车", "单车")) or (
        "骑" in subject and any(word in subject for word in ("电动车", "电瓶车"))
    )
    if not cycling:
        return question
    query = question
    if "斜穿" in q or "横穿" in q:
        # 同时检索横过道路的规则；行驶方向本身不足以确定横过或在车辆之间穿行。
        query += " 横过机动车道 下车推行"
    if any(word in q for word in ("车道", "穿行", "借道")) and any(
        word in q for word in ("罚", "怎么处理", "如何处理", "怎么判", "裁量")
    ):
        query += " 非机动车驾驶人 道路通行规定 警告 罚款"
    return query


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
            rank = len(seen)  # 相同片段不重复计分，也不占据排名。
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
        knowledge: "KnowledgeStore",
        settings: Settings,
        embedder: Embeddings | None = None,
    ):
        self.knowledge = knowledge
        self.settings = settings
        self.embedder = embedder or DashScopeEmbeddings(settings)
        self._snapshot = ([], BM25Retriever(ChunkRepository([])))

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        # 同一次检索共用一份 ready 快照；上传完成后下一次请求即可检索。
        chunks = self.knowledge.ready_chunks()
        previous, bm25 = self._snapshot
        if chunks != previous:
            bm25 = BM25Retriever(ChunkRepository(chunks))
            self._snapshot = (chunks, bm25)
        explicit = bm25.explicit_chunks(question)
        if explicit:
            return [SearchHit(chunk=chunk, score=1000) for chunk in explicit[:top_k]]
        if not chunks:
            return []
        limit = self.settings.hybrid_candidate_k
        query = _search_terms(question)

        def dense():
            vector = self.embedder.embed_query(query)
            return self.knowledge.rank_vectors(chunks, limit=limit, query_vector=vector)

        return fuse_rankers(
            lambda: bm25.rank_candidates(query, chunks, limit=limit),
            dense,
            top_k=top_k,
            rrf_k=self.settings.hybrid_rrf_k,
        )
