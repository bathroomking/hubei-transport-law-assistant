"""正式法规与上传资料的统一读取及混合检索。"""

from transport_law.config import Settings
from transport_law.embeddings import DashScopeEmbeddings, Embeddings
from transport_law.errors import AppError
from transport_law.hybrid import fuse_rankers
from transport_law.knowledge import KnowledgeStore
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.schemas import LegalChunk, SearchHit
from transport_law.vector_store import ChromaRetriever


class CombinedRepository:
    def __init__(self, curated: LawRepository, knowledge: KnowledgeStore):
        self.curated = curated
        self.knowledge = knowledge

    @property
    def all_chunks(self) -> list[LegalChunk]:
        return self.curated.all_chunks + self.knowledge.ready_chunks()

    @property
    def searchable(self) -> list[LegalChunk]:
        return self.curated.searchable + self.knowledge.ready_chunks()

    def get_article(self, chunk_id: str) -> LegalChunk:
        try:
            return self.curated.get_article(chunk_id)
        except AppError as exc:
            if exc.code != "article_not_found":
                raise
        return self.knowledge.get_article(chunk_id)


class UnifiedRetriever:
    def __init__(
        self,
        curated: LawRepository,
        knowledge: KnowledgeStore,
        settings: Settings,
        embedder: Embeddings | None = None,
    ):
        self.curated = curated
        self.knowledge = knowledge
        self.settings = settings
        self.embedder = embedder or DashScopeEmbeddings(settings)

    def search(self, question: str, top_k: int = 5) -> list[SearchHit]:
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")
        # 每次请求固定上传快照，后台新增 ready 文档无需重启服务。
        uploaded = self.knowledge.ready_chunks()
        bm25 = BM25Retriever(LawRepository(self.curated.searchable + uploaded))
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
            # 延迟打开索引：未建库时页面可启动，但提问明确报错且不消耗向量额度。
            retriever = (
                ChromaRetriever(self.curated, self.settings, self.embedder) if formal else None
            )
            # 同一个模型/维度的两座持久库共用一次问题 embedding。
            vector = self.embedder.embed_query(question)
            hits = []
            if retriever is not None:
                hits.extend(
                    retriever.rank_candidates(question, formal, limit=limit, query_vector=vector)
                )
            if extra:
                hits.extend(
                    self.knowledge._rank_vectors(
                        extra,
                        limit=limit,
                        query_vector=vector,
                    )
                )
            # 两库使用相同 embedding 合约和 cosine 距离，可合并成一路 dense 排名。
            return sorted(hits, key=lambda hit: (-hit.score, hit.chunk.id))[:limit]

        return fuse_rankers(
            lambda: bm25.rank_candidates(question, candidates, limit=limit),
            dense,
            top_k=top_k,
            rrf_k=self.settings.hybrid_rrf_k,
        )
