from dataclasses import dataclass

from transport_law.config import Settings
from transport_law.hybrid import HybridRetriever
from transport_law.knowledge import KnowledgeStore
from transport_law.llm import DeepSeekClient, LanguageModel
from transport_law.repository import KnowledgeRepository
from transport_law.retrieval import Retriever
from transport_law.service import LawService


@dataclass
class Runtime:
    settings: Settings
    repository: KnowledgeRepository
    service: LawService
    knowledge: KnowledgeStore


def build_runtime(
    settings: Settings | None = None,
    llm: LanguageModel | None = None,
    *,
    retriever: Retriever | None = None,
) -> Runtime:
    settings = settings or Settings()
    knowledge = KnowledgeStore(settings)
    repository = KnowledgeRepository(knowledge)
    return Runtime(
        settings=settings,
        repository=repository,
        service=LawService(
            retriever if retriever is not None else HybridRetriever(knowledge, settings),
            llm or DeepSeekClient(settings),
            settings=settings,
        ),
        knowledge=knowledge,
    )
