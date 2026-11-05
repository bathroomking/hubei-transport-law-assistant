from dataclasses import dataclass

from transport_law.config import Settings
from transport_law.knowledge import KnowledgeStore
from transport_law.llm import DeepSeekClient, LanguageModel
from transport_law.online import CombinedRepository, UnifiedRetriever
from transport_law.repository import LawRepository
from transport_law.retrieval import Retriever
from transport_law.service import LawService


@dataclass
class Runtime:
    settings: Settings
    repository: CombinedRepository
    service: LawService
    knowledge: KnowledgeStore


def build_runtime(
    settings: Settings | None = None,
    llm: LanguageModel | None = None,
    *,
    retriever: Retriever | None = None,
) -> Runtime:
    settings = settings or Settings()
    curated = LawRepository.from_jsonl(settings.law_data_path)
    knowledge = KnowledgeStore(
        settings, reserved_filenames=[chunk.source.file for chunk in curated.all_chunks]
    )
    repository = CombinedRepository(curated, knowledge)
    return Runtime(
        settings=settings,
        repository=repository,
        service=LawService(
            retriever if retriever is not None else UnifiedRetriever(curated, knowledge, settings),
            llm or DeepSeekClient(settings),
            settings=settings,
        ),
        knowledge=knowledge,
    )
