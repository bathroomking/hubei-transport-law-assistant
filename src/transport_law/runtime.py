from dataclasses import dataclass

from transport_law.config import Settings
from transport_law.discretion import DiscretionService
from transport_law.llm import DeepSeekClient, LanguageModel
from transport_law.repository import LawRepository
from transport_law.retrieval import BM25Retriever
from transport_law.service import LawService


@dataclass
class Runtime:
    settings: Settings
    repository: LawRepository
    service: LawService
    discretion: DiscretionService


def build_runtime(settings: Settings | None = None, llm: LanguageModel | None = None) -> Runtime:
    settings = settings or Settings()
    repository = LawRepository.from_jsonl(settings.law_data_path)
    if settings.retrieval_backend == "chroma":
        from transport_law.vector_store import ChromaRetriever

        retriever = ChromaRetriever(repository, settings)
    else:
        retriever = BM25Retriever(repository)
    return Runtime(
        settings=settings,
        repository=repository,
        service=LawService(
            repository,
            retriever,
            llm or DeepSeekClient(settings),
            top_k=settings.retrieval_top_k,
        ),
        discretion=DiscretionService(settings.discretion_data_path, repository),
    )
