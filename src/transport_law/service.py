import asyncio

from pydantic import ValidationError

from transport_law.errors import AppError
from transport_law.llm import LanguageModel
from transport_law.prompts import INSUFFICIENT_ANSWER
from transport_law.qa import StuffQAChain
from transport_law.retrieval import Retriever
from transport_law.schemas import AnswerResponse, GeneratedAnswer, SearchRequest


class LawService:
    def __init__(self, retriever: Retriever, llm: LanguageModel, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    async def ask(self, request: SearchRequest) -> AnswerResponse:
        # Chroma 检索包含同步 HTTP 请求，在工作线程执行，避免阻塞 FastAPI 事件循环。
        hits = await asyncio.to_thread(
            self.retriever.search, request.question, request.top_k or self.top_k
        )
        if not hits:
            return AnswerResponse(
                status="insufficient_evidence",
                answer=INSUFFICIENT_ANSWER,
                follow_up_questions=["请补充具体问题或导入相关资料。"],
            )
        source_documents = [hit.chunk for hit in hits]
        completion = await StuffQAChain(self.llm).ainvoke(request.question, source_documents)
        try:
            generated = GeneratedAnswer.model_validate_json(completion.content)
        except ValidationError as exc:
            raise AppError("llm_invalid_json", "模型输出未通过 JSON 结构校验，请重试。") from exc

        if generated.status == "insufficient_evidence":
            return AnswerResponse(
                status="insufficient_evidence",
                answer=INSUFFICIENT_ANSWER,
                source_documents=source_documents,
                follow_up_questions=generated.follow_up_questions,
                model=completion.model,
                usage=completion.usage,
            )

        allowed = {hit.chunk.id: hit.chunk for hit in hits}
        cited_ids = list(
            dict.fromkeys(
                citation_id for claim in generated.claims for citation_id in claim.citation_ids
            )
        )
        if any(citation_id not in allowed for citation_id in cited_ids):
            raise AppError("invalid_citation", "模型引用了本次检索以外的条目，答案已拦截。")
        # 原文和来源只来自库内记录，绝不使用模型生成的原文/页码/URL。
        return AnswerResponse(
            status="answered",
            answer="\n\n".join(claim.text for claim in generated.claims),
            claims=generated.claims,
            sources=[allowed[citation_id] for citation_id in cited_ids],
            source_documents=source_documents,
            follow_up_questions=generated.follow_up_questions,
            model=completion.model,
            usage=completion.usage,
        )
