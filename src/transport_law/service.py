import json

from pydantic import ValidationError

from transport_law.errors import AppError
from transport_law.llm import LanguageModel
from transport_law.prompts import SYSTEM_PROMPT
from transport_law.repository import LawRepository
from transport_law.retrieval import Retriever
from transport_law.schemas import AnswerResponse, GeneratedAnswer, SearchRequest, SearchResponse


class LawService:
    def __init__(
        self, repository: LawRepository, retriever: Retriever, llm: LanguageModel, top_k: int = 5
    ):
        self.repository = repository
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def retrieve(self, request: SearchRequest) -> SearchResponse:
        return SearchResponse(
            question=request.question,
            hits=self.retriever.search(request.question, request.top_k or self.top_k),
            coverage=sorted({f"{c.domain} / {c.title}" for c in self.repository.searchable}),
        )

    async def ask(self, request: SearchRequest) -> AnswerResponse:
        result = self.retrieve(request)
        if not result.hits:
            return AnswerResponse(
                status="insufficient_evidence",
                answer="当前资料中未检索到足够依据，不能据此给出处罚意见。",
                follow_up_questions=["请补充具体行为或导入该事项的法规与裁量资料。"],
            )
        evidence = [hit.chunk.model_dump(mode="json") for hit in result.hits]
        completion = await self.llm.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": request.question, "evidence": evidence}, ensure_ascii=False
                    ),
                },
            ],
            json_mode=True,
        )
        try:
            generated = GeneratedAnswer.model_validate_json(completion.content)
        except ValidationError as exc:
            raise AppError("llm_invalid_json", "模型输出未通过 JSON 结构校验，请重试。") from exc

        allowed = {hit.chunk.id: hit.chunk for hit in result.hits}
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
            follow_up_questions=generated.follow_up_questions,
            model=completion.model,
            usage=completion.usage,
        )
