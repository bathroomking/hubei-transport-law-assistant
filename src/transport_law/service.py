import asyncio
import json
from dataclasses import replace

from pydantic import ValidationError

from transport_law.config import Settings
from transport_law.conversation import ConversationManager, PreparedConversation, estimate_tokens
from transport_law.errors import AppError
from transport_law.llm import LanguageModel
from transport_law.prompts import INSUFFICIENT_ANSWER, build_qa_messages
from transport_law.retrieval import Retriever
from transport_law.schemas import (
    AnswerResponse,
    ChatMessage,
    ChatState,
    GeneratedAnswer,
    SearchRequest,
    Usage,
)


class LawService:
    def __init__(self, retriever: Retriever, llm: LanguageModel, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    async def ask(self, request: SearchRequest) -> AnswerResponse:
        if not self.settings.llm_configured:
            raise AppError("llm_not_configured", "在线问答需要配置 DEEPSEEK_API_KEY。", 503)
        if not self.settings.embedding_configured:
            raise AppError("embedding_not_configured", "在线问答需要配置 DASHSCOPE_API_KEY。", 503)
        manager = ConversationManager(self.llm, self.settings)
        prepared = await manager.prepare(request.question, request.chat_state)
        if prepared.clarification:
            return self._finish(
                request,
                prepared,
                AnswerResponse(
                    status="insufficient_evidence",
                    answer="还需要确认你这次追问的具体意思。",
                    follow_up_questions=[prepared.clarification],
                ),
            )
        # Chroma 检索包含同步 HTTP 请求，在工作线程执行，避免阻塞 FastAPI 事件循环。
        hits = await asyncio.to_thread(
            self.retriever.search, prepared.query, request.top_k or self.settings.retrieval_top_k
        )
        if not hits:
            return self._finish(
                request,
                prepared,
                AnswerResponse(
                    status="insufficient_evidence",
                    answer=INSUFFICIENT_ANSWER,
                ),
            )
        source_documents = [hit.chunk for hit in hits]

        def messages():
            return build_qa_messages(request.question, source_documents, prepared.state)

        def fits_budget():
            return (
                estimate_tokens(json.dumps(messages(), ensure_ascii=False))
                + self.settings.deepseek_max_tokens
                <= self.settings.chat_context_token_limit
            )

        if not fits_budget():
            state, compacted, usages = await manager.compact(prepared.state, force=True)
            prepared = replace(
                prepared,
                state=state,
                compacted=prepared.compacted or compacted,
                usages=[*prepared.usages, *usages],
            )
        # 只移除低排名的整条原文；送入模型的每条正文和出处仍完整对应。
        while not fits_budget() and len(source_documents) > 1:
            source_documents.pop()
        if not fits_budget():
            raise AppError(
                "context_too_large", "本轮上下文过长，请缩小问题范围或清空聊天后重试。", 413
            )
        completion = await self.llm.complete(messages(), json_mode=True)
        if completion.usage is not None:
            prepared.usages.append(completion.usage)
        try:
            generated = GeneratedAnswer.model_validate_json(completion.content)
        except ValidationError as exc:
            raise AppError("llm_invalid_json", "模型输出未通过 JSON 结构校验，请重试。") from exc

        if generated.status == "insufficient_evidence":
            return self._finish(
                request,
                prepared,
                AnswerResponse(
                    status="insufficient_evidence",
                    answer=INSUFFICIENT_ANSWER,
                    source_documents=source_documents,
                    follow_up_questions=generated.follow_up_questions,
                    model=completion.model,
                ),
            )

        allowed = {document.id: document for document in source_documents}
        cited_ids = list(
            dict.fromkeys(
                citation_id for claim in generated.claims for citation_id in claim.citation_ids
            )
        )
        if any(citation_id not in allowed for citation_id in cited_ids):
            raise AppError("invalid_citation", "模型引用了本次检索以外的条目，答案已拦截。")
        # 原文和来源只来自库内记录，绝不使用模型生成的原文/页码/URL。
        return self._finish(
            request,
            prepared,
            AnswerResponse(
                status="answered",
                answer="\n\n".join(claim.text for claim in generated.claims),
                claims=generated.claims,
                sources=[allowed[citation_id] for citation_id in cited_ids],
                source_documents=source_documents,
                follow_up_questions=generated.follow_up_questions,
                model=completion.model,
            ),
        )

    @staticmethod
    def _finish(
        request: SearchRequest, prepared: PreparedConversation, answer: AnswerResponse
    ) -> AnswerResponse:
        # 补问只显示在本轮。下一轮从用户实际输入的文字读取补充事实。
        answer.chat_state = ChatState(
            summary=prepared.state.summary,
            messages=[
                *prepared.state.messages,
                ChatMessage(role="user", content=request.question),
                ChatMessage(
                    role="assistant",
                    content=answer.answer,
                    citation_ids=[source.id for source in answer.sources],
                ),
            ],
        )
        answer.context_compacted = prepared.compacted
        answer.retrieval_query = prepared.query
        if prepared.usages:
            answer.usage = Usage(
                prompt_tokens=sum(item.prompt_tokens for item in prepared.usages),
                completion_tokens=sum(item.completion_tokens for item in prepared.usages),
                total_tokens=sum(item.total_tokens for item in prepared.usages),
            )
        return answer
