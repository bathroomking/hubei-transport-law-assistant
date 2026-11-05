"""stuff 问答链：把已排序原文一次放入模板，调用模型，不再执行检索。"""

import json
from functools import lru_cache

from transport_law.llm import Completion, LanguageModel
from transport_law.prompts import QA_TEMPLATE, SYSTEM_PROMPT
from transport_law.schemas import LegalChunk


@lru_cache(maxsize=1)
def _prompt_template():
    # 仅问答时加载；服务启动、资料读取和独立检索不引入 LangChain 或分块模块。
    from langchain_core.prompts import PromptTemplate

    return PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "question"],
        template_format="f-string",
    )


class StuffQAChain:
    def __init__(self, llm: LanguageModel):
        self.llm = llm

    async def ainvoke(self, question: str, source_documents: list[LegalChunk]) -> Completion:
        if not source_documents:
            raise ValueError("stuff 问答需要至少一条检索原文")
        context = [
            document.model_dump(
                mode="json",
                exclude={"search_text", "aliases", "document_aliases", "related_excerpt_ids"},
            )
            for document in source_documents
        ]
        # 原文完整保留、排名不变。JSON 编码区分正文和字段；输入中的花括号不二次插值。
        prompt = _prompt_template().format(
            context=json.dumps(context, ensure_ascii=False, indent=2),
            question=json.dumps(question, ensure_ascii=False),
        )
        return await self.llm.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
        )
