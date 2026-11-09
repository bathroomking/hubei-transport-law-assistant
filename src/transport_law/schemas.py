import re
from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_.:-]+$", max_length=120)]

# 实务提示没有引用。明显的处罚、强制措施、法定数值和引文应只进入带来源的 claims。
_LEGAL_NOTE = re.compile(
    r"罚(?!款?记录|决定书)|警告|拘留|禁驾|吊销|暂扣|扣留|扣押|没收|扣车|拖移|扣证|[记扣]分|"
    r"追究.{0,4}责任|(?:全部|主要|次要|同等)责任|全责|主责|"
    r"责令|强制|应当|必须|禁止|不得|有权|依法|法定|《|第[零〇一二三四五六七八九十百千\d]+条|"
    r"[\[【]\s*\d+\s*[\]】]|https?://|"
    r"[\d零〇一二三四五六七八九十百千万两]+\s*(?:元|块钱|分|日|天|个月|小时|年|%|％|公里/小时)"
)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupportingSource(Model):
    """合并整理条文所依赖的修改决定等，和主来源分别追溯。"""

    title: Text
    file: Text
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    official_url: HttpUrl
    locator: Text


class Source(Model):
    document_title: Text
    file: Text
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    pdf_page: int = Field(ge=1)
    # 单页来源保持原有序列化，兼容历史向量元数据中的 chunk_json。
    pdf_page_end: int | None = Field(default=None, ge=1, exclude_if=lambda value: value is None)
    source_type: Literal["pdf"] = "pdf"
    locator: Text | None = None
    retrieved_on: date | None = None
    legal_level: Text | None = None
    printed_page: str | None = None
    official_url: HttpUrl | None = None
    publisher: str | None = None
    published_on: date | None = None
    effective_on: date | None = None
    expires_on: date | None = None
    transcription_checked_on: date | None = None
    verification_status: Literal["pending", "verified", "superseded"] = "pending"
    verification_note: Text
    supporting_sources: list[SupportingSource] = Field(default_factory=list)


class LegalChunk(Model):
    id: Identifier
    document_id: Identifier
    version_id: Identifier
    item_id: Identifier
    title: Text
    article: Text
    kind: Literal["recognition", "penalty", "discretion", "provision", "procedure", "document"]
    region: Text
    domain: Text
    # 正文须与来源切片完全一致；缩进与末尾换行也是解析文本的一部分。
    text: Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
    search_text: Text
    aliases: list[Text] = Field(default_factory=list)
    document_aliases: list[Text] = Field(default_factory=list)
    article_number: int | None = Field(default=None, ge=1)
    # 指向相同法条的裁量表节选；仅用于检索去重，不表示旧法规被废止。
    related_excerpt_ids: list[Identifier] = Field(default_factory=list)
    source: Source


class ChatMessage(Model):
    role: Literal["user", "assistant"]
    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)
    ]
    citation_ids: list[Identifier] = Field(default_factory=list, max_length=10)

    @field_validator("content")
    @classmethod
    def remove_legacy_assistant_suffix(cls, content: str, info: ValidationInfo) -> str:
        # 旧客户端会原样回传服务生成的后缀。只剥离 assistant 消息中的固定分隔行。
        if info.data.get("role") == "assistant":
            body, marker, questions = content.rpartition("\n待补充：\n")
            if marker and body.strip() and questions.strip():
                return body
        return content


class ChatState(Model):
    """由页面内存携带；仅作未核实的聊天背景，不是法规证据。"""

    summary: str = Field(default="", max_length=6000)
    messages: list[ChatMessage] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_history(self):
        if len(self.messages) % 2:
            raise ValueError("历史须包含完整的用户/助手轮次")
        if any(
            message.role != ("user" if i % 2 == 0 else "assistant")
            for i, message in enumerate(self.messages)
        ):
            raise ValueError("历史消息须按用户、助手顺序排列")
        if sum(len(message.content) for message in self.messages) + len(self.summary) > 100000:
            raise ValueError("历史过长，请清空聊天后重试")
        return self


class SearchRequest(Model):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    top_k: int | None = Field(default=None, ge=1, le=10)
    chat_state: ChatState = Field(default_factory=ChatState)


class HybridScore(Model):
    """两路候选列表中的片段排名；RRF 分数不是相关概率。"""

    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    dense_score: float | None = None


class SearchHit(Model):
    chunk: LegalChunk
    score: float
    retrieval: HybridScore | None = None


class Claim(Model):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    citation_ids: list[Identifier] = Field(min_length=1, max_length=10)


class GeneratedAnswer(Model):
    """模型只写解释和引用 ID，不能自行编造来源字段或适用状态。"""

    status: Literal["answered", "insufficient_evidence"] = "answered"
    claims: list[Claim] = Field(max_length=8)
    practical_notes: list[Text] = Field(default_factory=list, max_length=2)
    follow_up_questions: list[Text] = Field(default_factory=list, max_length=3)

    @model_validator(mode="before")
    @classmethod
    def read_optional_questions(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        # 兼容上一版模型格式，只读取其中的问句。
        old = data.pop("follow_up_choices", [])
        if "follow_up_questions" not in data and isinstance(old, list):
            data["follow_up_questions"] = [
                item.get("question") for item in old if isinstance(item, dict)
            ]
        return data

    @field_validator("practical_notes", mode="before")
    @classmethod
    def short_practical_notes(cls, value):
        notes = []
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, str):
                    continue
                text = " ".join(item.split())
                if 0 < len(text) <= 240 and text not in notes and not _LEGAL_NOTE.search(text):
                    notes.append(text)
                    if len(notes) == 2:
                        break
        return notes

    @field_validator("follow_up_questions", mode="before")
    @classmethod
    def short_distinct_questions(cls, value):
        if not isinstance(value, list):
            return []
        questions = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                continue
            question = " ".join(item.split())
            key = question.rstrip("？?。").strip()
            if key and key not in seen and len(question) <= 120:
                questions.append(question)
                seen.add(key)
                if len(questions) == 3:
                    break
        return questions

    @model_validator(mode="after")
    def validate_answer_status(self):
        if self.status == "answered" and not self.claims:
            raise ValueError("有答案时必须提供带引用的陈述")
        if self.status == "insufficient_evidence" and self.claims:
            raise ValueError("资料不足时不得生成带引用的断言")
        if self.status == "insufficient_evidence":
            self.practical_notes = []
        return self


class Usage(Model):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class AnswerResponse(Model):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    practical_notes: list[Text] = Field(default_factory=list, max_length=2)
    sources: list[LegalChunk] = Field(default_factory=list)
    source_documents: list[LegalChunk] = Field(default_factory=list)
    follow_up_questions: list[Text] = Field(default_factory=list, max_length=3)
    applicability_status: Literal["not_verified"] = "not_verified"
    notice: str = "法律结论可通过引用核对。文件效力、适用地域和个案事实仍需核验。"
    model: str | None = None
    usage: Usage | None = None
    chat_state: ChatState = Field(default_factory=ChatState)
    retrieval_query: str = ""
    context_compacted: bool = False
