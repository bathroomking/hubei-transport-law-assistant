from datetime import date
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_.:-]+$", max_length=120)]


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


class FollowUpChoice(Model):
    question: Annotated[Text, Field(max_length=600)]
    options: list[Annotated[Text, Field(max_length=300)]] = Field(
        default_factory=list, max_length=3
    )

    @field_validator("options", mode="wrap")
    @classmethod
    def independently_answerable(cls, value, validate) -> list[str]:
        try:
            options = validate(value)
        except ValidationError:
            return []
        # 按钮发送后不携带问题；孤立的是/否、单个答案、重复答案均改为自由填写。
        bare = {"是", "否", "是的", "不是", "有", "没有", "不确定", "其他", "不知道"}
        if (
            len(options) == 1
            or len(set(options)) != len(options)
            or any(text.rstrip("。.!！") in bare for text in options)
        ):
            return []
        return options


class GeneratedAnswer(Model):
    """模型只写解释和引用 ID，不能自行编造来源字段或适用状态。"""

    status: Literal["answered", "insufficient_evidence"] = "answered"
    claims: list[Claim] = Field(max_length=8)
    follow_up_choices: list[FollowUpChoice] = Field(default_factory=list, max_length=2)

    @model_validator(mode="before")
    @classmethod
    def parse_optional_choices(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy = data.pop("follow_up_questions", [])
        if "follow_up_choices" in data:
            offered = data["follow_up_choices"]
        else:
            offered = (
                [{"question": question} for question in legacy] if isinstance(legacy, list) else []
            )
        choices = []
        if isinstance(offered, list):
            for item in offered:
                try:
                    choice = FollowUpChoice.model_validate(item)
                except ValidationError:
                    continue  # 可选控件缺少有效问题时仍保留有据回答。
                if any(previous.question == choice.question for previous in choices):
                    continue
                choices.append(choice)
                if len(choices) == 2:
                    break
        return {**data, "follow_up_choices": choices}

    @model_validator(mode="after")
    def validate_answer_status(self):
        if self.status == "answered" and not self.claims:
            raise ValueError("有答案时必须提供带引用的陈述")
        if self.status == "insufficient_evidence" and self.claims:
            raise ValueError("资料不足时不得生成带引用的断言")
        return self


class Usage(Model):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class AnswerResponse(Model):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[LegalChunk] = Field(default_factory=list)
    source_documents: list[LegalChunk] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    follow_up_choices: list[FollowUpChoice] = Field(default_factory=list, max_length=2)
    applicability_status: Literal["not_verified"] = "not_verified"
    notice: str = "当前仅解释已收录材料；文件效力、适用地域和个案事实尚未完成核验。"
    model: str | None = None
    usage: Usage | None = None
    chat_state: ChatState = Field(default_factory=ChatState)
    retrieval_query: str = ""
    context_compacted: bool = False
