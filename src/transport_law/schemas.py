from datetime import date
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_.:-]+$", max_length=120)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Model):
    document_title: Text
    file: Text
    sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    pdf_page: int = Field(ge=1)
    printed_page: str | None = None
    official_url: HttpUrl | None = None
    publisher: str | None = None
    published_on: date | None = None
    effective_on: date | None = None
    expires_on: date | None = None
    transcription_checked_on: date | None = None
    verification_status: Literal["pending", "verified", "superseded"] = "pending"
    verification_note: Text


class LegalChunk(Model):
    id: Identifier
    document_id: Identifier
    version_id: Identifier
    item_id: Identifier
    title: Text
    article: Text
    kind: Literal["recognition", "penalty", "discretion"]
    region: Text
    domain: Text
    text: Text
    search_text: Text
    aliases: list[Text] = Field(default_factory=list)
    source: Source


class SearchRequest(Model):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=2, max_length=2000)
    ]
    top_k: int | None = Field(default=None, ge=1, le=10)


class SearchHit(Model):
    chunk: LegalChunk
    score: float


class SearchResponse(Model):
    question: str
    hits: list[SearchHit]
    coverage: list[str]


class Claim(Model):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    citation_ids: list[Identifier] = Field(min_length=1, max_length=10)


class GeneratedAnswer(Model):
    """模型只写解释和引用 ID，不能自行编造来源字段或适用状态。"""

    claims: list[Claim] = Field(min_length=1, max_length=8)
    follow_up_questions: list[Annotated[Text, Field(max_length=300)]] = Field(max_length=8)


class Usage(Model):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class AnswerResponse(Model):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    sources: list[LegalChunk] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    applicability_status: Literal["not_verified"] = "not_verified"
    notice: str = "当前仅解释已收录材料；文件效力、适用地域和个案事实尚未完成核验。"
    model: str | None = None
    usage: Usage | None = None


class DiscretionBand(Model):
    id: Identifier
    condition_text: Text
    consequence_text: Text


class DiscretionRuleSet(Model):
    item_id: Identifier
    version_id: Identifier
    name: Text
    review_status: Literal["draft"] = "draft"
    source_ids: list[Identifier] = Field(min_length=1)
    bands: list[DiscretionBand] = Field(min_length=1)
    unresolved_boundaries_m2: list[Decimal]
    review_notes: list[Text] = Field(min_length=1)


class CaseFacts(Model):
    """只收集用户显式提供的事实；不将口语描述自动认定为法律事实。"""

    item_id: Identifier = "road_occupation_excavation"
    region: Text | None = None
    location: Text | None = None
    occurred_on: date | None = None
    road_type: Literal["ordinary_highway", "expressway", "urban_road", "other"] | None = None
    behavior: Literal["occupation", "excavation", "both"] | None = None
    has_permission: bool | None = Field(default=None, strict=True)
    area_m2: Decimal | None = Field(default=None, gt=0, max_digits=12, decimal_places=4)
    facts_verified: bool = Field(default=False, strict=True)


class DiscretionCheck(Model):
    status: Literal["needs_facts", "needs_review", "out_of_scope"]
    missing_fields: list[str]
    follow_up_questions: list[str]
    review_notes: list[str]
    rule_set: DiscretionRuleSet | None
    sources: list[LegalChunk]
    # 首版草稿规则没有确定金额的输出通道。
    applicability_status: Literal["not_verified"] = "not_verified"
