from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = Field(default="deepseek-flash", min_length=1)
    deepseek_timeout_seconds: float = Field(default=120, gt=0, le=300)
    deepseek_max_tokens: int = Field(default=3000, ge=128, le=16384)
    deepseek_max_retries: int = Field(default=1, ge=0, le=3)
    law_data_path: Path = PROJECT_ROOT / "data/processed/legal_chunks.jsonl"
    discretion_data_path: Path = PROJECT_ROOT / "data/rules/road_occupation.json"
    retrieval_top_k: int = Field(default=5, ge=1, le=10)
    # 维护工具兼容配置；在线 runtime 始终固定 hybrid。
    retrieval_backend: Literal["bm25", "chroma", "hybrid"] = "bm25"
    hybrid_candidate_k: int = Field(default=50, ge=10, le=200)
    hybrid_rrf_k: int = Field(default=60, ge=1, le=1000)
    dashscope_api_key: SecretStr = SecretStr("")
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    embedding_model: Literal["text-embedding-v4"] = "text-embedding-v4"
    # 百炼 v4 的支持维度；请求时通过 dimension 指定，同时校验返回长度。
    embedding_dimensions: int = Field(default=2048, ge=64, le=2048)
    embedding_batch_size: int = Field(default=10, ge=1, le=10)
    embedding_timeout_seconds: float = Field(default=60, gt=0, le=300)
    embedding_max_retries: int = Field(default=2, ge=0, le=3)
    embedding_query_instruction: str = Field(
        default=(
            "Given a question about Chinese transportation law, retrieve relevant "
            "legal provisions that help answer the question."
        ),
        min_length=1,
    )
    chroma_path: Path = PROJECT_ROOT / "data/local/chroma"
    chroma_collection: str = Field(default="transport_law_bailian_v4_2048_v1", min_length=3)
    knowledge_path: Path = PROJECT_ROOT / "data/local/knowledge"
    knowledge_collection: str = Field(default="transport_uploads_v4_2048_v1", min_length=3)
    # 供存储层算法测试使用，不是产品离线模式开关。
    knowledge_retrieval_backend: Literal["bm25", "chroma", "hybrid"] = "chroma"
    upload_max_bytes: int = Field(default=20 * 1024 * 1024, gt=0, le=100 * 1024 * 1024)

    @field_validator("deepseek_base_url", "dashscope_base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("模型接口地址必须是无认证信息、查询参数和片段的 HTTPS 地址")
        return value

    @field_validator("dashscope_base_url")
    @classmethod
    def validate_dashscope_path(cls, value: str) -> str:
        if urlsplit(value).path != "/api/v1":
            raise ValueError("DASHSCOPE_BASE_URL 必须使用以 /api/v1 结尾的原生接口地址")
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        if value not in (64, 128, 256, 512, 768, 1024, 1536, 2048):
            raise ValueError("EMBEDDING_DIMENSIONS 必须是百炼 text-embedding-v4 支持的维度")
        return value

    @field_validator("law_data_path", "discretion_data_path", "chroma_path", "knowledge_path")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def llm_configured(self) -> bool:
        return bool(self.deepseek_api_key.get_secret_value().strip())

    @property
    def embedding_configured(self) -> bool:
        return bool(self.dashscope_api_key.get_secret_value().strip())
