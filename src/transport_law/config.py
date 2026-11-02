from pathlib import Path
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

    @field_validator("deepseek_base_url")
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
            raise ValueError("DEEPSEEK_BASE_URL 必须是无认证信息、查询参数和片段的 HTTPS 地址")
        return value

    @field_validator("law_data_path", "discretion_data_path")
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def llm_configured(self) -> bool:
        return bool(self.deepseek_api_key.get_secret_value().strip())
