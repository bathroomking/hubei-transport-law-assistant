"""用 HTTPX 调用 DeepSeek 的 Chat Completions 接口，可注入 transport 离线测试。"""

import asyncio
from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import ValidationError

from transport_law.config import Settings
from transport_law.errors import AppError
from transport_law.schemas import Usage


@dataclass(frozen=True)
class Completion:
    content: str
    model: str
    usage: Usage | None = None


class LanguageModel(Protocol):
    async def complete(self, messages: list[dict[str, str]], *, json_mode: bool) -> Completion: ...


class DeepSeekClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    async def complete(self, messages: list[dict[str, str]], *, json_mode: bool) -> Completion:
        if not self.settings.llm_configured:
            raise AppError("llm_not_configured", "请在本地 .env 配置 DEEPSEEK_API_KEY。", 503)
        payload = {
            "model": self.settings.deepseek_model,
            "messages": messages,
            "stream": False,
            "thinking": {"type": "disabled"},
            "max_tokens": self.settings.deepseek_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            # 总时限覆盖全部重试，避免重试把等待时间无限放大。
            async with asyncio.timeout(self.settings.deepseek_timeout_seconds):
                async with httpx.AsyncClient(
                    timeout=self.settings.deepseek_timeout_seconds,
                    transport=self.transport,
                    follow_redirects=False,
                ) as client:
                    return await self._request(client, payload)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise AppError("llm_timeout", "模型请求超时，请稍后再试。", 504) from exc

    async def _request(self, client: httpx.AsyncClient, payload: dict) -> Completion:
        for attempt in range(self.settings.deepseek_max_retries + 1):
            try:
                response = await client.post(
                    self.settings.deepseek_base_url + "/chat/completions",
                    headers={
                        "Authorization": "Bearer "
                        + self.settings.deepseek_api_key.get_secret_value(),
                    },
                    json=payload,
                )
            except httpx.TimeoutException:
                raise
            except httpx.RequestError as exc:
                if attempt < self.settings.deepseek_max_retries:
                    await asyncio.sleep(0.5 * 2**attempt)
                    continue
                raise AppError(
                    "llm_unavailable", "无法连接模型服务，请检查网络和接口地址。", 503
                ) from exc

            status = response.status_code
            if status in (401, 403):
                raise AppError("llm_auth_error", "模型服务认证失败，请检查本地 API Key。", 503)
            if status == 402:
                raise AppError("llm_balance_error", "模型服务余额不足，请检查账户。", 503)
            if status == 429 or status >= 500:
                if attempt < self.settings.deepseek_max_retries:
                    await asyncio.sleep(0.5 * 2**attempt)
                    continue
                code = "llm_rate_limited" if status == 429 else "llm_unavailable"
                raise AppError(code, "模型服务暂时不可用或请求过多，请稍后重试。", 503)
            if status != 200:
                raise AppError("llm_request_error", "模型拒绝请求，请核对模型名称和接口配置。")
            return self._parse_response(response)
        raise AssertionError("unreachable")

    def _parse_response(self, response: httpx.Response) -> Completion:
        try:
            body = response.json()
            choice = body["choices"][0]
            if choice["finish_reason"] != "stop":
                raise AppError(
                    "llm_incomplete", "模型回复未完整结束，请缩短问题或调整 token 上限。"
                )
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise AppError("llm_empty_response", "模型返回空白内容，请重试。")
            model = body["model"]
            if not isinstance(model, str) or not model.strip():
                raise ValueError("invalid model")
            usage_body = body.get("usage")
            usage = (
                None
                if not usage_body
                else Usage.model_validate(
                    {
                        key: usage_body[key]
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                )
            )
            return Completion(content=content, model=model, usage=usage)
        except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
            raise AppError("llm_invalid_response", "模型响应格式不正确，无法读取。") from exc
