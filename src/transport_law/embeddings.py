"""阿里云百炼 DashScope 原生向量接口；显式区分 query 和 document。"""

import math
import time
from typing import Literal, Protocol

import httpx

from transport_law.config import Settings
from transport_law.errors import AppError


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, question: str) -> list[float]: ...


class DashScopeEmbeddings:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question], "query")[0]

    def _embed(
        self, texts: list[str], text_type: Literal["document", "query"]
    ) -> list[list[float]]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise AppError("invalid_embedding_input", "向量化输入必须是非空文本列表。", 400)
        if len(texts) > 10:
            raise AppError(
                "invalid_embedding_input", "百炼 text-embedding-v4 每批最多 10 条。", 400
            )
        if not self.settings.embedding_configured:
            raise AppError(
                "embedding_not_configured", "请在本地 .env 配置 DASHSCOPE_API_KEY。", 503
            )
        with httpx.Client(
            timeout=self.settings.embedding_timeout_seconds,
            transport=self.transport,
            follow_redirects=False,
        ) as client:
            return self._request(client, texts, text_type)

    def _request(self, client: httpx.Client, texts: list[str], text_type: str) -> list[list[float]]:
        parameters = {
            "dimension": self.settings.embedding_dimensions,
            "text_type": text_type,
            "output_type": "dense",
        }
        if text_type == "query":
            parameters["instruct"] = self.settings.embedding_query_instruction
        for attempt in range(self.settings.embedding_max_retries + 1):
            try:
                response = client.post(
                    self.settings.dashscope_base_url
                    + "/services/embeddings/text-embedding/text-embedding",
                    headers={
                        "Authorization": "Bearer "
                        + self.settings.dashscope_api_key.get_secret_value(),
                    },
                    json={
                        "model": self.settings.embedding_model,
                        "input": {"texts": texts},
                        "parameters": parameters,
                    },
                )
            except httpx.RequestError as exc:
                if attempt < self.settings.embedding_max_retries:
                    time.sleep(0.5 * 2**attempt)
                    continue
                code = (
                    "embedding_timeout"
                    if isinstance(exc, httpx.TimeoutException)
                    else "embedding_unavailable"
                )
                raise AppError(code, "无法取得向量，请检查网络和接口地址后重试。", 503) from exc
            status = response.status_code
            error_code = self._provider_error_code(response) if status != 200 else None
            if error_code == "AccessDenied.Unpurchased":
                raise AppError(
                    "embedding_not_activated",
                    "百炼服务尚未开通或当前账号不具备模型使用资格（AccessDenied.Unpurchased）。"
                    "请用阿里云主账号在该 Key 所属地域的百炼控制台完成服务开通。",
                    503,
                )
            if status == 402 or error_code in ("Arrearage", "isv.OUT_OF_SERVICE"):
                raise AppError("embedding_balance_error", "阿里云百炼账户欠费或余额不足。", 503)
            if status == 401:
                raise AppError(
                    "embedding_auth_error", "百炼认证失败，请检查 API Key 及其所属地域。", 503
                )
            if status == 403:
                raise AppError(
                    "embedding_access_denied",
                    "百炼拒绝访问，请检查该 API Key 的模型权限和业务空间授权。",
                    503,
                )
            if status == 429 or status >= 500:
                if attempt < self.settings.embedding_max_retries:
                    time.sleep(0.5 * 2**attempt)
                    continue
                raise AppError(
                    "embedding_unavailable", "向量服务限流或暂时不可用，请稍后重试。", 503
                )
            if status != 200:
                raise AppError(
                    "embedding_request_error", "向量请求被拒绝，请核对模型和接口配置。", 503
                )
            return self._parse_response(response, len(texts))
        raise AssertionError("unreachable")

    @staticmethod
    def _provider_error_code(response: httpx.Response) -> str | None:
        # 只匹配已知错误码，绝不把上游原始 message（可能含输入或密钥）回显。
        try:
            body = response.json()
        except ValueError:
            return None
        if not isinstance(body, dict):
            return None
        code = body.get("code")
        if not isinstance(code, str) and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
        return code if isinstance(code, str) else None

    def _parse_response(self, response: httpx.Response, count: int) -> list[list[float]]:
        try:
            rows = response.json()["output"]["embeddings"]
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError("wrong response count")
            by_index = {}
            for row in rows:
                # 新网关使用 index；旧 DashScope HTTP 响应使用 text_index。
                index = row.get("index", row.get("text_index"))
                if "index" in row and "text_index" in row and row["text_index"] != index:
                    raise ValueError("conflicting response indices")
                vector = row["embedding"]
                if type(index) is not int or not 0 <= index < count or index in by_index:
                    raise ValueError("invalid response index")
                if (
                    not isinstance(vector, list)
                    or len(vector) != self.settings.embedding_dimensions
                ):
                    raise ValueError("unexpected dimensions")
                if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
                    raise ValueError("non-finite vector")
                norm = math.hypot(*vector)
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError("invalid vector norm")
                by_index[index] = [x / norm for x in vector]
            # 服务端允许乱序返回，必须按 index 对齐原文，不能只依赖数组顺序。
            return [by_index[index] for index in range(count)]
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            raise AppError(
                "embedding_invalid_response",
                "向量响应的数量、顺序标识、维度或数值无效；本批次未写入。",
                502,
            ) from exc
