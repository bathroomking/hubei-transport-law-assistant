import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from transport_law.errors import AppError
from transport_law.llm import Completion, DeepSeekClient
from transport_law.schemas import SearchRequest
from transport_law.service import LawService


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def complete(self, messages, *, json_mode):
        self.calls.append((messages, json_mode))
        return Completion(content=self.content, model="offline-fixture")


def generated(ids=None):
    return json.dumps(
        {
            "claims": [
                {
                    "text": "材料载有擅自占用、挖掘公路的规定。",
                    "citation_ids": ids or ["cn-highway-2017-A044"],
                }
            ],
            "follow_up_questions": ["具体道路属性和许可情况是什么？"],
        },
        ensure_ascii=False,
    )


def envelope(content="测试回复", finish="stop"):
    return {
        "model": "deepseek-flash",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_cache_hit_tokens": 0,
        },
    }


def test_sources_are_filled_from_repository(repository, retriever, online_settings):
    llm = FakeLLM(generated())
    service = LawService(retriever, llm, online_settings)
    result = asyncio.run(service.ask(SearchRequest(question="公路挖沟怎么处理？")))
    assert result.sources == [repository.get_article("cn-highway-2017-A044")]
    assert result.applicability_status == "not_verified"
    assert llm.calls[0][1] is True
    assert "verification_status" in llm.calls[0][0][1]["content"]


def test_nonexistent_article_never_calls_llm(retriever, online_settings):
    llm = FakeLLM("should not be called")
    result = asyncio.run(
        LawService(retriever, llm, online_settings).ask(SearchRequest(question="公路法第999条"))
    )
    assert result.status == "insufficient_evidence"
    assert result.sources == [] and llm.calls == []


def test_rejects_citation_that_exists_but_was_not_retrieved(retriever, online_settings):
    request = SearchRequest(question="公路法第四十四条第一款", top_k=1)
    assert retriever.search(request.question, 1)[0].chunk.id == "cn-highway-2017-A044"
    llm = FakeLLM(generated(["HB-ROAD-003-BANDS"]))
    with pytest.raises(AppError) as error:
        asyncio.run(LawService(retriever, llm, online_settings).ask(request))
    assert error.value.code == "invalid_citation"


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "{}",
        " ",
        '{"claims": [], "follow_up_questions": []}',
        '{"claims": [{"text":"断言", "citation_ids":[]}], "follow_up_questions": []}',
        generated().replace('"claims":', '"invented_source": "example", "claims":'),
    ],
)
def test_rejects_bad_generated_structure(retriever, content, online_settings):
    with pytest.raises(AppError) as error:
        asyncio.run(
            LawService(retriever, FakeLLM(content), online_settings).ask(
                SearchRequest(question="公路挖沟怎么处理？")
            )
        )
    assert error.value.code == "llm_invalid_json"


def test_missing_key_is_local_error(settings):
    with pytest.raises(AppError) as error:
        asyncio.run(DeepSeekClient(settings).complete([], json_mode=True))
    assert error.value.code == "llm_not_configured"


def test_deepseek_payload_and_usage(settings):
    configured = settings.model_copy(update={"deepseek_api_key": SecretStr("test-key")})

    def handler(request):
        assert str(request.url) == "https://api.deepseek.com/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        assert body["response_format"] == {"type": "json_object"}
        assert body["stream"] is False
        return httpx.Response(200, json=envelope(generated()))

    client = DeepSeekClient(configured, httpx.MockTransport(handler))
    result = asyncio.run(client.complete([{"role": "user", "content": "json"}], json_mode=True))
    assert result.usage.total_tokens == 15


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "llm_auth_error"),
        (403, "llm_auth_error"),
        (402, "llm_balance_error"),
        (429, "llm_rate_limited"),
        (500, "llm_unavailable"),
        (400, "llm_request_error"),
        (302, "llm_request_error"),
    ],
)
def test_upstream_errors_do_not_leak_response(settings, status, code):
    configured = settings.model_copy(
        update={"deepseek_api_key": SecretStr("secret"), "deepseek_max_retries": 0}
    )
    client = DeepSeekClient(
        configured,
        httpx.MockTransport(
            lambda request: httpx.Response(status, text="private upstream body secret")
        ),
    )
    with pytest.raises(AppError) as error:
        asyncio.run(client.complete([], json_mode=True))
    assert error.value.code == code
    assert "secret" not in error.value.message


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (envelope(" "), "llm_empty_response"),
        (envelope("{}", "length"), "llm_incomplete"),
        ({"choices": []}, "llm_invalid_response"),
        ({}, "llm_invalid_response"),
    ],
)
def test_unusable_completion_rejected(settings, body, code):
    configured = settings.model_copy(update={"deepseek_api_key": SecretStr("test")})
    client = DeepSeekClient(
        configured, httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    )
    with pytest.raises(AppError) as error:
        asyncio.run(client.complete([], json_mode=True))
    assert error.value.code == code


def test_rate_limit_has_bounded_retry(settings, monkeypatch):
    configured = settings.model_copy(update={"deepseek_api_key": SecretStr("test")})
    calls = []

    async def no_sleep(_):
        pass

    def handler(request):
        calls.append(request)
        return httpx.Response(429) if len(calls) == 1 else httpx.Response(200, json=envelope())

    monkeypatch.setattr("transport_law.llm.asyncio.sleep", no_sleep)
    result = asyncio.run(
        DeepSeekClient(configured, httpx.MockTransport(handler)).complete([], json_mode=False)
    )
    assert result.content == "测试回复" and len(calls) == 2


@pytest.mark.parametrize(
    ("exception", "code"),
    [
        (httpx.ReadTimeout("private detail"), "llm_timeout"),
        (httpx.ConnectError("private detail"), "llm_unavailable"),
    ],
)
def test_network_errors(settings, exception, code):
    configured = settings.model_copy(
        update={"deepseek_api_key": SecretStr("test"), "deepseek_max_retries": 0}
    )

    def handler(request):
        raise exception

    with pytest.raises(AppError) as error:
        asyncio.run(
            DeepSeekClient(configured, httpx.MockTransport(handler)).complete([], json_mode=True)
        )
    assert error.value.code == code
