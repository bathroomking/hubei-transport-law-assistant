import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from transport_law.config import PROJECT_ROOT, Settings
from transport_law.embeddings import DashScopeEmbeddings
from transport_law.errors import AppError


def vector(x=1, y=0, z=0):
    return [x, y, z] + [0] * 61


def native_response(rows):
    return {"output": {"embeddings": rows}}


@pytest.fixture
def embedding_settings(settings):
    return settings.model_copy(
        update={
            "dashscope_api_key": SecretStr("offline-secret"),
            "embedding_dimensions": 64,
            "embedding_max_retries": 0,
        }
    )


@pytest.mark.parametrize("index_field", ["index", "text_index"])
def test_native_payload_order_normalization_and_query_instruction(embedding_settings, index_field):
    bodies = []

    def handler(request):
        assert str(request.url) == (
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        )
        assert request.headers["Authorization"] == "Bearer offline-secret"
        body = json.loads(request.content)
        assert body["model"] == "text-embedding-v4"
        assert body["parameters"]["dimension"] == 64
        assert body["parameters"]["output_type"] == "dense"
        bodies.append(body)
        if len(body["input"]["texts"]) == 2:
            rows = [
                {index_field: 1, "embedding": vector(0, 0, 5)},
                {index_field: 0, "embedding": vector(3, 4, 0)},
            ]
        else:
            rows = [{index_field: 0, "embedding": vector()}]
        return httpx.Response(200, json=native_response(rows))

    client = DashScopeEmbeddings(embedding_settings, httpx.MockTransport(handler))
    assert client.embed_documents(["法条一", "法条二"]) == [vector(0.6, 0.8, 0), vector(0, 0, 1)]
    assert client.embed_query("公路挖沟") == vector()
    assert bodies[0]["input"] == {"texts": ["法条一", "法条二"]}
    assert bodies[0]["parameters"]["text_type"] == "document"
    assert "instruct" not in bodies[0]["parameters"]
    assert bodies[1]["input"] == {"texts": ["公路挖沟"]}
    assert bodies[1]["parameters"]["text_type"] == "query"
    assert bodies[1]["parameters"]["instruct"] == embedding_settings.embedding_query_instruction


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 0, "embedding": [1, 0]}],
        [{"index": 0, "embedding": vector(0, 0, 0)}],
        [{"index": 0, "embedding": vector(True)}],
        [{"index": 0, "embedding": vector("1")}],
        [{"index": 1, "embedding": vector()}],
        [{"index": False, "embedding": vector()}],
        [{"index": 0, "text_index": 1, "embedding": vector()}],
        [{"embedding": vector()}],
        [None],
    ],
)
def test_bad_vector_response_is_rejected(embedding_settings, rows):
    client = DashScopeEmbeddings(
        embedding_settings,
        httpx.MockTransport(lambda request: httpx.Response(200, json=native_response(rows))),
    )
    with pytest.raises(AppError) as error:
        client.embed_documents(["法条"])
    assert error.value.code == "embedding_invalid_response"


def test_duplicate_indices_nonfinite_numbers_and_wrong_envelope_are_rejected(embedding_settings):
    client = DashScopeEmbeddings(embedding_settings)
    invalid_responses = [
        (
            native_response(
                [{"index": 0, "embedding": vector()}, {"index": 0, "embedding": vector()}]
            ),
            2,
        ),
        (native_response([{"index": 0, "embedding": vector(float("nan"))}]), 1),
        ({"data": [{"index": 0, "embedding": vector()}]}, 1),
    ]
    for body, count in invalid_responses:
        with pytest.raises(AppError):
            client._parse_response(httpx.Response(200, content=json.dumps(body)), count)


@pytest.mark.parametrize("status", [400, 401, 402, 403, 429, 500, 302])
def test_errors_do_not_echo_key_or_upstream_body(embedding_settings, status):
    client = DashScopeEmbeddings(
        embedding_settings,
        httpx.MockTransport(lambda request: httpx.Response(status, text="offline-secret private")),
    )
    with pytest.raises(AppError) as error:
        client.embed_documents(["法条"])
    assert "offline-secret" not in error.value.message
    assert "private" not in error.value.message


@pytest.mark.parametrize(
    ("status", "body", "expected_code"),
    [
        (403, {"code": "AccessDenied.Unpurchased"}, "embedding_not_activated"),
        (403, {"error": {"code": "AccessDenied.Unpurchased"}}, "embedding_not_activated"),
        (400, {"code": "Arrearage"}, "embedding_balance_error"),
        (403, {"code": "isv.OUT_OF_SERVICE"}, "embedding_balance_error"),
        (401, {"code": "InvalidApiKey"}, "embedding_auth_error"),
        (403, {"code": "Model.AccessDenied"}, "embedding_access_denied"),
        (403, {"error": "malformed-error-body"}, "embedding_access_denied"),
    ],
)
def test_provider_errors_identify_required_action_without_retry(
    embedding_settings, status, body, expected_code
):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={**body, "message": "private upstream offline-secret"})

    configured = embedding_settings.model_copy(update={"embedding_max_retries": 2})
    with pytest.raises(AppError) as error:
        DashScopeEmbeddings(configured, httpx.MockTransport(handler)).embed_documents(["法条"])
    assert error.value.code == expected_code
    assert "private" not in error.value.message and "offline-secret" not in error.value.message
    assert len(calls) == 1


def test_missing_key_does_not_send_a_request(settings):
    def handler(request):
        pytest.fail("missing key must not cause an HTTP request")

    with pytest.raises(AppError) as error:
        DashScopeEmbeddings(settings, httpx.MockTransport(handler)).embed_documents(["法条"])
    assert error.value.code == "embedding_not_configured"


def test_excess_batch_and_blank_query_are_rejected_locally(embedding_settings):
    def handler(request):
        pytest.fail("invalid input must not cause an HTTP request")

    client = DashScopeEmbeddings(embedding_settings, httpx.MockTransport(handler))
    with pytest.raises(AppError, match="最多 10 条"):
        client.embed_documents(["法条"] * 11)
    with pytest.raises(AppError):
        client.embed_query(" ")


@pytest.mark.parametrize("failure", ["rate_limit", "timeout"])
def test_retry_is_bounded(embedding_settings, monkeypatch, failure):
    calls = []
    monkeypatch.setattr("transport_law.embeddings.time.sleep", lambda seconds: None)
    configured = embedding_settings.model_copy(update={"embedding_max_retries": 1})

    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private error")
        return httpx.Response(429)

    with pytest.raises(AppError):
        DashScopeEmbeddings(configured, httpx.MockTransport(handler)).embed_documents(["法条"])
    assert len(calls) == 2


@pytest.mark.parametrize(
    "update",
    [
        {"embedding_dimensions": 4096},
        {"embedding_dimensions": 1000},
        {"embedding_batch_size": 32},
        {"embedding_model": "qwen/qwen3-embedding-8b"},
        {"dashscope_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
    ],
)
def test_settings_reject_incompatible_provider_options(settings, update):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **update)


def test_env_config_and_old_provider_key_are_not_confused(settings, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "obsolete-offline-key")
    assert not Settings(_env_file=None).embedding_configured
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-key")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "10")
    monkeypatch.setenv(
        "DASHSCOPE_BASE_URL", "https://example-workspace.cn-beijing.maas.aliyuncs.com/api/v1/"
    )
    loaded = Settings(_env_file=None)
    assert loaded.embedding_configured
    assert loaded.embedding_model == "text-embedding-v4"
    assert loaded.embedding_dimensions == 1024 and loaded.embedding_batch_size == 10
    assert loaded.dashscope_base_url.endswith("/api/v1")


def test_example_configuration_keeps_api_keys_empty():
    from dotenv import dotenv_values

    values = dotenv_values(PROJECT_ROOT / ".env.example")
    if any(value for key, value in values.items() if key.endswith("API_KEY")):
        pytest.fail("示例文件的 API Key 必须留空；真实密钥只能放在本地 .env。")
