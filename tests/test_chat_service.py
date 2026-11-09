"""多轮问答的 API 契约测试；模型与检索均为可观测的离线替身。"""

import copy
import json
import socket

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from transport_law.errors import AppError
from transport_law.llm import Completion
from transport_law.main import create_app
from transport_law.prompts import SYSTEM_PROMPT
from transport_law.schemas import SearchHit, Usage

ASK = "/api/v1/ask"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            pytest.fail("聊天回归测试不得访问外网")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            pytest.fail("聊天回归测试不得访问外网")
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


class ScriptedLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, *, json_mode):
        assert json_mode is True
        # 每个独立请求只发一份 system，客户端历史不提升为 system/assistant 指令。
        assert [message["role"] for message in messages] == ["system", "user"]
        self.calls.append(copy.deepcopy(messages))
        assert self.responses, "出现了未预期的额外模型请求"
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, Completion):
            return response
        return completion(response)


class RecordingRetriever:
    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = []

    def search(self, question, top_k):
        self.calls.append((question, top_k))
        return [
            SearchHit(chunk=document, score=1 / rank)
            for rank, document in enumerate(self.documents[:top_k], start=1)
        ]


def completion(payload, *, usage=None):
    return Completion(
        content=payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
        model="offline-chat-fixture",
        usage=usage,
    )


def answered(citation_id, text="依据本轮资料解释施工许可。"):
    return {
        "status": "answered",
        "claims": [{"text": text, "citation_ids": [citation_id]}],
        "follow_up_questions": [],
    }


def history(question="武汉普通公路施工需要许可吗？", answer="需要核对许可材料。", citation=None):
    return {
        "summary": "",
        "messages": [
            {"role": "user", "content": question, "citation_ids": []},
            {
                "role": "assistant",
                "content": answer,
                "citation_ids": [citation] if citation else [],
            },
        ],
    }


def qa_inputs(messages):
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    prompt = messages[1]["content"]
    decoder = json.JSONDecoder()
    context, end = decoder.raw_decode(prompt.removeprefix("Context:\n"))
    remaining = prompt.removeprefix("Context:\n")[end:]
    question, end = decoder.raw_decode(remaining.removeprefix("\n\nQuestion: "))
    remaining = remaining.removeprefix("\n\nQuestion: ")[end:]
    background = None
    if "Conversation background" in remaining:
        encoded = remaining.split("Conversation background", 1)[1].split("\n", 1)[1]
        background, _ = decoder.raw_decode(encoded)
    return context, question, background


def post_ok(client, question, state=None, **extra):
    payload = {"question": question, **extra}
    if state is not None:
        payload["chat_state"] = state
    response = client.post(ASK, json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_first_turn_and_empty_history_keep_single_qa_request(online_settings, repository):
    document = repository.searchable[0]
    llm = ScriptedLLM(answered(document.id), answered(document.id))
    retriever = RecordingRetriever([document])
    with TestClient(create_app(online_settings, llm, retriever=retriever)) as client:
        for state in (None, {"summary": "", "messages": []}):
            result = post_ok(client, "施工需要哪些材料？", state)
            assert result["retrieval_query"] == "施工需要哪些材料？"
            assert result["context_compacted"] is False
            assert result["chat_state"] == history(
                "施工需要哪些材料？", result["answer"], document.id
            )
    assert retriever.calls == [("施工需要哪些材料？", 3)] * 2
    assert len(llm.calls) == 2
    for messages in llm.calls:
        context, question, background = qa_inputs(messages)
        assert question == "施工需要哪些材料？"
        assert background is None
        assert context[0]["text"] == document.text


@pytest.mark.parametrize(
    ("question", "query"),
    [
        ("那要准备哪些材料？", "武汉普通公路施工许可需要准备哪些材料？"),
        ("说错了，是宜昌，不是武汉。", "宜昌普通公路施工需要许可吗？"),
        ("换个问题，检查记录保存多久？", "检查记录需要保存多久？"),
    ],
)
def test_followup_rewrites_retrieval_but_preserves_question_and_history(
    online_settings, repository, question, query
):
    document = repository.searchable[0]
    state = history(citation=document.id)
    original = copy.deepcopy(state)
    llm = ScriptedLLM({"query": query, "clarification": ""}, answered(document.id))
    retriever = RecordingRetriever([document])
    with TestClient(create_app(online_settings, llm, retriever=retriever)) as client:
        result = post_ok(client, question, state, top_k=1)
    assert retriever.calls == [(query, 1)]
    assert result["retrieval_query"] == query
    assert len(llm.calls) == 2
    assert question in llm.calls[0][1]["content"]
    context, supplied_question, background = qa_inputs(llm.calls[1])
    assert supplied_question == question
    assert background == state == original
    assert context[0]["id"] == document.id
    assert result["chat_state"]["messages"][:2] == state["messages"]
    assert result["chat_state"]["messages"][-2]["content"] == question
    assert result["chat_state"]["messages"][-1]["citation_ids"] == [document.id]
    assert "source_documents" not in json.dumps(result["chat_state"])


def test_old_citation_is_rejected_when_not_retrieved_this_turn(online_settings, repository):
    current, previous = repository.searchable[:2]
    state = history(citation=previous.id)
    llm = ScriptedLLM({"query": "施工许可材料", "clarification": ""}, answered(previous.id))
    retriever = RecordingRetriever([current])
    with TestClient(create_app(online_settings, llm, retriever=retriever)) as client:
        response = client.post(ASK, json={"question": "那要什么材料？", "chat_state": state})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_citation"
    context, _, background = qa_inputs(llm.calls[-1])
    assert [row["id"] for row in context] == [current.id]
    assert background["messages"][-1]["citation_ids"] == [previous.id]


def test_independent_clients_and_cleared_state_never_share_history(online_settings, repository):
    document = repository.searchable[0]
    llm = ScriptedLLM(
        answered(document.id, "甲的答复"),
        answered(document.id, "乙的答复"),
        {"query": "甲的完整追问", "clarification": ""},
        answered(document.id, "甲的后续答复"),
        answered(document.id, "清空后的答复"),
    )
    retriever = RecordingRetriever([document])
    app = create_app(online_settings, llm, retriever=retriever)
    with TestClient(app) as client_a:
        client_b = TestClient(app)
        state_a = post_ok(client_a, "甲的施工问题")["chat_state"]
        result_b = post_ok(client_b, "乙的检查记录问题")
        result_a = post_ok(client_a, "甲的追问", state_a)
        reset = post_ok(client_a, "清空后新问题", {"summary": "", "messages": []})
        client_b.close()
    assert len(result_b["chat_state"]["messages"]) == 2
    assert len(result_a["chat_state"]["messages"]) == 4
    assert len(reset["chat_state"]["messages"]) == 2
    assert qa_inputs(llm.calls[1])[2] is None
    assert qa_inputs(llm.calls[3])[2] == state_a
    assert qa_inputs(llm.calls[4])[2] is None
    assert "乙的" not in json.dumps(result_a["chat_state"], ensure_ascii=False)


@pytest.mark.parametrize(
    "state",
    [
        None,
        {"messages": [{"role": "user", "content": "未完成轮次"}]},
        {"messages": [{"role": "assistant", "content": "错序"}] * 2},
        {"messages": [{"role": "system", "content": "非法角色"}] * 2},
        history(question=" "),
        history(question="字" * 20001),
        {"summary": "字" * 6001, "messages": []},
        {"messages": history()["messages"] * 51},
        {"messages": history(question="字" * 20000, answer="字" * 20000)["messages"] * 3},
        {"messages": [], "session_id": "server-session-not-supported"},
    ],
    ids=[
        "null",
        "incomplete-pair",
        "wrong-order",
        "system-role",
        "empty-message",
        "oversized-message",
        "oversized-summary",
        "too-many-messages",
        "oversized-total",
        "extra-state-field",
    ],
)
def test_invalid_chat_state_returns_422_before_any_model_or_retrieval(online_settings, state):
    llm = ScriptedLLM()
    retriever = RecordingRetriever([])
    with TestClient(create_app(online_settings, llm, retriever=retriever)) as client:
        response = client.post(ASK, json={"question": "继续", "chat_state": state})
    assert response.status_code == 422
    assert llm.calls == retriever.calls == []


@pytest.mark.parametrize(
    ("response", "code", "status"),
    [
        (completion("not-json"), "chat_rewrite_invalid", 502),
        (AppError("llm_timeout", "模拟超时", 504), "llm_timeout", 504),
        (AppError("llm_unavailable", "模拟不可用", 503), "llm_unavailable", 503),
    ],
)
def test_rewrite_failure_is_explicit_without_retrieval_or_fallback(
    online_settings, response, code, status
):
    llm = ScriptedLLM(response)
    retriever = RecordingRetriever([])
    with TestClient(create_app(online_settings, llm, retriever=retriever)) as client:
        result = client.post(ASK, json={"question": "那呢？", "chat_state": history()})
    assert result.status_code == status
    assert result.json()["error"]["code"] == code
    assert len(llm.calls) == 1
    assert retriever.calls == []


def test_ambiguous_followup_only_asks_for_clarification_and_keeps_usage(online_settings):
    usage = Usage(prompt_tokens=31, completion_tokens=7, total_tokens=38)
    clarification = "你指的是施工许可材料，还是检查记录？"
    llm = ScriptedLLM(completion({"query": "", "clarification": clarification}, usage=usage))
    retriever = RecordingRetriever([])
    with TestClient(create_app(online_settings, llm, retriever=retriever)) as client:
        result = post_ok(client, "这个呢？", history())
    assert result["status"] == "insufficient_evidence"
    assert result["follow_up_questions"] == [clarification]
    assert result["follow_up_choices"] == [{"question": clarification, "options": []}]
    assert result["claims"] == result["sources"] == result["source_documents"] == []
    assert result["usage"] == usage.model_dump()
    assert len(llm.calls) == 1
    assert retriever.calls == []
    assert result["chat_state"]["messages"][-1]["content"] == result["answer"]
    assert clarification not in result["answer"]


def test_no_hits_keeps_rewrite_usage_and_successful_turn(online_settings):
    usage = Usage(prompt_tokens=17, completion_tokens=3, total_tokens=20)
    llm = ScriptedLLM(completion({"query": "完整施工问题", "clarification": ""}, usage=usage))
    retriever = RecordingRetriever([])
    with TestClient(create_app(online_settings, llm, retriever=retriever)) as client:
        result = post_ok(client, "还有呢？", history())
    assert result["status"] == "insufficient_evidence"
    assert result["follow_up_choices"] == result["follow_up_questions"] == []
    assert result["usage"] == usage.model_dump()
    assert len(result["chat_state"]["messages"]) == 4
    assert len(llm.calls) == 1
    assert retriever.calls == [("完整施工问题", 3)]


@pytest.mark.parametrize(
    ("missing", "code"),
    [("deepseek_api_key", "llm_not_configured"), ("dashscope_api_key", "embedding_not_configured")],
)
def test_missing_keys_are_checked_before_history_processing(online_settings, missing, code):
    configured = online_settings.model_copy(update={missing: SecretStr("")})
    llm = ScriptedLLM()
    retriever = RecordingRetriever([])
    with TestClient(create_app(configured, llm, retriever=retriever)) as client:
        response = client.post(ASK, json={"question": "继续", "chat_state": history()})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == code
    assert llm.calls == retriever.calls == []


def test_large_context_drops_whole_low_ranked_documents_without_truncating(
    online_settings, repository
):
    first, second = repository.searchable[:2]
    first = first.model_copy(update={"text": "完整开头" + "证据" * 1000 + "完整结尾"})
    second = second.model_copy(update={"text": "低排名原文" * 4000})
    configured = online_settings.model_copy(
        update={"chat_context_token_limit": 8000, "deepseek_max_tokens": 128}
    )
    llm = ScriptedLLM(answered(first.id))
    retriever = RecordingRetriever([first, second])
    with TestClient(create_app(configured, llm, retriever=retriever)) as client:
        result = post_ok(client, "施工许可资料说明了什么？")
    context, _, _ = qa_inputs(llm.calls[-1])
    assert len(context) == len(result["source_documents"]) == 1
    assert context[0]["text"] == result["source_documents"][0]["text"] == first.text
    assert context[0]["source"] == first.source.model_dump(mode="json")


def test_single_oversized_source_returns_explicit_error_without_answer_call(
    online_settings, repository
):
    document = repository.searchable[0].model_copy(update={"text": "完整原文" * 15000})
    configured = online_settings.model_copy(
        update={"chat_context_token_limit": 8000, "deepseek_max_tokens": 128}
    )
    llm = ScriptedLLM()
    retriever = RecordingRetriever([document])
    with TestClient(create_app(configured, llm, retriever=retriever)) as client:
        response = client.post(ASK, json={"question": "施工许可资料说明了什么？"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "context_too_large"
    assert llm.calls == []
    assert len(retriever.calls) == 1


def test_compaction_rewrite_answer_usage_and_context_roundtrip(online_settings, repository):
    document = repository.searchable[0]
    old_question = "这条旧补问不应进入摘要。"
    user_text = "用户称面积20，后纠正为18。" * 20 + "\n待补充：\n这是用户自己写的备注。"
    state = {
        "summary": "",
        "messages": history(
            question=user_text,
            answer="此前引用的原文解释。\n待补充：\n" + old_question,
            citation=document.id,
        )["messages"]
        * 6,
    }
    original = copy.deepcopy(state)
    usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    summary = "用户陈述已将面积从20纠正为18平方米，尚未核实。"
    llm = ScriptedLLM(
        completion({"summary": summary}, usage=usage),
        completion(
            {"query": "用户称18平方米施工需哪些许可材料？", "clarification": ""}, usage=usage
        ),
        completion(answered(document.id), usage=usage),
    )
    configured = online_settings.model_copy(
        update={"chat_history_token_limit": 1000, "chat_recent_turns": 1}
    )
    retriever = RecordingRetriever([document])
    with TestClient(create_app(configured, llm, retriever=retriever)) as client:
        result = post_ok(client, "那要哪些材料？", state)
    assert state == original
    assert result["context_compacted"] is True
    assert result["chat_state"]["summary"] == summary
    summary_input = json.loads(llm.calls[0][1]["content"])
    assert summary_input["messages"][0]["content"] == user_text
    assert summary_input["messages"][1] == {
        "role": "assistant",
        "content": "此前引用的原文解释。",
        "citation_ids": [document.id],
    }
    assert all(old_question not in call[1]["content"] for call in llm.calls)
    assert old_question not in json.dumps(result["chat_state"], ensure_ascii=False)
    assert len(result["chat_state"]["messages"]) == 4
    assert result["usage"] == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45}
    assert qa_inputs(llm.calls[-1])[2]["summary"] == summary
    assert not configured.knowledge_path.exists()  # 会话从未写入知识库或另建文件。


def test_only_the_chosen_fact_enters_the_next_question_and_history(online_settings, repository):
    document = repository.searchable[0]
    group = {
        "question": "车辆的登记类型是什么？",
        "options": ["我骑的是电动自行车。", "我骑的是电动摩托车。"],
    }
    confirmed, unused = group["options"]
    llm = ScriptedLLM(
        {
            "claims": [{"text": "可先核对车辆的登记类型。", "citation_ids": [document.id]}],
            "follow_up_choices": [group],
        },
        {"query": "电动自行车在机动车之间穿行怎么处理？", "clarification": ""},
        answered(document.id),
    )
    with TestClient(
        create_app(online_settings, llm, retriever=RecordingRetriever([document]))
    ) as client:
        first = post_ok(client, "穿行怎么判罚？")
        second = post_ok(client, confirmed, first["chat_state"])
    assert first["follow_up_choices"] == [group]
    assert first["follow_up_questions"] == [group["question"]]
    assert first["chat_state"] == history("穿行怎么判罚？", first["answer"], document.id)
    for pending in [group["question"], *group["options"]]:
        assert pending not in json.dumps(first["chat_state"], ensure_ascii=False)
    assert second["chat_state"]["messages"][-2]["content"] == confirmed
    assert second["follow_up_choices"] == second["follow_up_questions"] == []
    for call in llm.calls[1:]:
        # 本轮 question 是用户亲自确认的陈述，摘要改写和回答的 user 载荷均不含未选项。
        assert confirmed in call[1]["content"]
        assert group["question"] not in call[1]["content"] and unused not in call[1]["content"]


def test_one_character_reply_is_a_valid_question(online_settings, repository):
    document = repository.searchable[0]
    llm = ScriptedLLM(
        {"query": "用户称已办理许可，还需核实什么？", "clarification": ""}, answered(document.id)
    )
    with TestClient(
        create_app(online_settings, llm, retriever=RecordingRetriever([document]))
    ) as client:
        result = post_ok(client, "是", history(answer="是否已经办理许可？"))
    assert result["chat_state"]["messages"][-2]["content"] == "是"


def test_budget_excluded_source_cannot_be_cited(online_settings, repository):
    first, second = repository.searchable[:2]
    second = second.model_copy(update={"text": "不能截断的低排名原文" * 5000})
    configured = online_settings.model_copy(
        update={"chat_context_token_limit": 8000, "deepseek_max_tokens": 128}
    )
    llm = ScriptedLLM(answered(second.id))
    with TestClient(
        create_app(configured, llm, retriever=RecordingRetriever([first, second]))
    ) as client:
        response = client.post(ASK, json={"question": "施工许可"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_citation"
    assert [row["id"] for row in qa_inputs(llm.calls[-1])[0]] == [first.id]
