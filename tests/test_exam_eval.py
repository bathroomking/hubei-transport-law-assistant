import asyncio
import json

import httpx
import pytest

from transport_law.exam_eval import evaluate, load_cases, selection


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("答案为：Ａ。", "A"),
        ("答案为 A、C 和 B。", "ABC"),
        ("不考虑 A。正确答案为 B。", "B"),
        ("答案为 A。答案为 B。", None),
        ("在 A 与 B 中选择。", None),
        ("答案为 B 或 C。", None),
        ("答案为 X。", None),
        ("不能选择 A。", None),
        ("**正确答案：B**。", "B"),
        ("C。", "C"),
    ],
)
def test_selection(answer, expected):
    assert selection(answer, {"A": "甲", "B": "乙", "C": "丙"}) == expected


def test_tf_uses_supplied_options():
    assert selection("答案为错误。", {"A": "错误", "B": "正确"}) == "A"
    assert selection("错误。", {"A": "正确", "B": "错误"}) == "B"
    assert selection("该规则适用于纠正错误。", {"A": "正确", "B": "错误"}) is None


def run_eval(cases, output, handler, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await evaluate(
                cases,
                output,
                client,
                endpoint="http://local/ask",
                workers=1,
                limit=kwargs.get("limit"),
                retry_errors=kwargs.get("retry_errors", False),
            )

    return asyncio.run(run())


def make_case(i):
    return {
        "id": str(i),
        "question": f"题目{i}？",
        "options": {"A": "是", "B": "否"},
        "expected": "B",
        "explanation": "never reveal this answer key",
    }


def test_end_to_end_scoring_resume_and_error_retry(tmp_path):
    cases = [make_case(i) for i in range(6)]
    cases[-1]["skip_reason"] = "需图像"
    payloads = iter(
        [
            {"status": "answered", "answer": "答案为 B。"},
            {"status": "insufficient_evidence", "answer": "无法回答。"},
            {"status": "answered", "answer": "参考法规具体规定。"},
            {"status": "answered", "answer": "答案为 A。"},
        ]
    )
    requests = []

    def handler(request):
        body = json.loads(request.content)
        assert set(body) == {"question"}
        assert "never reveal" not in body["question"]
        assert "A. 是\nB. 否" in body["question"]
        requests.append(body["question"])
        if body["question"].startswith("题目4"):
            return httpx.Response(502, json={"error": {"code": "invalid_citation"}})
        return httpx.Response(200, json=next(payloads))

    out = tmp_path / "results.jsonl"
    pilot = run_eval(cases, out, handler, limit=1)
    assert pilot["attempted"] == 1 and pilot["pending"] == 4
    with out.open("ab") as fh:
        fh.write(b'{"id":')
    summary = run_eval(cases, out, handler)
    assert summary == {
        "total": 6,
        "eligible": 5,
        "excluded": 1,
        "attempted": 5,
        "pending": 0,
        "correct": 1,
        "wrong": 1,
        "abstained": 1,
        "unparsed": 1,
        "error": 1,
        "accuracy": 0.2,
        "stopped_on_errors": False,
    }
    assert len(set(requests)) == 5
    run_eval(cases, out, lambda _: pytest.fail("已完成的题目不应再次调用"))
    fixed = run_eval(
        cases,
        out,
        lambda _: httpx.Response(200, json={"status": "answered", "answer": "答案为 B。"}),
        retry_errors=True,
    )
    assert fixed["error"] == 0 and fixed["correct"] == 2
    assert len(out.read_text().splitlines()) == 7
    cases[0]["question"] = "不同的问题"
    with pytest.raises(ValueError, match="不匹配"):
        run_eval(cases, out, handler)


def test_stop_dispatch_after_persistent_errors(tmp_path):
    summary = run_eval(
        [make_case(i) for i in range(10)],
        tmp_path / "log",
        lambda _: httpx.Response(503, json={}),
    )
    assert summary["stopped_on_errors"] and summary["error"] == 3 and summary["pending"] == 7


def test_reject_duplicate_ids_before_requests(tmp_path):
    row = json.dumps(make_case(0))
    data = tmp_path / "exam"
    data.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ID 重复"):
        load_cases(data)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("应听从现场交通警察的指挥，正确选项为 C。", "C"),
        ("因此题干对应选项为C（交通警察）。", "C"),
        ("答案为 B“扣留”。", "B"),
        ("答案为 B项。", "B"),
        ("因此应选择C。", "C"),
        ("答案为 C（交通警察的指挥）。", "C"),
        ("选项 C 正确。", "C"),
        ("答案为 A 或选项 C。", None),
        ("答案为 A 或者 C。", None),
        ("正确选项为 A、选项 B。", "AB"),
        ("与选项 C 正确与否无关。", None),
    ],
)
def test_explicit_selection_phrasings(answer, expected):
    assert selection(answer, {"A": "甲", "B": "乙", "C": "丙"}) == expected


def test_offline_rescore_changes_only_existing_answer(tmp_path):
    cases = [make_case(1), make_case(2)]
    out = tmp_path / "results.jsonl"
    run_eval(
        cases,
        out,
        lambda _: httpx.Response(200, json={"status": "answered", "answer": "一般说明。"}),
        limit=1,
    )
    row = json.loads(out.read_text())
    row["answer"] = "正确选项为 B。"
    out.write_text(json.dumps(row) + "\n", encoding="utf-8")

    async def offline():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("重评分不应调用服务"))
        ) as client:
            return await evaluate(
                cases,
                out,
                client,
                endpoint="http://local/ask",
                workers=1,
                limit=None,
                retry_errors=False,
                rescore_only=True,
            )

    result = asyncio.run(offline())
    assert result["correct"] == result["rescored_changes"] == result["pending"] == 1
    assert asyncio.run(offline())["rescored_changes"] == 0


@pytest.mark.parametrize("kind,label", [("true_false", "判断题"), ("single_choice", "单选题")])
def test_single_selection_instruction_excludes_key(kind, label, tmp_path):
    case = {**make_case(1), "type": kind}

    def handler(request):
        q = json.loads(request.content)["question"]
        assert label in q and "只能选择一个选项" in q and "全部选项字母" not in q
        assert "答案为 B" not in q and "答案为 X" in q
        return httpx.Response(200, json={"status": "answered", "answer": "答案为 B。"})

    assert run_eval([case], tmp_path / "out", handler)["correct"] == 1


def test_item_errors_do_not_halt_other_questions(tmp_path):
    result = run_eval(
        [make_case(i) for i in range(5)],
        tmp_path / "out",
        lambda _: httpx.Response(502, json={"error": {"code": "invalid_citation"}}),
    )
    assert result["error"] == 5 and not result["stopped_on_errors"]


def test_bad_credentials_halt_without_repeated_requests(tmp_path):
    result = run_eval(
        [make_case(i) for i in range(5)],
        tmp_path / "out",
        lambda _: httpx.Response(503, json={"error": {"code": "embedding_auth_error"}}),
    )
    assert result["error"] == 1 and result["stopped_on_errors"]
