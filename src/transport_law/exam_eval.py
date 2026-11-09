"""Grade independent multiple-choice questions through the production /ask endpoint."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import httpx

SINGLE_INSTRUCTION = (
    "请根据资料作答。本题只能选择一个选项。有依据时，请写明“答案为 X”，"
    "用所选选项的字母替换 X。依据不足时请说明。"
)
MULTIPLE_INSTRUCTION = (
    "请根据资料作答。有依据时，请在结论中写明“答案为”及所选的全部选项字母；依据不足时请说明。"
)
ITEM_ERRORS = {"invalid_citation", "llm_invalid_json", "llm_incomplete", "context_too_large"}
CONFIG_ERRORS = {
    "llm_auth_error",
    "llm_balance_error",
    "llm_not_configured",
    "embedding_auth_error",
    "embedding_balance_error",
    "embedding_not_configured",
    "embedding_access_denied",
    "embedding_not_activated",
    "chroma_not_installed",
    "vector_index_mismatch",
}
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/api/v1/ask"
LETTERS = r"[A-Z](?:\s*(?:[,、和及与或/]\s*(?:选项\s*)?)?[A-Z])*"
START = r"(?:^|[\s。,:;!?])(?:所以|因此|故)?\s*(?:本题|题干|题目)?"
END = r"(?=$|[\s。,:;、!()【】「」“”\".])"
ANSWER_MARKER = re.compile(
    START
    + r"(?:(?:正确答案|答案|正确选项|对应选项)(?:应当|应)?(?:为|是|选)?|应(?:当)?选(?:择)?)"
    + r"\s*:?\s*(?:选项\s*)?[“\"「【(]?\s*("
    + LETTERS
    + r"|正确|错误|对|错)(?:选项|项)?"
    + END
)
OPTION_VERDICT = re.compile(
    START + r"(?:选项\s*)?(" + LETTERS + r")\s*(?:选项|项)?(?:为|是)?正确(?:的)?" + END
)
BINARY_VERDICT = re.compile(
    START
    + r"(?:题干|题目|题中|该|这个|这一|上述|这种|此|此题|题干中的)?"
    + r"(?:说法|描述|表述|判断|做法|行为)(?:是|为)?(不正确|正确|错误)(?:的)?"
    + END
)
ALTERNATIVE = re.compile(r"[\s,、]*[或/]")


def prompt_for(case: dict) -> str:
    options = [f"{key}. {value}" for key, value in case["options"].items()]
    kind = case.get("type")
    if kind in {"single_choice", "true_false"}:
        label = "单选题" if kind == "single_choice" else "判断题"
        instruction = f"{label}。{SINGLE_INSTRUCTION}"
    else:
        instruction = MULTIPLE_INSTRUCTION
    return "\n".join([case["question"], *options, instruction])


def selection(answer: str, options: dict[str, str]) -> str | None:
    text = unicodedata.normalize("NFKC", answer).replace("**", "").strip()
    matches = list(ANSWER_MARKER.finditer(text))
    if not matches:
        matches = list(OPTION_VERDICT.finditer(text))
    if not matches and set(options.values()) == {"正确", "错误"}:
        matches = list(BINARY_VERDICT.finditer(text))
    if any(ALTERNATIVE.match(text, match.end()) for match in matches):
        return None
    tokens = [match[1] for match in matches]
    if not tokens:
        # Accept a whole-answer label or option verbatim. Prose is never matched by substring.
        bare = text.strip("。.!！ ")
        tokens = [bare] if bare in options.values() or re.fullmatch(LETTERS, bare) else []
    choices = set()
    for token in tokens:
        if token in options.values() or token in {"对", "错", "不正确"}:
            value = {"对": "正确", "错": "错误", "不正确": "错误"}.get(token, token)
            keys = [key for key, option in options.items() if option == value]
            if len(keys) != 1:
                return None
        else:
            if "或" in token or "/" in token:
                return None
            keys = re.findall("[A-Z]", token)
        if not keys or any(key not in options for key in keys):
            return None
        choices.add("".join(sorted(set(keys))))
    return choices.pop() if len(choices) == 1 else None


def load_cases(path: Path) -> list[dict]:
    cases = []
    ids = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        ident = case["id"]
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError(f"第 {line_no} 行 ID 重复或无效")
        ids.add(ident)
        if not case.get("skip_reason"):
            options, expected = case["options"], case["expected"]
            if (
                not isinstance(case["question"], str)
                or not case["question"].strip()
                or not isinstance(options, dict)
                or len(options) < 2
                or any(
                    not re.fullmatch("[A-Z]", k) or not isinstance(v, str) or not v.strip()
                    for k, v in options.items()
                )
                or not isinstance(expected, str)
                or not expected
                or any(key not in options for key in expected)
                or len(set(expected)) != len(expected)
                or len(prompt_for(case)) > 2000
            ):
                raise ValueError(f"第 {line_no} 行题干、选项或答案无效")
            case["expected"] = "".join(sorted(expected))
        cases.append(case)
    if not cases:
        raise ValueError("题库为空")
    return cases


def identity(case: dict, endpoint: str) -> dict:
    prompt = prompt_for(case) if not case.get("skip_reason") else ""
    return {
        "id": case["id"],
        "expected": case.get("expected"),
        "request_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "skip_reason": case.get("skip_reason"),
        "endpoint": endpoint,
    }


def read_results(log: BinaryIO, identities: dict) -> dict:
    # flock also guards crash recovery and the interval between loading and appending.
    fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log.seek(0)
    rows = {}
    while True:
        start = log.tell()
        line = log.readline()
        if not line:
            break
        if not line.endswith(b"\n"):
            # A write interrupted before the final newline may contain partial UTF-8/JSON.
            log.truncate(start)
            break
        row = json.loads(line)
        ident = row["id"]
        if ident not in identities or any(row.get(k) != v for k, v in identities[ident].items()):
            raise ValueError(f"结果文件与数据或服务地址不匹配 ({ident})，请使用新的结果路径")
        rows[ident] = row
    return rows


def grade_answer(answer: str, case: dict) -> dict:
    predicted = selection(answer, case["options"])
    if predicted is None:
        return {"status": "unparsed", "predicted": None}
    status = "correct" if predicted == case["expected"] else "wrong"
    return {"status": status, "predicted": predicted}


async def ask(client: httpx.AsyncClient, endpoint: str, case: dict) -> dict:
    try:
        response = await client.post(endpoint, json={"question": prompt_for(case)})
        if not response.is_success:
            try:
                code = response.json()["error"]["code"]
            except (ValueError, KeyError, TypeError):
                code = "http_error"
            return {"status": "error", "http_status": response.status_code, "error": code}
        payload = response.json()
        answer, status = payload["answer"], payload["status"]
        if not isinstance(answer, str) or status not in {"answered", "insufficient_evidence"}:
            raise ValueError("Unexpected answer response")
        grade = (
            {"status": "abstained", "predicted": None}
            if status == "insufficient_evidence"
            else grade_answer(answer, case)
        )
        return {
            **grade,
            "answer": answer,
            "model": payload.get("model"),
            "usage": payload.get("usage"),
            "source_ids": [source["id"] for source in payload.get("sources", [])],
            "retrieved_ids": [source["id"] for source in payload.get("source_documents", [])],
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return {"status": "error", "error": type(exc).__name__}


def summarize(cases: list[dict], rows: Iterable[dict]) -> dict:
    counts = Counter(row["status"] for row in rows)
    eligible = sum(not c.get("skip_reason") for c in cases)
    attempted = sum(counts[key] for key in ("correct", "wrong", "unparsed", "abstained", "error"))
    return {
        "total": len(cases),
        "eligible": eligible,
        "excluded": len(cases) - eligible,
        "attempted": attempted,
        "pending": eligible - attempted,
        **{k: counts[k] for k in ("correct", "wrong", "abstained", "unparsed", "error")},
        "accuracy": counts["correct"] / attempted if attempted else None,
    }


async def evaluate(
    cases: list[dict],
    output: Path,
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    workers: int,
    limit: int | None,
    retry_errors: bool,
    rescore_only: bool = False,
) -> dict:
    identities = {case["id"]: identity(case, endpoint) for case in cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a+b") as log:
        rows = read_results(log, identities)

        def record(ident, result):
            row = {**identities[ident], **result}
            log.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
            log.flush()
            rows[ident] = row

        if rescore_only:
            changed = 0
            for case in cases:
                row = rows.get(case["id"])
                if row and row["status"] in {"correct", "wrong", "unparsed"}:
                    grade = grade_answer(row["answer"], case)
                    if grade["status"] != row["status"] or grade["predicted"] != row["predicted"]:
                        record(case["id"], {**row, **grade})
                        changed += 1
            return {**summarize(cases, rows.values()), "rescored_changes": changed}

        pending = []
        for case in cases:
            previous = rows.get(case["id"])
            if previous and not (retry_errors and previous["status"] == "error"):
                continue
            if case.get("skip_reason"):
                record(case["id"], {"status": "skipped"})
            else:
                pending.append(case)

        work = iter(pending[:limit])
        consecutive_errors = completed = 0
        halted = False

        async def worker():
            nonlocal consecutive_errors, completed, halted
            while not halted:
                case = next(work, None)
                if case is None:
                    return
                result = await ask(client, endpoint, case)
                record(case["id"], result)
                completed += 1
                if result["status"] == "error" and result["error"] not in ITEM_ERRORS:
                    consecutive_errors += 1
                    if (
                        consecutive_errors >= 3
                        or result["error"] in CONFIG_ERRORS
                        or result.get("http_status") in {401, 403, 404}
                    ):
                        halted = True
                else:
                    consecutive_errors = 0
                if completed % 25 == 0:
                    print(f"本轮已处理 {completed} 道", flush=True)

        await asyncio.gather(*(worker() for _ in range(workers)))
        return {**summarize(cases, rows.values()), "stopped_on_errors": halted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path, help="逐题 JSONL，重复运行时跳过已完成题目")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--limit", type=int, help="本轮最多请求的可测题数")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--retry-errors", action="store_true", help="重新请求上次调用失败的题目")
    parser.add_argument("--rescore", action="store_true", help="仅重判已保存的回答，无网络请求")
    args = parser.parse_args()
    if args.rescore and (not args.output.is_file() or args.retry_errors or args.limit):
        parser.error("--rescore 需要已有的结果文件，且与 --retry-errors、--limit 互斥")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须为正整数")
    if args.output.resolve() == args.dataset.resolve():
        parser.error("结果文件和题库不能相同")
    cases = load_cases(args.dataset)

    async def run():
        async with httpx.AsyncClient(timeout=210, trust_env=False) as client:
            return await evaluate(
                cases,
                args.output,
                client,
                endpoint=args.endpoint,
                workers=args.workers,
                limit=args.limit,
                retry_errors=args.retry_errors,
                rescore_only=args.rescore,
            )

    summary = asyncio.run(run())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("stopped_on_errors") or summary["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
