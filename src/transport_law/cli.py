import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from transport_law.config import PROJECT_ROOT, Settings
from transport_law.errors import AppError
from transport_law.evaluation import evaluate_retrieval
from transport_law.ingest import extract_pdf
from transport_law.llm import DeepSeekClient
from transport_law.runtime import build_runtime
from transport_law.schemas import CaseFacts, SearchRequest


def print_json(value):
    if isinstance(value, BaseModel):
        print(value.model_dump_json(indent=2))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="湖北交通运输法规与裁量助手")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("retrieve", "检索原文，不调用模型"),
        ("ask", "检索并调用 DeepSeek 生成有引用的解释"),
        ("chat", "单独检查 DeepSeek 模型连接，会使用 API 额度"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("question")
        if name != "chat":
            command.add_argument("--top-k", type=int, default=None)
    serve = commands.add_parser("serve", help="启动本地 API 和 /docs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    extract = commands.add_parser("extract-pdf", help="逐页提取 PDF 到待整理区")
    extract.add_argument("path", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    commands.add_parser("validate-data", help="校验条目、唯一 ID 和规则来源引用")
    evaluate = commands.add_parser("evaluate", help="运行离线检索回归集")
    evaluate.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evals/retrieval.jsonl")
    facts = commands.add_parser("check-facts", help="读取案件 JSON，检查缺失事实和规则核验状态")
    facts.add_argument("path", type=Path)
    return root


def run(args) -> int:
    if args.command == "serve":
        import uvicorn

        uvicorn.run("transport_law.main:app", host=args.host, port=args.port, reload=args.reload)
        return 0
    if args.command == "extract-pdf":
        print_json(extract_pdf(args.path, args.output))
        return 0
    settings = Settings()
    if args.command == "chat":
        question = SearchRequest(question=args.question).question
        completion = asyncio.run(
            DeepSeekClient(settings).complete(
                [{"role": "user", "content": question}], json_mode=False
            )
        )
        print_json({"answer": completion.content, "model": completion.model})
        return 0
    runtime = build_runtime(settings)
    if args.command in ("ask", "retrieve"):
        request = SearchRequest(question=args.question, top_k=args.top_k)
        if args.command == "ask":
            print_json(asyncio.run(runtime.service.ask(request)))
        else:
            print_json(runtime.service.retrieve(request))
    elif args.command == "validate-data":
        print_json(
            {
                "status": "ok",
                "chunks": len(runtime.repository.searchable),
                "rule_status": runtime.discretion.rule_set.review_status,
            }
        )
    elif args.command == "evaluate":
        report = evaluate_retrieval(args.dataset, runtime.service.retriever)
        print_json(report)
        return 0 if report["passed"] == report["total"] else 1
    elif args.command == "check-facts":
        facts = CaseFacts.model_validate_json(args.path.read_text(encoding="utf-8"))
        print_json(runtime.discretion.evaluate_rules(facts))
    return 0


def main() -> None:
    args = parser().parse_args()
    try:
        code = run(args)
    except AppError as exc:
        print_json({"error": {"code": exc.code, "message": exc.message}})
        code = 1
    except ValidationError:
        # Pydantic 的原始错误可能包含输入数据，不将配置中的敏感值打印到终端。
        print_json(
            {
                "error": {
                    "code": "invalid_input",
                    "message": "输入或配置格式错误，请检查字段类型和范围。",
                }
            }
        )
        code = 1
    except (OSError, ValueError):
        print_json({"error": {"code": "invalid_file", "message": "文件读取失败或格式不正确。"}})
        code = 1
    sys.exit(code)
