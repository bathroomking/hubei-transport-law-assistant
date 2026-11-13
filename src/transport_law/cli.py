import argparse
import asyncio
import json
import sys

from pydantic import BaseModel, ValidationError

from transport_law.config import Settings
from transport_law.errors import AppError
from transport_law.runtime import build_runtime
from transport_law.schemas import SearchRequest


def print_json(value):
    if isinstance(value, BaseModel):
        print(value.model_dump_json(indent=2))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="湖北交通运输法规与裁量助手")
    commands = root.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="检索已入库的上传资料并生成有出处的白话解释")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    serve = commands.add_parser("serve", help="启动本地前端、API 和 /docs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    commands.add_parser("graph-build", help="对当前已入库资料完整建图，并为后续上传启用自动建图")
    commands.add_parser("graph-status", help="查询当前图索引，不调用模型")
    return root


def run(args) -> int:
    if args.command in {"graph-build", "graph-status"}:
        from transport_law.knowledge import KnowledgeStore

        jobs = KnowledgeStore(Settings()).graph
        status = jobs.build() if args.command == "graph-build" else jobs.status()
        print_json(status)
        return int(args.command == "graph-build" and status["status"] not in {"ready", "empty"})
    if args.command == "serve":
        import uvicorn

        uvicorn.run("transport_law.main:app", host=args.host, port=args.port, reload=args.reload)
        return 0
    if args.command == "ask":
        request = SearchRequest(question=args.question, top_k=args.top_k)
        runtime = build_runtime(Settings())
        print_json(asyncio.run(runtime.service.ask(request)))
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
