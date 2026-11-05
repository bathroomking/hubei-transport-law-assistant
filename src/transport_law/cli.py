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
from transport_law.repository import LawRepository
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
        ("retrieve", "检索原文；BM25 离线，Chroma 会调用在线向量模型"),
        ("ask", "检索并调用 DeepSeek 生成有引用的解释"),
        ("chat", "单独检查 DeepSeek 模型连接，会使用 API 额度"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("question")
        if name != "chat":
            command.add_argument("--top-k", type=int, default=None)
            command.add_argument("--backend", choices=["bm25", "chroma"])
    serve = commands.add_parser("serve", help="启动本地前端、API 和 /docs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    extract = commands.add_parser("extract-pdf", help="逐页提取 PDF 到待整理区")
    extract.add_argument("path", type=Path)
    extract.add_argument("--output", type=Path, required=True)
    chunk = commands.add_parser("chunk-document", help="用 LangChain 将 PDF/TXT/MD 分块到待审核区")
    chunk.add_argument("path", type=Path)
    chunk.add_argument("--output", type=Path, required=True)
    chunk.add_argument("--document-id", help="可选的稳定文档标识，供同一文档不同版本使用")
    chunk.add_argument("--title", help="文档标题，默认使用文件名")
    chunk.add_argument("--chunk-size", type=int, default=800, help="单块字符数上限，默认 800")
    chunk.add_argument("--chunk-overlap", type=int, default=120, help="目标重叠字符数，默认 120")
    commands.add_parser("validate-data", help="校验条目、唯一 ID 和规则来源引用")
    index = commands.add_parser("index-vectors", help="调用阿里云百炼，把已有分块写入 Chroma")
    index.add_argument(
        "--documents", type=Path, required=True, help="完整 langchain_documents.jsonl"
    )
    index.add_argument("--dry-run", action="store_true", help="仅本地预检，不调用模型或写入索引")
    commands.add_parser("vector-status", help="校验本地向量索引和当前配置，不调用模型")
    evaluate = commands.add_parser("evaluate", help="运行检索回归集；Chroma 模式会调用向量模型")
    evaluate.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evals/retrieval.jsonl")
    evaluate.add_argument("--backend", choices=["bm25", "chroma"])
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
    if args.command == "chunk-document":
        from transport_law.chunking import chunk_document

        print_json(
            chunk_document(
                args.path,
                args.output,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                document_id=args.document_id,
                title=args.title,
            )
        )
        return 0
    settings = Settings()
    if getattr(args, "backend", None):
        settings = settings.model_copy(update={"retrieval_backend": args.backend})
    if args.command in ("index-vectors", "vector-status"):
        from transport_law.vector_store import build_vector_index, open_vector_index

        repository = LawRepository.from_jsonl(settings.law_data_path)
        if args.command == "index-vectors":
            print_json(
                build_vector_index(
                    settings,
                    repository,
                    args.documents,
                    dry_run=args.dry_run,
                    progress=lambda done, total: print(
                        f"向量入库：{done}/{total}", file=sys.stderr
                    ),
                )
            )
        else:
            collection = open_vector_index(settings, repository)
            print_json(
                {
                    "status": "ready",
                    "collection": collection.name,
                    "chunks": collection.count(),
                    "model": collection.metadata["embedding_model"],
                    "dimensions": collection.metadata["embedding_dimensions"],
                }
            )
        return 0
    if args.command == "chat":
        question = SearchRequest(question=args.question).question
        completion = asyncio.run(
            DeepSeekClient(settings).complete(
                [{"role": "user", "content": question}], json_mode=False
            )
        )
        print_json({"answer": completion.content, "model": completion.model})
        return 0
    if args.command in ("validate-data", "check-facts"):
        # 资料/规则检查与向量库和在线 Key 无关。
        settings = settings.model_copy(update={"retrieval_backend": "bm25"})
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
        report["backend"] = settings.retrieval_backend
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
