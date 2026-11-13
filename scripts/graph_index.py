"""在单独的 GraphRAG 3.3.0 环境中对一份已就绪原文快照执行标准索引。"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import lancedb
import pandas as pd
from graph_checkpoint import RESUME_WORKFLOWS, BuildObserver, has_extraction_error, snapshot_key
from graphrag.api import build_index
from graphrag.config.embeddings import default_embeddings
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag_llm.completion import register_completion
from graphrag_llm.completion.lite_llm_completion import LiteLLMCompletion
from pydantic import ValidationError


class ProviderBalanceError(RuntimeError):
    def __init__(self, stage):
        super().__init__("模型账户余额不足")
        self.stage = stage


class StopOnBalance(logging.Handler):
    """微软部分工作流会消费异常；首条真实 402 日志要取消整次流水线。"""

    def __init__(self, observer):
        super().__init__(logging.ERROR)
        self.observer = observer
        self.exhausted = False
        self.loop = asyncio.get_running_loop()

    def emit(self, record):
        if self.exhausted:
            return
        text = record.getMessage().lower()
        exc = record.exc_info[1] if record.exc_info else None
        if (
            getattr(exc, "status_code", None) == 402
            or "insufficient_balance" in text
            or "insufficient balance" in text
        ):
            self.exhausted = True
            self.loop.call_soon_threadsafe(self.observer.task.cancel)


class DeepSeekCompletion(LiteLLMCompletion):
    """保留微软的请求缓存和重试，将 Pydantic schema 映射为 DeepSeek JSON 输出。"""

    async def completion_async(self, /, **kwargs):
        schema = kwargs.pop("response_format", None)
        if schema is None:
            return await super().completion_async(**kwargs)
        messages = kwargs.pop("messages")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        instruction = (
            "以简体中文返回符合以下 JSON Schema 的 JSON 对象。保持字典键不变。\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        messages = [{"role": "system", "content": instruction}, *messages]
        extra = {"response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}}
        for attempt in range(3):
            result = await super().completion_async(messages=messages, extra_body=extra, **kwargs)
            try:
                result.formatted_response = schema.model_validate_json(result.content)
                return result
            except ValidationError as error:
                if attempt == 2:
                    raise
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.content},
                    {
                        "role": "user",
                        "content": "修正以下 JSON 格式错误："
                        + str(error.errors(include_input=False)),
                    },
                ]


def make_config(output, cache, concurrency):
    dims = int(os.getenv("EMBEDDING_DIMENSIONS", "2048"))
    chat_model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    retry = {"type": "exponential_backoff", "max_retries": 4, "base_delay": 2, "max_delay": 16}
    common = {"model_provider": "openai", "retry": retry, "metrics": None}
    completion = {
        **common,
        "type": "deepseek_json",
        "model": chat_model,
        "api_base": "https://api.deepseek.com",
        "api_key": os.environ["DEEPSEEK_API_KEY"],
        "call_args": {
            "temperature": 0,
            "timeout": 120,
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    }
    embedding = {
        **common,
        "model": embedding_model,
        "api_key": os.environ["DASHSCOPE_API_KEY"],
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "call_args": {
            "extra_body": {"dimensions": dims},
            "encoding_format": "float",
            "timeout": 120,
        },
    }
    cache_id = hashlib.sha256(f"3.3.0:{chat_model}:{embedding_model}:{dims}".encode()).hexdigest()[
        :20
    ]
    return GraphRagConfig.model_validate(
        {
            "completion_models": {"default_completion_model": completion},
            "embedding_models": {"default_embedding_model": embedding},
            "concurrent_requests": concurrency,
            "output_storage": {"base_dir": str(output)},
            "reporting": {"base_dir": str(output / "logs")},
            "cache": {
                "type": "json",
                "storage": {"type": "file", "base_dir": str(cache / cache_id)},
            },
            "vector_store": {
                "type": "lancedb",
                "db_uri": str(output / "lancedb"),
                "vector_size": dims,
            },
            "chunking": {"size": 1200, "overlap": 80, "prepend_metadata": ["title"]},
            "embed_text": {
                "batch_size": 8,
                "batch_max_tokens": 6000,
                "model_instance_name": f"dashscope_vectors_{dims}",
            },
            "extract_graph": {
                "prompt": str(Path(__file__).with_name("graph_extract_zh.txt").resolve()),
                "max_gleanings": 0,
                "entity_types": ["法规条款", "管理主体", "行为", "措施", "条件"],
            },
            "cluster_graph": {"use_lcc": False},
        }
    )


def build_documents(path):
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            c = json.loads(line)
            rows.append(
                {
                    "id": c["id"],
                    "human_readable_id": len(rows),
                    "title": f"{c['title']} {c['article']} [{c['id']}]",
                    "text": c["text"],
                    "creation_date": "",
                    "raw_data": None,
                }
            )
    docs = pd.DataFrame(rows)
    if docs.empty or not docs["id"].is_unique or not docs["text"].str.strip().all():
        raise ValueError("原文为空或原文 ID 重复")
    return docs


def export_graph(output, expected, config):
    def read(name):
        return pd.read_parquet(output / f"{name}.parquet")

    units, documents = read("text_units"), read("documents")
    unit_to_chunk = dict(zip(units.id, units.document_id, strict=True))
    if (
        set(documents.id) != expected
        or set(unit_to_chunk.values()) != expected
        or not units.id.is_unique
    ):
        raise ValueError("图谱原文来源与输入快照不一致")

    def refs(ids):
        return sorted({unit_to_chunk[ident] for ident in ids})

    ent, rel, communities, reports = map(
        read, ("entities", "relationships", "communities", "community_reports")
    )
    if (
        ent.empty
        or rel.empty
        or reports.empty
        or set(communities.community) != set(reports.community)
    ):
        raise ValueError("图谱实体、关系或社区报告不完整")
    nodes = [{"name": r.title, "chunk_ids": refs(r.text_unit_ids)} for r in ent.itertuples()]
    edges = [
        {
            "source": r.source,
            "target": r.target,
            "description": r.description,
            "chunk_ids": refs(r.text_unit_ids),
        }
        for r in rel.itertuples()
    ]
    if any(not obj["chunk_ids"] for obj in [*nodes, *edges]):
        raise ValueError("图谱关系或实体缺少原文出处")
    # Microsoft can log an extraction exception and continue with an empty dataframe.
    # This differs from a valid unit with no mentions; the latter may still be indexed.
    if has_extraction_error(output):
        raise RuntimeError("上游 GraphRAG 记录了部分抽取错误，不能发布该索引")
    db = lancedb.connect(output / "lancedb")
    counts = {}
    for name, rows in zip(default_embeddings, (ent, reports, units), strict=True):
        schema = config.vector_store.index_schema[name]
        table = db.open_table(schema.index_name)
        counts[name] = table.count_rows()
        if (
            counts[name] != len(rows)
            or table.schema.field(schema.vector_field).type.list_size != schema.vector_size
        ):
            raise ValueError(f"向量索引不完整：{name}")
    (output / "graph.json").write_text(
        json.dumps({"entities": nodes, "relationships": edges}, ensure_ascii=False)
    )
    return {
        "chunks": len(expected),
        "text_units": len(units),
        "entities": len(nodes),
        "relationships": len(edges),
        "chunks_without_relationships": len(expected - {k for e in edges for k in e["chunk_ids"]}),
        "communities": len(communities),
        "community_reports": len(reports),
        "vectors": counts,
        "graphrag_version": version("graphrag"),
        "method": "standard",
    }


async def run(args):
    if version("graphrag") != "3.3.0":
        raise RuntimeError("请在 GraphRAG 3.3.0 专用环境运行")
    output, cache = args.output.resolve(), args.cache.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录需为空；重试时使用新目录并复用缓存")
    output.mkdir(parents=True, exist_ok=True)
    docs = build_documents(args.input)
    register_completion("deepseek_json", DeepSeekCompletion)
    config = make_config(output, cache, args.concurrency)
    key = snapshot_key(args.input, config)
    checkpoint = cache.parent / "checkpoints" / key
    observer = BuildObserver(output, checkpoint)
    resumed = observer.restore()
    if resumed:
        config.workflows = RESUME_WORKFLOWS
    guard = StopOnBalance(observer)
    root_logger = logging.getLogger()
    root_logger.addHandler(guard)
    try:
        results = await build_index(
            config,
            input_documents=None if resumed else docs,
            callbacks=[observer],
            method="standard",
        )
    except asyncio.CancelledError:
        if guard.exhausted:
            raise ProviderBalanceError(observer.stage) from None
        raise
    finally:
        root_logger.removeHandler(guard)
    for result in results:
        if result.error is not None:
            raise RuntimeError(f"GraphRAG 工作流失败：{result.workflow}") from result.error
    manifest = export_graph(output, set(docs.id), config)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def _safe_failure(exc):
    if isinstance(exc, ProviderBalanceError):
        return {"stage": exc.stage, "category": "balance"}
    stage = "index"
    categories = {
        "AuthenticationError": "authentication",
        "PermissionDeniedError": "authentication",
        "RateLimitError": "rate_limit",
        "APITimeoutError": "timeout",
        "Timeout": "timeout",
        "TimeoutError": "timeout",
        "ConnectTimeout": "timeout",
        "ReadTimeout": "timeout",
        "APIConnectionError": "network",
        "ConnectError": "network",
        "ValidationError": "validation",
        "ValueError": "validation",
        "JSONDecodeError": "validation",
        "FileNotFoundError": "storage",
        "PermissionError": "storage",
        "OSError": "storage",
        "ImportError": "dependency",
        "ModuleNotFoundError": "dependency",
        "APIError": "upstream",
        "ServiceUnavailableError": "upstream",
        "BadRequestError": "upstream",
        "InternalServerError": "upstream",
    }
    current, category, seen = exc, "unknown", set()
    while current and id(current) not in seen:
        seen.add(id(current))
        for cls in type(current).__mro__:
            if cls.__name__ in categories:
                category = categories[cls.__name__]
                break
        tb = current.__traceback__
        while tb:
            frame = tb.tb_frame
            func = frame.f_code.co_name
            if frame.f_code.co_filename == __file__:
                if func == "export_graph":
                    stage = "export"
                elif func in {"make_config", "build_documents"}:
                    stage = "configure"
                elif func == "run" and (res := frame.f_locals.get("result")) is not None:
                    # PipelineRunResult.workflow is supplied by the installed standard pipeline.
                    stage = res.workflow
            tb = tb.tb_next
        current = current.__cause__ or current.__context__
    return {"stage": stage, "category": category}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "cache"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--concurrency", type=int, default=16, choices=range(1, 65), metavar="1..64"
    )
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except Exception as exc:
        failure = _safe_failure(exc)
        if args.output.is_dir() and not (args.output / "manifest.json").exists():
            (args.output / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
        print(json.dumps(failure), file=sys.stderr)
        raise SystemExit(1) from None
