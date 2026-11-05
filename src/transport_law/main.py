from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from transport_law import __version__
from transport_law.config import Settings
from transport_law.errors import AppError
from transport_law.llm import LanguageModel
from transport_law.runtime import build_runtime
from transport_law.schemas import (
    AnswerResponse,
    CaseFacts,
    DiscretionCheck,
    DiscretionRuleSet,
    LegalChunk,
    SearchRequest,
    SearchResponse,
)


def create_app(settings: Settings | None = None, llm: LanguageModel | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = build_runtime(settings, llm)
        yield

    app = FastAPI(
        title="湖北交通运输法规与裁量助手",
        version=__version__,
        description=(
            "学习原型：检索上传材料、返回原文与页码、收集裁量事实。"
            "目前覆盖普通公路‘擅自占用、挖掘公路’一个事项。"
            "问答接口在命中资料后调用 DeepSeek；文件效力及个案适用尚未核验。"
        ),
        lifespan=lifespan,
    )
    web_directory = Path(__file__).resolve().parent / "web"
    app.mount("/static", StaticFiles(directory=web_directory), name="static")

    @app.exception_handler(AppError)
    async def application_error(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web_directory / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health", tags=["状态"], summary="本地配置和资料状态（不调用模型）")
    def health(request: Request):
        runtime = request.app.state.runtime
        chunks = runtime.repository.searchable
        return {
            "status": "ok",
            "version": __version__,
            "chunk_count": len(chunks),
            "item_count": len({chunk.item_id for chunk in chunks}),
            "pending_verification_count": sum(
                chunk.source.verification_status == "pending" for chunk in chunks
            ),
            "llm_configured": runtime.settings.llm_configured,
            "llm_model": runtime.settings.deepseek_model,
            "llm_connectivity": "not_checked",
            "retrieval_backend": runtime.settings.retrieval_backend,
            "embedding_configured": runtime.settings.embedding_configured,
            "embedding_model": runtime.settings.embedding_model,
            "embedding_connectivity": "not_checked",
        }

    @app.post("/api/v1/retrieve", response_model=SearchResponse, tags=["法规检索"])
    def retrieve(payload: SearchRequest, request: Request):
        return request.app.state.runtime.service.retrieve(payload)

    @app.get("/api/v1/articles/{chunk_id}", response_model=LegalChunk, tags=["法规检索"])
    def get_article(chunk_id: str, request: Request):
        return request.app.state.runtime.repository.get_article(chunk_id)

    @app.post("/api/v1/ask", response_model=AnswerResponse, tags=["带依据问答"])
    async def ask(payload: SearchRequest, request: Request):
        return await request.app.state.runtime.service.ask(payload)

    @app.get("/api/v1/discretion/{item_id}", response_model=DiscretionRuleSet, tags=["裁量辅助"])
    def get_rules(item_id: str, request: Request):
        return request.app.state.runtime.discretion.get_discretion_rules(item_id)

    @app.post(
        "/api/v1/discretion/check",
        response_model=DiscretionCheck,
        tags=["裁量辅助"],
        summary="检查缺失事实和复核事项（首版不计算罚款）",
    )
    def check_facts(payload: CaseFacts, request: Request):
        return request.app.state.runtime.discretion.evaluate_rules(payload)

    return app


app = create_app()
