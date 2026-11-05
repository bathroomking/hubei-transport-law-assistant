from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from transport_law import __version__
from transport_law.config import Settings
from transport_law.documents import LawDocument, load_law_document
from transport_law.errors import AppError
from transport_law.knowledge_api import router as knowledge_router
from transport_law.llm import LanguageModel
from transport_law.retrieval import Retriever
from transport_law.runtime import build_runtime
from transport_law.schemas import (
    AnswerResponse,
    LegalChunk,
    SearchRequest,
)


def create_app(
    settings: Settings | None = None,
    llm: LanguageModel | None = None,
    *,
    retriever: Retriever | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = build_runtime(settings, llm, retriever=retriever)
        yield

    app = FastAPI(
        title="湖北交通运输法规与裁量助手",
        version=__version__,
        description=(
            "在线交通法规与裁量问答：白话提问，混合检索已入库的上传资料，"
            "由 DeepSeek 根据原文回答并提供出处。文件效力及个案适用尚未核验。"
        ),
        lifespan=lifespan,
    )
    web_directory = Path(__file__).resolve().parent / "web"
    app.mount("/static", StaticFiles(directory=web_directory), name="static")
    app.include_router(knowledge_router)

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
            "online_only": True,
            "version": __version__,
            "chunk_count": len(chunks),
            "item_count": len({chunk.item_id for chunk in chunks}),
            "pending_verification_count": sum(
                chunk.source.verification_status == "pending" for chunk in chunks
            ),
            "llm_configured": runtime.settings.llm_configured,
            "llm_model": runtime.settings.deepseek_model,
            "llm_connectivity": "not_checked",
            "retrieval_backend": "hybrid",
            "knowledge_retrieval_backend": "hybrid",
            "embedding_configured": runtime.settings.embedding_configured,
            "embedding_model": runtime.settings.embedding_model,
            "embedding_connectivity": "not_checked",
            "upload_max_bytes": runtime.settings.upload_max_bytes,
        }

    @app.get("/api/v1/articles/{chunk_id}", response_model=LegalChunk, tags=["出处原文"])
    def get_article(chunk_id: str, request: Request):
        return request.app.state.runtime.repository.get_article(chunk_id)

    @app.get("/api/v1/documents/{document_id}", response_model=LawDocument, tags=["出处原文"])
    def get_document(document_id: str, request: Request, version_id: str | None = None):
        return load_law_document(request.app.state.runtime.repository, document_id, version_id)

    @app.post("/api/v1/ask", response_model=AnswerResponse, tags=["在线问答"])
    async def ask(payload: SearchRequest, request: Request):
        return await request.app.state.runtime.service.ask(payload)

    return app


app = create_app()
