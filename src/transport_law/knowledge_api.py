"""上传知识库 API；接收后后台处理，以 ready 状态表示已完成入库。"""

from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Request, UploadFile
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/knowledge", tags=["上传知识库"])


class DocumentStatus(BaseModel):
    id: str
    filename: str
    status: Literal["queued", "processing", "ready", "failed"]
    chunk_count: int = Field(ge=0)
    embedded_count: int = Field(ge=0)
    error_code: str | None = None
    error_message: str | None = None


@router.post("/documents", response_model=DocumentStatus, status_code=202)
def upload_document(file: UploadFile, request: Request, background_tasks: BackgroundTasks):
    """上传 PDF。同名返回 409；202 只表示接收成功。

    处理状态通过 GET /documents/{id} 查询；只有 ready 文档进入检索。
    """
    store = request.app.state.runtime.knowledge
    try:
        result = store.receive(file.filename or "", file.file)
    finally:
        file.file.close()
    background_tasks.add_task(store.process, result["id"])
    return result


@router.get("/documents", response_model=list[DocumentStatus])
def list_documents(request: Request):
    return request.app.state.runtime.knowledge.list_documents()


@router.get("/documents/{document_id}", response_model=DocumentStatus)
def get_document(document_id: str, request: Request):
    return request.app.state.runtime.knowledge.get_document(document_id)


@router.post("/documents/{document_id}/retry", response_model=DocumentStatus, status_code=202)
def retry_document(document_id: str, request: Request, background_tasks: BackgroundTasks):
    """继续失败或被中断的导入，复用已保存的分块和成功向量批次。"""
    store = request.app.state.runtime.knowledge
    result = store.retry(document_id)
    background_tasks.add_task(store.process, document_id)
    return result
