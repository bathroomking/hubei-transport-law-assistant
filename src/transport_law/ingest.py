"""PDF 只提取至待整理区，不把未核对的跨页表格自动发布到知识库。"""

import hashlib
import json
from pathlib import Path

from transport_law.errors import AppError


def extract_pdf(path: Path, output: Path) -> dict:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise AppError(
            "pdf_dependency_missing", '请先安装 PDF 依赖：pip install -e ".[pdf]"', 500
        ) from exc
    if output.resolve() == path.resolve() or output.exists():
        raise AppError("output_exists", "输出文件已经存在，请选择新的路径。", 409)
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise AppError("pdf_encrypted", "暂不支持加密 PDF，请提供可读取的副本。", 400)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = []
        for page_number, page in enumerate(reader.pages, 1):
            content = page.extract_text() or ""
            rows.append(
                {
                    "file": path.name,
                    "sha256": digest,
                    "pdf_page": page_number,
                    "text": content,
                    "status": "needs_review" if content.strip() else "needs_ocr",
                }
            )
    except (OSError, ValueError, PdfReadError) as exc:
        raise AppError("pdf_read_failed", "PDF 读取失败，请检查文件。", 400) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except FileExistsError as exc:
        raise AppError("output_exists", "输出文件已经存在，请选择新的路径。", 409) from exc
    return {
        "page_count": len(rows),
        "needs_ocr_count": sum(row["status"] == "needs_ocr" for row in rows),
        "output": str(output),
        "status": "staged_not_published",
    }
