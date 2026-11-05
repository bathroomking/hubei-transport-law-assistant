# 湖北交通运输法规与裁量助手

基于本地法规与上传资料的 RAG 问答项目，支持自动分块、向量入库、混合检索和带原文引用的回答。

## 技术栈

| 用途 | 技术 |
| --- | --- |
| 后端 | Python 3.11+、FastAPI、Uvicorn |
| 前端 | 原生 HTML / CSS / JavaScript，无需前端构建 |
| 文档处理 | pypdf、LangChain Document、RecursiveCharacterTextSplitter |
| 向量模型 | 阿里云百炼 DashScope `text-embedding-v4`，默认 2048 维 |
| 检索 | BM25 + ChromaDB 向量检索 + RRF 排名融合 |
| 回答模型 | DeepSeek，通过 HTTPX 调用 |
| 数据与配置 | JSONL 法规条目、SQLite 上传记录、ChromaDB 持久化向量、Pydantic Settings |
| 开发检查 | pytest、Ruff |

资料经 chunking、embedding 写入向量库；问题经 embedding 与 BM25 召回，RRF 融合后将匹配原文加入 prompt，由 DeepSeek 生成回答，程序校验引用并回填原文。正式法规与已入库的上传资料共同参与检索。

## 运行项目

### 1. 安装

在项目根目录执行。完整上传流程依赖 `fcntl`，目前使用 macOS / Linux。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,ingest,vectors]" -c constraints.txt
```

后续命令均在项目根目录、已激活的虚拟环境中运行。PyCharm 选择 `.venv/bin/python` 作为解释器。

### 2. 配置模型

首次使用时复制 [.env.example](.env.example) 为 `.env`；已有 `.env` 时直接编辑，保留原配置。填写两个服务的密钥：

```dotenv
DASHSCOPE_API_KEY=你的百炼密钥
DEEPSEEK_API_KEY=你的DeepSeek密钥
```

其余模型、接口地址和检索参数见 `.env.example`。百炼默认使用北京地域的原生 API，密钥须与地域匹配。修改配置后重启服务，真实密钥不要提交到 Git。

入库会向百炼发送文档片段；问答会向百炼发送问题，向 DeepSeek 发送问题与检索原文，使用对应服务的额度。

### 3. 首次建立法规索引

仓库包含已整理的 [法规条目](data/processed/legal_chunks.jsonl)，本地向量库不随 Git 分发。首次运行时执行：

```bash
python scripts/chunk_legal_corpus.py --output-dir data/staging/current_document_batch
python -m transport_law index-vectors \
  --documents data/staging/current_document_batch/langchain_documents.jsonl --dry-run
python -m transport_law index-vectors \
  --documents data/staging/current_document_batch/langchain_documents.jsonl
python -m transport_law vector-status
```

分块输出目录必须尚不存在。重试已有批次时直接使用其 `langchain_documents.jsonl`。`--dry-run` 只做本地预检；正式入库调用百炼生成向量。若入库中断，重跑同一条 `index-vectors` 命令即可续传，无需重新分块。已有可用索引时跳过本节。

### 4. 启动与提问

```bash
python -m transport_law serve --reload
```

打开 [问答页面](http://127.0.0.1:8000/)。[API 文档](http://127.0.0.1:8000/docs) 提供接口说明，[健康检查](http://127.0.0.1:8000/health) 查看配置状态。

也可以通过命令行提问：

```bash
python -m transport_law ask "根据公路法第四十四条，占用、挖掘公路需要履行哪些手续？"
```

在线问答需要两项密钥和可用法规索引。每次提问独立处理，补充信息时请一并保留原问题背景。服务默认监听本机地址，目前没有用户鉴权。

### 5. 上传新资料

在页面点击“上传资料”，选择 PDF、TXT 或 Markdown，然后点击“上传并入库”。系统自动提取文字、分块、embedding 并写入 ChromaDB；状态变为“已入库”后即可参与下一次问答，无需重启。

默认文件上限为 20 MiB，同名文件不覆盖。TXT / Markdown 使用 UTF-8；扫描 PDF 需要先做 OCR。入库失败或中断时，在页面点击重试。

### 6. 数据维护与检查

原始资料和来源清单位于 `data/raw/`，法规条目位于 `data/processed/`，分块暂存文件位于 `data/staging/`。本地向量库与上传记录位于 `data/local/chroma/` 和 `data/local/knowledge/`，迁移时需一并备份。

更新正式法规时，先更新来源及条目；需从已登记来源重建条目时运行 `python scripts/build_legal_corpus.py`。正式语料或分块规则变化后，设置新的 `CHROMA_COLLECTION`，换用新的分块输出目录，重新执行第 3 步并重启服务。同一批次可续传，旧 collection 不支持跨语料版本直接更新。已有上传库绑定创建时的向量配置，目前不支持自动迁移。

常用检查：

```bash
python -m transport_law validate-data
python scripts/build_legal_corpus.py --check
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

前端改动可使用 Node.js 执行 `node --check src/transport_law/web/app.js`；运行服务本身不需要 Node.js。其他维护命令见 `python -m transport_law --help`。
