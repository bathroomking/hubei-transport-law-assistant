# 湖北交通运输法规与裁量助手

基于上传资料的 RAG 问答项目，支持自动分块、向量入库、混合检索和带原文引用的回答。

## 技术栈

| 用途 | 技术 |
| --- | --- |
| 后端 | Python 3.11+、FastAPI、Uvicorn |
| 前端 | 原生 HTML / CSS / JavaScript，无需前端构建 |
| 文档处理 | pypdf、LangChain Document、RecursiveCharacterTextSplitter |
| 向量模型 | 阿里云百炼 DashScope `text-embedding-v4`，默认 2048 维 |
| 检索 | BM25 + ChromaDB 向量检索 + RRF 排名融合 |
| 回答模型 | DeepSeek，通过 HTTPX 调用 |
| 数据与配置 | SQLite 文档和片段记录、ChromaDB 持久化向量、Pydantic Settings |
| 开发检查 | pytest、Ruff |

统一流程为：上传 PDF → 提取文字与自动分块 → embedding → SQLite / Chroma 入库 → 问答。只有完成入库的上传资料参与检索；BM25 从 SQLite 读取片段正文，向量检索使用同一批片段，RRF 融合后将匹配原文加入 prompt。DeepSeek 生成回答后，程序校验引用并回填保存的原文。

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

### 3. 启动

```bash
python -m transport_law serve --reload
```

打开 [问答页面](http://127.0.0.1:8000/)。[API 文档](http://127.0.0.1:8000/docs) 提供接口说明，[健康检查](http://127.0.0.1:8000/health) 查看配置和入库状态。

首次启动可以是空库，无需预置法规 JSONL、手动分块或提前建立向量索引。服务默认监听本机地址，目前没有用户鉴权。

### 4. 上传资料

在页面点击“上传资料”，选择 PDF 文件，然后点击“上传并入库”。系统自动提取文字、分块、生成向量并写入数据库；状态变为“已入库”后即可参与下一次问答，无需重启。

默认文件上限为 20 MiB，同名文件不覆盖。PDF 须可提取文字，扫描件需要先做 OCR。入库失败或中断时，在页面点击重试：已保存的分块和成功入库的向量可复用。

上传只完成文字处理与索引，不自动核验法律效力、表格关系或个案适用性。回答依据限于当前已入库的上传资料。

### 5. 提问

在网页输入问题，或通过命令行提问：

```bash
python -m transport_law ask "根据已上传资料，占用、挖掘公路需要履行哪些手续？"
```

网页与 `ask` 使用同一条问答链。在线问答需要两项密钥和已入库资料；每次提问独立处理，补充信息时请一并保留原问题背景。可通过回答中的引用查看片段原文及其所在文档的已收录内容。

### 6. 数据与检查

| 位置 | 内容 |
| --- | --- |
| `data/local/knowledge/` | SQLite 文档与片段记录、上传原文件、导入缓存和任务锁 |
| `data/local/chroma/` | 上传资料的持久化向量索引 |

这两个本地目录不随 Git 分发，迁移或备份时应一并保留。已存在的上传数据可继续使用；服务重启不会重新切分或生成已有文档的向量。新增资料继续走上传入口。

已有上传库绑定创建时的向量模型、维度和输入配置；修改这些配置前需另行规划重新入库，目前不支持自动迁移。

开发检查：

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

前端改动可使用 Node.js 执行 `node --check src/transport_law/web/app.js`；运行服务本身不需要 Node.js。命令行入口仅保留 `serve` 和 `ask`，参数见 `python -m transport_law --help`。
