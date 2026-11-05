# 上传知识库文档

上传接口接收 PDF、TXT 和 Markdown，保存原文件后在后台完成文字提取、分块、百炼 embedding 和 Chroma 入库。每份文档独立处理；新增文件不会重新计算已入库文件或现有法规库的向量。只有状态为 `ready` 的文档参与统一问答检索。

上传库与已整理法规库分别使用独立的 Chroma collection，但通过同一个网页、CLI `ask` 和 `POST /api/v1/ask` 提问。`/api/v1/knowledge/documents` 系列接口只负责上传、列表、状态和重试管理，不再提供独立的上传库检索或问答入口。上传资料自动标记为待核验，不自动判断法源类型、法律效力、地域或具体处罚档次。

## 安装与启动

上传任务当前使用 macOS/Linux 文件锁实现进程间互斥；Windows 尚不支持此上传流程，原有功能不受该限制。在项目根目录激活虚拟环境后安装上传所需的解析和向量依赖：

```bash
python -m pip install -e ".[ingest,vectors]" -c constraints.txt
python -m transport_law serve
```

在已有本地 `.env` 中配置 `DASHSCOPE_API_KEY`，沿用项目的 `text-embedding-v4`、维度和百炼接口设置。不要覆盖其他已有配置。上传分块会向百炼发送用于生成向量的文档内容并使用 API 额度；上传库语义检索也会发送问题文本。统一 `/api/v1/ask` 同时要求配置 `DEEPSEEK_API_KEY`，并在有候选依据时向 DeepSeek 发送问题和检索资料。精确条目 ID 可以直接定位原文、跳过问题 embedding，但仍执行两种密钥配置检查并调用在线回答模型。

打开 [上传资料页面](http://127.0.0.1:8000/#knowledge)，选择或拖入一份资料，再点击“上传并入库”。接收成功后，页面会从状态接口自动更新处理进度；仅 `ready` 显示“已入库”。失败时会显示后台错误原因，可点击“重试入库”；长时间等待的任务可以尝试“继续入库”，服务仍会拒绝重复启动正在执行的任务。

列表可手动刷新；离开上传页面或浏览器切到后台时暂停自动查询，回来后重新读取，所有已知任务完成或失败后停止轮询。读取状态失败时保留上次记录、暂停自动更新，提示手动刷新。网络中断后，上传是否接收可能尚未确认，应先刷新列表查看已有记录，再决定是否重传。

“查看入库原文”展示上传文档已处理的内容及覆盖说明；“去问答”回到唯一提问入口，问题仍在全部正式法规和已入库资料中检索，不限定为刚上传的文件。网页文件校验使用 `/health` 返回的 `upload_max_bytes`，实际接收仍以后端校验为准。也可在 [本机接口文档](http://127.0.0.1:8000/docs) 直接操作上传接口。

## 接口

| 方法与路径 | 用途 |
| --- | --- |
| `POST /api/v1/knowledge/documents` | 以 `multipart/form-data` 上传一个文件，字段名为 `file`；返回 `202` 和文档 ID |
| `GET /api/v1/knowledge/documents` | 列出上传记录及处理状态 |
| `GET /api/v1/knowledge/documents/{id}` | 查看一份文档的状态、分块数、已写向量数和错误信息 |
| `POST /api/v1/knowledge/documents/{id}/retry` | 恢复失败或已中断任务；返回 `202`，复用已保存的分块及成功批次 |
| `POST /api/v1/ask` | 在正式法规和 ready 上传资料中统一检索，在线生成解释与补问 |
| `GET /api/v1/articles/{chunk_id}` | 查看正式或上传资料的引用原文与来源 |
| `GET /api/v1/documents/{document_id}?version_id=...` | 查看指定文档版本的全部已收录内容及覆盖范围 |

统一问答接收 `{"question": "问题"}`，前端不展示检索参数。产品固定使用 hybrid：先对正式法规和 ready 上传条目建立同一套 BM25 统计，得到一路全局关键词排名；两座向量库复用同一个问题向量，在相同 embedding 合约与 cosine 度量下合并、排序成一路全局 dense 排名，然后对这两路执行 RRF。不会把两份已分别计算的 RRF 分数简单相加。

普通自然语言问题使用 `broad=True` 召回，不再要求命中旧的事项别名才能进入候选；明确的法规名称、条号和精确 ID 仍保留定位约束。两库使用相同模型与维度，查询不重新生成已有文档向量。新文档变为 `ready` 后，下一次问题就可在两路排名中召回，无需重启。

上传库没有经过业务标定的相似度拒答阈值。全库语义召回可以找到相关候选，但是否足以回答仍由基于材料的问答链判断，不能把“有向量命中”等同于“有足够依据”。算法说明及历史评测边界见 [混合检索说明](hybrid-retrieval.md)。

## 一次完整调用

上传一个文件；把示例路径替换成自己的实际文件路径：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/knowledge/documents \
  -F 'file=@/绝对路径/资料.pdf'
```

TXT、`.md`、`.markdown` 采用相同调用方式。上传响应中的 `id` 用于后续查询，例如：

```bash
document_id='替换为上传响应中的id'
curl -sS "http://127.0.0.1:8000/api/v1/knowledge/documents/$document_id"
```

**HTTP `202` 只表示已接收并安排处理，不表示 embedding 已完成。** 查询结果包含以下处理字段：

| 字段 | 含义 |
| --- | --- |
| `id` / `filename` | 上传文档 ID 和接收时规范化的文件名 |
| `status` | `queued`、`processing`、`ready` 或 `failed` |
| `chunk_count` | 已完成切分的片段数 |
| `embedded_count` | 已成功写入的向量数 |
| `error_code` / `error_message` | 失败时可供排查的错误，不含 API Key |

状态变为 `ready` 后，通过统一问答入口提问：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"这份资料对施工许可有哪些规定？"}'
```

该问题会同时检索正式法规与全部 ready 上传资料；当前请求不提供指定单份上传文档的过滤参数。返回的 `sources` 保留原文件名、文件 SHA-256、片段原文和位置；PDF 保留真实页码，TXT/Markdown 使用字符位置，不填造 PDF 页码。所有上传资料的核验状态均为 `pending`。引用详情通过 `/api/v1/articles/{chunk_id}` 查看；使用引用中的 `document_id` 与 `version_id` 查看文档时，覆盖类型为 `uploaded_document`，可能包含重叠片段，不代表整部法规全文。

失败或进程中断后，使用原文档 ID 重试：

```bash
curl -sS -X POST \
  "http://127.0.0.1:8000/api/v1/knowledge/documents/$document_id/retry"
```

仍在执行的任务和已完成任务不能重复启动，会返回 `409`。重试前先解决对应的文件、网络、密钥或账号问题；同名文件不能通过重新上传覆盖失败任务。

## 文件名与格式规则

- 支持 `.pdf`、`.txt`、`.md` 和 `.markdown`，扩展名不区分大小写。TXT/Markdown 必须为 UTF-8，可包含 UTF-8 BOM。
- 默认单文件上限为 20 MiB，即 20,971,520 字节。空文件、超限文件、路径形式的文件名或不支持的扩展名会被拒绝。
- 同名检查覆盖已登记的上传文件，以及现有已整理法规的来源文件名（取 `Source.file` 的文件名部分）。名称按 Unicode NFKC 规范化、去除首尾空白后进行 `casefold` 比较；扩展名属于文件名。因此 `说明.TXT` 与 `说明.txt` 重名，`说明.txt` 与 `说明.md` 是不同文件名。
- 同名上传返回 `409`，并发提交同名文件也只接收一份。已失败任务仍保留名称，需使用其 `retry` 接口；不会覆盖原文件。
- 文件名不同，即使内容相同，也按新文档处理。当前不提供跨文件名的内容去重或自动替换版本。
- PDF 必须能提取文字。加密 PDF 或含无法提取文字页面的 PDF 会记录为失败；当前不执行 OCR，也不把缺页提取结果悄悄标记为完成。空白页可能需要先从原件中确认或处理。

文件在接收后才开始解析，因此损坏 PDF、非 UTF-8 文本等问题可能通过状态接口以 `failed` 返回。立即拒绝的请求使用标准错误结构 `{"error":{"code":"…","message":"…"}}`；后台失败使用记录中的 `error_code` 和 `error_message`。

## 持久化与重复处理

默认保存位置：

| 数据 | 默认位置或名称 |
| --- | --- |
| 上传登记 | `data/local/knowledge/uploads.sqlite3` |
| 原文件、分块缓存 | `data/local/knowledge/documents/{id}/` 下的 `source.*` 和 `chunks.jsonl` |
| Chroma 持久化目录 | `data/local/chroma/`，与现有法规向量库共用客户端目录 |
| 上传文档 collection | `transport_uploads_v4_2048_v1`，与法规 collection 分开 |

这些本地产物都位于 Git 忽略目录中，不应提交到仓库。若需要迁移或备份，应同时保留上传目录与 Chroma 目录，不能只备份向量或只备份登记信息。

成功切分的结果会保存，重试时直接读取；每个成功的向量批次立即入库，重试只生成缺失的向量。任务全部成功后才切换为 `ready`，失败任务的部分向量不会参与检索。若远端已计费而客户端尚未保存成功响应时发生中断，重试该批次仍可能再次计费。

**服务重启不会重新切分或生成已有文档向量。** 中断的后台任务不会在启动时自动补跑，应查询状态后显式调用 `retry`；查询发现执行进程已退出时，原 `processing` 任务会标记为 `failed`，错误码为 `job_interrupted`。每次语义查询仍会生成问题向量，目前没有重复问题缓存。

以下配置均有默认值，无需为了启用上传再复制一份 `.env`：

```dotenv
KNOWLEDGE_PATH=data/local/knowledge
KNOWLEDGE_COLLECTION=transport_uploads_v4_2048_v1
UPLOAD_MAX_BYTES=20971520
```

上传库也使用现有的 `DASHSCOPE_*`、`EMBEDDING_*` 和 `CHROMA_PATH` 配置。模型、维度或输入规范不匹配时会拒绝混用；当前接口没有模型迁移、文档删除、替换同名版本或正式法规发布功能。

## 与现有分块命令的关系

`chunk-document` 仍是独立的离线草稿工具：同一输出路径拒绝覆盖，换输出路径会重新处理。上传接口则保存文档任务和分块结果，并继续完成 embedding、入库和上传库检索；不需要先手工运行 `chunk-document` 或 `index-vectors`。

上传资料以通用 `document` 片段收录，不会补造执法事项、法效核验或裁量规则。已有 `data/processed/legal_chunks.jsonl` 和法规索引的整理流程保持独立；把上传材料正式整理进法规资料库仍需核对原件、来源和业务字段。

## 2026-11-05 历史上传验证

以下记录对应统一入口改造前的独立上传库流程，测试数量、旧片段接口及检索结果仅说明当时验证情况，不是当前双库 `UnifiedRetriever` 的回归结果或准确率。

- 全套自动化测试 193 项通过，包含 27 项上传专项：四格式、同名与并发保护、来源、逐文档追加、失败续传、任务互斥、中断恢复、重启复用和问答引用。出现 41 条 Chroma 依赖库既有配置弃用提示。
- 在隔离临时目录启动真实本机 HTTP 服务，分别上传自造的 TXT、MD、Markdown 和文本 PDF。四次请求均先返回 `202 / queued`，随后变为 `ready`，真实百炼 `text-embedding-v4` 共生成 4 个 2048 维向量。
- 同名大小写变体返回 `409`；真实在线查询返回全部 4 个上传片段，原文、文件名、PDF 页码和文本字符位置均能通过片段接口回查。
- 关闭服务后由新的 Python 进程读取 4 份登记、4 个片段和 4 个持久化向量，文档及问题 embedding 调用均为 0。测试结束删除临时库，现有正式 collection 仍为 `ready`、1,338 个向量。
- 上传后的问答引用链通过离线模型验证；本次未真实调用 DeepSeek。完整本地在线报告为被 Git 忽略的 `data/local/validation/upload_live_smoke_20261105.json`。
