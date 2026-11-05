# 验证记录

## 2026-11-05：分批发布前检查

- 当前全部更新的 `python -m pytest -q`：166 项通过；ChromaDB 有 16 条既有的弃用提示。
- `ruff check src tests scripts`、`ruff format --check src tests scripts`、`node --check src/transport_law/web/app.js` 和 `git diff --check` 通过。
- 官网原始快照通过 `.gitattributes` 保留原始行尾与空白，以维持来源 SHA-256；快照不参与空白格式检查，代码与整理文档仍执行检查。
- `build_legal_corpus.py --check` 确认 17 份文档、1,302 条法规正文、29 项裁量表和总计 1,336 条记录可从本地来源重建；`pip check` 未发现依赖冲突。
- 本次仅验证本地代码与资料，没有再次调用在线模型或重新核验法规现行效力。分批检查、提交和推送的项目约定见根目录 `AGENTS.md`。

## 2026-11-05：阿里云百炼 text-embedding-v4 + Chroma 接入

- Python 3.12、ChromaDB 1.5.9；`python -m pip check` 未发现依赖冲突。
- 全量 `pytest -q`：166 项通过，其中 53 项覆盖向量接入。使用 DashScope 原生 HTTP 模拟响应验证 `output.embeddings`、`index`/`text_index` 对齐、文档/查询 `text_type`、查询 `instruct`、维度与每批 10 条限制，以及服务未开通、欠费、认证和权限错误分类。
- 真实 Chroma 临时库验证新进程读取、同批次复用、失败续传、服务商/模型/语料不一致拦截和 API 引用回填；另验证百炼原生模拟响应到 Chroma 入库及查询的完整链路。
- Chroma 的配置读取产生 16 条依赖库自身的 `legacy embedding function config` 弃用提示；本次调用均显式设置 `embedding_function=None` 并传入向量，不启用默认 embedding 模型。
- BM25 三个回归集分别 20/20、20/20、30/30 通过；`recall_at_k` 均为 1.0。这些结果只验证原基线没有回归，不能作为 Qwen 检索效果。
- 现有 `document_batch_20261104/langchain_documents.jsonl` 通过 `index-vectors --dry-run`：1,338 个子块、1,336 个父法条；配置为 `text-embedding-v4`、2048 维，每批 10 条；没有写正式向量库。
- Ruff、`git diff --check` 和前端 JavaScript 语法检查通过。
- 已按要求从本地 `.env` 删除未使用的 `OPENROUTER_API_KEY` 和 `OPENROUTER_BASE_URL`，其余有效配置及凭据保持原值。项目不存在 `.env.enviromet`；保留文档和测试引用的 `.env.example`，其中 API Key 均为空。
- 用户配置百炼密钥后，已真实请求北京地域的 `text-embedding-v4` 接口；诊断返回 HTTP 403、`AccessDenied.Unpurchased`。已确认运行时读取的密钥与 `.env` 一致，无环境变量覆盖或首尾空白；未取得有效向量，未开始全量入库，正式 Chroma 数据库尚未创建。
- 根据[百炼错误码说明](https://help.aliyun.com/zh/model-studio/error-code)，需核对该账号及地域的服务开通资格。客户端现单独返回 `embedding_not_activated`，不再将此错误归为密钥认证失败；不会回显上游原始消息或自动重试该类错误。
- 默认仍为 BM25。完成账号服务开通后，按 [向量检索说明](vector-retrieval.md) 继续在线入库及向量召回评测；当前离线测试结果不能作为真实模型连通性或检索效果的证明。

## 2026-11-04 全量已整理资料重新分块

使用新的 `Document + metadata` 流程处理当前全部 18 个已收录来源（17 份新增资料及原上传资料的已整理部分）。批次输出位于 `data/staging/document_batch_20261104/`，正式知识库未修改。

| 检查 | 结果 |
| --- | --- |
| 官网来源与文档重建检查 | `build_legal_corpus.py --check` 通过 |
| 批量分块 | 1,336 个父记录全部处理，生成 1,338 个候选 chunk，最长 791 字符 |
| 源文件校验 | 19 个主来源/修改依据文件 SHA-256 通过 |
| 内容覆盖 | 全部父记录非空白字符完整覆盖，逐块字符位置可回查，ID 无重复 |
| metadata 校验 | 所有 Document 的父条目字段、完整出处、修改依据与原记录一致 |
| 持久化回读 | 1,338 个 Document 和 1,338 个 DraftChunk 全部通过结构检查，分文档文件总数一致 |
| 正式知识库保护 | 处理前后原 `legal_chunks.jsonl` SHA-256 一致 |
| 全套 `python -m pytest -q` | 113 passed |
| `ruff check src tests scripts` / `ruff format --check src tests scripts` / `git diff --check` | 通过 |

只有 `cn-enforcement-2021-A018` 和 `hb-police-discretion-2026-G019` 被继续拆分。新增批处理测试覆盖相同文字不同父条目的 ID 隔离、metadata/修改依据保留、重复运行可复现、防覆盖和来源文件变动时拒绝处理。

本次使用已登记和整理的正文作为输入，未把网页导航、Markdown 说明段或 PDF 尚未整理的其他册当作法条。所有新结果保持 `needs_review`，没有调用模型、生成 embedding、切换检索库或重启服务。

## 2026-11-04 公安交管资料补充

本轮新增 6 份法规、规章全文 555 条，以及湖北公安 2026 年裁量基准总则 22 条、交通管理册 29 项（79 档），新增 606 条默认检索记录，总计 1,336 条。17 份整理文档及具体版本见 [法规资料库](legal-sources.md)。

| 检查 | 结果 |
| --- | --- |
| 全套 `python -m pytest -q` | 109 passed，包含现有 Document 数据流测试及新增公安交管测试 |
| 三套检索回归数据集 | 20/20、20/20、30/30；覆盖原事项、法规定位和公安交管样例 |
| `python scripts/build_legal_corpus.py --check` | 17 份文档、1,302 条条文及 29 项裁量表可从保存的快照/整理稿重建；哈希与结果一致 |
| `python -m transport_law validate-data` | 1,336 条结构校验通过，裁量规则仍为 draft |
| 原上传资料比对 | 前 5 条 JSONL 与 Git 原记录逐字节内容一致，ID、正文及出处未改 |
| PDF 来源检查 | 当前通知附件 236 页；总则物理第 3—12 页、交通表物理第 200—206 页逐页查看，印刷页码分别为 5—14、202—208 |
| PDF 文字交叉检查 | 用 pypdf 独立提取总则对应页，与 pdfplumber 整理的每一文本行比较，未发现缺行；表格 29 项 79 档保留完整条件及结果 |
| 湖北实施办法修改检查 | 第 56 条第 5 款按官方 2017 年修改决定替换，底本及修改决定均保留；第 73/74 条网页连段已分开 |
| API 离线检查 | 变道问题可检索实施条例第 44 条；湖北第 56 条返回两处来源；未生成新的公安自动裁量规则 |
| `ruff check src tests scripts/build_legal_corpus.py` / `ruff format --check src tests scripts/build_legal_corpus.py` | 通过 |
| `node --check src/transport_law/web/app.js` / `git diff --check` | 通过 |

法规名消歧检查覆盖道路交通安全法、国家实施条例、湖北实施办法；公安裁量总则第 20 条与交通管理册第 20 项分开定位，不存在的条号/表项返回空。红灯右转例外、变道不得妨碍正常行驶车辆、酒后事故检测条件等关键条文保留完整。2026 年事故程序征求意见稿未混入正文。

原来范围外的超速、超载、停车问题随资料扩充改为正例；无命中模型短路测试改用天气问题。醉酒驾驶已有行政法条，因此原刑事负例改为未覆盖的贪污罪，不能据此推断系统已具备醉驾量刑能力。

所有新增记录仍为 pending。上述结果证明来源保存、结构和样例召回，不证明全部法规现行有效、表中交叉引用无误或真实事故判断正确。公安裁量第 6、13 项保留原文疑点并提示复核。未调用真实 DeepSeek，未启动或重启用户服务，未完成现场图片识别、事故事实模型或事故定责能力。

## 2026-11-04 Document 数据流优化

文档加载入口改为返回 LangChain `Document`，正文和来源 metadata 一起经过条文整理与 `split_documents()`；保存前通过 `DraftChunk.from_document()` 校验，也支持从草稿转换回 Document。

| 检查 | 结果 |
| --- | --- |
| 全套 `python -m pytest -q` | 103 passed |
| 分块与重启专项测试 | 27 passed，覆盖 metadata 传递与持久化往返、输入不被修改、不同文档/版本/不连续页隔离，以及业务字段校验 |
| 两次 API lifespan 启动并检索 | 在禁止导入提取、分块和 LangChain 模块的条件下均成功，结果一致 |
| 215 页原 PDF 前后对比 | 两次均为 526 个候选块；JSONL 逐字节一致，正文、来源位置和 ID 均未变化 |
| 对比输出 SHA-256 | `8ea43f72d5b08c160eb5b6935efda5551bd9c4f9d8a5c273784be124d0c78224` |
| `ruff check src tests` / `ruff format --check src tests` / `git diff --check` | 通过 |

本次为了验证兼容性，主动运行了分块命令，输出到 `data/staging/hubei_document_chunks.jsonl`；没有覆盖原候选输出，也没有写入正式知识库。没有重启用户的服务或调用 DeepSeek。

服务启动仍只加载已整理资料并构建 BM25 内存索引。上传后后台导入、处理结果缓存、任务恢复和索引热更新仅补充为后续方案，本次没有实现这些功能。

## 2026-11-04 LangChain 自动分块

在项目 `.venv` 安装并验证 `langchain-text-splitters==1.1.3`、`langchain-core==1.7.4`；可选依赖入口为 `.[ingest]`，已将相关依赖版本补入 `constraints.txt`。

| 检查 | 结果 |
| --- | --- |
| 分块专项测试 + 原逐页提取回归 | 21 passed，其中本次新增分块测试 20 项 |
| 当前工作区全套 `python -m pytest -q` | 96 passed，包含同时更新的法规语料相关测试 |
| `ruff check src tests` / `ruff format --check src tests` | 通过 |
| `pip check` / `git diff --check` | 通过 |
| 实际上传版 PDF | 215 页生成 526 个候选块，单块最多 800 字符，ID 无重复 |
| 来源回查 | 全部 526 个块的页内字符范围可回查；覆盖所有页提取文本中的非空白字符 |
| 结果位置 | `data/staging/hubei_auto_chunks.jsonl`，已被 Git 忽略 |

专项测试覆盖条文边界、长条文重叠、跨页来源、无文字页隔断、重复文本位置与稳定 ID、Markdown/UTF-8 BOM、无效参数、不可读取文件、防覆盖，以及 CLI 不读取模型配置。此功能全程本地运行，没有调用 DeepSeek。

本次渲染核对第 2 页，确认通用分块仍可能跨越裁量表的不同行。526 个结果仅为待审核草稿，没有写入正式知识库；来源位置和文字覆盖检查不代表表格关系、法律含义或适用效力已核验。

## 2026-11-04 官方法规资料接入

新增 10 部法规、规章，按条整理为 725 条正文；保留原上传资料 5 条，总计 730 条。来源与收录版本见 [法规资料库](legal-sources.md)。测试结果针对本次当前工作区，包含已有前端和分块功能的测试。

| 检查 | 结果 |
| --- | --- |
| `python scripts/build_legal_corpus.py --check` | 10 份来源 SHA-256、725 条连续条号、生成文档和 JSONL 可重复重建，结果一致 |
| `python -m transport_law validate-data` | 730 条通过结构校验；裁量规则仍为 draft |
| `python -m transport_law evaluate` | 20/20，Recall@5=1.0；4 个旧节选期望改为相应官网全文，道路运输经营题因扩充语料改为正例 |
| `python -m transport_law evaluate --dataset evals/official_retrieval.jsonl` | 20/20，Recall@5=1.0 |
| `python -m pytest -q` | 96 passed |
| `ruff check src tests scripts/build_legal_corpus.py` | 通过 |
| `ruff format --check src tests scripts/build_legal_corpus.py` | 通过 |
| `node --check src/transport_law/web/app.js` | 通过 |
| `git diff --check` | 通过 |
| API 离线集成 | 新法规可检索，官网来源的页码为 null、条号及官方链接完整；旧 PDF ID 仍返回旧来源 |

另外核对了公路法第 44 条的完整后续内容、公路安全保护条例第 27 条七项内容、道路运输条例 2026 年年龄条款、湖北水路交通条例 2026 年施行条款。新全文优先于裁量表重复节选；国家和湖北农村公路条例分别定位；不存在的条号返回空，不以无词项匹配的条文填充结果。

本次未调用真实 DeepSeek，未启动或重启用户的服务，未做浏览器视觉验收。官方来源取得、文本结构完整和检索测试通过，均不等于逐字人工审定、独立确认全部法规现行有效或具体案件适用。所有新增来源仍为 pending；新旧法历史适用选择尚未实现。

## 2026-11-04 前端验收

在现有 macOS / Python 3.12 环境接入原生 HTML、CSS、JavaScript 前端，由 FastAPI 同源提供。

| 检查 | 结果 |
| --- | --- |
| `python -m pytest -q` | 61 passed，含新增页面、静态资源及目录隔离检查 |
| `node --check src/transport_law/web/app.js` | 通过 |
| `ruff check src tests` / `ruff format --check src tests` | 通过 |
| `git diff --check` | 通过 |
| 本机 HTTP | `/` 和 `/health` 返回 200，状态页显示 5 条记录、1 个事项 |
| 浏览器检索 | 公路开挖返回 5 条原文；“超速怎么罚”返回空结果与范围提示；详情显示页码、版本和核验记录 |
| 浏览器裁量辅助 | 示例缺失字段、明确无许可、20 平方米边界提示、事实填齐仍需复核、三档规则草稿展示均通过 |
| 浏览器问答 | 离线模拟模型验证回答、引用跳转、补充问题、Token 用量、密钥缺失提示、取消等待与依据不足；模型中的 HTML 标签按纯文本展示 |
| 窄屏布局 | 390 × 844 视口检查问答和裁量表单；文档宽度 390，无横向溢出 |
| 浏览器控制台 | 真实服务页面未发现 JavaScript 错误 |

本次没有向真实 DeepSeek 发送问答请求，不消耗实际模型额度。当前本地健康检查显示密钥已配置，但实际连通性、账户权限和真实生成质量仍未验证。以下为首次搭建时的历史记录。

## 2026-11-02 初版验证

验证日期：2026-11-02。环境：macOS、Python 3.12.0、项目 `.venv`；依赖版本保存在 `constraints.txt`。以下结果来自此仓库的实际执行，不沿用之前聊天中的测试数量。

## 已完成

| 检查 | 结果 |
| --- | --- |
| `python -m pytest -q` | 60 passed |
| `python -m transport_law evaluate` | 20 / 20 通过，正例 Recall@5 = 1.0 |
| `python -m transport_law validate-data` | 5 条记录有效，规则状态为 draft |
| `ruff check src tests` | 通过 |
| `ruff format --check src tests` | 20 个 Python 文件格式通过 |
| `pip check` | 无依赖冲突 |
| PDF 提取 | 实际读取 215 页，每页均有可提取文字；输出至 `data/staging/pages.jsonl` |
| 原文核对 | 渲染并查看上传 PDF 第 2 页，检查事项与三档条件、结果对应关系 |
| 原文件校验 | 仓库 PDF 与登记 SHA-256 一致 |
| Git 忽略 | `.env`、`.venv`、`.idea` 和待整理输出均被忽略 |

实际启动 `python -m transport_law serve --port 8000` 后，以 HTTP 请求检查：

| 请求 | 实际返回 |
| --- | --- |
| `/health` | 200；5 条记录、1 个事项、5 条效力待核验；llm_configured=false |
| `/docs` | 200 HTML |
| `/openapi.json` | 200 JSON |
| 检索路边挖沟问题 | 200，5 条候选 |
| 问答“超速怎么罚” | 200，insufficient_evidence |
| 无 Key 问答公路挖沟问题 | 503，llm_not_configured |
| 检查 region=湖北省、area_m2=20 | 200，needs_facts，含边界提醒 |

## 自动化覆盖

检索正例和范围外问题、条号与 ID 排序、作废条目排除、数据文件格式与重复 ID；每条解释的引用校验（包括库中存在但不在本次检索的 ID）；原文从仓库回填；无命中时不调用模型。

DeepSeek 客户端通过 HTTPX MockTransport 验证请求结构、认证、余额、限流、有限重试、连接失败、超时、空回复、截断回复、响应格式和 JSON 校验。测试只使用固定假 Key，不读取本地 `.env`，没有真实模型请求。

裁量检查覆盖缺失事实、布尔 false 的处理、错误省份和道路类型、无效面积，以及 20/40 平方米前后和恰好等于边界的情况。所有草稿路径均不生成具体处罚金额。

## 验证范围

- 没有真实 DeepSeek Key，尚未验证账户权限、实际网络连通性或真实生成质量。
- 测试输出有一条 Starlette TestClient 对 HTTPX 的弃用提示，测试仍通过；业务 HTTP 客户端和本机服务请求均正常。未为消除提示而更换业务依赖。
- 20 条题是单事项开发回归集，首批总计只有 5 个片段，Recall@5 不能代表真实业务能力。条号排序另外测试 top_k=1。
- PDF 提取成功不等于全部表格已核验；仅第 2 页第 3 项进入结构化资料，其他页仍在待整理区。
- 没有核验上传资料现行效力、完整法源链或具体案件适用，也未进行正式裁量计算。
