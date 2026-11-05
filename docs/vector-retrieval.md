# 阿里云百炼 text-embedding-v4 与 ChromaDB

向量生成直连阿里云百炼 DashScope 原生 HTTP API，模型为 `text-embedding-v4`（Qwen3 Embedding 系列），默认请求 2048 维、每批 10 条。ChromaDB 负责本地持久化和 cosine 检索。DeepSeek 仍负责读取检索到的原文、生成带引用的解释。在线问答固定使用 BM25 + Chroma + RRF；独立算法选项仅供 `evaluate --backend` 维护评测使用。本接入使用百炼的 v4 产品规格，不将其等同于开源 Qwen3 Embedding 8B 的参数或输出维度。

## 首次使用

在项目根目录、已激活的虚拟环境中运行：

```bash
python -m pip install -e ".[vectors]" -c constraints.txt
```

在已有 `.env` 中添加以下配置，不覆盖原有 DeepSeek 配置：

```dotenv
DASHSCOPE_API_KEY=在本地填写你的百炼API密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=2048
EMBEDDING_BATCH_SIZE=10
CHROMA_PATH=data/local/chroma
CHROMA_COLLECTION=transport_law_bailian_v4_2048_v1
```

默认地址对应北京地域，Key 必须在相应地域创建。百炼也提供业务空间专属域名，例如 `https://你的业务空间ID.cn-beijing.maas.aliyuncs.com/api/v1`；可以直接填入 `DASHSCOPE_BASE_URL`。新加坡传统地址为 `https://dashscope-intl.aliyuncs.com/api/v1`。使用原生接口地址，路径须为 `/api/v1`，不要填 `/compatible-mode/v1`。客户端通过 HTTPX 发请求，不需要安装 DashScope SDK。

真实密钥只写本地 `.env`，`.env.example` 保持空值并作为新环境的配置模板保留。向量接口使用 `DASHSCOPE_API_KEY`。

先校验已有批次。预检不需要 Key、不调用模型、不创建数据库：

```bash
python -m transport_law index-vectors \
  --documents data/staging/document_batch_20261104/langchain_documents.jsonl --dry-run
```

当前完整批次应报告 `chunks: 1338`、`parents: 1336`。这份文件被 Git 忽略；新检出仓库需要先按 [批量分块说明](chunking.md#批量处理已整理资料) 生成自己的批次，并把上述路径替换为实际输出。

填好密钥后执行正式入库；它会向阿里云百炼发送法规标题、条号和片段正文，并使用 API 额度：

```bash
python -m transport_law index-vectors \
  --documents data/staging/document_batch_20261104/langchain_documents.jsonl
python -m transport_law vector-status
python -m transport_law ask "公路挖沟有哪些依据？"
```

进度输出到 stderr，最后的 JSON 报告写到 stdout。`embedded` 是本次新生成的数量，`reused` 是已存在并核对一致的数量。密钥只用于服务端 HTTP 认证，不存入 Chroma，不发送到前端或报告。

向量索引须先完成；索引缺失、未完成或配置不一致时会明确报错。`ask` 还需要 DeepSeek 密钥，使用与网页相同的统一问答链。仅 `evaluate` 提供 `--backend` 算法选项；`validate-data` 只做本地资料和规则来源校验。

## 入库和检索规则

- 输入必须是 `chunk_legal_corpus.py` 生成的完整 Document 批次。程序对照当前 `LegalChunk` 核对父 ID、版本、完整出处、正文哈希、字符位置及非空白正文覆盖。普通上传草稿不会自动升级为正式知识库。
- 文档端向量化 `法规正式名称 + 条号 + 子块正文`，传 `parameters.text_type=document`。查询端发送原问题，传 `text_type=query` 和 `instruct` 任务说明；不在问题正文拼接开源模型的提示模板。两端使用同一个模型及维度。
- 明确传入 `embedding_function=None`，入库使用 `embeddings`，查询使用 `query_embeddings`。不下载 Chroma 的默认 embedding 模型。
- 从原生响应的 `output.embeddings` 读取向量，按 `index`（兼容旧响应的 `text_index`）重排，验证条数、维度、有限数值和非零向量，做 L2 归一化后写入。`EMBEDDING_DIMENSIONS` 通过 `parameters.dimension` 显式发送给服务端，同时校验返回长度；支持 64、128、256、512、768、1024、1536、2048。批次大小只允许 1–10，超出在本地拦截。
- Chroma 保存稳定子块 ID、向量、原文和 metadata。常用字段展开以供过滤，完整原法条出处保存在 `provenance_json` 字符串中。真正的 PDF 页码与官方 URL 来自父法条，分块中间 TXT 的字符位置不会充当 PDF 页码。
- 未指定法规时在全库召回；指定法规名、条号、裁量表项时按明确引用约束候选。父条目 ID 可直接定位，不存在的明确条号返回空。自然语言问题不再经过人工事项白名单；候选的法律相关性与资料不足拒答由问答阶段核对。
- 多个子块命中同一法条时只返回一次，以最佳子块相似度排序；候选不足时扩大向量结果集。返回完整父法条和其原有 ID，保持引用校验和出处展示。
- 向量检索的 `score = 1 - cosine distance`，不是法律适用概率，也不能与 BM25 得分直接比较。向量相近仍需核对原文；当前没有经真实数据标定的相似度拒答阈值。

## 重复执行、失败和更新

每个成功批次立即保存。认证失败、余额不足、接口响应异常会停止；网络失败、限流及服务端错误最多按 `EMBEDDING_MAX_RETRIES` 重试，默认 2 次。`EMBEDDING_TIMEOUT_SECONDS` 默认 60 秒，是单次网络操作的超时设置。失败批次不会被标记完成，已保存批次保留；重跑同一命令会继续缺失记录。若上游已经完成计费但响应丢失，重试仍可能再次计费。

如果返回 `embedding_not_activated`（上游 `403 / AccessDenied.Unpurchased`），请先用阿里云主账号在 API Key 所属地域的百炼控制台核对服务开通状态，按页面提示阅读协议并完成开通；反复更换分块或向量维度不能解决账号开通问题。`embedding_auth_error` 对应密钥/地域认证问题，`embedding_access_denied` 对应模型或业务空间权限，`embedding_balance_error` 对应欠费等账户状态。这些错误不会自动重试。排查依据见 [百炼错误码说明](https://help.aliyun.com/zh/model-studio/error-code)。

全部条目入库且数量一致后 collection 才变为 `ready`；未完成索引不会用于问答。已有记录会校验 ID 和 metadata，不删除或覆盖不同批次。同一 collection 不应并行运行多个入库进程。

collection 记录服务商、模型、维度、原生接口规范、输入模板、查询指令、语料和分块摘要。改变维度、接口地址、输入规范、语料或分块批次时，选择新的 `CHROMA_COLLECTION` 名称并重新入库，保留旧索引供回退。当前客户端只支持 `text-embedding-v4`；以后更换模型需先适配并验证对应规格。旧服务商的索引不会自动迁移或混用。当前只对同一批次做续传和复用，没有实现跨语料版本的增量更新或自动删除旧记录。

启动/重启不会重做文档向量；每个语义查询只生成问题向量。修改 `.env` 或正式 JSONL 后需要重启服务，语料变更还需生成对应的新分块及索引。法规核验状态原样保留，不会因为入库成功而变为已核验。

## 对照评测

```bash
python -m transport_law evaluate --dataset evals/retrieval.jsonl --backend bm25
python -m transport_law evaluate --dataset evals/retrieval.jsonl --backend chroma
python -m transport_law evaluate --dataset evals/official_retrieval.jsonl --backend chroma
python -m transport_law evaluate --dataset evals/traffic_police_retrieval.jsonl --backend chroma
```

Chroma 评测会使用阿里云百炼额度。比较各题的父法条 ID 和 `recall_at_k`。历史题集中的范围外空结果预期来自早期事项白名单，不能等同于全库问答阶段的拒答；原题和期望保持不变，以便追踪差异。离线测试使用模拟 API/向量及临时 Chroma 数据库，验证的是接线、持久化、续传、配置拦截和引用链路，不能证明真实模型的检索效果或连通性。

2026-11-05 接入阶段曾完成真实百炼调用、1,338 个向量入库、跨进程读取、重复入库复用和 API 原文来源回填检查。三组开发回归中 Chroma 分别通过 20/20、20/20、25/30，BM25 分别为 20/20、20/20、30/30；公安交管组有 5 题未找全前 5 条内的预期法条。当时已用精确余弦排序复核这些遗漏。上述数字属于已移除的事项限制路径，不适用于当前全库候选；历史详情见 [验证记录](validation.md)，当前链路检查见 [统一验收记录](unified-validation.md)。

接口依据：[百炼文本向量原生接口](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)、[text-embedding-v4 模型说明](https://help.aliyun.com/zh/model-studio/text-embedding-v4)、[百炼域名迁移说明](https://help.aliyun.com/zh/model-studio/embedding-interfaces-compatible-with-openai)、[Chroma 入库](https://docs.trychroma.com/docs/collections/add-data)、[Chroma collection 配置](https://docs.trychroma.com/docs/collections/configure)。
