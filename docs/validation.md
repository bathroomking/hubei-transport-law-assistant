# 验证记录

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
