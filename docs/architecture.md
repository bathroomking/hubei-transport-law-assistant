# 架构说明

本项目延续“学习路线建议”中确定的方向：Python、DeepSeek、可溯源的单事项 RAG，再逐步接入规则和单智能体。当前把可独立调试的模块先实现出来。

## 模块边界

| 层 | 输入 | 输出 | 扩展位置 |
| --- | --- | --- | --- |
| 数据仓库 | JSONL 文件 | 通过结构校验的 `LegalChunk` | 替换为 SQLite/PostgreSQL |
| 检索器 | 问题、top_k | `SearchHit[]` | 实现 `Retriever` 协议替换为向量/混合检索 |
| 模型客户端 | 消息、是否 JSON 模式 | `Completion` | 实现 `LanguageModel` 协议接入其他模型 |
| RAG 服务 | `SearchRequest` | `AnswerResponse` | 在检索后加入重排或证据审查 |
| 事实检查 | `CaseFacts` | `DiscretionCheck` | 核验规则后扩展确定性规则引擎 |
| API / CLI | HTTP / 命令行输入 | JSON | 可接展示界面和现有业务系统 |

`runtime.py` 只负责装配，API 和 CLI 使用同一份服务逻辑。测试可以直接注入 `FakeLLM` 或 HTTPX `MockTransport`，不会访问 DeepSeek。

## 数据和来源

每个片段保存 document_id、version_id、item_id、原文、检索文本、来源 SHA-256、PDF 物理页码和核验状态。物理页码从 1 开始，`printed_page` 单独保存印刷页码。

首批数据有两个不同的状态：

- **转录已核对**：2026-11-02 对照上传 PDF 第 2 页核对表格文本和列关系。
- **适用待核验**：尚未确认发布通知、施行日期、现行效力、地域适用和法条全文。

因此 `transcription_checked_on` 有日期，但 `verification_status` 仍为 `pending`；`effective_on` 为空。不得把前者当作后者。

`superseded` 条目保留按 ID 查询能力，但不进入默认检索。第一版尚未实现按案件日期选择法规版本；一般检索返回候选依据，不表示已判定其适用。

## 检索策略和局限

先以人工整理的事项别名和“道路＋挖掘/占用”组合召回事项，再用中文二/三字词项 BM25 排序。完整条号别名和条目 ID 命中获得额外权重。共享事项中的关联依据保留为候选，避免只取出罚款而丢掉条件。

这是可解释的单事项基线，没有 Embedding 或模型改写。扩大语料时需要扩充别名、评估误召回，并逐步替换为关键词与向量混合检索。当前组合匹配不能理解复杂否定、多事项混问或从对话补全省略信息。

检索可以返回湖北材料供比较，因此“湖南挖掘公路可以直接用湖北标准吗”仍会召回湖北条目。是否可适用不是检索结论；结构化事实检查会拒绝其他省份，RAG 固定返回 `not_verified`，不会标记案件已核准。

## 引用校验

模型每条解释返回 `text` 和 `citation_ids`。后端先检查 JSON 结构，再检查每个 ID 都属于**本次**检索结果，最后从仓库记录填充 `sources`。库中存在、但不在本次检索里的 ID 也会被拒绝。

这是出处完整性检查，不是语义蕴含检查。模型仍可能引用真实条目却作出不受支持的解释；现阶段需要对照原文复核，后续用标注回答评测集改进。资料中的提示词或用户要求不能改变系统提示词规则，但提示词也不是绝对保证。

## 裁量模块

三档的条件和处罚幅度按原表保存成文本，不自行改成数值区间。当前 `review_status` 只允许 `draft`，也没有“输入完整即执行处罚”的分支。

事实缺失时返回 `needs_facts`；事实齐全仍为 `needs_review`；省份、道路类型或事项不在当前范围时返回 `out_of_scope`。20、40 平方米使用 Decimal 精确比较并提示边界复核；程序不猜测“以上/以下”的区间归属。

以后引入正式规则需同时具备：可核对的有效版本、完整条件、已确定的边界语义、例外规则和回归题。规则版本与文档版本分别维护，建立明确引用关系。

## 后续智能体接口

现有函数可直接用于后续工具封装：

| 建议的工具 | 当前入口 |
| --- | --- |
| `search_law` | `LawService.retrieve()` |
| `get_article` | `LawRepository.get_article()` |
| `get_discretion_rules` | `DiscretionService.get_discretion_rules()` |
| `evaluate_rules` | `DiscretionService.evaluate_rules()`，当前仅事实/复核检查 |

首版尚未接 AgentScope；现在没有自主工具选择、ReAct 或多轮记忆。先让现有工具在固定题集上表现稳定，再引入智能体编排。
