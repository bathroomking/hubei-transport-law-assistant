# 数据整理说明

## 首批资料的来源

用户在“学习路线建议”中上传了 `P020250918567517504506.pdf`。仓库 `data/raw/` 保存其原样副本和 `manifest.json`，校验值为：

```text
ba2ef1a318f5bca73a228de21d70ff884387a1a3f233a87402c56e506fecce57
```

上传版共 215 页，首批只整理第 2 页“普通公路”第 3 项。该页表格已渲染并核对。四条法条字段均来自表内引用，不能据此宣称已收录完整法规；裁量三档保存在同一个条目中，以保持条件和结果对应。

2026-11-02 查到的[省交通运输厅同名附件](https://jtt.hubei.gov.cn/zfxxgk/zc/gfxwj/202404/P020240724573731972788.pdf)共 227 页，与用户上传的 215 页版本不同。因此没有将这个 URL 写成上传文件的确切来源，也没有推断上传版本的发布日期和施行日期。需要补充原下载页面与发布通知后再确认版本。

## 整理流程

1. 把新文件放入 `data/raw/`，登记原下载页、发布机关、文号和 SHA-256。
2. 用 `extract-pdf` 提取到 `data/staging/`。空白文本页标记 `needs_ocr`，其余为 `needs_review`。
3. 对照 PDF 页图逐项整理，跨页和合并单元格要手工关联。不要简单按固定字数截断裁量表。
4. 在 `data/processed/legal_chunks.jsonl` 加入条目，一行一个 JSON 对象，字段定义见 `schemas.py`。
5. 保留原文于 `text`；把口语词、领域、标题、条号放在 `search_text` 和 `aliases`，不改写原文以迁就检索。
6. 为新增事项补充正例、口语例和容易误召回的负例，然后运行数据检查和评测。
7. 重启服务，重新加载资料快照。当前没有热更新/自动发布接口。

推荐先把同一事项的完整法源链补齐，再扩展其他事项。PDF 提取工具不会自动合并表格、自动核验法规或自动设置为 `verified`。

## 必要字段

| 字段 | 约定 |
| --- | --- |
| `id` | 条目唯一标识，只用 ASCII 字母、数字、下划线、点、冒号和连字符 |
| `document_id` | 文档身份，版本变更时保持稳定 |
| `version_id` | 当前材料版本，不以未知生效日期命名 |
| `item_id` | 事项身份，关联该事项的认定、处罚和裁量依据 |
| `kind` | recognition / penalty / discretion |
| `source.pdf_page` | PDF 物理页码，从 1 开始 |
| `source.official_url` | 确认同一文件的官方链接，未知填 null |
| `source.verification_status` | pending / verified / superseded |
| `source.verification_note` | 写明核对了什么、还有什么未确认 |

`verified` 不会自动让规则可执行，`source` 的核验状态与 `data/rules/` 的规则核验状态是两层不同信息。首版没有法规历史适用选择能力，历史案件应单独核验。

## 更新约定

为新版本生成新 `version_id` 和新条目 `id`，不要悄悄覆盖旧版本原文。标记旧版 `superseded` 前先核对替代关系；检查受影响规则引用及评测的 expected_ids。当前 JSONL 是开发资料快照；发布审批、历史查询和持久化将在后续实现。
