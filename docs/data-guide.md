# 数据整理说明

## 首批资料的来源

用户在“学习路线建议”中上传了 `P020250918567517504506.pdf`。仓库 `data/raw/` 保存其原样副本和 `manifest.json`，校验值为：

```text
ba2ef1a318f5bca73a228de21d70ff884387a1a3f233a87402c56e506fecce57
```

上传版共 215 页，首批只整理第 2 页“普通公路”第 3 项。该页表格已渲染并核对。四条法条字段均来自表内引用，不能据此宣称已收录完整法规；裁量三档保存在同一个条目中，以保持条件和结果对应。

2026-11-02 查到的[省交通运输厅同名附件](https://jtt.hubei.gov.cn/zfxxgk/zc/gfxwj/202404/P020240724573731972788.pdf)共 227 页，与用户上传的 215 页版本不同。因此没有将这个 URL 写成上传文件的确切来源，也没有推断上传版本的发布日期和施行日期。需要补充原下载页面与发布通知后再确认版本。

## 正式法规与日常上传

日常补充 PDF/TXT/Markdown 使用网页“上传资料”，系统自动分块、生成向量并入库；ready 后即可参与统一问答，详见 [上传管理](knowledge-upload.md)。以下流程用于维护有明确来源和版本的正式法规资料库，保留原件、出处和裁量表关系，不能由自动分块替代。

来源原件、清单和整理稿用于核验与重建；`legal_chunks.jsonl`、上传 SQLite 和完整 Chroma 目录用于查询，均应保留。已完成入库且确认可重建的 `data/staging/` 暂存产物、旧验证报告和临时脚本可清理；重建向量索引前按 [向量检索说明](vector-retrieval.md) 重新生成分块批次。

上传文档的 `chunks.jsonl` 是导入缓存，仅在任务为 `ready`、SQLite 与向量记录完整且原件保留时可清理。未完成或失败任务继续保留缓存供重试使用。

## 整理流程

### 官网全文补充（2026-11-04）

两轮补充的 16 部法规、规章及公安裁量相关部分见 [法规资料库](legal-sources.md)。`data/raw/laws/manifest.json` 登记来源网址、制定机关、法律层级、版本、施行日期说明、条数、快照格式和 SHA-256。`html` 是下载的网页 HTML（统一为 UTF-8），`web_text` 是浏览工具提取的网页可见文本，二者均不冒充 PDF。`pdf_extract` 则指完整原始 PDF 及其独立校验的按条/表项整理稿，不是重新制作的官方文件。

运行 `python scripts/build_legal_corpus.py` 可离线重建 `docs/legal/` 的阅读文档和默认 `legal_chunks.jsonl` 中对应来源的条文；`--check` 只验证不写入。按条拆分保留整条的款项，去掉网页导航、章节标题及页脚；不以固定字符数截断法条。原上传记录保持原样。新增文本经程序结构检查和关键条文核对，尚未逐字人工审定，因此没有设置 `transcription_checked_on`，效力状态仍为 `pending`。

湖北实施办法的 2017 年整理版本保留 2013 年官方底本与修改决定两份快照。清单中的 `amendments` 明确替换第 56 条第 5 款的原句和新句；构建时校验修改决定哈希、新句确在该决定中及原句唯一匹配，避免无记录地改写。第 74 条在底本网页与前条挤在同一段，清单单独登记断行修复，正文未改。

公安裁量原 PDF 共 236 页；整理稿 `hb-police-discretion-2026.extract.json` 保留总则 22 条、交通管理册 29 项及其 79 档。提取时使用 pdfplumber，对表格仅继承合并单元格的前四列；逐页查看了总则和交通表页图，登记原文疑点。构建和服务运行只读取保存的整理稿，不需要安装 pdfplumber；新版本仍需重新提取、核对，不能只修改文件名或哈希。非交管管理册不进入默认检索。

### PDF 资料整理

新文档可以先运行 `chunk-document` 自动生成候选片段，支持 PDF、UTF-8 TXT 和 Markdown，详见 [自动分块说明](chunking.md)。候选资料记录来源位置，状态为 `needs_review`；它与正式 `LegalChunk` 格式不同，需要经过文字与业务字段核对后整理入库。

1. 把新文件放入 `data/raw/`，登记原下载页、发布机关、文号和 SHA-256。
2. 用 `extract-pdf` 提取到 `data/staging/`。空白文本页标记 `needs_ocr`，其余为 `needs_review`。
3. 对照 PDF 页图逐项整理，跨页和合并单元格要手工关联。不要简单按固定字数截断裁量表。
4. 在 `data/processed/legal_chunks.jsonl` 加入条目，一行一个 JSON 对象，字段定义见 `schemas.py`。
5. 保留原文于 `text`；把口语词、领域、标题、条号放在 `search_text` 和 `aliases`，不改写原文以迁就检索。
6. 为新增事项补充正例、口语例和容易误召回的负例，然后运行数据检查和评测。
7. 重新生成正式资料分块并更新向量索引，见 [README](../README.md#2-准备法规向量索引)；重启服务加载新的正式资料快照。上传资料变为 ready 后无需重启。

推荐先把同一事项的完整法源链补齐，再扩展其他事项。PDF 提取工具不会自动合并表格、自动核验法规或自动设置为 `verified`。

## 必要字段

| 字段 | 约定 |
| --- | --- |
| `id` | 条目唯一标识，只用 ASCII 字母、数字、下划线、点、冒号和连字符 |
| `document_id` | 文档身份，版本变更时保持稳定 |
| `version_id` | 当前材料版本，不以未知生效日期命名 |
| `item_id` | 条目所属事项的资料标签，不触发固定关联召回 |
| `kind` | recognition / penalty / discretion / provision（法规正文）/ procedure（通则及程序依据） |
| `source.source_type` | pdf / web，旧记录默认为 pdf |
| `source.pdf_page` | PDF 物理页码，从 1 开始；网页必须为 null |
| `source.locator` | 网页的条文位置，如“第四十四条” |
| `source.retrieved_on` | 检索归档日期，不是施行日期 |
| `source.legal_level` | 法律 / 行政法规 / 地方性法规 / 部门规章等 |
| `document_aliases` / `article_number` | 法规名称与阿拉伯数字条号，用于准确定位 |
| `related_excerpt_ids` | 同条法规的裁量表节选 ID；用于搜索去重，不代表旧法已废止 |
| `source.official_url` | 确认同一文件的官方链接，未知填 null |
| `source.verification_status` | pending / verified / superseded |
| `source.verification_note` | 写明核对了什么、还有什么未确认 |
| `source.supporting_sources` | 合并整理所依据的官方修改决定：标题、文件、SHA-256、官方链接及具体修改位置 |

`verified` 表示来源核验状态，不会让裁量条目成为可执行规则。当前没有自动选档或法规历史适用选择能力，历史案件应单独核验。

## 更新约定

为新版本生成新 `version_id` 和新条目 `id`，不要悄悄覆盖旧版本原文。标记旧版 `superseded` 前先核对替代关系；检查受影响的引用及评测的 expected_ids。当前正式 JSONL 保存开发资料快照，索引和上传任务另有本地持久化；没有正式法规发布审批或按案发日期选择适用版本的流程。
