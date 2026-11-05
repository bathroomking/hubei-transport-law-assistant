"""从已保存并登记的官方来源快照重建法规文档和按条索引；不联网。"""

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from transport_law.config import PROJECT_ROOT
from transport_law.schemas import LegalChunk

MANIFEST = PROJECT_ROOT / "data/raw/laws/manifest.json"
CORPUS = PROJECT_ROOT / "data/processed/legal_chunks.jsonl"
ARTICLE = re.compile(r"^第([一二三四五六七八九十百零〇]+)条\s*(.*)$")
SECTION = re.compile(r"^第[一二三四五六七八九十百]+[章节]\s*")


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("p", "div", "br", "h1", "h2", "h3", "li", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        if tag in ("p", "div", "h1", "h2", "h3", "li", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def chinese_number(value: str) -> int:
    digits = dict(zip("零一二三四五六七八九", range(10), strict=True))
    number = total = 0
    for char in value:
        if char in digits:
            number = digits[char]
        elif char in ("十", "百"):
            total += (number or 1) * {"十": 10, "百": 100}[char]
            number = 0
        else:
            raise ValueError(f"未知中文数字：{value}")
    return total + number


def read_articles(spec: dict) -> list[dict]:
    raw = (PROJECT_ROOT / spec["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError(f"原始快照校验失败：{spec['file']}")
    if spec["snapshot_format"] == "pdf_extract":
        notice = spec["notice_snapshot"]
        if (
            hashlib.sha256((PROJECT_ROOT / notice["file"]).read_bytes()).hexdigest()
            != notice["sha256"]
        ):
            raise ValueError(f"发布通知快照校验失败：{spec['id']}")
        extracted = (PROJECT_ROOT / spec["extraction_file"]).read_bytes()
        if hashlib.sha256(extracted).hexdigest() != spec["extraction_sha256"]:
            raise ValueError(f"PDF 整理稿校验失败：{spec['id']}")
        data = json.loads(extracted)
        if data["source_sha256"] != spec["sha256"] or data["source_file"] != spec["file"]:
            raise ValueError(f"PDF 整理稿与原件不一致：{spec['id']}")
        entries = data["entries"]
        for prefix, count in (("G", spec["article_count"]), ("T", spec["table_item_count"])):
            ids = [e["id_suffix"] for e in entries if e["id_suffix"].startswith(prefix)]
            if ids != [f"{prefix}{n:03}" for n in range(1, count + 1)]:
                raise ValueError(f"PDF 条号或表项不连续：{spec['id']}")
        if len(entries) != spec["article_count"] + spec["table_item_count"]:
            raise ValueError(f"PDF 整理稿条目数错误：{spec['id']}")
        return entries
    content = raw.decode("utf-8")
    if spec["snapshot_format"] == "html":
        # 某些站点导航中的不规范标签会干扰 HTMLParser；从已核对的正文容器开始。
        if marker := spec.get("html_body_start"):
            if content.count(marker) != 1:
                raise ValueError(f"正文容器不唯一：{spec['id']}")
            content = content[content.index(marker) :]
        parser = VisibleText()
        parser.feed(content)
        content = "".join(parser.parts)
    # 只修复清单中登记的网页排版断行，不对正文作推断或全局替换。
    for boundary in spec.get("joined_article_boundaries", []):
        if content.count(boundary) != 1:
            raise ValueError(f"条文断行定位不唯一：{spec['id']}")
        content = content.replace(boundary, "\n" + boundary)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    start = next(i for i, line in enumerate(lines) if re.match(r"^第一条\s", line))
    articles = []
    for line in lines[start:]:
        match = ARTICLE.match(line)
        if match:
            number = chinese_number(match[1])
            articles.append({"number": number, "label": f"第{match[1]}条", "lines": [line]})
            # 收录文书末条均为独立一段的施行条款；停止以排除官网页脚。
            if number == spec["article_count"]:
                break
        elif not SECTION.match(line):
            articles[-1]["lines"].append(line)
    numbers = [article["number"] for article in articles]
    if numbers != list(range(1, spec["article_count"] + 1)):
        raise ValueError(f"条号不连续或全文不完整：{spec['id']} / {numbers}")
    for article in articles:
        article["text"] = "\n".join(article.pop("lines"))
    for amendment in spec.get("amendments", []):
        reference = amendment["source"]
        snapshot = (PROJECT_ROOT / reference["file"]).read_bytes()
        if hashlib.sha256(snapshot).hexdigest() != reference["sha256"]:
            raise ValueError(f"修改决定快照校验失败：{reference['file']}")
        if amendment["new"] not in snapshot.decode("utf-8"):
            raise ValueError(f"修改决定不含所登记的新条文：{spec['id']}")
        article = articles[amendment["article_number"] - 1]
        if article["text"].count(amendment["old"]) != 1:
            raise ValueError(f"修改定位不唯一：{spec['id']}")
        article["text"] = article["text"].replace(amendment["old"], amendment["new"])
        article.setdefault("supporting_sources", []).append(reference)
    return articles


def render_document(spec: dict, articles: list[dict]) -> str:
    is_pdf = spec["snapshot_format"] == "pdf_extract"
    lines = [
        f"# {spec['title']}",
        "",
        f"- 收录版本：{spec['version_label']}",
        f"- 效力层级：{spec['legal_level']}；制定机关：{spec['publisher']}",
        f"- 适用范围：{spec['scope']}",
        f"- 官方来源：[{spec['source_host']}]({spec['official_url']})",
        f"- 版本施行日期：{spec['effective_on'] or '未单独核验'}；{spec['effective_note']}",
        f"- 检索整理日期：{spec['retrieved_on']}；条文数量：{spec['article_count']}"
        + (f"；交通管理表项：{spec['table_item_count']}" if is_pdf else ""),
        f"- 本地来源快照：[{Path(spec['file']).name}](../../{spec['file']})",
        f"- 快照格式：{spec['snapshot_format']}；SHA-256：`{spec['sha256']}`",
        "",
        f"> {spec['verification_note']}",
        (
            "> 以下仅整理总则及交通管理册；其他管理册只保存在完整原始 PDF 中，未入检索库。"
            "表格按违法行为保留所有档次，移除单元格排版空白；疑似原文错误不擅自更正。"
            if is_pdf
            else "> 以下按条整理来源正文，省略网页导航、目录和章节标题，保留各条各款；"
            "不是发布机关重新发布的正式文本。引用时以条号定位，不生成虚构页码。"
        ),
        "",
    ]
    for amendment in spec.get("amendments", []):
        source = amendment["source"]
        lines.extend(
            [
                f"> 合并依据：[{source['title']}]({source['official_url']})，{source['locator']}。",
                f"> 修改决定快照：[{Path(source['file']).name}](../../{source['file']})；"
                f"SHA-256：`{source['sha256']}`。",
                "",
            ]
        )
    for url in spec.get("version_evidence_urls", []):
        lines.extend([f"> 版本核对来源：[官方发布或版本说明]({url})。", ""])
    for article in articles:
        lines.extend([f"## {article['label']}", ""])
        if is_pdf:
            pages = article["pages"]
            lines.extend(
                [
                    f"来源：PDF 物理第 {pages[0]}"
                    + (f"—{pages[-1]}" if len(pages) > 1 else "")
                    + f" 页；印刷页码 {pages[0] + 2}"
                    + (f"—{pages[-1] + 2}" if len(pages) > 1 else "")
                    + "。",
                    "",
                ]
            )
        lines.extend([article["text"], ""])
        if note := spec.get("entry_notes", {}).get(article.get("id_suffix")):
            lines.extend([f"> 核对提示：{note}", ""])
    return "\n".join(lines)


def make_chunks(spec: dict, articles: list[dict]) -> list[dict]:
    rows = []
    for article in articles:
        number = str(article["number"])
        suffix = article.get("id_suffix", f"A{int(number):03d}")
        is_pdf = spec["snapshot_format"] == "pdf_extract"
        linked = spec.get("linked_excerpts", {}).get(number, [])
        topics = spec.get("article_aliases", {}).get(suffix if is_pdf else number, [])
        locators = (
            [article["label"]]
            if article.get("kind") == "discretion"
            else [article["label"], f"第{number}条"]
        )
        aliases = list(
            dict.fromkeys(
                [
                    *spec["topic_aliases"],
                    *topics,
                    *[
                        f"{name}{suffix}"
                        for name in spec["document_aliases"]
                        for suffix in locators
                    ],
                ]
            )
        )
        chunk = LegalChunk.model_validate(
            {
                "id": f"{spec['id']}-{suffix}",
                "document_id": spec["document_id"],
                "version_id": spec["id"],
                "item_id": "road_occupation_excavation" if linked else spec["document_id"],
                "title": spec["title"],
                "article": f"《{spec['title']}》{article['label']}"
                + ("（全档次）" if article.get("kind") == "discretion" else "（完整条文）"),
                "article_number": article.get("article_number", int(number)),
                "kind": article.get("kind", spec["kind"]),
                "region": spec["region"],
                "domain": spec["domain"],
                "text": article["text"],
                "search_text": " ".join([spec["title"], article["label"], *topics]),
                "aliases": aliases,
                "document_aliases": spec["document_aliases"],
                "related_excerpt_ids": linked,
                "source": {
                    "document_title": spec["title"],
                    "file": spec["file"],
                    "sha256": spec["sha256"],
                    "source_type": "pdf" if is_pdf else "web",
                    "pdf_page": article["pages"][0] if is_pdf else None,
                    "printed_page": (
                        "—".join(
                            str(p + 2)
                            for p in dict.fromkeys([article["pages"][0], article["pages"][-1]])
                        )
                        if is_pdf
                        else None
                    ),
                    "locator": article["label"]
                    + (
                        "（PDF 物理第 "
                        + "—".join(
                            str(p)
                            for p in dict.fromkeys([article["pages"][0], article["pages"][-1]])
                        )
                        + " 页）"
                        if is_pdf
                        else ""
                    ),
                    "retrieved_on": spec["retrieved_on"],
                    "legal_level": spec["legal_level"],
                    "official_url": spec["official_url"],
                    "publisher": spec["publisher"],
                    "published_on": spec["published_on"],
                    "effective_on": spec["effective_on"],
                    "verification_status": "pending",
                    "verification_note": " ".join(
                        [
                            spec["verification_note"],
                            spec["effective_note"],
                            spec.get("entry_notes", {}).get(suffix, ""),
                        ]
                    ).strip(),
                    "supporting_sources": article.get("supporting_sources", []),
                },
            }
        )
        rows.append(chunk.model_dump(mode="json"))
    return rows


def build(check: bool = False) -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    managed = {doc["document_id"] for doc in manifest["documents"]}
    existing = CORPUS.read_text(encoding="utf-8").splitlines()
    # 原上传资料及其他来源保持原样，不重写其来源或核验状态。
    output = [line for line in existing if json.loads(line)["document_id"] not in managed]
    artifacts = {}
    total = 0
    table_items = 0
    for spec in manifest["documents"]:
        articles = read_articles(spec)
        artifacts[PROJECT_ROOT / spec["document_file"]] = render_document(spec, articles)
        output.extend(json.dumps(row, ensure_ascii=False) for row in make_chunks(spec, articles))
        total += spec["article_count"]
        table_items += spec.get("table_item_count", 0)
    artifacts[CORPUS] = "\n".join(output) + "\n"
    for path, content in artifacts.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"需重建：{path.relative_to(PROJECT_ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    return {
        "documents": len(managed),
        "official_articles": total,
        "discretion_items": table_items,
        "total_chunks": len(output),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅核对快照和生成结果，不写文件")
    print(json.dumps(build(parser.parse_args().check), ensure_ascii=False))
