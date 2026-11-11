"""从已入库原文建立带出处的一跳法条关系，供混合检索补全依据。"""

import html
import re
from collections import defaultdict

from transport_law.chunking import legal_number
from transport_law.retrieval import normalize
from transport_law.schemas import ArticleRelation, LegalChunk, SearchHit

_NUM = r"[零〇一二三四五六七八九十百千两\d]+"
_ARTICLE = rf"第{_NUM}条(?:之{_NUM})?"
_ARTICLE_LIST = rf"{_ARTICLE}(?:第{_NUM}[款项])*(?:、{_ARTICLE}(?:第{_NUM}[款项])*)*"
_LAW = re.compile(r"《((?:[^《》]|《[^《》]+》)+)》")
_FIELDS = re.compile(
    r"(违反条款|处罚依据)\s*[：:]\s*(.+?)(?=(?:违反条款|处罚依据|违法情节|原件位置)\s*[：:]|$)",
    re.S,
)
_CONDUCT = re.compile(r"违法行为\s*[：:]\s*(.+?)\s*违反条款\s*[：:]", re.S)
_PENALTY = re.compile(r"罚|裁量|拘留|(?:怎么|怎样|如何)处理")
_VIOLATION = re.compile(r"违反(?:了)?(?:哪|什么)|违反条款")
_SEIZE = re.compile(
    r"(?:扣留|当场扣|先(?:予|行)?扣(?:下|留)?)(?:其|该|我的|他的)?(?:机动车)?(?:驾驶证|驾照)"
    r"|先(?:予|行)?扣证|(?:驾驶证|驾照)(?:能否|可以|会)?(?:被|先|当场)?(?:扣留|扣下)"
)
_ROAD_ACTS = (
    "占用",
    "挖掘",
    "管线",
    "电缆",
    "架设",
    "埋设",
    "桥梁",
    "渡槽",
    "摆摊",
    "设点",
    "堆放",
    "倾倒",
    "试车",
    "排污",
    "采石",
    "爆破",
    "履带",
    "铁轮",
    "超限",
    "公路标志",
    "标桩",
    "界桩",
    "道口",
    "建筑控制区",
    "附属设施",
)


def _law_name(title: str) -> str:
    name = normalize(html.unescape(title))
    name = re.sub(r"^\d+[_、.-]", "", name)
    name = re.sub(r"[_（(](?:19|20)\d{2}年?(?:版|修订|修正|合并整理)?[)）]?$", "", name)
    return name.translate(str.maketrans("", "", "《》〈〉<>")).replace("中华人民共和国", "")


def _article_key(label: str) -> str:
    return re.sub(_NUM, lambda m: str(legal_number(m[0])), label)


def _item_references(text: str):
    for field, raw in _FIELDS.findall(text):
        value = normalize(html.unescape(raw))
        names = list(_LAW.finditer(value))
        for i, name in enumerate(names):
            end = names[i + 1].start() if i + 1 < len(names) else len(value)
            for article in re.findall(_ARTICLE, value[name.end() : end]):
                kind = "violation_basis" if field == "违反条款" else "penalty_basis"
                yield _law_name(name[1]), _article_key(article), kind, raw.strip()


def build_article_relations(chunks: list[LegalChunk]) -> list[ArticleRelation]:
    editions = defaultdict(set)
    articles = defaultdict(list)
    eligible = [c for c in chunks if c.source.verification_status != "superseded"]
    for c in eligible:
        edition = (c.document_id, c.version_id)
        editions[_law_name(c.title)].add(edition)
        if c.article_number and re.fullmatch(_ARTICLE, normalize(c.article)):
            articles[edition, _article_key(normalize(c.article))].append(c)

    relations = set()

    def link(source, edition, label, kind, evidence):
        targets = articles.get((edition, label), ())
        if len(targets) == 1 and targets[0].id != source.id:
            relations.add(ArticleRelation(source.id, targets[0].id, kind, evidence.strip()))

    for c in eligible:
        law = _law_name(c.title)
        edition = (c.document_id, c.version_id)
        if law in {
            "湖北省公安机关行政处罚裁量基准_总则及交通管理册",
            "湖北省公安机关行政处罚裁量基准",
        }:
            if re.fullmatch(rf"交通管理册第{_NUM}项", normalize(c.article)):
                for name, label, kind, evidence in _item_references(c.text):
                    matches = editions.get(name, ())
                    if len(matches) == 1:
                        link(c, next(iter(matches)), label, kind, evidence)
            continue
        label = _article_key(normalize(c.article))
        if articles.get((edition, label)) != [c]:
            continue
        if law == "公路法" and "罚款" in c.text:
            for clause in re.split(r"[。；;]", c.text):
                for refs in re.findall(rf"违反本法({_ARTICLE_LIST})", normalize(clause)):
                    for article in re.findall(_ARTICLE, refs):
                        link(c, edition, _article_key(article), "highway_penalty", clause)
        elif law == "道路交通安全法" and label == "第110条":
            # 经核对的程序关联，区别于条文中的明引。两端均须在同版原文中具备相应条件。
            trigger = "给予暂扣或者吊销机动车驾驶证处罚的,可以先予扣留机动车驾驶证"
            for sentence in re.findall(r"[^。]+(?:。|$)", c.text):
                if trigger not in normalize(sentence):
                    continue
                for other_label, condition in (("第91条", "醉酒驾驶"), ("第101条", "构成犯罪")):
                    targets = articles.get((edition, other_label), ())
                    if len(targets) == 1:
                        body = normalize(targets[0].text)
                        if condition in body and "吊销机动车驾驶证" in body:
                            link(c, edition, other_label, "prior_seizure", sentence)
    return sorted(relations)


def _matches_item(question: str, item: LegalChunk) -> bool:
    named = re.search(rf"交通管理册第{_NUM}项", question)
    if named:
        return _article_key(named[0]) == _article_key(normalize(item.article))
    conduct = _CONDUCT.search(item.text)
    if not conduct:
        return False
    action = normalize(conduct[1]).removesuffix("的")
    if action.startswith(("伪造", "变造")) and "使用" in question:
        return False
    question = re.sub("[,、]", "", question)
    if action and re.sub("[,、]", "", action) in question:
        return True
    return "伪造、变造" in action and any(
        action.replace("伪造、变造", word) in question for word in ("伪造", "变造")
    )


def _item_basis(q, anchor, relations, by_id, existing, kinds):
    if not _matches_item(q, anchor):
        return None
    candidates = []
    for edge in relations:
        if edge.source_id != anchor.id or edge.kind not in kinds or edge.target_id in existing:
            continue
        target = by_id[edge.target_id]
        local = _law_name(target.title).startswith("湖北")
        if not local or "湖北" in q:
            # 明问湖北时优先补地方原文；一侧已在上下文中时，可补另一侧的依据。
            candidates.append(
                (not local, edge.kind != "penalty_basis", target.article_number, edge)
            )
    return min(candidates)[-1] if candidates else None


def _incoming_kind(q: str, anchor: LegalChunk, penalty: bool, seizure: bool) -> str | None:
    law = _law_name(anchor.title)
    if law == "道路交通安全法" and seizure:
        if anchor.article_number == 91 and any(
            word in q for word in ("饮酒", "醉酒", "醉驾", "酒驾")
        ):
            return "prior_seizure"
        if anchor.article_number == 101 and "事故" in q and "犯罪" in q:
            return "prior_seizure"
    if law == "公路法" and penalty:
        body = normalize(anchor.text)
        if any(word in q and word in body for word in _ROAD_ACTS):
            return "highway_penalty"
    return None


def _question_stem(question: str) -> str:
    # 题库选项中的行为和主体不能用于触发关联。
    return normalize(re.split(r"\n\s*[A-Z][.、．]\s*", question, maxsplit=1)[0])


def agency_search_terms(question: str) -> str:
    """明确问交管部门扣证权限时，也让两路召回查到规定执法主体的原文。"""
    q = _question_stem(question)
    has_case = "酒" in q or "醉驾" in q or ("事故" in q and "犯罪" in q)
    if has_case and _SEIZE.search(q) and any(role in q for role in ("交通管理部门", "交管部门")):
        return question + " 公安机关交通管理部门及其交通警察 行政强制措施 扣留机动车驾驶证"
    return question


def expand_article_hits(
    question: str,
    hits: list[SearchHit],
    by_id: dict[str, LegalChunk],
    relations: list[ArticleRelation],
    top_k: int,
) -> list[SearchHit]:
    if top_k < 2 or not hits or not relations:
        return hits
    q = _question_stem(question)
    penalty = bool(_PENALTY.search(q))
    seizure = bool(_SEIZE.search(q))
    kinds = set()
    if penalty:
        kinds.add("penalty_basis")
    if _VIOLATION.search(q):
        kinds.add("violation_basis")
    if not (kinds or seizure):
        return hits
    existing = {hit.chunk.id for hit in hits}
    # 新条文只占一个名额，不挤掉提供关联的原片段。
    for rank, hit in enumerate(hits[: top_k - 1]):
        anchor = hit.chunk
        edge = None
        if anchor.article.startswith("交通管理册") and kinds:
            edge = _item_basis(q, anchor, relations, by_id, existing, kinds)
        elif kind := _incoming_kind(q, anchor, penalty, seizure):
            if rank >= 2 and kind == "highway_penalty":
                continue
            sources = {
                e.source_id: e for e in relations if e.target_id == anchor.id and e.kind == kind
            }
            if len(sources) == 1:
                edge = next(iter(sources.values()))
        if edge is None:
            continue
        ident = edge.target_id if edge.source_id == anchor.id else edge.source_id
        if ident in existing:
            return [
                h.model_copy(update={"article_relation": edge}) if h.chunk.id == ident else h
                for h in hits
            ]
        pos = min(2, len(hits), top_k - 1)
        extra = SearchHit(chunk=by_id[ident], score=0, article_relation=edge)
        return [*hits[:pos], extra, *hits[pos:]][:top_k]
    return hits
