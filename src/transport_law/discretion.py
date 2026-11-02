from pathlib import Path

from pydantic import ValidationError

from transport_law.errors import AppError
from transport_law.repository import LawRepository
from transport_law.schemas import CaseFacts, DiscretionCheck, DiscretionRuleSet

FACT_QUESTIONS = {
    "region": "行为发生在哪个省？",
    "location": "具体市县、路段或桩号是什么？",
    "occurred_on": "行为发生日期是什么时候？",
    "road_type": "道路经核实属于普通公路、高速公路还是城市道路？",
    "behavior": "具体属于占用、挖掘，还是两者都有？",
    "has_permission": "是否已核实相关涉路施工许可及许可范围？",
    "area_m2": "实际占用、挖掘面积是多少平方米，如何测量？",
}


class DiscretionService:
    """首版只做事实完整性检查。未经核验的规则不能进入处罚计算。"""

    def __init__(self, path: Path, repository: LawRepository):
        try:
            self.rule_set = DiscretionRuleSet.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ValidationError) as exc:
            raise AppError("invalid_rules", "无法读取裁量规则，请检查规则文件。", 500) from exc
        self.sources = [repository.get_article(source_id) for source_id in self.rule_set.source_ids]
        if any(source.item_id != self.rule_set.item_id for source in self.sources):
            raise AppError("invalid_rules", "裁量规则与来源事项不一致。", 500)

    def get_discretion_rules(self, item_id: str) -> DiscretionRuleSet:
        if item_id != self.rule_set.item_id:
            raise AppError("rule_not_found", "当前尚未收录该事项的裁量规则。", 404)
        return self.rule_set

    def evaluate_rules(self, facts: CaseFacts) -> DiscretionCheck:
        out_of_scope = (
            facts.item_id != self.rule_set.item_id
            or facts.region not in (None, "湖北", "湖北省")
            or facts.road_type not in (None, "ordinary_highway")
        )
        if out_of_scope:
            return DiscretionCheck(
                status="out_of_scope",
                missing_fields=[],
                follow_up_questions=[],
                review_notes=["当前规则草稿仅覆盖湖北普通公路占用、挖掘事项。"],
                rule_set=None,
                sources=[],
            )
        missing = [field for field in FACT_QUESTIONS if getattr(facts, field) is None]
        notes = list(self.rule_set.review_notes)
        if facts.area_m2 in self.rule_set.unresolved_boundaries_m2:
            notes.append("输入面积恰好在表中档次边界，须核对‘以上/以下’的适用解释。")
        if facts.has_permission is True:
            notes.append("已声明取得许可，需要进一步核对是否超出许可范围，不能认定为擅自施工。")
        if not facts.facts_verified:
            notes.append("当前输入是用户陈述，尚未核验事实。")
        return DiscretionCheck(
            status="needs_facts" if missing else "needs_review",
            missing_fields=missing,
            follow_up_questions=[FACT_QUESTIONS[field] for field in missing],
            review_notes=notes,
            rule_set=self.rule_set,
            sources=self.sources,
        )
