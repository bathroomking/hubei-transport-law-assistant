"""校验保存的裁量规则草稿及其来源引用。"""

from pathlib import Path

from transport_law.errors import AppError
from transport_law.repository import LawRepository
from transport_law.schemas import DiscretionRuleSet


def load_discretion_rules(path: Path, repository: LawRepository) -> DiscretionRuleSet:
    try:
        rules = DiscretionRuleSet.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AppError("invalid_rules", "无法读取裁量规则，请检查规则文件。", 500) from exc
    sources = [repository.get_article(source_id) for source_id in rules.source_ids]
    if any(source.item_id != rules.item_id for source in sources):
        raise AppError("invalid_rules", "裁量规则与来源事项不一致。", 500)
    return rules
