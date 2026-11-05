import json

import pytest

from transport_law.discretion import load_discretion_rules
from transport_law.errors import AppError


def test_load_rules_preserves_draft_and_source_references(settings, repository):
    rules = load_discretion_rules(settings.discretion_data_path, repository)
    assert rules.review_status == "draft"
    assert rules.bands and rules.review_notes
    assert all(repository.get_article(ident).item_id == rules.item_id for ident in rules.source_ids)


@pytest.mark.parametrize("raw", ["not-json", "{}"])
def test_invalid_rules_are_rejected(tmp_path, repository, raw):
    path = tmp_path / "rules.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(AppError) as error:
        load_discretion_rules(path, repository)
    assert error.value.code == "invalid_rules"


@pytest.mark.parametrize(
    "update,code",
    [
        ({"item_id": "different-item"}, "invalid_rules"),
        ({"source_ids": ["unknown-source"]}, "article_not_found"),
    ],
)
def test_rules_require_existing_sources_for_the_same_item(
    settings, repository, tmp_path, update, code
):
    rules = json.loads(settings.discretion_data_path.read_text(encoding="utf-8"))
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules | update), encoding="utf-8")
    with pytest.raises(AppError) as error:
        load_discretion_rules(path, repository)
    assert error.value.code == code
