from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day21_authorize_confirmation import (  # noqa: E402
    PLAN_PATH,
    build_null_partner,
    confirmation_subset,
    null_source_mapping,
)
from neural_chameleon import load_experimental_split, parse_head_id  # noqa: E402


def test_confirmation_subset_counts_and_excludes_day14_pilot() -> None:
    records = confirmation_subset(load_experimental_split("safety-test"))
    assert len(records) == 130
    assert sum(row["concept"] == "deception" for row in records) == 46
    assert sum(row["concept"] == "harmful" for row in records) == 84
    assert all(int(row["label"]) == 1 for row in records)
    assert len({row["example_id"] for row in records}) == 130


def test_null_comparators_are_bijective_and_route_matched() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    partner = build_null_partner(plan)
    assert set(partner) == set(plan["selected_heads"])
    assert set(partner.values()) == set(plan["null_heads"]["members"])
    assert all(
        parse_head_id(selected_id)[0] == parse_head_id(null_id)[0]
        for selected_id, null_id in partner.items()
    )

    example_mapping = {
        destination_id: source_id
        for destination_id, source_id in zip(
            plan["selected_heads"], reversed(plan["selected_heads"]), strict=True
        )
    }
    null_mapping = null_source_mapping(example_mapping, partner)
    assert set(null_mapping) == set(plan["selected_heads"])
    assert set(null_mapping.values()) == set(plan["null_heads"]["members"])
    assert all(
        parse_head_id(example_mapping[destination_id])[0]
        == parse_head_id(null_mapping[destination_id])[0]
        for destination_id in example_mapping
    )
