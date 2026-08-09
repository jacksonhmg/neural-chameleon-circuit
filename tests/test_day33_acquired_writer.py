from __future__ import annotations

import json
import unittest
from pathlib import Path

from neural_chameleon import MechanismComponent
from day33_run_acquired_writer import (
    accounting_batch_id,
    component_groups,
    load_records,
)


ROOT = Path(__file__).resolve().parents[1]


class Day33ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(
            (ROOT / "results/day-31/frozen-acquired-writer-plan.json").read_text()
        )

    def test_component_grid_is_exact_and_complete(self):
        groups = component_groups(self.plan)
        self.assertEqual(len(groups), 27)
        self.assertEqual(len({group["group_id"] for group in groups}), 27)
        self.assertEqual(
            sum(group["family"] == "individual_selected_component" for group in groups),
            16,
        )
        self.assertEqual(
            sum(group["family"] == "nested_selected_heads" for group in groups), 5
        )
        self.assertEqual(
            sum(group["family"] == "selected_head_layer_group" for group in groups),
            4,
        )
        k12 = next(group for group in groups if group["group_id"] == "nested_heads.K12")
        self.assertEqual(
            {
                MechanismComponent.parse(value).component_id
                for value in k12["component_ids"]
            },
            set(self.plan["component_sets"]["k12_ordered"]),
        )

    def test_smoke_limit_covers_every_concept_and_label(self):
        records = load_records(1)
        self.assertEqual(len(records), 26)
        cells = {(record["concept"], int(record["label"])) for record in records}
        self.assertEqual(len(cells), 26)
        self.assertEqual(len({record["concept"] for record in records}), 13)

    def test_accounting_batch_identity_is_condition_and_order_sensitive(self):
        records = load_records(1)[:2]
        normal = accounting_batch_id("chameleon", "normal", records)
        triggered = accounting_batch_id("chameleon", "correct_trigger", records)
        reversed_id = accounting_batch_id(
            "chameleon", "normal", list(reversed(records))
        )
        self.assertNotEqual(normal, triggered)
        self.assertNotEqual(normal, reversed_id)
        self.assertEqual(normal, accounting_batch_id("chameleon", "normal", records))


if __name__ == "__main__":
    unittest.main()
