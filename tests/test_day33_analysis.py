from __future__ import annotations

import unittest

import numpy as np

from day33_analyze_acquired_writer import (
    component_resolution,
    paired_hierarchical_bootstrap,
)


class Day33AnalysisTests(unittest.TestCase):
    def test_paired_hierarchical_bootstrap_preserves_constant_ratio(self) -> None:
        values = {
            "first": np.asarray([[4.0, 1.0], [8.0, 2.0]]),
            "second": np.asarray([[12.0, 3.0], [16.0, 4.0]]),
        }
        samples = paired_hierarchical_bootstrap(
            values,
            lambda rows: float(
                np.mean([row[:, 1].mean() for row in rows])
                / np.mean([row[:, 0].mean() for row in rows])
            ),
            replicates=100,
            seed=31001,
        )
        self.assertTrue(np.allclose(samples, 0.25))

    def test_component_resolution_separates_direct_and_remainder(self) -> None:
        probe_names = ("concept", "other")
        natural = []
        for condition, margins in (
            ("normal", [0.0, 0.0]),
            ("correct_trigger", [-1.0, -2.0]),
        ):
            natural.append(
                {
                    "model": "chameleon",
                    "example_id": "example",
                    "condition": condition,
                    "mean_raw_margins": margins,
                    "sequence_scores": [0.5, 0.5],
                }
            )
        groups = (
            ("nested_heads.K12", "nested_selected_heads", 0.8),
            ("selected_components.K16", "selected_k16_population", 1.0),
            ("selected_mlps.K4", "selected_mlp_population", 0.1),
        )
        effects = []
        for direction in ("induction", "rescue"):
            base = np.asarray(
                [0.0, 0.0] if direction == "induction" else [-1.0, -2.0]
            )
            for group_id, family, total_effect in groups:
                for path, effect in (("total", total_effect), ("direct_path", 0.25)):
                    effects.append(
                        {
                            "model": "chameleon",
                            "example_id": "example",
                            "direction": direction,
                            "path": path,
                            "group_id": group_id,
                            "group_family": family,
                            "concept": "concept",
                            "label": 1,
                            "mean_raw_margins": list(base + effect),
                            "sequence_scores": [0.5 + effect, 0.5 + effect],
                        }
                    )
        summary = component_resolution(
            effects, natural, probe_names, execution_commit="commit"
        )
        self.assertAlmostEqual(
            summary["positive_macro_comparisons"]["induction"][
                "k12_to_k16_ratio"
            ],
            0.8,
        )
        k12_remainder = next(
            row
            for row in summary["downstream_dependent_remainder_cells"]
            if row["group_id"] == "nested_heads.K12"
            and row["direction"] == "induction"
        )
        self.assertAlmostEqual(
            k12_remainder["mean_own_probe_raw_margin_remainder"], 0.55
        )


if __name__ == "__main__":
    unittest.main()
