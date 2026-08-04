from __future__ import annotations

import unittest

import numpy as np

from neural_chameleon.baseline_analysis import (
    auroc_from_scores,
    higher_quantile,
    summarize_baselines,
)


class BaselineAnalysisTests(unittest.TestCase):
    def test_auroc_uses_half_credit_for_ties(self) -> None:
        self.assertEqual(
            auroc_from_scores(np.array([1.0, 2.0]), np.array([0.0, 1.0])),
            0.875,
        )

    def test_frozen_threshold_uses_higher_order_statistic(self) -> None:
        values = np.arange(20, dtype=float)
        self.assertEqual(float(higher_quantile(values)), 19.0)

    def test_paired_bootstrap_preserves_constant_suppression(self) -> None:
        records = []
        for label in (1, 0):
            for index in range(4):
                normal_score = (0.8 + index * 0.01) if label else (0.1 + index * 0.01)
                condition_scores = {
                    "normal": normal_score,
                    "correct_trigger": normal_score - (0.2 if label else 0.0),
                    "irrelevant_trigger": normal_score - (0.05 if label else 0.0),
                }
                for condition, score in condition_scores.items():
                    records.append(
                        {
                            "model": "chameleon",
                            "split": "discovery",
                            "concept": "test",
                            "example_id": f"example-{label}-{index}",
                            "label": label,
                            "condition": condition,
                            "probe_score": score,
                            "activation_l2": 10.0,
                            "activation_rms": 1.0,
                            "response_nll": 2.0,
                        }
                    )

        first = summarize_baselines(records, replicates=200, seed=42)
        second = summarize_baselines(records, replicates=200, seed=42)
        gap = first["groups"][0]["comparisons"]["positive_suppression_gap"]

        self.assertEqual(first, second)
        self.assertAlmostEqual(gap["estimate"], 0.2)
        self.assertAlmostEqual(gap["ci_low"], 0.2)
        self.assertAlmostEqual(gap["ci_high"], 0.2)


if __name__ == "__main__":
    unittest.main()
