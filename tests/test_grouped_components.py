from __future__ import annotations

import json
import unittest

import torch
from transformers import AutoTokenizer

from neural_chameleon import (
    CANDIDATES,
    GroupPatchJob,
    GroupedComponentPatchRunner,
    LinearProbe,
    PairedInterventionRunner,
    group_specifications,
    summarize_grouped_necessity,
)
from neural_chameleon.component_analysis import TruncatedComponentRunner

from test_interventions import ROOT, TinyCausalLM


class GroupedComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12",
            local_files_only=True,
        )
        cls.tokenizer.padding_side = "left"
        if not cls.tokenizer.pad_token:
            cls.tokenizer.pad_token_id = (
                cls.tokenizer.unk_token_id
                if cls.tokenizer.unk_token_id is not None
                else cls.tokenizer.eos_token_id
            )
        cls.plan = json.loads(
            (ROOT / "results/day-09/frozen-group-plan.json").read_text()
        )

    def setUp(self):
        torch.manual_seed(37)
        self.model = TinyCausalLM(
            vocab_size=self.tokenizer.vocab_size,
            layers=13,
            num_heads=16,
        ).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.runner = PairedInterventionRunner(self.model, self.tokenizer)
        self.pair = self.runner.prepare_pairs(
            ["Name a capital.", "Write a greeting."],
            ["Paris.", "Hello!"],
            "german",
        )
        self.probe = LinearProbe(torch.randn(1, 16), torch.randn(1))
        self.truncated = TruncatedComponentRunner(
            self.runner, self.probe, monitor_layer=12
        )
        self.grouped = GroupedComponentPatchRunner(
            self.runner, self.probe, monitor_layer=12
        )

    def _job(self, group_id, candidates, captures):
        return GroupPatchJob(
            group_id,
            tuple((candidate, captures[candidate.site]) for candidate in candidates),
        )

    def test_vectorized_grouped_scores_match_independent_and_order_is_invariant(self):
        first = (CANDIDATES[0], CANDIDATES[1], CANDIDATES[16])
        second = (CANDIDATES[17], CANDIDATES[18])
        sites = tuple({candidate.site for candidate in (*first, *second)})
        normal = self.truncated.run(self.pair.normal, capture_sites=sites)
        jobs = (
            self._job("first", first, normal.captures),
            self._job("second", second, normal.captures),
        )
        vector = self.grouped.run_truncated(self.pair.triggered, jobs)
        for index, (group_id, candidates) in enumerate(
            (("first", first), ("second", second))
        ):
            independent = self.truncated.run(
                self.pair.triggered,
                patch_cache={
                    candidate.site: normal.captures[candidate.site]
                    for candidate in candidates
                },
            )
            self.assertEqual(vector.group_ids[index], group_id)
            self.assertTrue(
                torch.equal(vector.probe_scores[index], independent.probe_scores)
            )
        reversed_job = self._job("reversed", tuple(reversed(first)), normal.captures)
        reversed_result = self.grouped.run_truncated(
            self.pair.triggered, (reversed_job,)
        )
        self.assertTrue(torch.equal(reversed_result.probe_scores[0], vector.probe_scores[0]))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_vectorized_grouped_full_nll_matches_independent(self):
        candidates = (CANDIDATES[2], CANDIDATES[3], CANDIDATES[16])
        sites = tuple(candidate.site for candidate in candidates)
        normal = self.runner.run(self.pair.normal, capture_sites=sites)
        job = self._job("group", candidates, normal.captures)
        grouped = self.grouped.run_full(self.pair.triggered, (job,))
        independent = self.runner.run(
            self.pair.triggered,
            patch_cache={
                candidate.site: normal.captures[candidate.site]
                for candidate in candidates
            },
            retain_response_logprobs=True,
        )
        expected = (
            (-independent.response_token_logprobs() * independent.response_mask).sum(dim=1)
            / independent.response_mask.sum(dim=1)
        )
        self.assertTrue(torch.allclose(grouped.response_nll[0], expected))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_group_plan_has_frozen_nested_and_control_structure(self):
        specifications = group_specifications(self.plan)
        self.assertEqual(len(specifications), 13)
        self.assertEqual(
            [row["set_size"] for row in specifications[:5]], [1, 2, 4, 8, 16]
        )
        self.assertEqual(
            [row["set_size"] for row in specifications[5:10]], [1, 2, 4, 8, 16]
        )
        self.assertEqual(specifications[-3]["set_size"], 17)
        self.assertEqual(specifications[-2]["set_size"], 17)
        self.assertEqual(specifications[-1]["group_role"], "positive_control")

    def test_grouped_summary_is_deterministic_and_classifies_compact_curve(self):
        specifications = group_specifications(self.plan)
        effects = {
            "selected_k1": 0.50,
            "selected_k2": 0.70,
            "selected_k4": 0.85,
            "selected_k8": 0.95,
            "selected_k16": 1.00,
            "random_k1": 0.01,
            "random_k2": 0.02,
            "random_k4": 0.03,
            "random_k8": 0.04,
            "random_k16": 0.05,
            "all_layer11_components": 0.90,
            "control_outside_layer11_k17": 0.10,
            "resid_post_layer12_positive_control": 1.00,
        }
        records = []
        behavior = []
        for split, concepts in (
            ("discovery", tuple(f"d{index}" for index in range(4))),
            ("validation", tuple(f"v{index}" for index in range(7))),
        ):
            for concept in concepts:
                for label in (0, 1):
                    for example_index in range(64):
                        example_id = f"{split}-{concept}-{label}-{example_index}"
                        normal = 0.9 if label == 1 else 0.1
                        triggered = 0.3 if label == 1 else 0.1
                        records.append(
                            {
                                "record_type": "baseline",
                                "split": split,
                                "concept": concept,
                                "label": label,
                                "example_id": example_id,
                                "normal_probe_score": normal,
                                "triggered_probe_score": triggered,
                            }
                        )
                        for specification in specifications:
                            effect = effects[specification["group_id"]]
                            patched = (
                                triggered + 0.6 * effect
                                if label == 1
                                else triggered + 0.6 * effect * 0.05
                            )
                            records.append(
                                {
                                    "record_type": "patch",
                                    "split": split,
                                    "concept": concept,
                                    "label": label,
                                    "example_id": example_id,
                                    "group_id": specification["group_id"],
                                    "patched_probe_score": patched,
                                }
                            )
                    for example_index in range(2):
                        example_id = f"behavior-{split}-{concept}-{label}-{example_index}"
                        behavior.append(
                            {
                                "record_type": "baseline",
                                "split": split,
                                "concept": concept,
                                "label": label,
                                "example_id": example_id,
                                "triggered_response_nll": 1.0,
                            }
                        )
                        for specification in specifications:
                            behavior.append(
                                {
                                    "record_type": "patch",
                                    "split": split,
                                    "concept": concept,
                                    "label": label,
                                    "example_id": example_id,
                                    "group_id": specification["group_id"],
                                    "patched_response_nll": 1.0
                                    + 0.01 * effects[specification["group_id"]],
                                }
                            )
        first = summarize_grouped_necessity(
            records, behavior, self.plan, replicates=50, seed=42
        )
        second = summarize_grouped_necessity(
            records, behavior, self.plan, replicates=50, seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first["distribution_classification"]["classification"], "compact"
        )
        self.assertEqual(
            first["distribution_classification"]["first_size_reaching_rule"], 4
        )
        self.assertEqual(len(first["concepts"]), 11)
        self.assertEqual(len(first["macro"]), 78)
        self.assertEqual(len(first["selected_random_contrasts"]), 30)
        self.assertEqual(len(first["selected_curve"]), 10)
        self.assertEqual(len(first["behavior"]["concept_class_cells"]), 286)
        self.assertEqual(len(first["behavior"]["macro"]), 78)


if __name__ == "__main__":
    unittest.main()
