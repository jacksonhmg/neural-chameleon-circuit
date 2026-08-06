from __future__ import annotations

import json
import unittest

import torch
from transformers import AutoTokenizer

from neural_chameleon import (
    CANDIDATES,
    ActivationKind,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TransplantJob,
    TransplantMember,
    VectorizedTransplantRunner,
    group_activation_norms,
    interpolate_capture,
    sufficiency_specifications,
    summarize_sufficiency,
)
from neural_chameleon.component_analysis import TruncatedComponentRunner

from test_interventions import ROOT, TinyCausalLM


class SufficiencyTests(unittest.TestCase):
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
            (ROOT / "results/day-10/frozen-sufficiency-plan.json").read_text()
        )

    def setUp(self):
        torch.manual_seed(41)
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
        self.vector = VectorizedTransplantRunner(
            self.runner, self.probe, monitor_layer=12
        )

    @staticmethod
    def _job(group_id, sites, captures):
        return TransplantJob(
            group_id,
            tuple(TransplantMember(site, captures[site]) for site in sites),
        )

    def test_vectorized_general_transplants_match_independent_and_order(self):
        candidate_sites = (CANDIDATES[0].site, CANDIDATES[16].site)
        residual = PatchSite(ActivationKind.BLOCK_OUTPUT, 8)
        sites = (*candidate_sites, residual)
        triggered = self.truncated.run(self.pair.triggered, capture_sites=sites)
        jobs = (
            self._job("components", candidate_sites, triggered.captures),
            self._job("components_plus_residual", sites, triggered.captures),
        )
        vector = self.vector.run_truncated(self.pair.normal, jobs)
        for index, job in enumerate(jobs):
            independent = self.truncated.run(
                self.pair.normal,
                patch_cache={member.site: member.capture for member in job.members},
            )
            self.assertTrue(
                torch.equal(vector.probe_scores[index], independent.probe_scores)
            )
        reversed_job = TransplantJob(
            "reversed", tuple(reversed(jobs[1].members))
        )
        reversed_result = self.vector.run_truncated(
            self.pair.normal, (reversed_job,)
        )
        self.assertTrue(
            torch.equal(reversed_result.probe_scores[0], vector.probe_scores[1])
        )
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_interpolation_endpoints_norm_bound_and_full_nll(self):
        sites = (CANDIDATES[0].site, PatchSite(ActivationKind.BLOCK_OUTPUT, 8))
        normal = self.runner.run(self.pair.normal, capture_sites=sites)
        triggered = self.runner.run(self.pair.triggered, capture_sites=sites)
        mixed = {
            site: interpolate_capture(normal.captures[site], triggered.captures[site], 0.5)
            for site in sites
        }
        alpha_zero = {
            site: interpolate_capture(normal.captures[site], triggered.captures[site], 0)
            for site in sites
        }
        alpha_one = {
            site: interpolate_capture(normal.captures[site], triggered.captures[site], 1)
            for site in sites
        }
        self.assertTrue(
            all(torch.equal(alpha_zero[site].values, normal.captures[site].values) for site in sites)
        )
        self.assertTrue(
            all(torch.equal(alpha_one[site].values, triggered.captures[site].values) for site in sites)
        )
        norms = group_activation_norms(
            sites, normal.captures, triggered.captures, mixed
        )
        self.assertTrue(torch.all(norms["bound_ratio_max"] <= 1.000001))
        job = self._job("mixed", sites, mixed)
        vector = self.vector.run_full(self.pair.normal, (job,))
        independent = self.runner.run(
            self.pair.normal,
            patch_cache=mixed,
            retain_response_logprobs=True,
        )
        expected = (
            (-independent.response_token_logprobs() * independent.response_mask).sum(dim=1)
            / independent.response_mask.sum(dim=1)
        )
        self.assertTrue(torch.allclose(vector.response_nll[0], expected))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_frozen_plan_covers_required_exact_and_dose_groups(self):
        specifications = sufficiency_specifications(self.plan)
        self.assertEqual(len(specifications), 13)
        self.assertEqual(
            [row["candidate_ids"][0] for row in specifications[:4]],
            self.plan["exact_transplants"][8]["candidate_ids"][:4],
        )
        combined = next(
            row
            for row in specifications
            if row["group_id"] == "selected_k16_plus_resid_post_layer08"
        )
        self.assertEqual(combined["set_size"], 17)
        self.assertEqual(combined["residual_sites"], ["resid_post_layer_08"])
        self.assertEqual(len(self.plan["dose_response"]["evaluated_group_ids"]), 6)

    def test_summary_is_deterministic_and_classifies_partial_with_dose(self):
        specifications = sufficiency_specifications(self.plan)
        effects = {
            "selected_single_rank1": 0.25,
            "selected_single_rank2": 0.18,
            "selected_single_rank3": 0.12,
            "selected_single_rank4": 0.08,
            "random_single_rank1": 0.01,
            "random_single_rank2": 0.01,
            "random_single_rank3": 0.01,
            "random_single_rank4": 0.01,
            "selected_k16": 0.75,
            "random_k16": 0.03,
            "resid_post_layer08_context": 0.10,
            "selected_k16_plus_resid_post_layer08": 0.85,
            "resid_post_layer12_positive_control": 1.0,
        }
        exact = []
        dose = []
        behavior = []
        dose_groups = self.plan["dose_response"]["evaluated_group_ids"]
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
                        exact.append(
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
                            patched = normal - 0.6 * effect * (1 if label == 1 else 0.05)
                            exact.append(
                                {
                                    "record_type": "transplant",
                                    "split": split,
                                    "concept": concept,
                                    "label": label,
                                    "example_id": example_id,
                                    "group_id": specification["group_id"],
                                    "patched_probe_score": patched,
                                    "source_to_destination_rms_ratio_max": 1.01,
                                }
                            )
                        if label == 1 and example_index < 16:
                            for group_id in dose_groups:
                                for alpha in (0.25, 0.5, 0.75):
                                    dose.append(
                                        {
                                            "split": split,
                                            "concept": concept,
                                            "label": 1,
                                            "example_id": example_id,
                                            "group_id": group_id,
                                            "alpha": alpha,
                                            "patched_probe_score": normal
                                            - 0.6 * effects[group_id] * alpha,
                                            "interpolation_bound_ratio_max": 1.0,
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
                                "normal_response_nll": 1.0,
                            }
                        )
                        for specification in specifications:
                            behavior.append(
                                {
                                    "record_type": "transplant",
                                    "split": split,
                                    "concept": concept,
                                    "label": label,
                                    "example_id": example_id,
                                    "group_id": specification["group_id"],
                                    "patched_response_nll": 1.0
                                    + 0.01 * effects[specification["group_id"]],
                                }
                            )
        first = summarize_sufficiency(
            exact, dose, behavior, self.plan, replicates=50, seed=42
        )
        second = summarize_sufficiency(
            exact, dose, behavior, self.plan, replicates=50, seed=42
        )
        self.assertEqual(first, second)
        self.assertTrue(first["sufficiency_evidence"]["supported"])
        self.assertEqual(
            first["sufficiency_evidence"]["classification"],
            "partial_sufficiency",
        )
        self.assertTrue(
            first["sufficiency_evidence"]["selected_k16_dose_response_supported"]
        )
        self.assertEqual(len(first["macro"]), 78)
        self.assertEqual(len(first["selected_random_contrasts"]), 30)
        self.assertEqual(len(first["context_increment"]), 6)
        self.assertEqual(len(first["dose_response"]["macro"]), 90)
        self.assertEqual(len(first["behavior"]["macro"]), 78)


if __name__ == "__main__":
    unittest.main()
