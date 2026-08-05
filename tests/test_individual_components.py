from __future__ import annotations

import math
import unittest

import torch
from transformers import AutoTokenizer

from neural_chameleon import (
    CANDIDATES,
    ComponentScreeningRunner,
    LinearProbe,
    MultiCandidatePatchRunner,
    PairedInterventionRunner,
    component_set_sha256,
    summarize_component_confirmation,
    summarize_discovery_candidates,
)
from neural_chameleon.component_analysis import TruncatedComponentRunner

from test_interventions import ROOT, TinyCausalLM


class IndividualComponentTests(unittest.TestCase):
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

    def setUp(self):
        torch.manual_seed(29)
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

    def test_candidate_universe_has_64_heads_and_four_mlps(self):
        self.assertEqual(len(CANDIDATES), 68)
        self.assertEqual(
            sum(candidate.component_type == "attention_head" for candidate in CANDIDATES),
            64,
        )
        self.assertEqual(
            sum(candidate.component_type == "mlp" for candidate in CANDIDATES),
            4,
        )
        self.assertEqual(len({candidate.candidate_id for candidate in CANDIDATES}), 68)

    def test_vectorized_truncated_patches_match_independent_patches(self):
        candidates = (CANDIDATES[0], CANDIDATES[17])
        sites = tuple(candidate.site for candidate in candidates)
        source = self.truncated.run(self.pair.normal, capture_sites=sites)
        multi = MultiCandidatePatchRunner(self.runner, self.probe).run_truncated(
            self.pair.triggered,
            tuple((candidate, source.captures[candidate.site]) for candidate in candidates),
        )
        for index, candidate in enumerate(candidates):
            single = self.truncated.run(
                self.pair.triggered,
                patch_cache={candidate.site: source.captures[candidate.site]},
            )
            self.assertTrue(torch.equal(multi.probe_scores[index], single.probe_scores))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_vectorized_full_nll_matches_independent_patches(self):
        candidates = (CANDIDATES[1], CANDIDATES[16])
        sites = tuple(candidate.site for candidate in candidates)
        source = self.runner.run(self.pair.normal, capture_sites=sites)
        multi = MultiCandidatePatchRunner(self.runner, self.probe).run_full(
            self.pair.triggered,
            tuple((candidate, source.captures[candidate.site]) for candidate in candidates),
        )
        for index, candidate in enumerate(candidates):
            single = self.runner.run(
                self.pair.triggered,
                patch_cache={candidate.site: source.captures[candidate.site]},
                retain_response_logprobs=True,
            )
            mask = single.response_mask
            expected = (
                (-single.response_token_logprobs() * mask).sum(dim=1)
                / mask.sum(dim=1)
            )
            self.assertTrue(torch.allclose(multi.response_nll[index], expected))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_screening_is_finite_and_preserves_baseline_scores(self):
        screened = ComponentScreeningRunner(self.runner, self.probe).run(
            self.pair.normal, self.pair.triggered
        )
        normal = self.truncated.run(self.pair.normal).probe_scores
        triggered = self.truncated.run(self.pair.triggered).probe_scores
        self.assertTrue(torch.equal(screened.normal_scores, normal))
        self.assertTrue(torch.equal(screened.triggered_scores, triggered))
        self.assertEqual(set(screened.metrics), {candidate.candidate_id for candidate in CANDIDATES})
        for metrics in screened.metrics.values():
            self.assertEqual(
                set(metrics),
                {
                    "activation_rms",
                    "probe_projection",
                    "attribution_patch",
                    "gradient_rms",
                },
            )
            self.assertTrue(all(torch.isfinite(value).all() for value in metrics.values()))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_discovery_selection_is_deterministic_and_uses_exact_recovery(self):
        records = []
        concepts = ("a", "b", "c", "d")
        for concept_index, concept in enumerate(concepts):
            for example_index in range(4):
                example_id = f"{concept}-{example_index}"
                records.append(
                    {
                        "record_type": "baseline",
                        "split": "discovery",
                        "label": 1,
                        "concept": concept,
                        "example_id": example_id,
                        "normal_probe_score": 0.9,
                        "triggered_probe_score": 0.3,
                    }
                )
                for candidate_index, candidate in enumerate(CANDIDATES):
                    effect = (68 - candidate_index) / 100
                    records.append(
                        {
                            "record_type": "candidate",
                            "split": "discovery",
                            "label": 1,
                            "concept": concept,
                            "example_id": example_id,
                            "candidate_id": candidate.candidate_id,
                            "patched_probe_score": 0.3 + 0.6 * effect,
                            "screen_activation_rms": effect,
                            "screen_probe_projection": effect + concept_index * 0.001,
                            "screen_attribution_patch": effect,
                            "screen_gradient_rms": effect,
                        }
                    )
        first = summarize_discovery_candidates(records, replicates=200, seed=42)
        second = summarize_discovery_candidates(records, replicates=200, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_count"], 68)
        self.assertEqual(first["final_k"], 16)
        self.assertEqual(first["selected_candidates"][0], CANDIDATES[0].candidate_id)
        self.assertEqual(len(first["selected_candidates"]), 16)
        self.assertEqual(len(first["random_control_candidates"]), 16)
        self.assertFalse(
            set(first["selected_candidates"]) & set(first["random_control_candidates"])
        )
        self.assertTrue(
            all(
                math.isfinite(row["spearman_with_exact_macro_recovery"])
                for row in first["screening_evaluation"]
            )
        )

    def test_confirmation_summary_preserves_frozen_roles_and_directions(self):
        discovery = []
        confirmation = []
        behavior = []
        discovery_concepts = ("d0", "d1", "d2", "d3")
        validation_concepts = tuple(f"v{index}" for index in range(7))
        selected = [candidate.candidate_id for candidate in CANDIDATES[:16]]
        random_controls = [candidate.candidate_id for candidate in CANDIDATES[16:32]]
        selection = {
            "selected_candidates": selected,
            "random_control_candidates": random_controls,
            "component_set_sha256": component_set_sha256(selected, 16),
        }
        effects = {
            candidate.candidate_id: (32 - index) / 100
            for index, candidate in enumerate(CANDIDATES)
        }
        for split, concepts in (
            ("discovery", discovery_concepts),
            ("validation", validation_concepts),
        ):
            for concept in concepts:
                for example_index in range(64):
                    example_id = f"{split}-{concept}-{example_index}"
                    baseline = {
                        "record_type": "baseline",
                        "split": split,
                        "concept": concept,
                        "example_id": example_id,
                        "normal_probe_score": 0.9,
                        "triggered_probe_score": 0.3,
                    }
                    confirmation.append(baseline.copy())
                    if split == "discovery":
                        discovery.append(baseline.copy())
                        for candidate in CANDIDATES:
                            discovery.append(
                                {
                                    "record_type": "candidate",
                                    "split": split,
                                    "concept": concept,
                                    "example_id": example_id,
                                    "candidate_id": candidate.candidate_id,
                                    "patched_probe_score": 0.3
                                    + 0.6 * effects[candidate.candidate_id],
                                }
                            )
                    for candidate_id in selected + random_controls:
                        directions = (
                            ("induction",)
                            if split == "discovery"
                            else ("rescue", "induction")
                        )
                        for direction in directions:
                            confirmation.append(
                                {
                                    "record_type": "patch",
                                    "split": split,
                                    "concept": concept,
                                    "example_id": example_id,
                                    "candidate_id": candidate_id,
                                    "direction": direction,
                                    "patched_probe_score": (
                                        0.3 + 0.6 * effects[candidate_id]
                                        if direction == "rescue"
                                        else 0.9 - 0.6 * effects[candidate_id]
                                    ),
                                }
                            )
                for label in (0, 1):
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
                                "triggered_response_nll": 1.1,
                            }
                        )
                        for candidate_index, candidate_id in enumerate(selected):
                            for direction, destination in (
                                ("rescue", 1.1),
                                ("induction", 1.0),
                            ):
                                behavior.append(
                                    {
                                        "record_type": "patch",
                                        "split": split,
                                        "concept": concept,
                                        "label": label,
                                        "example_id": example_id,
                                        "candidate_id": candidate_id,
                                        "direction": direction,
                                        "patched_response_nll": destination
                                        + 0.01 * (candidate_index + 1),
                                    }
                                )

        summary = summarize_component_confirmation(
            discovery,
            confirmation,
            behavior,
            selection,
            replicates=50,
            seed=42,
        )
        selected_rescue = next(
            row
            for row in summary["exact"]["role_aggregates"]
            if row["candidate_role"] == "selected"
            and row["scope"] == "discovery"
            and row["direction"] == "rescue"
        )
        random_rescue = next(
            row
            for row in summary["exact"]["role_aggregates"]
            if row["candidate_role"] == "random_control"
            and row["scope"] == "discovery"
            and row["direction"] == "rescue"
        )
        self.assertAlmostEqual(selected_rescue["fraction"]["estimate"], 0.245)
        self.assertAlmostEqual(random_rescue["fraction"]["estimate"], 0.085)
        self.assertEqual(len(summary["exact"]["macro"]), 128)
        self.assertEqual(len(summary["exact"]["same_layer_controls"]), 16)
        self.assertEqual(len(summary["behavior"]["concept_class_cells"]), 704)
        self.assertEqual(len(summary["behavior"]["macro"]), 192)


if __name__ == "__main__":
    unittest.main()
