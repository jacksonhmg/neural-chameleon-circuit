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


if __name__ == "__main__":
    unittest.main()
