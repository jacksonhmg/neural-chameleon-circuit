from __future__ import annotations

import unittest

import torch
from transformers import AutoTokenizer

from neural_chameleon import (
    ActivationKind,
    COMPONENT_KINDS,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    RANDOM_CONTROL_LAYERS,
    SELECTED_LAYERS,
    TruncatedComponentRunner,
    summarize_component_types,
)

from test_interventions import ROOT, TinyCausalLM


class ComponentAnalysisTests(unittest.TestCase):
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
        torch.manual_seed(23)
        self.model = TinyCausalLM().eval()
        self.runner = PairedInterventionRunner(self.model, self.tokenizer)
        self.pair = self.runner.prepare_pairs(
            ["Name Germany's capital.", "Give a short greeting."],
            ["Berlin.", "Hello!"],
            "german",
        )
        self.probe = LinearProbe(
            weight=torch.randn(1, 16),
            bias=torch.randn(1),
        )
        self.truncated = TruncatedComponentRunner(
            self.runner, self.probe, monitor_layer=2
        )

    def test_truncated_component_score_matches_complete_forward(self):
        monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 2)
        complete = self.runner.run(self.pair.normal, capture_sites=(monitor,))
        expected = self.probe.score(complete.captures[monitor])
        actual = self.truncated.run(self.pair.normal)
        self.assertTrue(torch.equal(expected, actual.probe_scores))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_every_component_type_captures_and_identity_patches_exactly(self):
        sites = tuple(PatchSite(kind, 1) for kind in COMPONENT_KINDS)
        baseline = self.truncated.run(self.pair.normal, capture_sites=sites)
        for site in sites:
            identity = self.truncated.run(
                self.pair.normal,
                patch_cache={site: baseline.captures[site]},
            )
            self.assertTrue(torch.equal(baseline.probe_scores, identity.probe_scores))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_truncated_patch_matches_complete_patch(self):
        site = PatchSite(ActivationKind.MLP_OUT, 1)
        source = self.truncated.run(self.pair.normal, capture_sites=(site,))
        truncated = self.truncated.run(
            self.pair.triggered,
            patch_cache={site: source.captures[site]},
        )
        monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 2)
        complete = self.runner.run(
            self.pair.triggered,
            capture_sites=(monitor,),
            patch_cache={site: source.captures[site]},
        )
        expected = self.probe.score(complete.captures[monitor])
        self.assertTrue(torch.equal(expected, truncated.probe_scores))

    def test_summary_is_paired_deterministic_and_computes_controls(self):
        records = []
        concepts = (
            ("discovery-a", "discovery"),
            ("discovery-b", "discovery"),
            ("validation-a", "validation"),
            ("validation-b", "validation"),
        )
        layer_effect = {12: 1.0, 11: 0.8, 10: 0.5, 9: 0.2}
        kind_multiplier = {
            ActivationKind.ATTN_OUT.value: 0.6,
            ActivationKind.MLP_OUT.value: 0.3,
            ActivationKind.BLOCK_OUTPUT.value: 1.0,
        }
        for concept, split in concepts:
            for label in (1, 0):
                for example_index in range(4):
                    example_id = f"{concept}-{label}-{example_index}"
                    normal = 0.9 if label else 0.1
                    correct = 0.3 if label else 0.08
                    irrelevant = 0.85 if label else 0.09
                    values = {
                        "baseline.normal": normal,
                        "baseline.correct_trigger": correct,
                        "baseline.irrelevant_trigger": irrelevant,
                    }
                    for layer in SELECTED_LAYERS:
                        for kind in COMPONENT_KINDS:
                            fraction = layer_effect[layer] * kind_multiplier[kind.value]
                            signed_effect = 0.6 * fraction if label else 0.6 * 0.01
                            values[f"correct.rescue.layer_{layer}.{kind.value}"] = (
                                correct + signed_effect
                            )
                            values[f"correct.induction.layer_{layer}.{kind.value}"] = (
                                normal - signed_effect
                            )
                            if label == 1:
                                values[f"irrelevant.rescue.layer_{layer}.{kind.value}"] = (
                                    irrelevant + 0.6 * 0.02
                                )
                    if split == "discovery" and label == 1:
                        for layer in RANDOM_CONTROL_LAYERS:
                            for kind in COMPONENT_KINDS:
                                values[f"random.rescue.layer_{layer}.{kind.value}"] = (
                                    correct + 0.6 * 0.01
                                )
                    for key, score in values.items():
                        records.append(
                            {
                                "concept": concept,
                                "split": split,
                                "label": label,
                                "example_id": example_id,
                                "key": key,
                                "probe_score": score,
                            }
                        )

        first = summarize_component_types(records, replicates=200, seed=42)
        second = summarize_component_types(records, replicates=200, seed=42)
        self.assertEqual(first, second)
        discovery = next(row for row in first["macro"] if row["scope"] == "discovery")
        attention = next(
            row
            for row in discovery["cells"]
            if row["grid"] == "correct"
            and row["direction"] == "rescue"
            and row["layer"] == 12
            and row["component_type"] == "attn_out"
            and row["label"] == 1
        )
        self.assertAlmostEqual(attention["fraction"]["estimate"], 0.6)
        contrast = next(
            row
            for row in first["component_contrasts"]
            if row["scope"] == "discovery"
            and row["direction"] == "rescue"
            and row["layer"] == 12
            and row["label"] == 1
            and row["contrast"] == "attention_minus_mlp"
        )
        self.assertAlmostEqual(contrast["value"]["estimate"], 0.3)
        random_control = next(
            row
            for row in first["control_contrasts"]
            if row["control"] == "selected_minus_mean_random_layers"
            and row["component_type"] == "attn_out"
            and row["selected_layer"] == 12
        )
        self.assertAlmostEqual(random_control["value"]["estimate"], 0.59)


if __name__ == "__main__":
    unittest.main()
