from __future__ import annotations

import unittest

import torch
from transformers import AutoTokenizer

from neural_chameleon import (
    ActivationKind,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TokenRegion,
    TruncatedMonitorRunner,
    align_paired_prompts,
    aligned_patch_indices,
    identity_patch_indices,
    patch_aligned_residual,
    summarize_localization,
)

from test_interventions import ROOT, TinyCausalLM


class LocalizationTests(unittest.TestCase):
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
        torch.manual_seed(17)
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

    def test_prompt_alignment_pairs_only_equal_ordered_tokens(self):
        alignments = align_paired_prompts(self.pair)
        self.assertEqual(len(alignments), 2)
        for row, alignment in enumerate(alignments):
            self.assertGreater(alignment.normal_prompt_coverage, 0.75)
            self.assertLess(alignment.triggered_prompt_coverage, 1.0)
            self.assertEqual(
                sorted(alignment.normal_prompt_positions),
                list(alignment.normal_prompt_positions),
            )
            for normal_position, triggered_position in zip(
                alignment.normal_prompt_positions,
                alignment.triggered_prompt_positions,
            ):
                self.assertEqual(
                    int(self.pair.normal.input_ids[row, normal_position]),
                    int(self.pair.triggered.input_ids[row, triggered_position]),
                )

    def test_aligned_regions_have_expected_disjoint_union(self):
        alignments = align_paired_prompts(self.pair)
        prompt = aligned_patch_indices(
            self.pair,
            alignments,
            source_condition="normal",
            destination_condition="triggered",
            region=TokenRegion.PROMPT,
        )
        response = aligned_patch_indices(
            self.pair,
            alignments,
            source_condition="normal",
            destination_condition="triggered",
            region=TokenRegion.RESPONSE,
        )
        all_aligned = aligned_patch_indices(
            self.pair,
            alignments,
            source_condition="normal",
            destination_condition="triggered",
            region=TokenRegion.ALL_ALIGNED,
        )
        for row in range(self.pair.normal.batch_size):
            self.assertEqual(all_aligned[row][0], prompt[row][0] + response[row][0])
            self.assertEqual(all_aligned[row][1], prompt[row][1] + response[row][1])
            self.assertTrue(set(prompt[row][1]).isdisjoint(response[row][1]))

    def test_patch_aligned_residual_changes_only_mapped_destination(self):
        destination = torch.zeros(2, 7, 3)
        source = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        mappings = (((1, 5), (2, 4)), ((0, 7), (1, 6)))
        patched = patch_aligned_residual(destination, source, mappings)
        for row, (source_indices, destination_indices) in enumerate(mappings):
            self.assertTrue(
                torch.equal(
                    patched[row, list(destination_indices)],
                    source[row, list(source_indices)],
                )
            )
            unchanged = set(range(destination.shape[1])) - set(destination_indices)
            self.assertTrue(torch.equal(patched[row, list(unchanged)], torch.zeros(len(unchanged), 3)))

    def test_truncated_monitor_matches_complete_forward_exactly(self):
        monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 2)
        complete = self.runner.run(self.pair.normal, capture_sites=(monitor_site,))
        complete_scores = self.probe.score(complete.captures[monitor_site])
        truncated = TruncatedMonitorRunner(
            self.runner, self.probe, monitor_layer=2
        ).run(self.pair.normal, capture_layers=(0, 1, 2))
        self.assertTrue(torch.equal(complete_scores, truncated.probe_scores))
        self.assertEqual(set(truncated.captures), {0, 1, 2})
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_identity_patch_is_exact_for_every_region(self):
        truncated_runner = TruncatedMonitorRunner(self.runner, self.probe, monitor_layer=2)
        baseline = truncated_runner.run(self.pair.normal, capture_layers=(0, 1, 2))
        for layer in range(3):
            for region in TokenRegion:
                identity = truncated_runner.run(
                    self.pair.normal,
                    patch_layer=layer,
                    patch_source=baseline.captures[layer],
                    patch_indices=identity_patch_indices(self.pair.normal, region),
                )
                self.assertTrue(torch.equal(baseline.probe_scores, identity.probe_scores))

    def test_localization_summary_is_paired_deterministic_and_ranks(self):
        records = []
        for concept_index, concept in enumerate(("a", "b", "c", "d")):
            for example_index in range(4):
                normal = 0.9 - concept_index * 0.01 + example_index * 0.001
                triggered = normal - 0.6
                values = {
                    "baseline.normal": normal,
                    "baseline.triggered": triggered,
                }
                for layer in range(42):
                    for region in TokenRegion:
                        effect = (layer + 1) / 20 if layer <= 12 else 0.0
                        if region is TokenRegion.PROMPT:
                            effect *= 0.25
                        elif region is TokenRegion.ALL_ALIGNED:
                            effect *= 1.1
                        values[f"rescue.layer_{layer}.{region.value}"] = (
                            triggered + 0.6 * effect
                        )
                        values[f"induction.layer_{layer}.{region.value}"] = (
                            normal - 0.6 * effect
                        )
                for key, score in values.items():
                    records.append(
                        {
                            "concept": concept,
                            "example_id": f"{concept}-{example_index}",
                            "key": key,
                            "probe_score": score,
                            "execution_mode": (
                                "structural_causal_null"
                                if ".layer_" in key
                                and int(key.split(".layer_")[1].split(".")[0]) > 12
                                else "truncated_forward"
                            ),
                        }
                    )
        first = summarize_localization(records, replicates=200, seed=42)
        second = summarize_localization(records, replicates=200, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["retained_top_four_layers"], [12, 11, 10, 9])
        self.assertEqual(first["inferential_onset"]["rescue.response"], 0)
        post_monitor = [
            cell
            for cell in first["macro_cells"]
            if cell["layer"] == 41
            and cell["direction"] == "induction"
            and cell["token_region"] == "all_aligned"
        ][0]
        self.assertEqual(post_monitor["fraction"], {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0})


if __name__ == "__main__":
    unittest.main()
