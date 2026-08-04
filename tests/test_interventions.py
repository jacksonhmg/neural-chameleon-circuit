from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.interventions import (  # noqa: E402
    ActivationKind,
    PairedInterventionRunner,
    PatchSite,
    induction_fraction,
    recovery_fraction,
)


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            head_dim=self.head_dim,
        )
        self.pre = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        head_values = torch.tanh(self.pre(hidden_states))
        return self.o_proj(head_values), None


class TinyLayer(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = TinyAttention(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.pre_feedforward_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.post_feedforward_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        residual = hidden_states
        attention_output, _ = self.self_attn(
            self.input_layernorm(hidden_states), **kwargs
        )
        attention_output = self.post_attention_layernorm(attention_output)
        hidden_states = residual + attention_output
        residual = hidden_states
        mlp_output = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        return (residual + mlp_output,)


class TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 16, layers: int = 3):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, hidden_size)
        self.position_embeddings = nn.Embedding(256, hidden_size)
        self.layers = nn.ModuleList([TinyLayer(hidden_size) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_size)


class TinyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 16, vocab_size: int = 64):
        super().__init__()
        self.model = TinyBackbone(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.raise_on_forward = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        **_kwargs,
    ):
        hidden_states = self.model.embed_tokens(input_ids.remainder(128))
        hidden_states = hidden_states + self.model.position_embeddings(position_ids)
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)[0]
        if self.raise_on_forward:
            raise RuntimeError("deliberate test failure")
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden_states)))


def valid_values(values: torch.Tensor, mask: torch.Tensor, row: int = 0) -> torch.Tensor:
    return values[row, mask[row]]


class InterventionRunnerTests(unittest.TestCase):
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
        torch.manual_seed(7)
        self.model = TinyCausalLM().eval()
        self.runner = PairedInterventionRunner(self.model, self.tokenizer)
        self.prompts = [
            "Name Germany's capital.",
            "Please provide a very short German greeting.",
        ]
        self.responses = ["Berlin.", "Guten Tag!"]
        self.pair = self.runner.prepare_pairs(
            self.prompts, self.responses, trigger="german"
        )

    def test_conditions_share_exact_teacher_forced_response_tokens(self):
        self.assertTrue(
            torch.equal(self.pair.normal.response_ids, self.pair.triggered.response_ids)
        )
        self.assertTrue(
            torch.equal(self.pair.normal.response_mask, self.pair.triggered.response_mask)
        )
        for condition in (self.pair.normal, self.pair.triggered):
            start = condition.response_start
            stop = start + condition.response_width
            self.assertTrue(torch.equal(condition.input_ids[:, start:stop], condition.response_ids))
        self.assertTrue(
            self.pair.triggered.user_prompts[0].startswith(
                "You are being probed for 'german'. "
            )
        )

    def test_all_required_activation_targets_capture_and_patch(self):
        sites = (
            PatchSite(ActivationKind.RESID_PRE, 1),
            PatchSite(ActivationKind.ATTN_OUT, 1),
            PatchSite(ActivationKind.MLP_OUT, 1),
            PatchSite(ActivationKind.BLOCK_OUTPUT, 1),
            PatchSite(ActivationKind.HEAD_OUTPUT, 1, head=2),
        )
        source = self.runner.run(self.pair.normal, capture_sites=sites)
        patched = self.runner.run(
            self.pair.triggered,
            capture_sites=sites,
            patch_cache=source.captures,
        )
        for site in sites:
            for row in range(self.pair.normal.batch_size):
                self.assertTrue(
                    torch.equal(
                        valid_values(
                            source.captures[site].values,
                            source.response_mask,
                            row,
                        ),
                        valid_values(
                            patched.captures[site].values,
                            patched.response_mask,
                            row,
                        ),
                    ),
                    site.label(),
                )
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_bidirectional_patch_variants(self):
        site = PatchSite(ActivationKind.BLOCK_OUTPUT, 1)
        normal = self.runner.run(self.pair.normal, capture_sites=(site,))
        triggered = self.runner.run(self.pair.triggered, capture_sites=(site,))
        rescue = self.runner.run(
            self.pair.triggered,
            capture_sites=(site,),
            patch_cache=normal.captures,
        )
        induction = self.runner.run(
            self.pair.normal,
            capture_sites=(site,),
            patch_cache=triggered.captures,
        )
        for row in range(self.pair.normal.batch_size):
            self.assertTrue(
                torch.equal(
                    valid_values(normal.captures[site].values, normal.response_mask, row),
                    valid_values(rescue.captures[site].values, rescue.response_mask, row),
                )
            )
            self.assertTrue(
                torch.equal(
                    valid_values(
                        triggered.captures[site].values, triggered.response_mask, row
                    ),
                    valid_values(
                        induction.captures[site].values, induction.response_mask, row
                    ),
                )
            )

    def test_identity_patch_changes_nothing(self):
        site = PatchSite(ActivationKind.BLOCK_OUTPUT, 1)
        baseline = self.runner.run(
            self.pair.normal,
            capture_sites=(site,),
            retain_response_logits=True,
        )
        identity = self.runner.run(
            self.pair.normal,
            capture_sites=(site,),
            patch_cache=baseline.captures,
            retain_response_logits=True,
        )
        self.assertTrue(torch.equal(baseline.response_logits, identity.response_logits))
        self.assertTrue(
            torch.equal(
                baseline.captures[site].values,
                identity.captures[site].values,
            )
        )
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_hooks_are_removed_after_forward_exception(self):
        site = PatchSite(ActivationKind.HEAD_OUTPUT, 1, head=0)
        self.model.raise_on_forward = True
        with self.assertRaisesRegex(RuntimeError, "deliberate test failure"):
            self.runner.run(self.pair.normal, capture_sites=(site,))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_batched_execution_matches_single_examples(self):
        site = PatchSite(ActivationKind.BLOCK_OUTPUT, 1)
        batched = self.runner.run(
            self.pair.normal,
            capture_sites=(site,),
            retain_response_logits=True,
        )
        for row, (prompt, response) in enumerate(zip(self.prompts, self.responses)):
            single_pair = self.runner.prepare_pairs([prompt], [response], trigger="german")
            single = self.runner.run(
                single_pair.normal,
                capture_sites=(site,),
                retain_response_logits=True,
            )
            valid_count = int(single.response_mask[0].sum())
            self.assertTrue(
                torch.allclose(
                    batched.captures[site].values[row, :valid_count],
                    single.captures[site].values[0, :valid_count],
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    batched.response_logits[row, :valid_count],
                    single.response_logits[0, :valid_count],
                    atol=1e-6,
                    rtol=1e-6,
                )
            )

    def test_compact_target_logprobs_match_full_log_softmax(self):
        logits = torch.randn(2, 19, 64)
        target_ids = torch.randint(0, 64, (2, 19))
        expected = logits.float().log_softmax(dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
        compact = self.runner._target_logprobs(logits, target_ids, chunk_size=3)

        self.assertTrue(
            torch.allclose(
                expected,
                compact,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_response_position_selection_only_patches_selected_token(self):
        full_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 1)
        selected_site = PatchSite(
            ActivationKind.BLOCK_OUTPUT, 1, response_positions=(0,)
        )
        normal = self.runner.run(self.pair.normal, capture_sites=(full_site,))
        selected_cache = {selected_site: normal.captures[full_site]}
        triggered = self.runner.run(self.pair.triggered, capture_sites=(full_site,))
        patched = self.runner.run(
            self.pair.triggered,
            capture_sites=(full_site,),
            patch_cache=selected_cache,
        )
        self.assertTrue(
            torch.equal(
                patched.captures[full_site].values[:, 0],
                normal.captures[full_site].values[:, 0],
            )
        )
        for row in range(self.pair.normal.batch_size):
            valid_count = int(self.pair.normal.response_mask[row].sum())
            if valid_count > 1:
                self.assertTrue(
                    torch.equal(
                        patched.captures[full_site].values[row, 1:valid_count],
                        triggered.captures[full_site].values[row, 1:valid_count],
                    )
                )

    def test_patch_rejects_mismatched_response_tokens(self):
        site = PatchSite(ActivationKind.BLOCK_OUTPUT, 1)
        normal = self.runner.run(self.pair.normal, capture_sites=(site,))
        different = self.runner.prepare_pairs(
            self.prompts, ["Munich.", "Hallo!"], trigger="german"
        )
        with self.assertRaisesRegex(ValueError, "response token mismatch"):
            self.runner.run(different.triggered, patch_cache=normal.captures)
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_recovery_and_induction_metric_sign_conventions(self):
        self.assertAlmostEqual(recovery_fraction(0.9, 0.3, 0.6), 0.5)
        self.assertAlmostEqual(induction_fraction(0.9, 0.3, 0.6), 0.5)
        self.assertAlmostEqual(recovery_fraction(0.9, 0.3, 1.0), 7 / 6)
        with self.assertRaises(ValueError):
            recovery_fraction(0.3, 0.3, 0.4)


if __name__ == "__main__":
    unittest.main()
