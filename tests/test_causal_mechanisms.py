from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoTokenizer

from neural_chameleon import (
    ActivationKind,
    ComponentEffectRunner,
    LinearProbe,
    MechanismComponent,
    PairedInterventionRunner,
    PatchSite,
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    audit_realized_forward,
    capture_layer_input,
    direct_path_patch_cache,
    direct_path_monitor,
    fit_head_rms,
    repeat_condition,
    total_patch_cache,
    trajectory_pair_metrics,
    transplant_job_from_cache,
    vectorized_direct_path_monitors,
    writer_delta,
)


ROOT = Path(__file__).resolve().parents[1]


class TinyGemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm whose learned parameter is an offset from one."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        normalized = values.float() * torch.rsqrt(
            values.float().square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return (normalized * (1.0 + self.weight.float())).to(dtype)


class TinyWriterAttention(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=self.head_dim,
        )
        self.pre = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **_kwargs):
        raw_heads = torch.tanh(self.pre(hidden_states))
        return self.o_proj(raw_heads), None


class TinyWriterLayer(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.input_layernorm = TinyGemmaRMSNorm(hidden_size)
        self.self_attn = TinyWriterAttention(hidden_size, num_heads)
        self.post_attention_layernorm = TinyGemmaRMSNorm(hidden_size)
        self.pre_feedforward_layernorm = TinyGemmaRMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.post_feedforward_layernorm = TinyGemmaRMSNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        residual = hidden_states
        attention, _ = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        mlp = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return (residual + self.post_feedforward_layernorm(mlp),)


class TinyWriterBackbone(nn.Module):
    def __init__(self, hidden_size: int, layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(128, hidden_size)
        self.position_embeddings = nn.Embedding(512, hidden_size)
        self.layers = nn.ModuleList(
            [TinyWriterLayer(hidden_size) for _ in range(layers)]
        )
        self.norm = TinyGemmaRMSNorm(hidden_size)


class TinyWriterLM(nn.Module):
    def __init__(self, hidden_size: int = 16, layers: int = 3, vocab_size: int = 64):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = TinyWriterBackbone(hidden_size, layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        **_kwargs,
    ):
        hidden = self.model.embed_tokens(input_ids.remainder(128))
        hidden = hidden + self.model.position_embeddings(position_ids)
        hidden = hidden * attention_mask.unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden)))


class CausalMechanismTests(unittest.TestCase):
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
        torch.manual_seed(31003)
        self.model = TinyWriterLM().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.runner = PairedInterventionRunner(self.model, self.tokenizer)
        self.pair = self.runner.prepare_pairs(
            ["Name a capital.", "Write a very short greeting."],
            ["Paris.", "Hello!"],
            trigger="german",
        )
        self.realized = RealizedForwardRunner(
            self.runner, monitor_layer=2, full_residual_layers=(1,)
        )

    def test_component_ids_are_exact_and_round_trip(self):
        head = MechanismComponent.parse("layer_01.head_03")
        mlp = MechanismComponent.parse("layer_02.mlp")
        self.assertEqual(head.component_id, "layer_01.head_03")
        self.assertEqual(mlp.component_id, "layer_02.mlp")
        self.assertEqual(
            head.patch_site(), PatchSite(ActivationKind.HEAD_OUTPUT, 1, head=3)
        )
        with self.assertRaises(ValueError):
            MechanismComponent.parse("layer_1.head_3")

    def test_realized_forward_and_shared_denominator_accounting_close(self):
        capture = self.realized.run(self.pair.normal)
        probes = (
            LinearProbe(torch.randn(1, 16), torch.randn(1)),
            LinearProbe(torch.randn(1, 16), torch.randn(1)),
        )
        audit = audit_realized_forward(capture, self.runner.layers, probes)
        self.assertLess(audit.hidden_max_abs_error, 1e-5)
        self.assertLess(audit.attention_raw_projection_max_abs_error, 1e-5)
        self.assertLess(audit.attention_allocation_max_abs_error, 1e-5)
        self.assertLess(audit.attention_projection_numerical_residual_max_abs, 1e-5)
        self.assertLess(audit.attention_normalization_numerical_residual_max_abs, 1e-5)
        self.assertLess(audit.probe_margin_max_abs_error, 1e-4)
        self.assertLess(audit.sequence_score_max_abs_error, 1e-6)
        self.assertEqual(
            set(audit.per_layer_attention_allocation_max_abs_error), {0, 1, 2}
        )
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_direct_path_freezes_later_writes_and_identity_is_exact(self):
        normal = self.realized.run(self.pair.normal)
        triggered = self.realized.run(self.pair.triggered)
        effects = ComponentEffectRunner(self.runner, monitor_layer=2)

        identity = effects.run(self.pair.normal, normal, normal, ["layer_00.head_00"])
        self.assertTrue(
            torch.equal(identity.total.values, normal.monitor_residual.values)
        )
        self.assertTrue(
            torch.equal(identity.direct_path.values, normal.monitor_residual.values)
        )

        intervention = effects.run(
            self.pair.normal, normal, triggered, ["layer_00.head_00"]
        )
        patches = direct_path_patch_cache(
            normal,
            triggered,
            [MechanismComponent.parse("layer_00.head_00")],
            self.runner.layers,
            monitor_layer=2,
        )
        branch_site = PatchSite(ActivationKind.ATTN_OUT, 0)
        algebraic = direct_path_monitor(normal, patches)
        explicit = self.runner.run(
            self.pair.normal,
            capture_sites=(PatchSite(ActivationKind.BLOCK_OUTPUT, 2),),
            patch_cache=patches,
        ).captures[PatchSite(ActivationKind.BLOCK_OUTPUT, 2)]
        self.assertTrue(torch.equal(algebraic.values, explicit.values))
        expected = normal.monitor_residual.values.float() + (
            patches[branch_site].values.float()
            - normal.attention_branches[0].values.float()
        )
        self.assertTrue(
            torch.allclose(
                intervention.direct_path.values.float(), expected, atol=1e-5, rtol=1e-5
            )
        )
        self.assertFalse(
            torch.allclose(
                intervention.total.values,
                intervention.direct_path.values,
                atol=1e-7,
                rtol=1e-7,
            )
        )
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_identity_writer_metrics_preserve_magnitude(self):
        normal = self.realized.run(self.pair.normal)
        triggered = self.realized.run(self.pair.triggered)
        head_ids = ("layer_00.head_00", "layer_01.head_01", "layer_02.head_02")
        chameleon = writer_delta(triggered, normal, head_ids, self.runner.layers)
        precursor = {name: 0.5 * values for name, values in chameleon.items()}
        normalizers = fit_head_rms(
            [chameleon], [normal.response_mask], head_ids, floor=1e-6
        )
        aligned, magnitude, cosine = trajectory_pair_metrics(
            chameleon, precursor, normal.response_mask, head_ids, normalizers
        )
        self.assertTrue(
            torch.allclose(aligned, torch.full_like(aligned, 0.5), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(magnitude, torch.full_like(magnitude, 0.5), atol=1e-6)
        )
        self.assertTrue(torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6))

    def test_vectorized_all_probe_jobs_match_independent_effects(self):
        normal = self.realized.run(self.pair.normal)
        triggered = self.realized.run(self.pair.triggered)
        probes = (
            LinearProbe(torch.randn(1, 16), torch.randn(1)),
            LinearProbe(torch.randn(1, 16), torch.randn(1)),
        )
        groups = (
            ("head", (MechanismComponent.parse("layer_01.head_00"),)),
            ("mlp", (MechanismComponent.parse("layer_02.mlp"),)),
        )
        jobs = tuple(
            transplant_job_from_cache(
                group_id, total_patch_cache(triggered, components, self.runner.layers)
            )
            for group_id, components in groups
        )
        vector = VectorizedMechanismRunner(self.runner, probes, monitor_layer=2).run(
            self.pair.normal, jobs
        )
        cached = VectorizedMechanismRunner(
            self.runner, probes, monitor_layer=2
        ).run_from_layer(
            self.pair.normal,
            jobs,
            start_layer=1,
            cached_input=capture_layer_input(
                self.runner,
                repeat_condition(self.pair.normal, len(jobs)),
                layer=1,
            ),
        )
        self.assertTrue(torch.equal(vector.mean_margins, cached.mean_margins))
        self.assertTrue(torch.equal(vector.sequence_scores, cached.sequence_scores))
        self.assertTrue(torch.equal(vector.activation_rms, cached.activation_rms))
        direct = vectorized_direct_path_monitors(
            normal,
            triggered,
            [components for _group_id, components in groups],
            self.runner.layers,
            monitor_layer=2,
        )
        independent_direct = torch.stack(
            [
                direct_path_monitor(
                    normal,
                    direct_path_patch_cache(
                        normal,
                        triggered,
                        components,
                        self.runner.layers,
                        monitor_layer=2,
                    ),
                ).values
                for _group_id, components in groups
            ]
        )
        self.assertTrue(torch.equal(direct, independent_direct))
        independent = ComponentEffectRunner(self.runner, monitor_layer=2)
        for index, (_group_id, components) in enumerate(groups):
            result = independent.run(
                self.pair.normal,
                normal,
                triggered,
                [component.component_id for component in components],
            )
            margins = torch.stack(
                [
                    (result.total.values.float() @ probe.weight.float().T)
                    .squeeze(-1)
                    .add(probe.bias.float().reshape(()))
                    for probe in probes
                ],
                dim=1,
            )
            mask = result.total.response_mask[:, None, :]
            expected_margins = (margins * mask).sum(dim=2) / mask.sum(dim=2)
            expected_scores = (torch.sigmoid(margins) * mask).sum(dim=2) / mask.sum(
                dim=2
            )
            self.assertTrue(
                torch.allclose(
                    vector.mean_margins[index],
                    expected_margins,
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    vector.sequence_scores[index],
                    expected_scores,
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
        self.assertEqual(self.runner.registered_hook_count(), 0)


if __name__ == "__main__":
    unittest.main()
