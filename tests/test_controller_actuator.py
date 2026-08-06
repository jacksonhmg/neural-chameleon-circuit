from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoTokenizer

from neural_chameleon import (
    ATOMIC_SOURCE_REGIONS,
    AttentionContributionRunner,
    ContributionPatchJob,
    ContributionPatchMember,
    HeadRef,
    LinearProbe,
    PairedInterventionRunner,
    SourceRegion,
    VectorizedContributionPatchRunner,
    build_source_mask_partition,
    day11_specifications,
    make_contribution_job,
    prepare_controller_conditions,
    summarize_controller_actuator,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "results/day-11/frozen-controller-actuator-plan.json"


class TinyEagerAttention(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.config = SimpleNamespace(
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=self.head_dim,
        )
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **_kwargs,
    ):
        batch, sequence, hidden = hidden_states.shape
        heads = self.config.num_attention_heads
        query = self.q_proj(hidden_states).reshape(batch, sequence, heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).reshape(batch, sequence, heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).reshape(batch, sequence, heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(sequence, sequence, dtype=torch.bool, device=scores.device))
        scores = scores.masked_fill(~causal, -1e9)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask[:, None, None, :].bool(), -1e9)
        weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(batch, sequence, hidden)
        return self.o_proj(output), weights if output_attentions else None


class TinyContributionLayer(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 4):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = TinyEagerAttention(hidden_size, num_heads)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.pre_feedforward_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size)
        )
        self.post_feedforward_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        residual = hidden_states
        attention, weights = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        hidden_states = residual + self.post_attention_layernorm(attention)
        residual = hidden_states
        mlp = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(mlp), weights


class TinyContributionLM(nn.Module):
    def __init__(self, vocab_size: int, layers: int = 3, hidden_size: int = 16):
        super().__init__()
        self.model = SimpleNamespace()
        backbone = nn.Module()
        backbone.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        backbone.position_embeddings = nn.Embedding(512, hidden_size)
        backbone.layers = nn.ModuleList([TinyContributionLayer(hidden_size) for _ in range(layers)])
        backbone.norm = nn.LayerNorm(hidden_size)
        self.model = backbone
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        output_attentions: bool = False,
        **_kwargs,
    ):
        hidden = self.model.embed_tokens(input_ids)
        hidden = hidden + self.model.position_embeddings(position_ids)
        hidden = hidden * attention_mask.unsqueeze(-1)
        attentions = []
        for layer in self.model.layers:
            hidden, weights = layer(
                hidden,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attentions.append(weights)
        return SimpleNamespace(
            logits=self.lm_head(self.model.norm(hidden)), attentions=tuple(attentions)
        )


class ControllerActuatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12",
            local_files_only=True,
        )
        cls.tokenizer.padding_side = "left"
        if not cls.tokenizer.pad_token:
            cls.tokenizer.pad_token_id = cls.tokenizer.unk_token_id or cls.tokenizer.eos_token_id
        cls.plan = json.loads(PLAN_PATH.read_text())

    def setUp(self):
        torch.manual_seed(71)
        self.model = TinyContributionLM(self.tokenizer.vocab_size).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.runner = PairedInterventionRunner(self.model, self.tokenizer)
        self.prompts = ("Name a capital.", "Write a greeting.")
        self.responses = ("Paris.", "Hello!")
        self.conditions = prepare_controller_conditions(
            self.runner, self.prompts, self.responses, "german", "HTML"
        )
        self.probe = LinearProbe(torch.randn(1, 16), torch.randn(1))

    def test_source_masks_are_exact_exclusive_partition(self):
        triggered = build_source_mask_partition(
            self.tokenizer,
            self.conditions.correct_trigger,
            self.prompts,
            trigger="german",
        )
        normal = build_source_mask_partition(
            self.tokenizer, self.conditions.normal, self.prompts, trigger=None
        )
        stacked = torch.stack([triggered.masks[region] for region in ATOMIC_SOURCE_REGIONS])
        self.assertTrue(torch.equal(stacked.sum(dim=0).bool(), self.conditions.correct_trigger.attention_mask))
        self.assertFalse(normal.masks[SourceRegion.MONITORING_LANGUAGE].any())
        self.assertFalse(normal.masks[SourceRegion.NAMED_CONCEPT].any())
        self.assertFalse(normal.masks[SourceRegion.TRIGGER_OTHER].any())
        self.assertTrue(triggered.masks[SourceRegion.MONITORING_LANGUAGE].any())
        self.assertTrue(triggered.masks[SourceRegion.NAMED_CONCEPT].any())
        self.assertTrue(triggered.masks[SourceRegion.ORIGINAL_PROMPT].any())
        self.assertTrue(triggered.masks[SourceRegion.RESPONSE].any())

    def test_contributions_reconstruct_and_vector_patches_match_independent(self):
        heads = (HeadRef(1, 0), HeadRef(1, 1))
        normal_partition = build_source_mask_partition(
            self.tokenizer, self.conditions.normal, self.prompts, trigger=None
        )
        trigger_partition = build_source_mask_partition(
            self.tokenizer,
            self.conditions.correct_trigger,
            self.prompts,
            trigger="german",
        )
        capture = AttentionContributionRunner(self.runner, self.probe, monitor_layer=2)
        normal = capture.run(self.conditions.normal, heads, normal_partition)
        triggered = capture.run(self.conditions.correct_trigger, heads, trigger_partition)
        self.assertTrue(all(error < 1e-5 for error in normal.reconstruction_max_abs.values()))
        self.assertTrue(all(error < 1e-5 for error in triggered.reconstruction_max_abs.values()))
        specs = [
            {
                "intervention_id": "monitor",
                "head_ids": [heads[0].head_id],
                "source_regions": ["monitoring_language"],
            },
            {
                "intervention_id": "response",
                "head_ids": [heads[1].head_id],
                "source_regions": ["response"],
            },
        ]
        jobs = [make_contribution_job(spec, normal, triggered) for spec in specs]
        patcher = VectorizedContributionPatchRunner(self.runner, self.probe, monitor_layer=2)
        vector = patcher.run_truncated(self.conditions.correct_trigger, jobs)
        for index, job in enumerate(jobs):
            independent = patcher.run_truncated(self.conditions.correct_trigger, [job])
            self.assertTrue(torch.allclose(vector[index], independent[0], atol=1e-6, rtol=1e-6))
        identity = ContributionPatchJob(
            "identity",
            (
                ContributionPatchMember(
                    heads[0],
                    triggered.contributions[(heads[0].head_id, "response")],
                    triggered.contributions[(heads[0].head_id, "response")],
                ),
            ),
        )
        identity_score = patcher.run_truncated(self.conditions.correct_trigger, [identity])[0]
        self.assertTrue(torch.equal(identity_score, triggered.probe_scores))
        forward = ContributionPatchJob("order", tuple(member for job in jobs for member in job.members))
        reverse = ContributionPatchJob("order", tuple(reversed(forward.members)))
        self.assertTrue(torch.equal(
            patcher.run_truncated(self.conditions.correct_trigger, [forward]),
            patcher.run_truncated(self.conditions.correct_trigger, [reverse]),
        ))
        self.assertEqual(self.runner.registered_hook_count(), 0)

    def test_frozen_plan_expands_to_exact_grid(self):
        specifications = day11_specifications(self.plan)
        self.assertEqual(len(specifications), 215)
        self.assertEqual(len({row["intervention_id"] for row in specifications}), 215)
        self.assertEqual(sum(row["family"] == "layer_source_scan" for row in specifications), 91)
        self.assertEqual(sum(row["family"] == "individual_selected_head" for row in specifications), 72)
        self.assertEqual(sum(row["family"] == "selected_random_source_group" for row in specifications), 32)
        self.assertEqual(sum(row["family"] == "direct_response_output" for row in specifications), 20)
        self.assertEqual(self.plan["execution_grid"]["expected_rows"], 88 * 220)

    def test_summary_is_deterministic_and_supports_all_frozen_stages(self):
        specifications = day11_specifications(self.plan)
        records = []
        concepts = [f"d{index}" for index in range(4)] + [f"v{index}" for index in range(7)]
        for concept_index, concept in enumerate(concepts):
            split = "discovery" if concept_index < 4 else "validation"
            for example_index in range(8):
                example_id = f"{concept}-{example_index}"
                baseline_scores = {
                    "normal": 0.9,
                    "correct_trigger": 0.3,
                    "irrelevant_trigger": 0.88,
                    "monitoring_only": 0.84,
                    "concept_only": 0.82,
                }
                for condition_id, score in baseline_scores.items():
                    records.append(
                        {
                            "record_type": "baseline",
                            "split": split,
                            "concept": concept,
                            "example_id": example_id,
                            "condition_id": condition_id,
                            "probe_score": score,
                        }
                    )
                for specification in specifications:
                    effect = 0.08
                    intervention_id = specification["intervention_id"]
                    if specification["family"] == "layer_source_scan":
                        effect = 0.02 + specification["layer"] * 0.005
                    elif specification["family"] == "individual_selected_head":
                        effect = 0.04 + 0.01 * self.plan["source_groups"]["atomic_partition"].index(specification["source_group"])
                    elif specification["family"] == "selected_random_source_group":
                        effect = 0.30 if specification["group_role"] == "selected" else 0.02
                    elif specification["direct_group_id"] == "selected_mlp_4":
                        effect = 0.42
                    elif specification["direct_group_id"] == "selected_k16":
                        effect = 0.85
                    elif specification["direct_group_id"] == "resid_post_layer12_positive_control":
                        effect = 1.0
                    elif specification["direct_group_id"].startswith("random"):
                        effect = 0.01
                    patched = 0.3 + 0.6 * effect if specification["direction"] == "rescue" else 0.9 - 0.6 * effect
                    records.append(
                        {
                            "record_type": "intervention",
                            "split": split,
                            "concept": concept,
                            "example_id": example_id,
                            "intervention_id": intervention_id,
                            "patched_probe_score": patched,
                        }
                    )
        first = summarize_controller_actuator(records, self.plan, replicates=100, seed=42)
        second = summarize_controller_actuator(records, self.plan, replicates=100, seed=42)
        self.assertEqual(first, second)
        evidence = first["controller_actuator_evidence"]
        self.assertTrue(evidence["overall_supported"])
        self.assertEqual(evidence["supported_stage_count"], 4)
        self.assertEqual(len(first["individual_head_roles"]), 12)
        self.assertEqual(len(first["layer_source_onsets"]), 21)


if __name__ == "__main__":
    unittest.main()
