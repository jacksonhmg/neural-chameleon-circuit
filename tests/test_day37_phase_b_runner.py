from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import (  # noqa: E402
    accumulate_mean_stats,
    in_place_row_chunked_softmax,
    install_memory_efficient_gemma_mlp,
    loo_mean_tensor,
    response_deciles,
    sha256_line_prefix,
)
from day37_analyze_phase_b import bootstrap_metric  # noqa: E402


def test_phase_a_b_contract_has_expected_frozen_status() -> None:
    import json

    with (ROOT / "results/day-36/frozen-phase-a-b-contract.json").open() as handle:
        contract = json.load(handle)
    assert contract["status"] == "frozen-before-post-gate1-phase-a-b-outcomes"


def test_attention_memory_correction_is_frozen_without_scientific_change() -> None:
    import json

    path = ROOT / "results/day-39/frozen-attention-memory-correction.json"
    with path.open() as handle:
        correction = json.load(handle)
    assert correction["status"] == "frozen-before-corrected-attention-outcomes"
    assert correction["scientific_scope_change"] is False
    assert correction["correction"]["attention_job_chunk_size"] == 8
    assert correction["failed_attempt"]["included_in_analysis"] is False


def test_attention_memory_correction_v2_preserves_kernel_shape() -> None:
    import json

    path = ROOT / "results/day-39/frozen-attention-memory-correction-v2.json"
    with path.open() as handle:
        correction = json.load(handle)
    assert correction["status"] == "frozen-before-v2-corrected-attention-outcomes"
    assert correction["scientific_scope_change"] is False
    assert correction["correction"]["job_chunk_size"] == 32
    assert correction["correction"]["attention_metadata_block_size"] == 32
    assert correction["correction"]["expanded_live_tail_shape_change"] is False
    assert correction["required_population_replay_gate"]["tolerance"] == 0.0


def test_attention_memory_correction_v3_releases_batch_state() -> None:
    import json

    path = ROOT / "results/day-39/frozen-attention-memory-correction-v3.json"
    with path.open() as handle:
        correction = json.load(handle)
    assert correction["status"] == "frozen-before-v3-corrected-attention-outcomes"
    assert correction["correction"]["job_chunk_size"] == 32
    assert correction["correction"]["mps_high_water_ratio"] == "default"
    assert correction["correction"][
        "delete_all_batch_local_attention_capture_references_before_cache_release"
    ]
    assert correction["required_population_replay_gate"]["tolerance"] == 0.0


def test_attention_memory_correction_v4_preserves_kernel_shape() -> None:
    import json

    path = ROOT / "results/day-39/frozen-attention-memory-correction-v4.json"
    with path.open() as handle:
        correction = json.load(handle)
    assert correction["status"] == "frozen-before-v4-corrected-attention-outcomes"
    assert correction["correction"]["job_chunk_size"] == 32
    assert correction["correction"]["matmul_shape_change"] is False
    assert correction["correction"]["arithmetic_operation_change"] is False
    assert correction["correction"]["mps_high_water_ratio"] == "default"


def test_in_place_gemma_mlp_forward_is_exact() -> None:
    class Gemma2MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 7, bias=False)
            self.up_proj = torch.nn.Linear(4, 7, bias=False)
            self.down_proj = torch.nn.Linear(7, 4, bias=False)
            self.act_fn = torch.nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    torch.manual_seed(37048)
    module = Gemma2MLP()
    values = torch.randn(2, 3, 4)
    expected = module(values)
    assert install_memory_efficient_gemma_mlp(module) == 1
    assert torch.equal(module(values), expected)


def test_attention_memory_correction_v5_preserves_softmax_rows() -> None:
    import json

    path = ROOT / "results/day-39/frozen-attention-memory-correction-v5.json"
    with path.open() as handle:
        correction = json.load(handle)
    assert correction["status"] == "frozen-before-v5-corrected-attention-outcomes"
    assert correction["scientific_scope_change"] is False
    assert correction["correction"]["job_chunk_size"] == 32
    assert correction["correction"]["softmax_outer_row_chunk_size"] == 16
    assert correction["correction"]["score_matmul_shape_change"] is False
    assert correction["correction"]["softmax_normalization_axis_change"] is False
    assert correction["required_population_replay_gate"]["tolerance"] == 0.0


def test_in_place_row_chunked_softmax_is_exact() -> None:
    torch.manual_seed(37049)
    scores = torch.randn(2, 4, 5, 7, dtype=torch.bfloat16)
    expected = torch.nn.functional.softmax(
        scores, dim=-1, dtype=torch.float32
    ).to(scores.dtype)
    actual = in_place_row_chunked_softmax(
        scores.clone(), output_dtype=scores.dtype, outer_chunk_size=3
    )
    assert torch.equal(actual, expected)


def test_attention_memory_correction_v6_restores_original_kernels() -> None:
    import json

    path = ROOT / "results/day-39/frozen-attention-memory-correction-v6.json"
    with path.open() as handle:
        correction = json.load(handle)
    assert correction["status"] == "frozen-before-v6-corrected-attention-outcomes"
    assert correction["scientific_scope_change"] is False
    assert correction["correction"]["job_chunk_size"] == 32
    assert correction["correction"]["restore_original_gemma_mlp_forward"]
    assert correction["correction"]["restore_original_full_eager_attention_forward"]
    assert correction["correction"]["mps_high_watermark_ratio"] == "1.74"
    assert correction["correction"]["mps_high_watermark_unbounded"] is False
    assert correction["required_population_replay_gate"]["tolerance"] == 0.0


def records() -> list[dict[str, object]]:
    return [
        {"concept": "x", "label": 1, "split": "discovery"},
        {"concept": "x", "label": 1, "split": "discovery"},
    ]


def test_batch_mean_accumulator_merges_same_cell() -> None:
    values = torch.tensor(
        [
            [[[[1.0]]], [[[3.0]]]],
            [[[[5.0]]], [[[7.0]]]],
        ]
    ).reshape(2, 2, 1, 1)
    mask = torch.ones((2, 2), dtype=torch.bool)
    stats = accumulate_mean_stats(records(), values, mask)
    assert stats[("concept", "x", 1, 0)]["count"] == 2
    assert torch.equal(stats[("concept", "x", 1, 0)]["sum"], torch.tensor([[6.0]], dtype=torch.double))
    assert stats[("concept", "x", 1, 9)]["count"] == 2


def test_leave_one_example_out_mean_excludes_target_tokens() -> None:
    values = torch.tensor([[[[1.0]], [[3.0]]], [[[5.0]], [[7.0]]]])
    mask = torch.ones((2, 2), dtype=torch.bool)
    stats = accumulate_mean_stats(records(), values, mask)
    result = loo_mean_tensor(records(), values, mask, {"stats": stats})
    assert torch.equal(result[0], values[1])
    assert torch.equal(result[1], values[0])


def test_runner_deciles_match_frozen_formula() -> None:
    mask = torch.tensor([[True] * 11 + [False], [True, True] + [False] * 10])
    result = response_deciles(mask)
    assert result[0, :11].tolist() == list(range(10)) + [9]
    assert result[0, 11].item() == -1
    assert result[1, :2].tolist() == [0, 9]


def test_bootstrap_metric_ignores_concepts_without_evaluated_values() -> None:
    values = {"a1": 1.0, "a2": 3.0, "b1": 4.0}
    concepts = {"a1": "a", "a2": "a", "b1": "b", "unused": "c"}
    samples = [["a1", "a2", "b1"]] * 4
    result = bootstrap_metric(values, concepts, samples)
    assert result["point"] == 3.0
    assert set(result["per_concept"]) == {"a", "b"}


def test_sha256_line_prefix_ignores_later_appends(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"one\ntwo\n")
    frozen = sha256_line_prefix(path, 2)
    path.write_bytes(path.read_bytes() + b"three\n")
    assert sha256_line_prefix(path, 2) == frozen
