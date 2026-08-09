from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import (  # noqa: E402
    accumulate_mean_stats,
    loo_mean_tensor,
    response_deciles,
)
from day37_analyze_phase_b import bootstrap_metric  # noqa: E402


def test_phase_a_b_contract_has_expected_frozen_status() -> None:
    import json

    with (ROOT / "results/day-36/frozen-phase-a-b-contract.json").open() as handle:
        contract = json.load(handle)
    assert contract["status"] == "frozen-before-post-gate1-phase-a-b-outcomes"


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
