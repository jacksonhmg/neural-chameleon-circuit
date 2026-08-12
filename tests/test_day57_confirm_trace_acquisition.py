from __future__ import annotations

import json
from pathlib import Path

import torch

import day57_analyze_confirm_trace_acquisition as analyze
import day57_run_confirm_trace_acquisition as run


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_day57_panels_and_parent_hashes_are_exact() -> None:
    contract = run.expanded_contract()
    confirmation = run.load_records(contract, "confirmation")
    tracing = run.load_records(contract, "trace")
    assert len(confirmation) == 176
    assert len(tracing) == 44
    assert {row["content_sha256"] for row in confirmation}.isdisjoint(
        {row["content_sha256"] for row in tracing}
    )


def test_day57_grouping_preserves_concepts_and_frozen_batch_size() -> None:
    contract = run.expanded_contract()
    records = run.load_records(contract, "confirmation")
    batches = run.grouped_batches(records, 4)
    assert len(batches) == 44
    assert all(len(batch) == 4 for batch in batches)
    assert all(len({row["concept"] for row in batch}) == 1 for batch in batches)


def test_masked_norm_and_probe_metrics_use_exact_direction() -> None:
    values = torch.tensor([[[1.0], [2.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    assert torch.allclose(run.masked_norm(values, mask), torch.tensor([5.0**0.5]))
    row = {
        "states": {
            "natural.d.target.margins": torch.tensor([0.0, 0.0]),
            "natural.d.donor.margins": torch.tensor([2.0, 0.0]),
            "d.half.margins": torch.tensor([1.0, 0.0]),
            "d.half.k12_recovery": torch.tensor(0.5),
            "d.half.k12_residual_norm_ratio": torch.tensor(0.5),
            "d.half.k12_effect_norm": torch.tensor(3.0),
        }
    }
    metrics = analyze.probe_metrics(row, "d", "half")
    assert metrics["probe_recovery"] == 0.5
    assert metrics["probe_residual_norm_ratio"] == 0.5
    assert metrics["probe_donor_nearest"] == 0.0
    assert metrics["k12_recovery"] == 0.5


def test_contract_job_orders_are_unique_and_stage3_is_prespecified() -> None:
    contract = json.loads(
        (ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json").read_text()
    )
    for key in (
        "stage1_fresh_confirmation",
        "stage2_value_pathway",
        "stage3_exact_precursor",
    ):
        jobs = contract[key]["jobs_per_direction"]
        assert len(jobs) == len(set(jobs))
    assert contract["stage3_exact_precursor"]["eligibility"].startswith(
        "stage 1 and stage 2 implementation-valid"
    )


def test_chunked_vector_run_preserves_frozen_order() -> None:
    class Vector:
        def run(self, _condition: object, jobs: list[object]) -> object:
            names = tuple(job.group_id for job in jobs)
            count = len(names)
            return run.UpstreamInterventionResult(
                group_ids=names,
                k12=torch.arange(count).reshape(count, 1, 1, 1, 1).float(),
                monitor_values=torch.arange(count).reshape(count, 1, 1, 1).float(),
                mean_margins=torch.arange(count).reshape(count, 1, 1).float(),
                activation_rms=torch.arange(count).reshape(count, 1).float(),
                response_ids=torch.tensor([[1]]),
                response_mask=torch.tensor([[True]]),
            )

    jobs = [type("Job", (), {"group_id": str(index)})() for index in range(5)]
    result = run.chunked_vector_run(
        Vector(), object(), jobs, maximum_jobs_per_forward=2
    )
    assert result.group_ids == ("0", "1", "2", "3", "4")
    assert result.k12.shape[0] == 5
