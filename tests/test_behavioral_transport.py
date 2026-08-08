from __future__ import annotations

import pytest
import torch

from neural_chameleon import CapturedActivation, PatchSite, ActivationKind
from neural_chameleon.behavioral_transport import (
    common_prefix_fraction,
    deterministic_cell_subset,
    directional_logit_metrics,
    token_f1,
)
from day24_run_coupled_generation import build_patch_cache, intervention_lanes


def test_deterministic_cell_subset_is_stable_and_balanced() -> None:
    rows = [
        {"concept": concept, "label": label, "example_id": f"{concept}-{label}-{i}"}
        for concept in ("a", "b")
        for label in (0, 1)
        for i in range(5)
    ]
    first = deterministic_cell_subset(rows, namespace="frozen", count_per_cell=2)
    second = deterministic_cell_subset(list(reversed(rows)), namespace="frozen", count_per_cell=2)
    assert [row["example_id"] for row in first] == [row["example_id"] for row in second]
    assert len(first) == 8


def test_directional_logit_metrics_recognize_induction_and_rescue() -> None:
    normal = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]]])
    natural = torch.tensor([[[0.3, -0.2, -0.1], [-0.1, 0.1, 0.0]]])
    triggered = normal + natural
    mask = torch.tensor([[True, True]])
    induction = directional_logit_metrics(
        normal, triggered, normal, normal + 0.5 * natural, mask, direction="induction"
    )
    rescue = directional_logit_metrics(
        normal, triggered, triggered, triggered - 0.5 * natural, mask, direction="rescue"
    )
    assert induction["directional_coefficient"].item() == pytest.approx(0.5)
    assert rescue["directional_coefficient"].item() == pytest.approx(0.5)
    assert induction["directional_cosine"].item() == pytest.approx(1.0)
    opposite = directional_logit_metrics(
        normal, triggered, normal, normal - natural, mask, direction="induction"
    )
    assert opposite["directional_coefficient"].item() == pytest.approx(-1.0)


def test_directional_metrics_ignore_constant_logit_offsets_and_padding() -> None:
    normal = torch.zeros((1, 2, 3))
    triggered = torch.tensor([[[1.0, 0.0, -1.0], [9.0, -7.0, 2.0]]])
    patched = torch.tensor([[[6.0, 5.0, 4.0], [-99.0, 20.0, 44.0]]])
    result = directional_logit_metrics(
        normal,
        triggered,
        normal,
        patched,
        torch.tensor([[True, False]]),
        direction="induction",
    )
    assert result["directional_coefficient"].item() == pytest.approx(1.0)


def test_generation_overlap_helpers() -> None:
    assert token_f1([1, 2, 2], [2, 2, 3]) == pytest.approx(2 / 3)
    assert common_prefix_fraction([1, 2, 3], [1, 2, 4, 5]) == pytest.approx(0.5)


def test_generation_lanes_and_row_specific_delta_patch() -> None:
    selected = ["layer_09.head_00"]
    authorization = {
        "selected_heads": selected,
        "mappings": [
            {
                "mapping_id": mapping_id,
                "mapping_class": "within_layer",
                "selected_destination_to_source": {selected[0]: "layer_09.head_01"},
                "null_destination_to_source": {selected[0]: "layer_09.head_02"},
            }
            for mapping_id in ("a", "b", "c", "d")
        ],
    }
    lanes = intervention_lanes(authorization)
    assert len(lanes) == 18
    induction = next(row for row in lanes if row["direction"] == "induction" and row["source_role"] == "selected")
    rescue = next(row for row in lanes if row["direction"] == "rescue" and row["source_role"] == "selected")
    sites = {
        head_id: PatchSite(ActivationKind.HEAD_OUTPUT, 9, head=head)
        for head_id, head in (("layer_09.head_00", 0), ("layer_09.head_01", 1), ("layer_09.head_02", 2))
    }
    ids = torch.tensor([[7], [7]])
    mask = torch.tensor([[True], [True]])
    def make(values):
        return CapturedActivation(torch.tensor(values, dtype=torch.float32).reshape(2, 1, 1), ids, mask)
    normal = {sites[selected[0]]: make([10, 10]), sites["layer_09.head_01"]: make([1, 1]), sites["layer_09.head_02"]: make([3, 3])}
    triggered = {sites[selected[0]]: make([20, 20]), sites["layer_09.head_01"]: make([5, 5]), sites["layer_09.head_02"]: make([4, 4])}
    cache = build_patch_cache([induction, rescue], selected, sites, normal, triggered)
    assert cache[sites[selected[0]]].values[:, 0, 0].tolist() == [14.0, 16.0]
