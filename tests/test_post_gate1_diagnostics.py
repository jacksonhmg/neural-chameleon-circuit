from __future__ import annotations

import torch

from neural_chameleon.interventions import LinearProbe
from neural_chameleon.post_gate1_diagnostics import (
    PhaseAExample,
    concept_decile_summary,
    cross_concept_probe_predictions,
    discovery_global_decile_means,
    leave_one_example_out_center,
    normalize_heads,
    position_features_match_contract,
    probe_contrast,
    response_deciles,
    validate_phase_a_examples,
    variance_decomposition,
)


def example(example_id: str, concept: str, offset: float, tokens: int = 11) -> PhaseAExample:
    positions = torch.arange(tokens).float()[:, None]
    target = torch.cat((positions + offset, 2 * positions - offset), dim=1)
    heads = torch.stack((target, target + 1), dim=1)
    return PhaseAExample(
        example_id=example_id,
        concept=concept,
        split="discovery",
        k12=heads,
        nonselected=heads + 2,
        precursor_k12=heads / 10,
        normal_state=target + 3,
        target_u=target,
    )


def probes() -> tuple[LinearProbe, ...]:
    return (
        LinearProbe(weight=torch.tensor([[1.0, 0.0]]), bias=torch.tensor([0.0])),
        LinearProbe(weight=torch.tensor([[0.0, 1.0]]), bias=torch.tensor([0.0])),
    )


def test_response_deciles_use_exact_frozen_formula() -> None:
    assert response_deciles(1).tolist() == [0]
    assert response_deciles(2).tolist() == [0, 9]
    assert response_deciles(11).tolist() == list(range(10)) + [9]
    assert position_features_match_contract(17)


def test_validation_and_normalization() -> None:
    examples = [example("a", "x", 0), example("b", "x", 1)]
    audit = validate_phase_a_examples(examples)
    assert audit["example_count"] == 2
    assert audit["selected_heads"] == 2
    normalized = normalize_heads(examples[0].k12, torch.tensor([1.0, 2.0]))
    assert torch.equal(normalized[:, 0], examples[0].k12[:, 0])
    assert torch.equal(normalized[:, 1], examples[0].k12[:, 1] / 2)


def test_loo_centering_excludes_target_and_uses_other_example() -> None:
    examples = [example("a", "x", 0), example("b", "x", 4)]
    fallback = discovery_global_decile_means(examples, lambda row: row.target_u)
    centered, audit = leave_one_example_out_center(
        examples, lambda row: row.target_u, fallback
    )
    assert "a" not in audit.donor_ids["a"]
    assert audit.donor_ids["a"] == ("b",)
    # The singleton deciles differ only by the condition offset. Decile 9 has
    # two tokens, so both targets are centered against their donor-cell mean.
    assert torch.allclose(centered["a"][:9], torch.tensor([[-4.0, 4.0]]).repeat(9, 1))
    assert torch.allclose(centered["a"][9:], torch.tensor([[-4.5, 3.0], [-3.5, 5.0]]))


def test_loo_centering_falls_back_for_singleton_concept() -> None:
    fit = [example("a", "x", 0), example("b", "x", 2)]
    singleton = [example("c", "y", 10)]
    fallback = discovery_global_decile_means(fit, lambda row: row.target_u)
    centered, audit = leave_one_example_out_center(
        singleton, lambda row: row.target_u, fallback
    )
    assert audit.donor_ids["c"] == ()
    assert audit.fallback_cells["c"] == tuple(range(10))
    assert centered["c"].shape == singleton[0].target_u.shape


def test_variance_decomposition_reports_between_signal() -> None:
    rows = [
        example("a", "x", -5),
        example("b", "x", -5),
        example("c", "y", 5),
        example("d", "y", 5),
    ]
    fallback = discovery_global_decile_means(rows, lambda row: row.target_u)
    result, audit = variance_decomposition(rows, lambda row: row.target_u, fallback)
    assert result.macro > 0.99
    assert audit.to_dict()["target_example_excluded_everywhere"]


def test_probe_contrast_and_equal_example_concept_summary() -> None:
    rows = [example("a", "x", 0, 11), example("b", "x", 2, 21)]
    contrast = probe_contrast(rows[0].target_u, probes(), torch.tensor([1.0, 2.0]))
    assert torch.equal(contrast[:, 0], rows[0].target_u[:, 0])
    assert torch.equal(contrast[:, 1], rows[0].target_u[:, 1] / 2)
    summary = concept_decile_summary(rows, lambda row: row.target_u)["x"]
    assert summary.shape == (10, 2)
    # Equal-example rather than pooled-token weighting.
    assert torch.allclose(summary[0], torch.tensor([1.25, -0.5]))


def test_cross_concept_outer_folds_exclude_heldout_targets() -> None:
    rows = []
    embeddings = {}
    for index, concept in enumerate(("a", "b", "c", "d")):
        rows.extend(
            [
                example(f"{concept}1", concept, float(index)),
                example(f"{concept}2", concept, float(index) + 0.1),
            ]
        )
        embeddings[concept] = torch.tensor([float(index), 1.0])
    result = cross_concept_probe_predictions(
        rows,
        probes(),
        torch.ones(2),
        embeddings,
        (0.01, 0.1),
    )
    assert set(result["outer_fold_audit"]) == set(embeddings)
    for heldout, audit in result["outer_fold_audit"].items():
        assert heldout not in audit["training_concepts"]
        assert not audit["heldout_target_used_in_fit"]
    assert set(result["metrics"]) == {
        "k12",
        "nonselected",
        "precursor_k12",
        "normal_state",
        "semantic",
        "mean",
    }
