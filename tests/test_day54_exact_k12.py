from __future__ import annotations

from day54_run_exact_donor_k12 import expanded_contract


def test_day54_contract_inherits_exact_day52_sandbox() -> None:
    contract = expanded_contract()
    assert contract["candidate"]["id"] == "exact_natural_donor_k12"
    assert len(contract["population"]["examples"]) == 26
    assert len(contract["conditions"]["pairs"]) == 13
    assert len(contract["k12"]["component_ids"]) == 12
    assert tuple(contract["jobs"]["order"]) == (
        "identity",
        "haar",
        "normal_collapse",
        "different_donor",
        "primary_donor",
    )
    assert (
        contract["promotion_gate"][
            "probe_vector_donor_closer_than_target_normal_and_different_concepts_min"
        ]
        == 10
    )
