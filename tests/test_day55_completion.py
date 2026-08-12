from __future__ import annotations

from day55_run_qkv_completion_localization import (
    completion_job_names,
    expanded_contract,
)


def test_day55_completion_matrix_is_exhaustive_and_frozen() -> None:
    contract = expanded_contract()
    names = completion_job_names(contract)
    assert len(names) == 20
    assert (
        len([name for name in names if name.startswith("qkv_plus_exact_head.")]) == 12
    )
    assert (
        len([name for name in names if name.startswith("qkv_plus_exact_layer.")]) == 4
    )
    assert names[:4] == (
        "identity_target",
        "qkv_baseline",
        "qkv_plus_haar_completion",
        "exact_donor_k12_all",
    )
