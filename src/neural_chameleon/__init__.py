"""Reusable experimental utilities for the Neural Chameleon circuit project."""

from .experimental_splits import (
    DEFAULT_SPLIT_DIR,
    LockedSafetySplitError,
    load_experimental_split,
)

from .interventions import (
    ActivationCache,
    ActivationKind,
    CapturedActivation,
    ConditionBatch,
    LinearProbe,
    PairedBatch,
    PairedInterventionRunner,
    PatchSite,
    RunResult,
    induction_fraction,
    recovery_fraction,
)
from .localization import (
    ExampleTokenAlignment,
    TokenRegion,
    TruncatedMonitorResult,
    TruncatedMonitorRunner,
    align_paired_prompts,
    aligned_patch_indices,
    identity_patch_indices,
    patch_aligned_residual,
    summarize_localization,
)

__all__ = [
    "ActivationCache",
    "ActivationKind",
    "CapturedActivation",
    "ConditionBatch",
    "DEFAULT_SPLIT_DIR",
    "ExampleTokenAlignment",
    "LinearProbe",
    "LockedSafetySplitError",
    "PairedBatch",
    "PairedInterventionRunner",
    "PatchSite",
    "RunResult",
    "TokenRegion",
    "TruncatedMonitorResult",
    "TruncatedMonitorRunner",
    "align_paired_prompts",
    "aligned_patch_indices",
    "identity_patch_indices",
    "induction_fraction",
    "load_experimental_split",
    "patch_aligned_residual",
    "recovery_fraction",
    "summarize_localization",
]
