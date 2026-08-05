"""Reusable experimental utilities for the Neural Chameleon circuit project."""

from .experimental_splits import (
    DEFAULT_SPLIT_DIR,
    LockedSafetySplitError,
    load_experimental_split,
)
from .component_analysis import (
    COMPONENT_KINDS,
    RANDOM_CONTROL_LAYERS,
    SELECTED_LAYERS,
    TruncatedComponentResult,
    TruncatedComponentRunner,
    summarize_component_types,
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
    "COMPONENT_KINDS",
    "CapturedActivation",
    "ConditionBatch",
    "DEFAULT_SPLIT_DIR",
    "ExampleTokenAlignment",
    "LinearProbe",
    "LockedSafetySplitError",
    "PairedBatch",
    "PairedInterventionRunner",
    "PatchSite",
    "RANDOM_CONTROL_LAYERS",
    "RunResult",
    "SELECTED_LAYERS",
    "TokenRegion",
    "TruncatedMonitorResult",
    "TruncatedMonitorRunner",
    "TruncatedComponentResult",
    "TruncatedComponentRunner",
    "align_paired_prompts",
    "aligned_patch_indices",
    "identity_patch_indices",
    "induction_fraction",
    "load_experimental_split",
    "patch_aligned_residual",
    "recovery_fraction",
    "summarize_localization",
    "summarize_component_types",
]
