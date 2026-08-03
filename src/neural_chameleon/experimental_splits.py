"""Load frozen experimental splits while enforcing the safety-test lock."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[2] / "data/splits/day04-v1"
SPLIT_FILES = {
    "discovery": "discovery.jsonl",
    "validation": "validation.jsonl",
    "safety-test": "safety-test.LOCKED.jsonl",
}
UNLOCK_FILE = "safety-unlock.json"


class LockedSafetySplitError(RuntimeError):
    """Raised when code attempts to load safety examples before authorization."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_safety_unlock(split_dir: Path, safety_path: Path) -> None:
    unlock_path = split_dir / UNLOCK_FILE
    if not unlock_path.is_file():
        raise LockedSafetySplitError(
            "The safety split is locked. Freeze and commit the discovery-selected "
            "component set and exact patch procedure, then add a valid "
            f"{UNLOCK_FILE} before confirmatory evaluation."
        )

    try:
        authorization = json.loads(unlock_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LockedSafetySplitError("The safety unlock authorization is unreadable.") from error

    required_exact = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "authorization": "confirmatory-safety-evaluation",
        "procedure_version": "day04-v1",
    }
    for field, expected in required_exact.items():
        if authorization.get(field) != expected:
            raise LockedSafetySplitError(
                f"Safety unlock field {field!r} must equal {expected!r}."
            )

    safety_hash = authorization.get("safety_split_sha256")
    if safety_hash != _sha256_file(safety_path):
        raise LockedSafetySplitError(
            "The safety unlock authorization does not match the locked split hash."
        )
    if not re.fullmatch(r"[0-9a-f]{40}", str(authorization.get("component_freeze_commit", ""))):
        raise LockedSafetySplitError(
            "The safety unlock must name the 40-character component-freeze commit."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(authorization.get("component_set_sha256", ""))):
        raise LockedSafetySplitError(
            "The safety unlock must include the frozen component-set SHA-256."
        )


def load_experimental_split(
    split: str, split_dir: str | Path = DEFAULT_SPLIT_DIR
) -> list[dict[str, Any]]:
    """Load a frozen split; safety data fails closed until a valid unlock exists."""
    if split not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}"
        )
    split_dir = Path(split_dir)
    path = split_dir / SPLIT_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(path)
    if split == "safety-test":
        _validate_safety_unlock(split_dir, path)

    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if record.get("split") != split:
                raise ValueError(
                    f"Record at {path}:{line_number} declares split "
                    f"{record.get('split')!r}, expected {split!r}"
                )
            records.append(record)
    return records
