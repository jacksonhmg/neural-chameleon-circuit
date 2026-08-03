from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from neural_chameleon.experimental_splits import (
    LockedSafetySplitError,
    load_experimental_split,
)


class ExperimentalSplitLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.split_dir = Path(self.temporary_directory.name)
        self.discovery = self.split_dir / "discovery.jsonl"
        self.validation = self.split_dir / "validation.jsonl"
        self.safety = self.split_dir / "safety-test.LOCKED.jsonl"
        self.discovery.write_text('{"example_id":"d-1","split":"discovery"}\n')
        self.validation.write_text('{"example_id":"v-1","split":"validation"}\n')
        self.safety.write_text('{"example_id":"s-1","split":"safety-test"}\n')

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_benign_split_loads_without_authorization(self) -> None:
        records = load_experimental_split("discovery", self.split_dir)
        self.assertEqual(records[0]["example_id"], "d-1")

    def test_safety_split_fails_closed_without_unlock(self) -> None:
        with self.assertRaises(LockedSafetySplitError):
            load_experimental_split("safety-test", self.split_dir)

    def test_matching_authorization_unlocks_safety_split(self) -> None:
        safety_hash = hashlib.sha256(self.safety.read_bytes()).hexdigest()
        authorization = {
            "schema_version": 1,
            "freeze_id": "day04-v1",
            "authorization": "confirmatory-safety-evaluation",
            "procedure_version": "day04-v1",
            "safety_split_sha256": safety_hash,
            "component_freeze_commit": "a" * 40,
            "component_set_sha256": "b" * 64,
        }
        (self.split_dir / "safety-unlock.json").write_text(
            json.dumps(authorization)
        )

        records = load_experimental_split("safety-test", self.split_dir)

        self.assertEqual(records[0]["example_id"], "s-1")

    def test_mismatched_hash_keeps_safety_split_locked(self) -> None:
        authorization = {
            "schema_version": 1,
            "freeze_id": "day04-v1",
            "authorization": "confirmatory-safety-evaluation",
            "procedure_version": "day04-v1",
            "safety_split_sha256": "0" * 64,
            "component_freeze_commit": "a" * 40,
            "component_set_sha256": "b" * 64,
        }
        (self.split_dir / "safety-unlock.json").write_text(
            json.dumps(authorization)
        )

        with self.assertRaises(LockedSafetySplitError):
            load_experimental_split("safety-test", self.split_dir)


if __name__ == "__main__":
    unittest.main()
