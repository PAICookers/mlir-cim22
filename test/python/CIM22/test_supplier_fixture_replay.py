"""Manifest provenance checks for supplier fixture replay."""

import copy
import json
import unittest
from pathlib import Path

import supplier_fixture_replay as replay

MANIFEST = json.loads(
    Path(__file__).parents[2].joinpath("Inputs/CIM22/fixtures.json").read_text()
)


class SupplierManifestTest(unittest.TestCase):
    def test_current_manifest_is_valid(self):
        self.assertEqual(len(replay.validate_manifest(MANIFEST)), 4)

    def test_invalid_provenance_is_rejected(self):
        mutations = {
            "duplicate case": lambda value: value["cases"].__setitem__(
                1, copy.deepcopy(value["cases"][0])
            ),
            "invalid Macro": lambda value: value["cases"][0].__setitem__(
                "supplier_macro", 3
            ),
            "path traversal": lambda value: value["cases"][0]["files"][
                "weight"
            ].__setitem__("path", "../cim_w.txt"),
            "source root mismatch": lambda value: value["cases"][0]["files"][
                "weight"
            ].__setitem__("source_path", "docs/other/cim1_w.txt"),
            "malformed source ID": lambda value: value["cases"][0].__setitem__(
                "source_id", "HWSRC-X"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                candidate = copy.deepcopy(MANIFEST)
                mutate(candidate)
                with self.assertRaises(replay.FixtureError):
                    replay.validate_manifest(candidate)


if __name__ == "__main__":
    unittest.main()
