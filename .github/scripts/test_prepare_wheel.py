"""Check CI versioning without compiling native components."""

import importlib.util
import tempfile
import tomllib
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "prepare_wheel", Path(__file__).with_name("prepare-wheel.py")
)
assert spec is not None and spec.loader is not None
prepare_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_wheel)

SOURCE = """[project]
name = "mlir-cim22"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
[tool.scikit-build]
wheel.py-api = "py3"
"""


class WheelVersionTest(unittest.TestCase):
    """Keep version, dependency and compatibility metadata in agreement."""

    def test_release_tag_must_match_source(self) -> None:
        self.assertEqual(
            prepare_wheel.wheel_version(
                "1.0.0", "tag", "v1.0.0", "20261028", "42"
            ),
            "1.0.0",
        )
        for tag in ("v1.0.1", "1.0.0", "v1.0.0rc1", "nightly"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                prepare_wheel.wheel_version(
                    "1.0.0", "tag", tag, "20261028", "42"
                )

    def test_branch_snapshots_use_target_date_and_run(self) -> None:
        for branch in ("main", "feature/manual-build"):
            with self.subTest(branch=branch):
                self.assertEqual(
                    prepare_wheel.wheel_version(
                        "1.0.0", "branch", branch, "20261028", "42"
                    ),
                    "1.0.0.dev20261028000042",
                )
        for run, suffix in (("1", "000001"), ("999999", "999999")):
            with self.subTest(run=run):
                self.assertEqual(
                    prepare_wheel.wheel_version(
                        "1.0.0", "branch", "main", "20261028", run
                    ),
                    f"1.0.0.dev20261028{suffix}",
                )

    def test_invalid_version_inputs(self) -> None:
        for base, ref_type, date, run in (
            ("1.0", "branch", "20261028", "42"),
            ("01.0.0", "branch", "20261028", "42"),
            ("1.0.0.dev1", "branch", "20261028", "42"),
            ("1.0.0", "pull", "20261028", "42"),
            ("1.0.0", "branch", "20260230", "42"),
            ("1.0.0", "branch", "2026102", "42"),
            ("1.0.0", "branch", "20261028", "0"),
            ("1.0.0", "branch", "20261028", "-1"),
            ("1.0.0", "branch", "20261028", "x"),
            ("1.0.0", "branch", "20261028", "01"),
            ("1.0.0", "branch", "20261028", "1000000"),
        ):
            with (
                self.subTest(base=base, ref_type=ref_type, date=date, run=run),
                self.assertRaises(ValueError),
            ):
                prepare_wheel.wheel_version(base, ref_type, "main", date, run)

    def test_version_preserves_other_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pyproject.toml"
            path.write_text(SOURCE, encoding="utf-8")
            prepare_wheel.prepare_wheel(root, "1.0.0.dev20261028000042")
            actual = tomllib.loads(path.read_text(encoding="utf-8"))
            expected = tomllib.loads(SOURCE)
            expected["project"]["version"] = "1.0.0.dev20261028000042"
            self.assertEqual(actual, expected)
            prepare_wheel.prepare_wheel(root, "1.0.0.dev20261028000042")
            self.assertEqual(tomllib.loads(path.read_text()), expected)
            prepare_wheel.prepare_wheel(root, "1.0.0")
            self.assertEqual(
                tomllib.loads(path.read_text()), tomllib.loads(SOURCE)
            )

    def test_invalid_version_does_not_modify_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pyproject.toml"
            path.write_text(SOURCE, encoding="utf-8")
            for version in (
                "20261028",
                "1.0.0rc1",
                "1.0.0.devx",
                '1.0.0"',
            ):
                with (
                    self.subTest(version=version),
                    self.assertRaises(ValueError),
                ):
                    prepare_wheel.prepare_wheel(root, version)
                self.assertEqual(path.read_text(), SOURCE)

    def test_ambiguous_version_does_not_modify_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "pyproject.toml"
            source = SOURCE + '\n[other]\nversion = "2.0"\n'
            path.write_text(source, encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_wheel.prepare_wheel(root, "1.0.0.dev20261028000042")
            self.assertEqual(path.read_text(), source)


if __name__ == "__main__":
    unittest.main()
