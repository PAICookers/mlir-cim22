"""Replay normalized supplier fixtures through the test-private INT8 model."""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import int8_reference as reference
import numpy as np


class FixtureError(ValueError):
    pass


def validate_manifest(manifest: dict) -> list[dict]:
    """Validate provenance metadata without touching fixture bytes."""
    if manifest.get("schema_version") != 1:
        raise FixtureError("unsupported fixture schema")
    if manifest.get("evidence_level") != "supplier-fixture-match":
        raise FixtureError(
            "fixture evidence level must be supplier-fixture-match"
        )
    if not re.fullmatch(
        r"HWSRC-\d{3}", str(manifest.get("snapshot_source_id"))
    ):
        raise FixtureError("snapshot_source_id must be HWSRC-NNN")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise FixtureError("manifest must contain four cases")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise FixtureError("manifest case_id values must be unique")
    for case in cases:
        case_id = case.get("case_id")
        if not re.fullmatch(
            r"(?:cim22_tb_0p9|test_data_1_17)_macro[12]", str(case_id)
        ):
            raise FixtureError(f"{case_id}: malformed case_id")
        if not re.fullmatch(r"HWSRC-\d{3}", str(case.get("source_id"))):
            raise FixtureError(f"{case_id}: malformed source_id")
        if case.get("supplier_macro") not in (1, 2):
            raise FixtureError(f"{case_id}: supplier_macro must be 1 or 2")
        source_root = PurePosixPath(str(case.get("source_root", "")))
        if source_root.is_absolute() or ".." in source_root.parts:
            raise FixtureError(f"{case_id}: invalid source_root")
        files = case.get("files")
        if not isinstance(files, dict) or set(files) != {
            "weight",
            "input",
            "output",
        }:
            raise FixtureError(
                f"{case_id}: files must contain weight/input/output"
            )
        for entry in files.values():
            local_path = PurePosixPath(str(entry.get("path", "")))
            source_path = PurePosixPath(str(entry.get("source_path", "")))
            if local_path.is_absolute() or ".." in local_path.parts:
                raise FixtureError(f"{case_id}: fixture path escapes root")
            if source_path.is_absolute() or ".." in source_path.parts:
                raise FixtureError(f"{case_id}: source path escapes root")
            if not str(source_path).startswith(f"{source_root}/"):
                raise FixtureError(
                    f"{case_id}: source_path is outside source_root"
                )
            if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256"))):
                raise FixtureError(f"{case_id}: malformed SHA-256")
    return cases


def _checked_path(root: Path, entry: dict) -> Path:
    path = (root / entry["path"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise FixtureError(
            f"fixture path escapes manifest root: {entry['path']}"
        ) from error
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise FixtureError(f"SHA-256 mismatch: {entry['path']}")
    return path


def _parse_addressed_bits(
    path: Path, count: int, width: int
) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open(newline="", encoding="ascii") as stream:
        for line_number, record in enumerate(csv.reader(stream), 1):
            if len(record) != 2:
                raise FixtureError(
                    f"{path}:{line_number}: expected address,bits"
                )
            try:
                address = int(record[0])
            except ValueError as error:
                raise FixtureError(
                    f"{path}:{line_number}: invalid address"
                ) from error
            bits = record[1].strip()
            if len(bits) != width or set(bits) - {"0", "1"}:
                raise FixtureError(
                    f"{path}:{line_number}: expected {width} binary bits"
                )
            rows.append((address, bits))
    addresses = [address for address, _ in rows]
    if len(rows) != count or sorted(addresses) != list(range(count)):
        raise FixtureError(
            f"{path}: addresses must cover 0..{count - 1} exactly once"
        )
    return sorted(rows)


def _parse_weight(path: Path) -> np.ndarray:
    rows = _parse_addressed_bits(path, reference.WEIGHT_WORDS, 32)
    return np.asarray([int(bits, 2) for _, bits in rows], dtype=np.uint32)


def _parse_expected(path: Path) -> np.ndarray:
    lines = [
        line.split()
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(lines) != reference.CACHE_ROWS or any(
        len(line) != reference.LANES for line in lines
    ):
        raise FixtureError(f"{path}: expected 8 rows of 16 i21 tokens")
    result = np.zeros((reference.CACHE_ROWS, reference.LANES), dtype=np.int64)
    for row, tokens in enumerate(lines):
        for lane, token in enumerate(reversed(tokens)):
            if len(token) != 21 or set(token) - {"0", "1"}:
                raise FixtureError(f"{path}: row {row} has invalid i21 token")
            result[row, lane] = reference.decode_signed_i21(int(token, 2))
    return result


def replay(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    cases = validate_manifest(manifest)

    root = manifest_path.parent
    for case in cases:
        if case.get("mode") != "int8" or case.get("shapes") != {
            "weight": [16, 64],
            "input": [8, 64],
            "output": [8, 16],
        }:
            raise FixtureError(
                f"{case.get('case_id')}: unsupported mode or shapes"
            )
        files = case["files"]
        word_path = _checked_path(root, files["weight"])
        input_path = _checked_path(root, files["input"])
        output_path = _checked_path(root, files["output"])

        words = _parse_weight(word_path)
        weight = reference.decode_int8_weight_words(words)
        np.testing.assert_array_equal(
            reference.encode_int8_weight_words(weight), words
        )
        inputs = reference.decode_int8_cache_lines(
            _parse_addressed_bits(
                input_path, reference.CACHE_ROWS, reference.CACHE_LINE_BITS
            )
        )
        expected = _parse_expected(output_path)
        for row in range(reference.CACHE_ROWS):
            actual = reference.simulate_int8_tile(inputs[row], weight)
            np.testing.assert_array_equal(actual, expected[row])
            flits = reference.encode_int8_output_response(actual)
            np.testing.assert_array_equal(
                reference.decode_int8_output_response(flits), actual
            )

    return "supplier-fixture-match cases=4 words=1024 slots=2048 tokens=512"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} fixtures.json")
    print(f"PASS {replay(Path(sys.argv[1]))}")


if __name__ == "__main__":
    main()
