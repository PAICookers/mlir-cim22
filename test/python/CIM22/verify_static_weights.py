"""Verify the M2.4b static-weight section and encoded INT8 words."""

import argparse
import re
from pathlib import Path
from typing import NamedTuple

import int8_reference as reference
import numpy as np

PLAN_BINDING_FIELDS = {
    "function",
    "resource",
    "m_tile",
    "n_tile",
    "k_tile",
    "work_id",
    "group_id",
    "core_slot",
    "macro_slot",
    "mapping",
}
WORK_FIELDS = (
    "m_tile",
    "n_tile",
    "k_tile",
    "work_id",
    "group_id",
    "core_slot",
    "macro_slot",
)


class VerificationError(ValueError):
    pass


class Record(NamedTuple):
    function: str
    resource: str
    work: tuple[int, ...]
    mapping: tuple[tuple[int, ...], ...]
    route: tuple[int, ...]
    macro: int
    words: np.ndarray | None


def _balanced(text: str, start: int, line_number: int) -> str:
    opening = text.find("{", start)
    if opening < 0:
        raise VerificationError(
            f"line {line_number}: missing attribute dictionary"
        )
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index]
    raise VerificationError(
        f"line {line_number}: unterminated attribute dictionary"
    )


def _integer(body: str, name: str, bits: int, line_number: int) -> int:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(-?\d+)\s*:\s*i{bits}\b", body
    )
    if not match:
        raise VerificationError(f"line {line_number}: missing {name} : i{bits}")
    return int(match.group(1))


def _array(
    body: str, name: str, bits: int, size: int, line_number: int
) -> tuple[int, ...]:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*array<i{bits}:\s*([^>]*)>", body
    )
    if not match:
        raise VerificationError(f"line {line_number}: missing {name}")
    try:
        values = tuple(
            int(value.strip()) for value in match.group(1).split(",")
        )
    except ValueError as error:
        raise VerificationError(
            f"line {line_number}: invalid {name}"
        ) from error
    if len(values) != size:
        raise VerificationError(
            f"line {line_number}: {name} expects {size} values"
        )
    return values


def _mapping(
    body: str, name: str, line_number: int
) -> tuple[tuple[int, ...], ...]:
    match = re.search(rf"\b{re.escape(name)}\s*=", body)
    if not match:
        raise VerificationError(f"line {line_number}: missing {name}")
    nested = _balanced(body, match.start(), line_number)
    return tuple(
        _array(nested, field, 64, size, line_number)
        for field, size in (
            ("core_coord", 2),
            ("ingress", 2),
            ("source", 2),
            ("destination", 2),
            ("route", 6),
        )
    )


def _reference(body: str, name: str, line_number: int) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*@([A-Za-z_][A-Za-z0-9_.$-]*)", body
    )
    if not match:
        raise VerificationError(f"line {line_number}: missing {name} symbol")
    return match.group(1)


def _top_level_names(body: str) -> set[str]:
    parts = []
    start = 0
    depth = 0
    pairs = {"{": "}", "[": "]", "<": ">", "(": ")"}
    closers = set(pairs.values())
    for index, char in enumerate(body):
        if char in pairs:
            depth += 1
        elif char in closers:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index])
            start = index + 1
    parts.append(body[start:])
    names = set()
    for part in parts:
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.]*)\s*=", part)
        if not match:
            raise VerificationError(
                f"malformed plan binding field: {part.strip()}"
            )
        names.add(match.group(1))
    return names


def map_tiles(weight_tiles: np.ndarray) -> np.ndarray:
    """Map [B,16,64] opaque bytes to signed [B,256] MLIR i32 words."""
    tiles = np.asarray(weight_tiles, dtype=np.uint8).reshape(-1, 16, 64)
    return np.stack(
        [
            reference.encode_int8_weight_words(tile.view(np.int8)).view(
                np.int32
            )
            for tile in tiles
        ]
    )


def _parse_dense(line: str, line_number: int) -> tuple[str, np.ndarray] | None:
    match = re.search(
        r"\bcim\.static_weight\s+@([A-Za-z_][A-Za-z0-9_.$-]*)\s*=\s*"
        r"dense<(.*)>\s*:\s*tensor<16x64xi8>",
        line,
    )
    if not match:
        return None
    payload = match.group(2).strip()
    raw = re.fullmatch(r'"0x([0-9A-Fa-f]+)"', payload)
    if raw:
        values = np.frombuffer(
            bytes.fromhex(raw.group(1)), dtype=np.uint8
        ).copy()
    else:
        values = np.fromstring(
            re.sub(r"[\[\]]", "", payload), sep=",", dtype=np.int16
        )
    if values.size == 1:
        values = np.full(1024, values[0], dtype=np.int16)
    if values.size != 1024 or np.any(values < -128) or np.any(values > 255):
        raise VerificationError(
            f"line {line_number}: invalid static weight payload"
        )
    return match.group(1), values.astype(np.uint8).reshape(16, 64)


def _parse_configure(
    line: str, function: str, line_number: int
) -> Record | None:
    match = re.search(
        r"\bcim\.configure_weight\s+@([A-Za-z_][A-Za-z0-9_.$-]*)", line
    )
    if not match:
        return None
    body = _balanced(line, match.start(), line_number)
    if _integer(body, "cim.segment_id", 64, line_number) != 0:
        raise VerificationError(
            f"line {line_number}: expected cim.segment_id 0"
        )
    work = tuple(_integer(body, name, 64, line_number) for name in WORK_FIELDS)
    mapping = _mapping(body, "cim.mapping", line_number)
    return Record(
        function, match.group(1), work, mapping, mapping[4], work[-1], None
    )


def _parse_command(line: str, line_number: int) -> Record | None:
    match = re.search(r"\bcimframe\.write_int8_weights\b", line)
    if not match:
        return None
    body = _balanced(line, match.start(), line_number)
    marker = re.search(r"\bcim\.plan_binding\s*=", body)
    if not marker:
        raise VerificationError(f"line {line_number}: missing cim.plan_binding")
    plan_binding = _balanced(body, marker.start(), line_number)
    names = _top_level_names(plan_binding)
    if names != PLAN_BINDING_FIELDS:
        raise VerificationError(
            f"line {line_number}: plan binding fields mismatch: {sorted(names)}"
        )
    function = _reference(plan_binding, "function", line_number)
    resource = _reference(plan_binding, "resource", line_number)
    work = tuple(
        _integer(plan_binding, name, 64, line_number) for name in WORK_FIELDS
    )
    mapping = _mapping(plan_binding, "mapping", line_number)
    route = _array(body, "route", 32, 6, line_number)
    macro = _integer(body, "macro", 32, line_number)
    words_match = re.search(
        r"\bwords\s*=\s*dense<(?P<data>.*?)>\s*:\s*tensor<(?P<size>\d+)xi32>",
        body,
    )
    if not words_match:
        raise VerificationError(f"line {line_number}: missing words")
    payload = words_match.group("data").strip()
    size = int(words_match.group("size"))
    if payload.startswith('"0x') and payload.endswith('"'):
        raw = bytes.fromhex(payload[3:-1])
        if len(raw) != size * 4:
            raise VerificationError(
                f"line {line_number}: dense i32 hex payload has wrong size"
            )
        words = np.frombuffer(raw, dtype="<i4").astype(np.int64)
    elif payload.startswith("[") and payload.endswith("]"):
        words = np.fromstring(payload[1:-1], sep=",", dtype=np.int64)
    else:
        try:
            words = np.full(size, int(payload), dtype=np.int64)
        except ValueError as error:
            raise VerificationError(
                f"line {line_number}: unsupported dense words payload"
            ) from error
    if (
        size != 256
        or words.size != 256
        or np.any(words < -(1 << 31))
        or np.any(words >= (1 << 31))
    ):
        raise VerificationError(
            f"line {line_number}: words expects 256 signed i32 values"
        )
    return Record(
        function, resource, work, mapping, route, macro, words.astype(np.int32)
    )


def parse_records(
    lines, expected_commands: int | None = None
) -> tuple[dict[str, np.ndarray], list[Record]]:
    """Parse and validate static resources and their command records."""
    weights = {}
    configurations = []
    commands = []
    function = ""
    for line_number, line in enumerate(lines, 1):
        if line.lstrip().startswith("//"):
            continue
        if re.search(
            r"\b(?:uint64|flit|flatbuffer|runtime)\b", line, re.IGNORECASE
        ):
            raise VerificationError(
                "typed M2.4b dump contains a forbidden raw/runtime marker"
            )
        function_match = re.search(
            r"\bfunc\.func\s+@([A-Za-z_][A-Za-z0-9_.$-]*)", line
        )
        if function_match:
            function = function_match.group(1)
        dense = _parse_dense(line, line_number)
        if dense:
            if dense[0] in weights:
                raise VerificationError(
                    f"line {line_number}: duplicate resource {dense[0]}"
                )
            weights[dense[0]] = dense[1]
        configure = _parse_configure(line, function, line_number)
        if configure:
            configurations.append(configure)
        command = _parse_command(line, line_number)
        if command:
            commands.append(command)

    if expected_commands is not None and len(commands) != expected_commands:
        raise VerificationError(
            f"expected {expected_commands} commands, found {len(commands)}"
        )
    if len(commands) != len(configurations):
        raise VerificationError("command/configure_weight count mismatch")
    for index, (command, configure) in enumerate(
        zip(commands, configurations, strict=True)
    ):
        if command[:-1] != configure[:-1]:
            raise VerificationError(
                f"command {index}: plan binding/route/Macro mismatch"
            )
        if command.resource not in weights:
            raise VerificationError(
                f"command {index}: unknown resource {command.resource}"
            )

    for start in range(0, len(commands), 128):
        chunk = commands[start : start + 128]
        expected = map_tiles(
            np.stack([weights[command.resource] for command in chunk])
        )
        actual = np.stack([command.words for command in chunk])
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)[0]
            raise VerificationError(
                f"command {start + int(mismatch[0])}: word {int(mismatch[1])} mismatch"
            )
    return weights, commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an M2.4b typed section"
    )
    parser.add_argument("mlir", type=Path)
    parser.add_argument("--expect-commands", type=int)
    args = parser.parse_args()
    with args.mlir.open() as stream:
        weights, commands = parse_records(stream, args.expect_commands)
    print(
        f"PASS software-only commands={len(commands)} resources={len(weights)}"
    )


if __name__ == "__main__":
    main()
