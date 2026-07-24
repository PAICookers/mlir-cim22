"""Independent software-only oracle for canonical M4 mapped MLIR dumps."""

import argparse
import re
from collections import deque
from pathlib import Path
from typing import NamedTuple

PROFILE_ID = "cim22-4x5-v1"
PROFILE_VERSION = 1
PLACEMENT_POLICY = "core-major-dual-macro-v1"
ROUTE_POLICY = "lower-left-maximal-xy-v1"
LOWER_LEFT = (0, 0)
UPPER_RIGHT = (3, 4)

# Contract entries are deliberately explicit: slot numbering is policy, not an
# address formula.
PROFILE_SLOTS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
    (3, 0), (3, 1), (3, 2), (3, 3), (3, 4),
)
NODES = frozenset(PROFILE_SLOTS)
NEIGHBOR_DELTAS = (
    (1, 1), (-1, -1),
    (0, 1), (0, -1),
    (1, 0), (-1, 0),
)
ROUTE_DELTAS = ((1, 1), (0, 1), (1, 0))


class OracleError(ValueError):
    pass


class Profile(NamedTuple):
    profile_id: str
    version: int
    placement_policy: str
    route_policy: str


class Mapping(NamedTuple):
    work_id: int
    core_slot: int
    macro_slot: int
    core_coord: tuple[int, int]
    ingress: tuple[int, int]
    source: tuple[int, int]
    destination: tuple[int, int]
    route: tuple[int, int, int, int, int, int]


def _required_string_attr(text: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\"([^\"]+)\"", text)
    if not match:
        raise OracleError(f"missing function attribute {name}")
    return match.group(1)


def _required_int_attr(text: str, name: str) -> int:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+)\s*:\s*i64\b", text)
    if not match:
        raise OracleError(f"missing function attribute {name}")
    return int(match.group(1))


def parse_profile(text: str) -> Profile:
    return Profile(
        _required_string_attr(text, "cim.target_profile"),
        _required_int_attr(text, "cim.target_profile_version"),
        _required_string_attr(text, "cim.placement_policy"),
        _required_string_attr(text, "cim.route_policy"),
    )


def _mapping_array(mapping: str, name: str, size: int, line_number: int) -> tuple[int, ...]:
    match = re.search(rf"\b{name}\s*=\s*array<i64:\s*([^>]*)>", mapping)
    if not match:
        raise OracleError(f"line {line_number}: incomplete cim.mapping: missing {name}")
    try:
        values = tuple(int(value.strip()) for value in match.group(1).split(","))
    except ValueError as error:
        raise OracleError(f"line {line_number}: non-integer cim.mapping field {name}") from error
    if len(values) != size:
        raise OracleError(
            f"line {line_number}: {name} expects {size} values, got {len(values)}")
    return values


def parse_records(text: str) -> tuple[Mapping, ...]:
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("//") or not re.search(r"\bcim\.vmm\b", line):
            continue
        values = []
        for name in ("work_id", "core_slot", "macro_slot"):
            match = re.search(rf"\b{name}\s*=\s*(-?\d+)\s*:\s*i64\b", line)
            if not match:
                raise OracleError(f"line {line_number}: missing canonical i64 attribute {name}")
            values.append(int(match.group(1)))
        mapping_match = re.search(r"\bcim\.mapping\s*=\s*\{([^}]*)\}", line)
        if not mapping_match:
            raise OracleError(f"line {line_number}: missing cim.mapping")
        mapping = mapping_match.group(1)
        records.append(Mapping(
            *values,
            _mapping_array(mapping, "core_coord", 2, line_number),
            _mapping_array(mapping, "ingress", 2, line_number),
            _mapping_array(mapping, "source", 2, line_number),
            _mapping_array(mapping, "destination", 2, line_number),
            _mapping_array(mapping, "route", 6, line_number),
        ))
    return tuple(records)


def shortest_distance(source: tuple[int, int], destination: tuple[int, int]) -> int:
    pending = deque([(source, 0)])
    visited = {source}
    while pending:
        node, distance = pending.popleft()
        if node == destination:
            return distance
        for delta_row, delta_col in NEIGHBOR_DELTAS:
            neighbor = (node[0] + delta_row, node[1] + delta_col)
            if neighbor in NODES and neighbor not in visited:
                visited.add(neighbor)
                pending.append((neighbor, distance + 1))
    raise OracleError(f"unreachable destination {destination} from {source}")


def replay_route(source: tuple[int, int], route: tuple[int, ...]) -> tuple[int, int]:
    node = source
    for distance, (delta_row, delta_col) in zip(route[:3], ROUTE_DELTAS):
        for _ in range(distance):
            node = (node[0] + delta_row, node[1] + delta_col)
            if node not in NODES:
                raise OracleError(f"route leaves 4x5 topology at {node}")
    return node


def validate_profile(profile: Profile) -> None:
    expected = Profile(PROFILE_ID, PROFILE_VERSION, PLACEMENT_POLICY, ROUTE_POLICY)
    if profile != expected:
        raise OracleError(f"profile mismatch: expected {expected}, actual {profile}")


def validate_records(
        records: tuple[Mapping, ...],
        slot_table: tuple[tuple[int, int], ...] = PROFILE_SLOTS,
        require_all_slots: bool = False) -> None:
    if tuple(slot_table) != PROFILE_SLOTS or len(set(slot_table)) != len(slot_table):
        raise OracleError("profile slot table is not the frozen explicit 4x5 table")
    if not records:
        raise OracleError("missing mapped work")
    work_ids = [record.work_id for record in records]
    if work_ids != list(range(len(records))):
        raise OracleError(f"missing or non-dense work_id: {work_ids[:12]}")

    used_slots = set()
    for record in records:
        if not 0 <= record.core_slot < len(slot_table):
            raise OracleError(f"work {record.work_id}: invalid core_slot {record.core_slot}")
        if record.macro_slot not in (0, 1):
            raise OracleError(f"work {record.work_id}: invalid macro_slot {record.macro_slot}")
        expected_coord = slot_table[record.core_slot]
        if record.core_coord != expected_coord:
            raise OracleError(
                f"work {record.work_id}: stale core_coord {record.core_coord} "
                f"for slot {record.core_slot}, expected {expected_coord}")
        if record.ingress != LOWER_LEFT or record.source != LOWER_LEFT:
            raise OracleError(
                f"work {record.work_id}: unsupported ingress/source "
                f"{record.ingress}/{record.source}")
        if record.destination != expected_coord:
            raise OracleError(
                f"work {record.work_id}: destination {record.destination} "
                f"does not match core {expected_coord}")
        if any(component < 0 or component > 31 for component in record.route):
            raise OracleError(f"work {record.work_id}: route field outside [0,31]")
        if record.route[3:] != (0, 0, 0):
            raise OracleError(f"work {record.work_id}: nonzero Copy route {record.route[3:]}")
        endpoint = replay_route(record.source, record.route)
        if endpoint != record.destination:
            raise OracleError(
                f"work {record.work_id}: route endpoint {endpoint} "
                f"does not match destination {record.destination}")
        distance = shortest_distance(record.source, record.destination)
        if sum(record.route[:3]) != distance:
            raise OracleError(
                f"work {record.work_id}: route length {sum(record.route[:3])} "
                f"is not BFS shortest distance {distance}")
        used_slots.add(record.core_slot)

    if require_all_slots and used_slots != set(range(len(PROFILE_SLOTS))):
        raise OracleError(f"missing core slots: {sorted(set(range(20)) - used_slots)}")


def validate_dump(text: str, require_all_slots: bool = False) -> tuple[Mapping, ...]:
    validate_profile(parse_profile(text))
    records = parse_records(text)
    validate_records(records, require_all_slots=require_all_slots)
    return records


def _route_text(route: tuple[int, ...]) -> str:
    return "[" + ",".join(map(str, route)) + "]"


def _boundary(records: tuple[Mapping, ...], destination: tuple[int, int]) -> str:
    record = next(
        (record for record in records if record.destination == destination), None)
    if record is None:
        return "NA"
    return (f"w{record.work_id}/c{record.core_slot}/m{record.macro_slot} "
            f"route={_route_text(record.route)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical M4 mappings by graph replay and BFS")
    parser.add_argument("mlir", type=Path)
    parser.add_argument("--require-all-slots", action="store_true")
    args = parser.parse_args()

    records = validate_dump(args.mlir.read_text(), args.require_all_slots)
    macros = "[" + ",".join(map(str, sorted({record.macro_slot for record in records}))) + "]"
    print(f"PASS software-only profile={PROFILE_ID} version={PROFILE_VERSION} "
          f"work={len(records)} cores={len({record.core_slot for record in records})} "
          f"macros={macros}")
    print(f"boundaries zero={_boundary(records, LOWER_LEFT)} "
          f"corner={_boundary(records, UPPER_RIGHT)}")


if __name__ == "__main__":
    main()
