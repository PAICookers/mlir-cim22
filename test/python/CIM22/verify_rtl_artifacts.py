"""Independently decode the supplier INT8 RTL artifact profile.

Field expectations come from MODEL_TEAM_FRAME_FEEDBACK (2026-09-05) and
CIM22_tb/tb_noc_20_cores_edge.sv, not from the production encoder. This checks
delivered bytes and recipient sets, not RTL execution or NoC progress.
"""

import argparse
from pathlib import Path


def read_words(path: Path, width: int) -> list[int]:
    lines = path.read_text().splitlines()
    assert lines, f"{path}: empty file"
    assert all(len(line) == width and set(line) <= {"0", "1"} for line in lines)
    return [int(line, 2) for line in lines]


def read_source(path: Path, width: int, count: int) -> dict[int, int]:
    result = {}
    for line in path.read_text().splitlines():
        address, bits = line.split(",")
        address = int(address)
        assert address not in result, f"{path}: repeated source address"
        assert len(bits) == width and set(bits) <= {"0", "1"}
        result[address] = int(bits, 2)
    assert set(result) == set(range(count)), f"{path}: incomplete source"
    return result


def signed_magnitude(value: int) -> int:
    return -(value & 31) if value & 32 else value


def destinations(head: int) -> set[int]:
    xy, x, y, cxy, cx, cy = [
        signed_magnitude((head >> shift) & 63)
        for shift in (54, 48, 42, 36, 30, 24)
    ]
    # Only the evidenced positive X/Y Copy subset is part of this profile.
    assert cxy == 0 and cx >= 0 and cy >= 0, "unsupported Copy expansion"
    result = set()
    for row in range(xy + y, xy + y + cy + 1):
        for col in range(xy + x, xy + x + cx + 1):
            assert 0 <= row < 4 and 0 <= col < 5, "route leaves chip"
            result.add(row * 5 + col)
    return result


def packets(path: Path):
    words = read_words(path, 64)
    cursor = 0
    while cursor < len(words):
        head = words[cursor]
        kind = head >> 60
        count = head & 0x3FFF if kind in (1, 2) else 0
        assert cursor + count < len(words), "truncated packet"
        yield kind, head, words[cursor + 1 : cursor + 1 + count]
        cursor += count + 1


def verify_weights(directory: Path, active: set[int]) -> None:
    sources = [
        read_source(directory / "sources" / f"cim{macro + 1}_w.txt", 32, 256)
        for macro in range(2)
    ]
    selected = {}
    seen = set()
    for kind, head, body in packets(directory / "01_config.frames.txt"):
        targets = destinations(head)
        if kind == 8:
            selected.update((core, head & 1) for core in targets)
        elif kind == 2:
            assert len(body) == 256, "CIM write must have 256 body flits"
            actual = {}
            for word in body:
                address = word & 255
                assert address not in actual, f"repeated CIM address {address}"
                actual[address] = (word >> 8) & 0xFFFFFFFF
            assert set(actual) == set(range(256)), "missing CIM addresses"
            for core in targets:
                macro = selected[core]
                assert actual == sources[macro], "CIM data differs from source"
                assert (core, macro) not in seen, "duplicate CIM write"
                seen.add((core, macro))
            # The supplier multicast task retains type=0010 in each body.
            assert all(word >> 40 == 0x200000 for word in body), "CIM body prefix"
    assert seen == {(core, macro) for core in active for macro in range(2)}


def verify_returns(directory: Path, active: set[int]) -> None:
    seen = set()
    for kind, head, _ in packets(directory / "01_config.frames.txt"):
        if kind != 10:
            continue
        targets = destinations(head)
        assert len(targets) == 1, "return configuration must be onecast"
        core = targets.pop()
        assert core not in seen, "duplicate return configuration"
        seen.add(core)
        row, col = divmod(core, 5)
        xy, x, y = [signed_magnitude((head >> s) & 63) for s in (12, 6, 0)]
        assert (row + xy + y, col + xy + x) == (-1, 4), "return is not off-chip"
        assert (head >> 18) & 31 == core, "wrong five-bit response index"
    assert seen == active, "return configuration does not match responders"


def verify_targets(directory: Path, active: set[int]) -> None:
    inputs = [
        read_source(directory / "sources" / f"cache{macro + 1}_in.txt", 1024, 8)
        for macro in range(2)
    ]
    writes = set()
    starts = []
    stops = []
    reads = set()
    cim_reads = set()
    for stage in ("01_config", "02_work", "03_readback"):
        selected = {}
        for kind, head, body in packets(directory / f"{stage}.frames.txt"):
            targets = destinations(head)
            assert targets <= active, f"{stage}: request reaches inactive core"
            if kind == 8:
                assert (head & 3) in (0, 1), "expected INT8 control"
                selected.update((core, head & 1) for core in targets)
            elif kind == 1:
                assert stage == "01_config" and len(body) == 16
                address = (head >> 14) & 15
                assert address & 8, "input cache selector bit"
                row = address & 7
                bits = int("".join(f"{word:064b}" for word in body), 2)
                for core in targets:
                    macro = selected[core]
                    assert bits == inputs[macro][row], "input differs from source"
                    key = (core, macro, row)
                    assert key not in writes, "duplicate input write"
                    writes.add(key)
            elif kind == 9:
                assert stage == "02_work" and (head & 3) in (0, 1)
                (starts if head & 1 else stops).extend(targets)
            elif kind in (5, 6):
                assert stage == "03_readback" and head & (1 << 23)
                for core in targets:
                    macro = selected[core]
                    if kind == 5:
                        row = (head >> 14) & 15
                        assert row < 8
                        key = (core, macro, row)
                        assert key not in reads, "duplicate cache read"
                        reads.add(key)
                    else:
                        key = (core, macro)
                        assert key not in cim_reads, "duplicate CIM read"
                        cim_reads.add(key)
            else:
                assert stage == "01_config" and kind in (2, 10)
    pairs = {(core, macro) for core in active for macro in range(2)}
    rows = {(core, macro, row) for core, macro in pairs for row in range(8)}
    assert writes == rows and reads == rows and cim_reads == pairs
    assert sorted(starts) == sorted(active), "work starts do not match active cores"
    assert sorted(stops) == sorted(active), "work stops do not match active cores"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument(
        "--check",
        choices=("weights", "returns", "targets", "all"),
        default="all",
    )
    args = parser.parse_args()
    active = set(range(args.cores))
    checks = {
        "weights": verify_weights,
        "returns": verify_returns,
        "targets": verify_targets,
    }
    for name, check in checks.items():
        if args.check in (name, "all"):
            check(args.directory, active)
    print(
        "PASS RTL artifact fields, sources and targets: "
        f"{args.cores} cores ({args.check})"
    )


if __name__ == "__main__":
    main()
