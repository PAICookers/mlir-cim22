"""Feed immutable supplier stages to the production C++ BF16 implementation.

This adapter parses fixtures, not BF16 arithmetic. The bundles supply aligned
INT8 weights and independent exponent bytes, not original BF16 weights.
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import int8_reference as reference
import supplier_fixture_replay as supplier


def bit_rows(path: Path, rows: int, columns: int, width: int, delimiter=None):
    lines = path.read_text(encoding="ascii").splitlines()
    records = [line for line in lines if line.strip()]
    tokens = (
        list(csv.reader(records, delimiter=delimiter))
        if delimiter
        else [line.split() for line in records]
    )
    if len(tokens) != rows or any(len(row) != columns for row in tokens):
        raise supplier.FixtureError(f"{path}: expected {rows}x{columns} tokens")
    for row in tokens:
        if any(not re.fullmatch(f"[01]{{{width}}}", token) for token in row):
            raise supplier.FixtureError(f"{path}: expected {width}-bit tokens")
    return tokens


def load_cases(root: Path) -> list[dict]:
    result = []
    for case_id, macro in (
        ("cim22_tb_0p9_macro1", 1),
        ("cim22_tb_0p9_macro2", 2),
        ("test_data_1_17_macro1", 1),
        ("test_data_1_17_macro2", 2),
    ):
        directory = root / "bf16" / case_id
        files = {
            "input": directory / f"cache{macro}_in_bf16.txt",
            "aligned": directory / f"cache{macro}_in_bf16_processed.txt",
            "input_exponents": directory / f"cache{macro}_expmax.txt",
            "input_words": directory / f"cache{macro}_in_bf16_splited.txt",
            "weight_exponents": directory / f"cim{macro}_w_exp.txt",
            "exponent_words": directory
            / f"tb_program_modifier_w{macro}_exp.txt",
            "intermediate": directory / f"output_{macro}.txt",
            "output": directory / f"output_bf16_{macro}.txt",
            "weight": root / case_id / "cim_w.txt",
        }
        inputs = supplier._parse_addressed_bits(files["input"], 8, 1024)
        aligned = bit_rows(files["aligned"], 8, 1, 1024)
        aligned_values = reference.decode_int8_cache_lines(
            [(row, tokens[0]) for row, tokens in enumerate(aligned)]
        ).tolist()
        weight = (
            reference.decode_int8_weight_words(
                supplier._parse_weight(files["weight"])
            )
            .reshape(-1)
            .tolist()
        )
        exponents = [
            int(row[0], 2)
            for row in bit_rows(files["weight_exponents"], 16, 1, 8)
        ]
        exponent_words = re.findall(
            r"64'b([01]{64})", files["exponent_words"].read_text()
        )
        if len(exponent_words) != 2:
            raise supplier.FixtureError(
                "expected two supplier Weight_EXP words"
            )
        input_exponents = bit_rows(files["input_exponents"], 8, 1, 8)
        input_words = bit_rows(files["input_words"], 8, 16, 64, ",")
        intermediate = supplier._parse_expected(files["intermediate"]).tolist()
        output_tokens = bit_rows(files["output"], 8, 16, 21)
        for row, (_, bits) in enumerate(inputs):
            output = [int(token, 2) for token in reversed(output_tokens[row])]
            if any(value > 0xFFFF for value in output):
                raise supplier.FixtureError(
                    "BF16 output has nonzero upper bits"
                )
            # Supplier text is lane 15 first. Append the adopted 48-bit padding
            # to build an independent stream, not the C++ lane-offset formula.
            response_bits = "".join(output_tokens[row]) + "0" * 48
            result.append(
                {
                    "case": case_id,
                    "row": row,
                    "macro": macro - 1,
                    "weight": weight,
                    "weight_exponents": exponents,
                    "exponent_words": [int(word, 2) for word in exponent_words],
                    "input": [
                        int(bits[k : k + 16], 2) for k in range(0, 1024, 16)
                    ],
                    "input_words": [int(word, 2) for word in input_words[row]],
                    "aligned": aligned_values[row],
                    "input_exponent": int(input_exponents[row][0], 2),
                    "intermediate": intermediate[row],
                    "output": output,
                    "response": [
                        int(response_bits[k : k + 64], 2)
                        for k in range(0, 384, 64)
                    ],
                }
            )
    return result


def run(binary: str, cases: list[dict]):
    return subprocess.run(
        [binary, "--bf16-fixtures"],
        input=json.dumps(cases),
        text=True,
        capture_output=True,
        check=False,
    )


def main():
    cases = load_cases(Path(sys.argv[1]))
    actual = run(sys.argv[2], cases)
    if actual.returncode:
        raise SystemExit(actual.stderr)
    print(actual.stdout.strip())

    # Prove each independent stage is checked. Never regenerate expected values
    # from C++ output or silently bless a changed fixture.
    for field in (
        "aligned",
        "input_exponent",
        "intermediate",
        "output",
        "input_words",
        "exponent_words",
        "response",
    ):
        changed = dict(cases[0])
        if isinstance(changed[field], list):
            changed[field] = list(changed[field])
            changed[field][0] ^= (1 << 43) if field == "response" else 1
        else:
            changed[field] ^= 1
        failed = run(sys.argv[2], [changed])
        if failed.returncode == 0 or f"stage={field}" not in failed.stderr:
            raise SystemExit(
                f"mutation was not detected at {field}: {failed.stderr}"
            )
    print("PASS BF16 stage mutations=7")


if __name__ == "__main__":
    main()
