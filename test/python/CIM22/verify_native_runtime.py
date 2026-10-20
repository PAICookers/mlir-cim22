"""Test actual native dispatch, Host epilogue, ownership and error recovery."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
from cim22.runtime import Session


def main() -> None:
    # Native compilation must reject the same malformed split IR as the passes;
    # choosing the last Host module would silently ignore another module.
    packet = Path(sys.argv[1]).read_text(encoding="utf-8")
    duplicate = packet.replace(
        "module @__cim_host",
        'module @ignored attributes {cim.pipeline_part = "host"} {}\n'
        "module @__cim_host",
        1,
    )
    if duplicate == packet:
        raise AssertionError("Test did not introduce a duplicate Host module")
    with tempfile.TemporaryDirectory() as directory:
        invalid = Path(directory) / "duplicate.mlir"
        invalid.write_text(duplicate, encoding="utf-8")
        try:
            with Session(invalid, Path(sys.argv[2])):
                raise AssertionError("Duplicate Host module accepted")
        except RuntimeError as error:
            if "exactly one Host and one device module" not in str(error):
                raise
    with Session(Path(sys.argv[1]), Path(sys.argv[2])) as session:
        if session.num_transactions == 0:
            raise AssertionError("Test silently bypassed CIM")
        # Small normal inputs keep the supplier's raw-exponent reconstruction
        # finite: after Host doubling, the 64/1-item tiles give 16384 and 256.
        first_input = np.full((2, 65), 2.0**-126, dtype=np.float32)
        original_input = first_input.copy()
        first_expected = np.full((2, 17), 16641, np.float32)
        second_expected = np.full((2, 17), 33281, np.float32)
        first = session(first_input)[0]
        np.testing.assert_array_equal(first_input, original_input)
        np.testing.assert_array_equal(first, first_expected)
        second = session(first_input * 2)[0]
        np.testing.assert_array_equal(second, second_expected)
        np.testing.assert_array_equal(first, first_expected)
        np.testing.assert_array_equal(session(first_input)[0], first)
        np.testing.assert_array_equal(first_input, original_input)
        if session.dispatch_count == 0 or np.shares_memory(first, second):
            raise AssertionError("Missing dispatch or shared output storage")
        try:
            session(np.full((2, 65), np.nan, np.float32))
        except RuntimeError as error:
            if "BF16" not in str(error):
                raise
        else:
            raise AssertionError("Native runner failure was swallowed")
        np.testing.assert_array_equal(session(first_input)[0], first)
        for invalid in (first_input.astype(np.float64), first_input[:, :-1]):
            try:
                session(invalid)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("Invalid ABI input accepted")
    np.testing.assert_array_equal(first, first_expected)
    try:
        session(first_input)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Closed session accepted invocation")


if __name__ == "__main__":
    main()
