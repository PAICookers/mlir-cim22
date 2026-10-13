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
        first_input = np.ones((2, 65), dtype=np.float32)
        first = session(first_input)[0]
        np.testing.assert_array_equal(first_input, np.ones((2, 65), np.float32))
        np.testing.assert_array_equal(first, np.full((2, 17), 66, np.float32))
        second = session(first_input * 2)[0]
        np.testing.assert_array_equal(second, np.full((2, 17), 131, np.float32))
        np.testing.assert_array_equal(first, np.full((2, 17), 66, np.float32))
        np.testing.assert_array_equal(session(first_input)[0], first)
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
    np.testing.assert_array_equal(first, np.full((2, 17), 66, np.float32))
    try:
        session(first_input)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Closed session accepted invocation")


if __name__ == "__main__":
    main()
