"""Time one simulator artifact per process and report its deployment plan.

Run from the Python project with the repository's dev dependencies. Use the
same input archive for each artifact and compare their hashes with --baseline.
"""

import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np

from cim22 import CIMTimingConfig, Session
from cim22.runtime import _load_library, runtime_library
from tests._generated import cim22_model_pb2 as proto
from tests.conftest import Descriptor


def read_artifact(path: Path) -> proto.ModelArtifact:
    """Read the embedded protobuf from a compiled simulator model.

    Args:
        path: Path to a native model library from the current build.

    Returns:
        The model's decoded deployment plan.

    Raises:
        ValueError: The ABI is unsupported or the model targets USB.
    """
    library = _load_library(path)
    getter = library.cim22_model_v1
    getter.restype = ctypes.POINTER(Descriptor)
    descriptor = getter().contents
    if descriptor.abi_version != 1:
        raise ValueError(f"{path}: unsupported model ABI")
    model = proto.ModelArtifact.FromString(
        ctypes.string_at(descriptor.data, descriptor.data_size)
    )
    if model.backend != proto.SOFTWARE:
        raise ValueError(f"{path}: expected a simulator artifact")
    return model


def _readback_operations(wave: proto.Transaction) -> tuple[int, int, int]:
    """Estimate Macro batches, selection flits, and Cache read requests."""
    placements = {
        (group.group_id, work.work_id): work.core_slot
        for group in wave.groups
        for work in group.works
    }
    selection = [{}, {}]
    row_cores = [Counter(), Counter()]
    canonical = [True, True]
    for binding in wave.readbacks:
        macro = binding.macro_slot
        core = placements[binding.group_id, binding.work_id]
        selection[macro][core] = binding.data_type
        row_cores[macro][binding.output_cache_address] += 1
        y, x = divmod(core, 5)
        diagonal = min(x, y)
        canonical[macro] &= list(binding.route) == [
            diagonal,
            x - diagonal,
            y - diagonal,
            0,
            0,
            0,
        ]
    batches = selection_flits = requests = 0
    for cores, rows, routes_match in zip(
        selection, row_cores, canonical, strict=True
    ):
        if not cores:
            continue
        batches += 1
        if routes_match and len(cores) == 20 and len(set(cores.values())) == 1:
            selection_flits += 1
        else:
            selection_flits += len(cores)
        requests += sum(
            1 if routes_match and count == 20 else count
            for count in rows.values()
        )
    return batches, selection_flits, requests


def operation_counts(
    model: proto.ModelArtifact, calls: list[int] | None = None
) -> dict[str, int]:
    """Estimate operation counts and USB costs with full weight writes.

    Args:
        model: Decoded plan with kernels in native dispatch-index order.
        calls: Optional invocation counts in the same order as the kernels.
            The default counts each stored kernel once.

    Returns:
        Summed work, routing and readback counts, peak active slots, and
        modeled USB bytes and fixed waits with the default board timing.
    """
    counts: Counter[str] = Counter()
    timing = CIMTimingConfig()
    if calls is None:
        calls = [1] * len(model.kernels)
    for kernel, invocations in zip(model.kernels, calls, strict=True):
        if not invocations:
            continue
        wave = kernel.transaction
        work = sum(len(group.works) for group in wave.groups)
        cores = len(wave.groups)
        readbacks = len(wave.readbacks)
        batches, selection_flits, requests = _readback_operations(wave)
        wave_counts = dict(
            transactions=1,
            physical_works=work,
            work_once=cores,
            active_core_slots=cores,
            active_macro_slots=work,
            readback_batches=batches,
            readback_selection_flits=selection_flits,
            readback_requests=requests,
            readback_response_packets=readbacks,
            configuration_batches=1 + 3 * work,
            host_send_operations=2 + 3 * work + 2 * batches,
            weight_write_packets=work,
            # Per Core: route and Once. Per Work: 4 select/exp, 17 input,
            # and 257 Weight records. Readback adds selection and requests.
            usb_out_bytes=12
            * (2 * cores + 278 * work + selection_flits + requests),
            usb_in_bytes=12 * 7 * readbacks,
            fixed_wait_ms=(
                timing.work_delay_ms
                + work
                * (timing.cache_write_delay_ms + timing.weight_write_delay_ms)
                + batches * timing.readback_select_delay_ms
            ),
        )
        counts.update(
            {key: value * invocations for key, value in wave_counts.items()}
        )
        counts["peak_active_cores"] = max(counts["peak_active_cores"], cores)
        counts["peak_active_macros"] = max(counts["peak_active_macros"], work)
    return dict(counts)


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of the file at path."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_case(
    path: Path, repeats: int, sample: Path | None
) -> dict[str, object]:
    """Time one warmup and repeated calls on a single simulator Session.

    Args:
        path: Compiled native simulator artifact.
        repeats: Positive number of timed calls after warmup.
        sample: NPZ with positional arrays in Session signature order. If
            omitted, generate seeded inputs from the Session signature.

    Returns:
        JSON-compatible timing samples, input/output digests, artifact and
        runtime hashes, native dispatch count and the static deployment plan.

    Raises:
        RuntimeError: Results differ between calls or CIM dispatches are absent
            or inconsistent.
        ValueError: Repeats is less than one or an input shape is invalid.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    runtime = runtime_library()
    native = _load_library(runtime)
    counts = operation_counts(read_artifact(path))
    started = time.perf_counter()
    with Session(path) as session:
        load = time.perf_counter() - started
        rng = np.random.default_rng(20261104)
        if sample is None:
            inputs = tuple(
                np.ascontiguousarray(rng.standard_normal(shape).astype(dtype))
                for shape, dtype in zip(
                    session.input_shapes, session.input_dtypes, strict=True
                )
            )
        else:
            with np.load(sample, allow_pickle=False) as archive:
                inputs = tuple(
                    np.ascontiguousarray(archive[key]) for key in archive.files
                )
        started = time.perf_counter()
        outputs = session(*inputs)
        warmup = time.perf_counter() - started
        dispatches = session.dispatch_count
        if not dispatches:
            raise RuntimeError(f"{path}: missing native dispatches")
        output_hashes = [
            hashlib.sha256(output).hexdigest() for output in outputs
        ]
        del outputs
        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            outputs = session(*inputs)
            timings.append(time.perf_counter() - started)
            if session.dispatch_count != dispatches:
                raise RuntimeError(f"{path}: inconsistent native dispatches")
            if [
                hashlib.sha256(output).hexdigest() for output in outputs
            ] != output_hashes:
                raise RuntimeError(f"{path}: output changed after warmup")
            del outputs
    # Keep the selected runtime loaded while inspecting the model.
    del native
    return {
        "artifact": str(path),
        "artifact_sha256": sha256(path),
        "runtime_sha256": sha256(runtime),
        "python": platform.python_version(),
        "input_shapes": [list(value.shape) for value in inputs],
        "input_dtypes": [value.dtype.name for value in inputs],
        "input_sha256": [hashlib.sha256(value).hexdigest() for value in inputs],
        "output_shapes": [list(shape) for shape in session.output_shapes],
        "output_dtypes": [dtype.name for dtype in session.output_dtypes],
        "output_sha256": output_hashes,
        "static_plan": counts,
        "dynamic_dispatches": dispatches,
        "load_seconds": load,
        "warmup_seconds": warmup,
        "timings_seconds": timings,
        "median_seconds": statistics.median(timings),
        "range_seconds": [min(timings), max(timings)],
    }


def main() -> None:
    """Run one variant and write its results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--sample", type=Path, help="Ordered native inputs in NPZ"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, help="Require matching input and output hashes"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run_case(args.artifact.resolve(), args.repeats, args.sample)
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        for field in (
            "input_shapes",
            "input_dtypes",
            "input_sha256",
            "output_shapes",
            "output_dtypes",
            "output_sha256",
        ):
            if result[field] != baseline.get(field):
                parser.exit(1, f"Benchmark baseline mismatch in {field}\n")
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
