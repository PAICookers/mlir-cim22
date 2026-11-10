"""Compare BF16 compensation on deterministic 64-term simulator dot products."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from benchmarks.build_cases import Case, _linalg
from cim22.compiler import BF16Compensation, _compile_linalg
from cim22.runtime import MODEL_SUFFIX, compiler_library, runtime_library

_CASE = Case("matvec", (64,), (16, 64), (16,))
_CASES = (
    "positive",
    "signed",
    "wide",
    "sparse",
    "cancellation",
    "tiny",
    "large",
    "zero-collision",
    "overflow",
    "overflow-cancel",
)
_MODES: tuple[BF16Compensation, ...] = ("none", "legacy", "sign-split")
_DIRECTED = ("zero-collision", "overflow", "overflow-cancel")
_SEED = 20261106
_UNIT = 2.0**-266
_BF16_MAX = 255 << 386  # 255 * 2^120 in product units of 2^-266.
_REJECTIONS = (
    "CIM BF16 compensated tile exceeds BF16",
    "CIM BF16 chip output is not finite",
    "CIM legacy BF16 group scaling changed prealigned input mantissas",
    "CIM BF16 result exceeds signed i21 range",
    "CIM software result exceeds signed i21 range",
)


def _bf16(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Round binary32 to BF16, keeping the widened binary32 representation."""
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _fixture(
    name: str, samples: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    rng = np.random.default_rng(_SEED)
    if name in _DIRECTED:
        samples = 1

    def magnitudes(shape: tuple[int, ...]) -> NDArray[np.float32]:
        return rng.integers(128, 256, size=shape).astype(np.float32) / 128

    inputs = magnitudes((samples, 64))
    weights = magnitudes((16, 64))
    if name != "positive":
        inputs *= rng.choice((-1, 1), size=inputs.shape)
        weights *= rng.choice((-1, 1), size=weights.shape)
    if name == "wide":
        inputs = np.ldexp(inputs, rng.integers(-24, 1, size=inputs.shape))
        weights = np.ldexp(weights, rng.integers(-24, 1, size=weights.shape))
    elif name == "sparse":
        inputs *= rng.random(inputs.shape) < 0.125
        weights *= rng.random(weights.shape) < 0.125
        inputs[0] = 0
    elif name == "cancellation":
        inputs.fill(0)
        inputs[:, :3] = 1
        # Exact dot varies from 0 to a signed small residual on every lane.
        offsets = [0.0] + [
            sign * 2.0**exp for exp in range(-12, 1) for sign in (1, -1)
        ]
        inputs[:, 3] = np.resize(offsets, samples)
        weights.fill(0)
        weights[:, :4] = (129 / 128, 127 / 128, -2, 1)
        weights = np.ldexp(weights, (np.arange(16) % 8 - 4)[:, None])
    elif name == "tiny":
        inputs = np.ldexp(
            inputs, np.resize([-133, -126, -12, -5, 0, 4], samples)[:, None]
        )
        weights = np.ldexp(weights, np.repeat([-131, -125], 8)[:, None])
    elif name == "large":
        inputs = np.ldexp(inputs, np.resize([0, 30, 56, 60], samples)[:, None])
        weights = np.ldexp(weights, 60)
    elif name == "zero-collision":
        inputs.fill(0)
        inputs[0, :2] = (1, 1 / 32)
        weights.fill(0)
        weights[:8, 2] = 1
        weights[8:, 1] = 1
    elif name == "overflow-cancel":
        inputs.fill(0)
        inputs[0, :2] = 2
        weights.fill(0)
        weights[:, :2] = (2.0**127, -(2.0**127))
    elif name == "overflow":
        inputs.fill(2.0**126)
        weights.fill(1)
    return _bf16(inputs.astype(np.float32)), _bf16(weights.astype(np.float32))


def _operand_units(values: NDArray[np.float32]) -> list[int]:
    # Every finite BF16 value is an integer multiple of 2^-133.
    return [int(float(value) * 2.0**133) for value in values]


def _quantized_units(values: list[int]) -> list[int]:
    shift = max(0, max(map(abs, values)).bit_length() - 7)
    # A finer grid for subnormal inputs leaves operand-unit values unchanged.
    return [
        (abs(value) >> shift << shift) * (-1 if value < 0 else 1)
        for value in values
    ]


def _dot(x: list[int], w: list[int]) -> int:
    return sum(a * b for a, b in zip(x, w, strict=True))


def _split_units(values: list[int]) -> list[int]:
    positive = _quantized_units([max(v, 0) for v in values])
    negative = _quantized_units([max(-v, 0) for v in values])
    return [p - n for p, n in zip(positive, negative, strict=True)]


def _references(
    inputs: NDArray[np.float32], weights: NDArray[np.float32]
) -> tuple[list[int], list[int], dict[str, list[int]]]:
    exact, mass, signed, split = [], [], [], []
    lanes = [_operand_units(lane) for lane in weights]
    signed_weights = [_quantized_units(lane) for lane in lanes]
    split_weights = [_split_units(lane) for lane in lanes]
    for row in inputs:
        x = _operand_units(row)
        signed_x = _quantized_units(x)
        split_x = _split_units(x)
        for w, sw, pw in zip(lanes, signed_weights, split_weights, strict=True):
            products = [a * b for a, b in zip(x, w, strict=True)]
            exact.append(sum(products))
            mass.append(sum(map(abs, products)))
            signed.append(_dot(signed_x, sw))
            split.append(_dot(split_x, pw))
    return exact, mass, {"signed-exact": signed, "split-exact": split}


def _metrics(
    actual: Sequence[int | None],
    exact: list[int],
    mass: list[int],
    paired: Sequence[bool],
) -> dict[str, object]:
    absolute, normalized, matched, relative = [], [], [], []
    condition: dict[str, list[float]] = {}
    false_zero = zero_floor = false_nonzero = sign_flip = compared = 0
    for a, y, scale, pair in zip(actual, exact, mass, paired, strict=True):
        if a is None or abs(y) > _BF16_MAX:
            continue
        compared += 1
        error = abs(a - y)
        absolute.append(error * _UNIT)
        if scale:
            normalized.append(error / scale)
            if pair:
                matched.append(error / scale)
        if y:
            error_ratio = error / abs(y)
            relative.append(error_ratio)
            bucket = next(
                (
                    str(limit)
                    for limit in (2, 16, 256)
                    if scale <= limit * abs(y)
                ),
                ">256",
            )
            condition.setdefault(bucket, []).append(error_ratio)
            false_zero += a == 0
            zero_floor += a == 0 and abs(y) <= (1 << 132)
            sign_flip += a != 0 and (a < 0) != (y < 0)
        else:
            false_nonzero += a != 0
    return {
        "in_range_finite": compared,
        "in_range_finite_nonzero": len(relative),
        "normalized_count": len(normalized),
        "normalized_p50": float(np.median(normalized)) if normalized else None,
        "normalized_p99": float(np.percentile(normalized, 99))
        if normalized
        else None,
        "normalized_max": max(normalized, default=None),
        "abs_max": max(absolute, default=None),
        "relative_p99": float(np.percentile(relative, 99))
        if relative
        else None,
        "legacy_sign_split_common_count": len(matched),
        "legacy_sign_split_common_p99": (
            float(np.percentile(matched, 99)) if matched else None
        ),
        "relative_by_condition": {
            bucket: {
                "count": len(errors),
                "p99": float(np.percentile(errors, 99)),
            }
            for bucket, errors in condition.items()
        },
        "false_zero": false_zero,
        "false_zero_bf16_rounds_to_zero": zero_floor,
        "false_nonzero": false_nonzero,
        "sign_flip": sign_flip,
    }


def _run(
    source: str,
    path: Path,
    inputs: NDArray[np.float32],
    mode: BF16Compensation,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], int, dict[str, int]]:
    outputs = np.full((len(inputs), 16), np.nan, dtype=np.float32)
    rejected = np.zeros(len(inputs), dtype=np.bool_)
    reasons: Counter[str] = Counter()
    dispatches = 0
    with _compile_linalg(
        source, artifact_path=path, runtime="simulator", bf16_compensation=mode
    ) as session:
        for index, row in enumerate(inputs):
            try:
                outputs[index] = session(row)[0]
            except RuntimeError as error:
                reason = next(
                    (msg for msg in _REJECTIONS if msg in str(error)), None
                )
                if reason is None:
                    raise
                reasons[reason] += 1
                rejected[index] = True
                continue
            count = session.dispatch_count
            if count <= 0:
                raise RuntimeError(f"{path}: invocation did not dispatch CIM")
            dispatches += count
    return outputs, rejected, dispatches, dict(reasons)


def _digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _benchmark(
    name: str, samples: int, folder: Path
) -> list[dict[str, object]]:
    inputs, weights = _fixture(name, samples)
    exact, mass, oracles = _references(inputs, weights)
    source = _linalg(_CASE, weights)
    folder.mkdir()
    (folder / "model.linalg.mlir").write_text(source, encoding="utf-8")
    arrays = {"inputs": inputs, "weights": weights}
    values: dict[str, Sequence[int | None]] = dict(oracles)
    counts: dict[str, dict[str, object]] = {}
    for mode in _MODES:
        path = folder / f"{mode}{MODEL_SUFFIX}"
        output, rejected, dispatches, reasons = _run(source, path, inputs, mode)
        arrays[mode] = output
        arrays[f"{mode}_rejected"] = rejected
        valid = output[~rejected]
        # Convert each finite binary32 result to exact product units.
        values[mode] = [
            int(float(value) / _UNIT) if np.isfinite(value) else None
            for value in output.flat
        ]
        counts[mode] = {
            "finite": int(np.isfinite(valid).sum()),
            "nan": int(np.isnan(valid).sum()),
            "pos_inf": int(np.isposinf(valid).sum()),
            "neg_inf": int(np.isneginf(valid).sum()),
            "rejected_calls": int(rejected.sum()),
            "rejected_outputs": int(rejected.sum()) * 16,
            "rejection_reasons": reasons,
            "dispatches_successful_calls": dispatches,
            "artifact_sha256": _digest(path),
        }
    np.savez_compressed(folder / "sample.npz", **arrays)
    paired = [
        a is not None and b is not None
        for a, b in zip(values["legacy"], values["sign-split"], strict=True)
    ]
    split_residual = [
        abs(a - q) / scale
        for a, q, y, scale in zip(
            values["sign-split"],
            oracles["split-exact"],
            exact,
            mass,
            strict=True,
        )
        if a is not None and scale and abs(y) <= _BF16_MAX
    ]
    counts["sign-split"]["split_exact_residual_p99"] = (
        float(np.percentile(split_residual, 99)) if split_residual else None
    )
    return [
        {
            "case": name,
            "mode": mode,
            "dots": len(exact),
            "exact_zero": exact.count(0),
            "exact_outside_bf16": sum(abs(y) > _BF16_MAX for y in exact),
            **counts.get(mode, {}),
            **_metrics(values[mode], exact, mass, paired),
        }
        for mode in (*_MODES, *oracles)
    ]


def main() -> None:
    """Compile, execute and record selected cases in a fresh output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=_CASES)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = runtime_library()
    rows = []
    for name in dict.fromkeys(args.case or _CASES):
        rows.extend(_benchmark(name, args.samples, args.output_dir / name))
        print(name, flush=True)
    report = {
        "seed": _SEED,
        "random_and_sweep_samples_per_case": args.samples,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "benchmark_sha256": _digest(Path(__file__)),
        "runtime_sha256": _digest(runtime),
        "compiler_sha256": _digest(compiler_library(runtime)),
        "results": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, dict)
                    else v
                    for k, v in row.items()
                }
            )


if __name__ == "__main__":
    main()
