"""Native model publication, typed boundaries and software execution."""

import os
import re
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from cim22.compiler import _compile_linalg, _emit_object
from cim22.runtime import MODEL_SUFFIX, Session, compiler_library

from ._generated import cim22_model_pb2 as proto

if TYPE_CHECKING:
    from .conftest import EmbeddedModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_runtime"),
]


@pytest.mark.skipif(sys.platform != "linux", reason="ELF export policy")
def test_native_export_boundary(native_runtime: Path) -> None:
    """Only the three shared error tags may escape the native LLVM copies."""
    allowed = {
        "llvm::ErrorInfoBase::ID",
        "llvm::ErrorList::ID",
        "llvm::StringError::ID",
    }
    for library in (native_runtime, compiler_library(native_runtime)):
        output = subprocess.run(
            ["nm", "-D", "-C", "--defined-only", str(library)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        symbols = {line.split(maxsplit=2)[2] for line in output.splitlines()}
        assert {s for s in symbols if s.startswith("llvm::")} == allowed
        assert not any(
            s.startswith(("google::protobuf::", "absl::", "mlir::cl::"))
            for s in symbols
        )
        assert "cim22_compile_object" in symbols or (
            "cim22_session_load_shared" in symbols
        )


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    """A minimal split program isolates the native compiler/file boundary."""
    path = tmp_path / "model.packet.mlir"
    path.write_text(
        """
module attributes {cim.split_program} {
  module @host attributes {cim.pipeline_part = "host"} {
    func.func @main(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {
      %y = arith.addf %x, %x : tensor<2x3xf32>
      return %y : tensor<2x3xf32>
    }
  }
  module @device attributes {cim.pipeline_part = "device"} {}
}
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("destination", ["missing", "directory"])
def test_object_destination_failure(
    packet: Path, tmp_path: Path, destination: str, native_runtime: Path
) -> None:
    output = tmp_path / destination / "model.o"
    if destination == "directory":
        output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="cannot .* object file"):
        _emit_object(packet, output, native_runtime)
    assert not list(tmp_path.rglob("*.tmp-*"))
    if destination == "missing":
        assert not output.exists()
    else:
        assert output.is_dir()


@pytest.mark.skipif(
    sys.platform == "win32", reason="requires POSIX RLIMIT_FSIZE"
)
def test_object_write_failure_preserves_previous_output(
    packet: Path, tmp_path: Path, native_runtime: Path
) -> None:
    output = tmp_path / "model.o"
    output.write_bytes(b"previous object")
    # Limit only a child process: ignore SIGXFSZ so the file write returns EFBIG
    # and the native API has to propagate the error instead of being killed.
    script = """
import resource
import signal
import sys
from pathlib import Path
from cim22.compiler import _compile_linalg, _emit_object
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (128, 128))
try:
    _emit_object(Path(sys.argv[1]), Path(sys.argv[2]),
                 Path(sys.argv[3]))
except RuntimeError as error:
    assert 'cannot write object file' in str(error), str(error)
else:
    raise AssertionError('native object write failure was swallowed')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(packet),
            str(output),
            str(native_runtime),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_bytes() == b"previous object"
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.parametrize(
    "visibility", ["", "private "], ids=["public", "private"]
)
def test_runtime_entry_preserves_input(packet: Path, visibility: str) -> None:
    packet.write_text(
        packet.read_text().replace(
            "func.func @main", f"func.func {visibility}@main"
        )
    )
    value = np.ones((2, 3), dtype=np.float32)
    with Session(packet) as session:
        np.testing.assert_array_equal(session(value)[0], np.full_like(value, 2))
    np.testing.assert_array_equal(value, np.ones_like(value))


@pytest.mark.parametrize("dtype", [np.int8, np.int32])
def test_integer_host_boundary(
    packet: Path, dtype: type[np.signedinteger]
) -> None:
    """Native JIT boundaries preserve storage width without casting."""
    element_type = "i8" if dtype == np.int8 else "i32"
    packet.write_text(
        packet.read_text().replace("f32", element_type).replace("addf", "addi")
    )
    value = np.array([[-128, -1, 0], [1, 63, 127]], dtype=dtype)
    with Session(packet) as session:
        assert session.input_dtypes == (np.dtype(dtype),)
        assert session.output_dtypes == (np.dtype(dtype),)
        np.testing.assert_array_equal(session(value)[0], value + value)
        with pytest.raises(TypeError, match=np.dtype(dtype).name):
            session(value.astype(np.float32))


@pytest.mark.usefixtures("native_toolchain")
@pytest.mark.parametrize("artifact_format", ["packet", "shared"])
def test_int8_conv_execution(
    tmp_path: Path, native_runtime: Path, artifact_format: str
) -> None:
    """Native INT8 Conv preserves exact values through JIT and shared models."""
    weights = (np.arange(17 * 2 * 6 * 6).reshape(17, 2, 6, 6) % 15 - 7).astype(
        np.int8
    )
    source = tmp_path / "conv.linalg.mlir"
    source.write_text(
        f"""
func.func @main(%input: tensor<1x2x8x8xi8>) -> tensor<1x17x3x3xi32> {{
  %weight = arith.constant dense<{weights.tolist()}> : tensor<17x2x6x6xi8>
  %zero = arith.constant dense<0> : tensor<1x17x3x3xi32>
  %result = linalg.conv_2d_nchw_fchw
      ins(%input, %weight : tensor<1x2x8x8xi8>, tensor<17x2x6x6xi8>)
      outs(%zero : tensor<1x17x3x3xi32>) -> tensor<1x17x3x3xi32>
  return %result : tensor<1x17x3x3xi32>
}}
""",
        encoding="utf-8",
    )
    packet = tmp_path / "conv.packet.mlir"
    subprocess.run(
        [
            os.environ.get("CIM22_OPT")
            or str(
                files("cim22")
                / "bin"
                / (
                    "mlir-cim22-opt.exe"
                    if os.name == "nt"
                    else "mlir-cim22-opt"
                )
            ),
            str(source),
            "--cim-bf16-pipeline=stage=packet max-kernel-columns=1",
            "-o",
            str(packet),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    artifact = packet
    if artifact_format == "shared":
        object_file = tmp_path / ("conv.obj" if os.name == "nt" else "conv.o")
        artifact = tmp_path / f"conv{MODEL_SUFFIX}"
        _emit_object(packet, object_file, native_runtime, "simulator")
        subprocess.run(
            [
                os.environ.get("CIM22_CC", "cl"),
                "/nologo",
                "/LD",
                "/MD",
                str(object_file),
                str(native_runtime.with_suffix(".lib")),
                "/link",
                f"/OUT:{artifact}",
                "/EXPORT:cim22_model_v1",
                f"/IMPLIB:{tmp_path / 'conv.lib'}",
                "msvcrt.lib",
            ]
            if os.name == "nt"
            else [
                os.environ.get("CIM22_CC", "cc"),
                "-shared",
                "-Wl,-z,defs",
                "-Wl,-Bsymbolic-functions",
                "-Wl,-rpath,$ORIGIN",
                str(object_file),
                str(native_runtime),
                "-lm",
                "-o",
                str(artifact),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    value = (np.arange(128).reshape(1, 2, 8, 8) * 13 % 256 - 128).astype(
        np.int8
    )
    inputs = [value, np.flip(value, axis=3).copy(), value]
    outputs = []
    with Session(artifact, library=native_runtime) as session:
        assert session.input_shapes == ((1, 2, 8, 8),)
        assert session.output_shapes == ((1, 17, 3, 3),)
        assert session.input_dtypes == (np.dtype(np.int8),)
        assert session.output_dtypes == (np.dtype(np.int32),)
        assert session.num_transactions > 0
        for current in inputs:
            original = current.copy()
            # An int64 spatial oracle is independent of compiler im2col/tiling.
            patches = np.lib.stride_tricks.sliding_window_view(
                current.astype(np.int64), (6, 6), axis=(2, 3)
            )
            expected = np.einsum(
                "nchwkl,fckl->nfhw", patches, weights.astype(np.int64)
            ).astype(np.int32)
            (actual,) = session(current)
            assert session.dispatch_count > 0
            assert actual.dtype == np.int32
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(current, original)
            assert not np.shares_memory(actual, current)
            for previous in outputs:
                assert not np.shares_memory(actual, previous)
            outputs.append(actual)
    np.testing.assert_array_equal(outputs[0], outputs[2])


def test_packet_parser_rejects_invalid_return(packet: Path) -> None:
    packet.write_text(
        packet.read_text().replace(
            ") -> tensor<2x3xf32>", ") -> tensor<1x3xf32>"
        )
    )
    with pytest.raises(RuntimeError, match="return"):
        Session(packet)


@pytest.mark.usefixtures("native_toolchain")
def test_multiple_kernels_retain_distinct_weight_resources(
    tmp_path: Path, embedded_model: "EmbeddedModel"
) -> None:
    """Per-kernel extraction retains every referenced weight and packet."""

    class TwoWeights(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(64, 32, bias=False)
            self.second = torch.nn.Linear(64, 16, bias=False)
            with torch.no_grad():
                self.first.weight.fill_(1.0)
                self.second.weight.fill_(2.0)

        def forward(
            self, value: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.first(value), self.second(value)

    compiled = torch.compile(
        TwoWeights().eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "simulator", "artifact_dir": tmp_path},
    )
    # Unit/double-weight kernels must retain their distinct resources.
    value = torch.ones(1, 64)
    expected = (torch.full((1, 32), 64.0), torch.full((1, 16), 128.0))
    with torch.inference_mode():
        for actual, reference in zip(compiled(value), expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    (artifact,) = tmp_path.glob(f"*{MODEL_SUFFIX}")
    model = embedded_model(artifact)
    assert len(model.kernels) >= 2
    assert {len(k.transaction.weights) for k in model.kernels} == {1, 2}
    assert (
        len({w.resource for k in model.kernels for w in k.transaction.weights})
        >= 3
    )
    for kernel in model.kernels:
        transaction = kernel.transaction
        assert len(transaction.inputs) == len(transaction.weights)
        assert len(transaction.readbacks) == len(transaction.weights)
        assert sum(p.kind == proto.WEIGHT for p in transaction.packets) == len(
            transaction.weights
        )
        assert transaction.flits
    with Session(artifact) as session:
        assert session.num_transactions >= 2
        for actual, reference in zip(
            session(value.numpy()), expected, strict=True
        ):
            np.testing.assert_array_equal(actual, reference.numpy())


def test_obsolete_correction_marker_rejected(packet: Path) -> None:
    packet.write_text(
        packet.read_text().replace(
            "cim.split_program}",
            "cim.split_program, cim.bf16_corrected = false}",
        )
    )
    with pytest.raises(RuntimeError, match="obsolete cim.bf16_corrected"):
        Session(packet)


@pytest.mark.parametrize("mode", ['"unknown"', "true"])
def test_invalid_compensation_marker_rejected(packet: Path, mode: str) -> None:
    packet.write_text(
        packet.read_text().replace(
            "cim.split_program}",
            f"cim.split_program, cim.bf16_compensation = {mode}}}",
        )
    )
    with pytest.raises(RuntimeError, match="BF16 compensation must"):
        Session(packet)


@pytest.mark.usefixtures("native_toolchain")
@pytest.mark.parametrize(
    ("columns", "k", "n", "fixed_waves", "auto_waves"),
    [
        (1, 65, 17, 1, 1),
        (31, 65, 17, 4, 4),
        (32, 65, 17, 4, 4),
        (33, 65, 17, 5, 4),
        (65, 65, 17, 9, 7),
        (65, 64, 16, 3, 2),
        (66, 64, 33, 7, 6),  # The static Work budget excludes a five-wave plan.
    ],
)
@pytest.mark.parametrize("integer", [False, True], ids=["bf16", "int8"])
def test_column_blocks(
    columns: int,
    k: int,
    n: int,
    fixed_waves: int,
    auto_waves: int,
    integer: bool,
) -> None:
    """Check exact tails, wave counts and column order across blocking modes."""
    element, result = ("i8", "i32") if integer else ("f32", "f32")
    marker = "{cim.int8}" if integer else ""
    weights = np.arange(n * k).reshape(n, k) % 5 - 2
    data = (np.arange(k * columns).reshape(k, columns) % 19 - 9).astype(
        np.int8 if integer else np.float32
    )
    weight_values = (
        weights.tolist() if integer else weights.astype(float).tolist()
    )
    activation = f"tensor<{k}x{columns}x{element}>"
    weight_type = f"tensor<{n}x{k}x{element}>"
    output = f"tensor<{n}x{columns}x{result}>"
    source = f"""
func.func @main(%x: {activation}) -> {output} {{
  %w = arith.constant dense<{weight_values}> : {weight_type}
  %z = arith.constant dense<{"0" if integer else "0.0"}> : {output}
  %r = linalg.matmul {marker} ins(%w, %x : {weight_type}, {activation})
       outs(%z : {output}) -> {output}
  return %r : {output}
}}
"""
    with (
        _compile_linalg(
            source, runtime="simulator", max_kernel_columns=1
        ) as serial,
        _compile_linalg(
            source, runtime="simulator", max_kernel_columns=32
        ) as blocked,
        _compile_linalg(source, runtime="simulator") as automatic,
    ):
        branches = 1 if integer else 4
        for value in (data, np.ascontiguousarray(-data[:, ::-1])):
            expected = serial(value)[0]
            for model in (blocked, automatic):
                np.testing.assert_array_equal(
                    model(value)[0].view(np.uint32), expected.view(np.uint32)
                )
            if integer:
                np.testing.assert_array_equal(expected, weights @ value)
            assert serial.dispatch_count == columns * branches
            assert blocked.dispatch_count == fixed_waves * branches
            assert automatic.dispatch_count == auto_waves * branches


@pytest.mark.usefixtures("native_toolchain")
@pytest.mark.parametrize(
    ("integer", "k", "n", "columns", "full_writes", "grouped_writes"),
    [
        (True, 256, 320, 3, 240, 80),
        (True, 256, 320, 1, 80, 80),
        (False, 256, 512, 1, 512, 256),
        (False, 512, 110, 1, 224, 112),
    ],
    ids=["int8-linear", "int8-no-reuse", "bf16-fc1", "bf16-fc2"],
)
def test_weight_residency_schedule(
    tmp_path: Path,
    embedded_model: "EmbeddedModel",
    integer: bool,
    k: int,
    n: int,
    columns: int,
    full_writes: int,
    grouped_writes: int,
) -> None:
    """Replay actual Host dispatch order and check unchanged logical outputs."""
    rng = np.random.default_rng(622)
    weights = rng.integers(-4, 5, size=(n, k), dtype=np.int8)
    data = rng.integers(
        -8, 9, size=(k, columns) if integer else (k,), dtype=np.int8
    )
    element, result = ("i8", "i32") if integer else ("f32", "f32")
    activation = (
        f"tensor<{k}x{columns}x{element}>" if integer else f"tensor<{k}xf32>"
    )
    weight_type = f"tensor<{n}x{k}x{element}>"
    output = (
        f"tensor<{n}x{columns}x{result}>" if integer else f"tensor<{n}xf32>"
    )
    kernel = "linalg.matmul {cim.int8}" if integer else "linalg.matvec"
    values = weights.tolist() if integer else weights.astype(float).tolist()
    if not integer:
        data = data.astype(np.float32)
    source = f"""
func.func @main(%x: {activation}) -> {output} {{
  %w = arith.constant dense<{values}> : {weight_type}
  %z = arith.constant dense<{"0" if integer else "0.0"}> : {output}
  %r = {kernel} ins(%w, %x : {weight_type}, {activation})
       outs(%z : {output}) -> {output}
  return %r : {output}
}}
"""
    counts = []
    outputs = []
    schedules = []
    bindings = []
    for enabled in (False, True):
        debug = tmp_path / str(enabled)
        artifact = tmp_path / f"test-{enabled}{MODEL_SUFFIX}"
        with _compile_linalg(
            source,
            artifact_path=artifact,
            debug_dir=debug,
            runtime="simulator",
            weight_residency_schedule=enabled,
        ) as session:
            outputs.append(session(data)[0].copy())
            dispatches = session.dispatch_count
        model = embedded_model(artifact)
        transactions = {
            kernel.kernel_name: kernel.transaction for kernel in model.kernels
        }
        bindings.append(
            {
                name: transaction.SerializeToString(deterministic=True)
                for name, transaction in transactions.items()
            }
        )
        text = (debug / "model.packet.mlir").read_text(encoding="utf-8")
        host = text.split("module @__cim_device", maxsplit=1)[0]
        calls = {}
        for line in host.splitlines():
            if match := re.match(r"    func.func (?:private )?@(\w+)\(", line):
                current = match[1]
                calls[current] = []
            elif match := re.search(r"\bcall @(\w+)\(", line):
                calls[current].append(match[1])

        pending = ["main"]
        order = []
        while pending:
            name = pending.pop()
            if name in transactions:
                order.append(name)
            else:
                pending.extend(reversed(calls[name]))
        assert len(order) == dispatches
        if not integer and enabled:
            (helper,) = calls["main"]
            assert helper.startswith("__cim_resident_linear")
            assert calls[helper] == order
        schedules.append(order)
        resident = {}
        writes = 0
        works = 0
        for name in order:
            transaction = transactions[name]
            assert len(transaction.weights) == len(transaction.readbacks)
            for weight in transaction.weights:
                slot = weight.core_slot, weight.macro_slot
                identity = (
                    weight.data_type,
                    tuple(weight.route),
                    tuple(weight.words),
                    weight.exponents,
                )
                writes += resident.get(slot) != identity
                resident[slot] = identity
                works += 1
        counts.append((dispatches, works, writes))

    assert counts[0][:2] == counts[1][:2]
    assert bindings[0] == bindings[1]
    assert counts[0][1] == full_writes
    assert [count[2] for count in counts] == [full_writes, grouped_writes]
    if full_writes == grouped_writes:
        assert schedules[0] == schedules[1]
    else:
        assert schedules[0] != schedules[1]
    assert outputs[0].tobytes() == outputs[1].tobytes()
    if integer:
        np.testing.assert_array_equal(
            outputs[0], weights.astype(np.int32) @ data
        )
        if columns == 3:
            assert schedules[1] == [schedules[0][i] for i in [0, 2, 4, 1, 3, 5]]


@pytest.mark.usefixtures("native_toolchain")
def test_weight_residency_schedule_respects_host_effect() -> None:
    """Leave sign branches around an opaque Host effect in their original order."""
    optimizer = os.environ.get("CIM22_OPT") or str(
        files("cim22")
        / "bin"
        / ("mlir-cim22-opt.exe" if os.name == "nt" else "mlir-cim22-opt")
    )
    source = """
func.func @main(%x: tensor<65xf32>) -> tensor<17xf32> {
  %w = arith.constant dense<1.0> : tensor<17x65xf32>
  %z = arith.constant dense<0.0> : tensor<17xf32>
  %r = linalg.matvec ins(%w, %x : tensor<17x65xf32>, tensor<65xf32>)
       outs(%z : tensor<17xf32>) -> tensor<17xf32>
  return %r : tensor<17xf32>
}
"""
    split = subprocess.run(
        [optimizer, "--cim-bf16-pipeline=stage=split"],
        input=source,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    split = split.replace(
        "\n    func.func @main",
        "\n    func.func private @opaque()\n    func.func @main",
        1,
    )
    split, count = re.subn(
        r"^(.*= call @__cim_kernel_0\(.*)$",
        r"\1\n      func.call @opaque() : () -> ()",
        split,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1
    packet = subprocess.run(
        [optimizer, "--compile-cim-parts=stage=packet"],
        input=split,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    main = packet.split("func.func @main", maxsplit=1)[1].split(
        "\n    }", maxsplit=1
    )[0]
    assert re.findall(r"\bcall @(\w+)", main) == [
        "__cim_kernel_0",
        "opaque",
        "__cim_kernel_1",
        "__cim_kernel_2",
        "__cim_kernel_3",
    ]
