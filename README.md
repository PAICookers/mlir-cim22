# mlir-cim22

An early-stage, out-of-tree MLIR compiler project for CIM22. The current
implementation is software-only and is not a hardware deployment tool.

## Current scope

- BF16 MatVec/MatMul tiling, static Linear and shared-weight `batch_matmul`
  normalization, including resource-backed and transposed constant weights
- Nine named 2D convolution forms: ordinary, grouped and depthwise, including
  channel multiplier, static batch, stride and dilation
- Mixed Linalg compilation: eligible FP32 contractions use FP32/BF16 casts and
  CIM VMM; remaining operations lower to LLVM CPU code
- Linux `torch.compile(backend="cim22")` inference and standalone model `.so`
  generation/loading, with embedded weights and binary metadata
- Candidate BF16 software execution through the native runtime
- Quantized ONNX import, INT8 weight layout and typed CIM frame planning

## Prerequisites

- LLVM/MLIR 22.1.8, including `mlir-runner` and `mlir_c_runner_utils` for tests
- CMake 3.28 or newer
- C++17 compiler
- Ninja
- uv
- Python 3 interpreter, used by the upstream ONNX protobuf generator
- Python test packages locked by `pyproject.toml` and `uv.lock`

The default configuration downloads the pinned ONNX v1.21.0 source archive,
verifies its SHA-256, and lets ONNX build its matching Protobuf toolchain and
transitive dependencies. To use an installed ONNX v1.21.0 package instead,
configure with
`-DMLIRCIM22_USE_SYSTEM_ONNX=ON -DONNX_DIR=/path/to/lib/cmake/ONNX`.
The installed package is responsible for exporting its complete transitive
link interface.

For an offline source build, pre-populate CMake's FetchContent cache or set the
standard `FETCHCONTENT_SOURCE_DIR_ONNX`, `FETCHCONTENT_SOURCE_DIR_PROTOBUF`, and
`FETCHCONTENT_SOURCE_DIR_ABSL` cache variables.

After changing dependency profiles in an existing build directory, clear the
cached ONNX package hint once and reconfigure:

```sh
cmake -S . -B build -UONNX_DIR \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
```

## Build and test

```sh
uv sync --locked
uv run --locked cmake -G Ninja -S . -B build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --target mlir-cim22-opt
cmake --build build --target mlir-cim22-onnx-import
cmake --build build --target check-mlir-cim22
build/bin/mlir-cim22-opt --help
uv lock --check
```

## Python inference backend

The Linux Python package has its own Python 3.12 uv environment and lockfile
under `python/`. See [Python setup and usage](python/README.md) for the pinned
CPU PyTorch/torch-mlir wheels, `torch.compile(backend="cim22")`, native `.so`
loading, and pytest/Ruff commands. The root uv environment above remains for
the compiler's ONNX/lit tests.

## Execution contract and limits

The Python backend requires frozen weights and static contiguous CPU FP32
inputs. Eligible contractions use 16x64 BF16 VMM tiles; cross-K results are
accumulated on the Host. Static time/batch dimensions are flattened and restored
around stateless contractions. Reset state before each complete SNN inference;
cross-call state continuation, training and dynamic shapes are unsupported.

The selected BF16 software model uses shared exponents, magnitude truncation
before sign, INT21 accumulation, zero-preserving reconstruction, gradual
underflow and finite saturation. It is approximate and not a bit-exact hardware
model. Historical board zero/cancellation observations differ from it.
`bf16-reference=true` enables diagnostic RTL comparison exports; those exports
and software tests do not establish board equivalence.

USB/board transport and general ONNX operator coverage remain unimplemented.
Native model artifacts currently target Linux x86-64; Host code generation uses
O0 to bound compilation cost for large statically unrolled graphs.

## License

This project is licensed under [GPL-3.0](LICENSE).
