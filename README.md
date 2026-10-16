# mlir-cim22

An early-stage, out-of-tree MLIR compiler project for CIM22. The current
implementation supports software simulation and Linux USB diagnostics; BF16
hardware tensor output is not yet qualified.

## Current scope

- BF16 MatVec/MatMul tiling, static Linear and shared-weight `batch_matmul`
  normalization, including resource-backed and transposed constant weights
- Nine named 2D convolution forms: ordinary, grouped and depthwise, including
  channel multiplier, static batch, stride and dilation
- Mixed Linalg compilation: eligible FP32 contractions use FP32/BF16 casts and
  CIM VMM; remaining operations lower to LLVM CPU code
- Linux `torch.compile(backend="cim22")` inference, with embedded weights and
  binary metadata in internal native model artifacts
- Candidate BF16 software execution through the native runtime
- Quantized ONNX import, INT8 weight layout and typed CIM frame planning

## Build and install

The only Python project and lockfile live in `python/`. Its scikit-build-core
backend invokes CMake and installs one wheel containing the `cim22` package,
`cim22/bin/mlir-cim22-opt`, and `cim22/lib/libcim22_{runtime,compiler}.so`.
The compiler discovers these installed tools automatically. `CIM22_OPT`,
`CIM22_RUNTIME_LIBRARY`, and `CIM22_COMPILER_LIBRARY` can select a separate native
build during development.

Build prerequisites:

- LLVM/MLIR **22.1.8**, supplied as an existing CMake package
- CMake **3.28+**, Ninja, and a C++17 compiler
- Python **3.12** and [uv](https://docs.astral.sh/uv/) **0.11+**
- Linux `libusb-1.0` development files for USB support

From the repository root:

```sh
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
export CC=clang CXX=clang++
uv build python --wheel
uv sync --project python --locked --extra compiler
```

The compiler extra uses a local daily torch-mlir wheel in the repository root,
referenced by a relative filename in `python/pyproject.toml`. Update that filename
and version when replacing the wheel; local wheels are not committed.
The default build is Release, excludes tests and the ONNX importer, and installs
no Python development group. To reuse a build directory and bound parallelism:

```sh
uv build python --wheel \
  -Cbuild-dir=../build/wheel-release -Cbuild.tool-args=-j2
```

CMake downloads Protobuf **31.1** directly. The optional ONNX importer uses ONNX
**1.21.0** and shares that Protobuf toolchain. Offline builds can set the standard
`FETCHCONTENT_SOURCE_DIR_PROTOBUF`, `FETCHCONTENT_SOURCE_DIR_ABSL`, and, when
needed, `FETCHCONTENT_SOURCE_DIR_ONNX` cache variables. Protobuf and its
transitive dependencies are built from pinned sources and linked statically;
no system Protobuf/ONNX package selection is needed.

USB support is enabled by default and requires Linux, pkg-config, and libusb
development files; configuration fails if they are unavailable. Add
`-Ccmake.define.CIM22_ENABLE_USB=OFF` for a software-only wheel without libusb.
The current native build targets Linux x86-64. A successful local wheel build
does not establish manylinux compliance or portability across LLVM installs.

## Linux user prerequisites

Installing a prebuilt wheel does not require CMake, Ninja, LLVM/MLIR development
files, protoc, ONNX or Python Protobuf. The wheel includes the optimizer and both
native libraries. For the supported `torch.compile` workflow, users need:

- Linux x86-64 and Python 3.12, with a system ABI compatible with the wheel
- NumPy and CPU PyTorch 2.14.0 (package dependencies), plus the matching
  torch-mlir wheel (`compiler` extra; currently a local development wheel)
- A working `cc` compiler driver and system linker for generating model `.so`
  files on first compilation, including the standard C development files
- System C/C++ runtime libraries, zlib and zstd; the USB-enabled wheel also
  needs libusb and its platform dependencies (libudev on this build)

USB access additionally requires device permissions, for example a suitable
udev rule. A software-only wheel built with `CIM22_ENABLE_USB=OFF` does not need
libusb. Model execution after compilation does not rerun CMake or protoc.
CIM22 has no standalone Python deployment API; ordinary application use still
follows `torch.compile`.

Building from source additionally requires the build prerequisites above;
scikit-build-core is installed in the isolated build environment. CMake fetches
and builds the pinned native dependencies. Only developers running tests need
the Python `dev` group and optional native test tools. Debug symbols alone do
not require ONNX or test dependencies.

## Development and tests

Opt into Debug and the ONNX diagnostic tool when needed:

```sh
uv build python --wheel \
  -Ccmake.build-type=Debug -Ccmake.define.MLIRCIM22_ENABLE_ONNX=ON
```

This adds `cim22/bin/mlir-cim22-onnx-import`; ordinary PyTorch inference does not
need it. Native regression tests use a separate CMake build and the same Python
environment:

```sh
uv sync --project python --locked --extra compiler --group dev --no-install-project
cmake -G Ninja -S . -B build/debug \
  -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR="$MLIR_DIR" \
  -DMLIRCIM22_INCLUDE_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/python/.venv/bin/python"
cmake --build build/debug --target check-mlir-cim22 -j2
```

The native suite additionally needs the LLVM `FileCheck`, `split-file`,
`mlir-runner`, and `mlir_c_runner_utils` targets. In a fresh build directory,
enabling tests also enables the ONNX importer; when reusing a directory with
ONNX disabled, also pass `-DMLIRCIM22_ENABLE_ONNX=ON`. See
[Python setup and usage](python/README.md) for inference, pytest, and formatting
commands. Use the host-installed `ruff` and
`clang-format`; neither is a Python package dependency.

## Execution contract and limits

The Python backend accepts static contiguous CPU FP32 inputs and freezes
read-only model parameters for CIM lowering. Runtime weights remain on Host.
Eligible contractions use 16x64 BF16 VMM tiles; cross-K results are
accumulated on the Host. Static time/batch dimensions are flattened and restored
around stateless contractions. Reset state before each complete SNN inference;
tensor state updates persist across calls. Training and dynamic shapes are
unsupported.

The selected BF16 software model uses shared exponents, magnitude truncation
before sign, INT21 accumulation, zero-preserving reconstruction, gradual
underflow and finite saturation. It is approximate and not a bit-exact hardware
model. Historical board zero/cancellation observations differ from it.
`bf16-reference=true` enables diagnostic RTL comparison exports; those exports
and software tests do not establish board equivalence.

Linux USB transport supports raw BF16 diagnostics; BF16 tensor output still
requires confirmation of the hardware response layout. General ONNX operator
coverage is incomplete.
Native model artifacts currently target Linux x86-64; Host code generation uses
O0 to bound compilation cost for large statically unrolled graphs.
