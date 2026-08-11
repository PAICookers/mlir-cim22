# mlir-cim22

An early-stage, out-of-tree MLIR compiler project for CIM22. The current
implementation is software-only and is not a hardware deployment tool.

## Current scope

- `cim.vmm` logical lowering
- Quantized ONNX `MatMulInteger` import
- Provisional typed CIM frame planning
- INT8 weight-layout helper

## Prerequisites

- LLVM/MLIR 22.1.8
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

## Not implemented

There is no general ONNX operator coverage, raw flit codec, board runtime, or
deployable artifact flow yet.

## License

This project is licensed under [GPL-3.0](LICENSE).
