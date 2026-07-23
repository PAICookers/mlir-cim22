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
- Protobuf 4.25.1 or newer, including `protoc` and CMake config targets
- Python 3 interpreter, used by the upstream ONNX protobuf generator
- Python test packages declared in `requirements-test.txt`

The default configuration downloads the pinned ONNX v1.21.0 source archive,
verifies its SHA-256, and builds only the required `onnx_proto` target. To use
an installed ONNX v1.21.0 package instead, configure with
`-DMLIRCIM22_USE_SYSTEM_ONNX=ON -DONNX_DIR=/path/to/lib/cmake/ONNX`.

## Build and test

```sh
python3 -m pip install -r requirements-test.txt
cmake -G Ninja -S . -B build -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build --target mlir-cim22-opt
cmake --build build --target mlir-cim22-onnx-import
cmake --build build --target check-mlir-cim22
build/bin/mlir-cim22-opt --help
```

## Not implemented

There is no general ONNX operator coverage, raw flit codec, board runtime, or
deployable artifact flow yet.

## License

This project is licensed under [GPL-3.0](LICENSE).
