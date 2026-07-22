# mlir-cim22

An early-stage, out-of-tree MLIR compiler project for CIM22. The current
implementation is software-only and is not a hardware deployment tool.

## Current scope

- `cim.vmm` logical lowering
- Provisional typed CIM frame planning
- INT8 weight-layout helper

## Prerequisites

- LLVM/MLIR 22.1.8
- CMake 3.20 or newer
- C++17 compiler
- Ninja

## Build and test

```sh
cmake -G Ninja -S . -B build -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
cmake --build build --target mlir-cim22-opt
cmake --build build --target check-mlir-cim22
build/bin/mlir-cim22-opt --help
```

## Not implemented

There is no ONNX importer, raw flit codec, board runtime, or deployable
artifact flow yet.

## License

This project is licensed under [GPL-3.0](LICENSE).
