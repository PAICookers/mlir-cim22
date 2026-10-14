# MLIR-CIM22 Tests

- `Dialect`, `Conversion`, `Frontend`, and `Target` mirror production ownership.
- `Integration` exercises compiler pipelines across components.
- `Unit` contains C++ API tests invoked by lit wrappers.
- `python` contains numeric oracles, fixture parsers and artifact checks.
- `Inputs` holds immutable fixtures and is excluded from lit discovery.

Run the compiler suite with `cmake --build build --target check-mlir-cim22`.
LLVM/MLIR 22.1.8 supplies `mlir-runner` and `mlir_c_runner_utils`; CMake provides
their paths. The native packet test requires Linux and Python 3.12 with NumPy.
See [Python verification](../python/README.md#verification) for the separate uv,
pytest and Ruff workflow covering the public Python backend and model `.so` ABI.

| Coverage | Tests |
| --- | --- |
| Conv layouts, groups, channel multiplier, stride/dilation/padding | `normalize-cim-conv*.mlir`, `normalize-cim-conv-numerics.test`: original and normalized CPU execution against a scalar oracle |
| Shared constant batch weights, resource tensors, transposes and Host fallback | `shared-batch-linear.mlir`; Python backend test checks distinct T/B rows and temporal reset |
| Mixed CIM/Host boundaries, casts, bias, reduction and bounded column reuse | `mixed-bf16-pipeline.mlir`, `mixed-bf16-numerics.test`, `prepare-cim-bf16.mlir` |
| Candidate BF16 arithmetic | `BF16SupportTest.cpp` checks finite encodings, sign symmetry, zero/subnormals and saturation; `CIMSoftwareRunnerTest.cpp` checks execution |
| Frame layouts and exports | BF16 fixture replay, reference/multicore export tests, and existing INT8 frame tests |
| Native runtime contracts | `native-bf16.mlir` checks packet compilation, dispatch and malformed split IR; pytest checks AOT publication, relocated reload, output ownership, ABI rejection and backend caching |

Supplier snapshots retain their original bytes and historical arithmetic
outputs. BF16 replay now checks only Cache/Weight_EXP/response layouts, with
three mutations proving rejection; it does not use supplier arithmetic as the
candidate runner's oracle. See [fixture provenance](Inputs/CIM22/bf16/README.md).

Mathematical BF16 references test compiler indexing and precision boundaries;
candidate-runner tests cover the selected shared-exponent approximation.
Neither establishes bit-exact hardware behavior. All executable fixtures live
in this repository; model-specific experiments and local tools are not required.
