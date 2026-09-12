# MLIR-CIM22 Test Organization

- `Dialect`, `Conversion`, `Frontend`, and `Target` mirror production ownership.
- `Integration` contains cross-component compiler pipelines. `ONNX` includes the
  importer; target-stage directories such as ExecutionPlan, Mapping, and
  Schedule start from project MLIR.
- `python` contains test-only numeric, layout, and model reference code.
- `Inputs` contains immutable fixtures and manifests excluded from lit discovery.
- `Unit` contains C++ API/helper tests invoked through lit wrappers.

Test-only names describe their role:

- `reference` implements evidenced numeric or layout behavior.
- `verify` checks compiler IR or artifacts, optionally using a reference model.
- `replay` runs immutable supplier fixtures through a reference model.
- `runner` is reserved for a future executable-artifact consumer.

Do not encode temporary milestone names in directory paths. Record milestone,
source provenance, and evidence level in task records, test comments, or fixture
manifests instead.

BF16 supplier replay checks all four bundle/Macro cases against production C++
prealignment, reconstruction, Cache/Weight_EXP helpers and software runner.
Expected stages are fixed supplier files; the Python adapter only parses
them. See [snapshot provenance and boundaries](Inputs/CIM22/bf16/README.md).
The fixed response/padding and signed exponent-gap tests are in
`Unit/Support/BF16SupportTest.cpp`. Neither suite establishes RTL execution or
raw BF16 weight preprocessing equivalence.
