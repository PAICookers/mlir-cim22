# MLIR-CIM22 Test Organization

- `Dialect`, `Conversion`, `Frontend`, and `Target` mirror production ownership.
- `Integration` contains cross-component compiler pipelines. `ONNX` includes the
  importer; target-stage directories such as ExecutionPlan, Mapping, and
  Schedule start from project MLIR.
- `python` contains test-only numeric, layout, and model reference code.
- `Inputs` contains immutable fixtures and manifests excluded from lit discovery.
- `Unit` contains C++ API/helper tests invoked through lit wrappers.

Do not encode temporary milestone names in directory paths. Record milestone,
source provenance, and evidence level in task records, test comments, or fixture
manifests instead.
