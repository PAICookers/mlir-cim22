MLIR-CIM22 test organization

- Dialect, Conversion, Frontend, and Target mirror production ownership.
- Integration contains cross-component compiler pipelines grouped by domain.
- python contains test-only numeric, layout, and model reference code.
- Inputs contains immutable fixtures and manifests excluded from lit discovery.
- Unit contains C++ API/helper tests invoked through lit wrappers.

Do not encode temporary milestone names in directory paths. Record milestone,
source provenance, and evidence level in task records, test comments, or fixture
manifests instead.
