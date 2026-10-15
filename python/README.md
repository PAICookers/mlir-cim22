# CIM22 Python inference backend (Linux)

The backend builds a native Linux model `.so`. MLIR selects supported
contractions for FP32 → BF16 → FP32 CIM execution and compiles the remaining
graph to CPU code. `CIMSoftwareRunner` uses the selected candidate BF16 model:
per-lane shared weight exponents, magnitude truncation before sign, INT21 MAC,
zero-preserving reconstruction, gradual underflow and finite saturation.
Shared-exponent loss remains; this is not conventional BF16 matmul or a
bit-exact hardware model. Previously confirmed board demo zero/cancellation
outputs differ; historical supplier/board evidence is retained separately.

## One-time environment setup

Tested: Linux x86-64 (including WSL), Python **3.12**, CPU PyTorch **2.14.0**,
the **20260930.892** torch-mlir binary wheel, and LLVM/MLIR **22.1.8**.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.13+**
and build the native targets from the repository root, then create the Python
environment:

```sh
cmake --build build --target mlir-cim22-opt cim22_compiler
export CIM22_OPT=/absolute/path/to/build/bin/mlir-cim22-opt
export CIM22_RUNTIME_LIBRARY=/absolute/path/to/build/lib/libcim22_runtime.so
cd python
uv sync --locked --extra compiler
```

`python/pyproject.toml` and `python/uv.lock` define this environment separately
from the repository-root C++/ONNX test environment. `.python-version` selects
Python 3.12; uv creates `.venv/` and installs this package in editable mode.
The default dev group supplies pytest and Ruff. PyTorch uses an
[explicit CPU index](https://docs.astral.sh/uv/guides/integration/pytorch/), and
the compiler extra uses a pinned binary wheel from
[llvm/torch-mlir-release](https://github.com/llvm/torch-mlir-release/releases).
The lockfile records resolved versions and download hashes. No CUDA packages
or torch-mlir source checkout are used. Compilation also needs a C linker
(`cc`, or `CIM22_CC`). Building the native compiler requires LLVM/MLIR;
its dependencies follow the configured LLVM/MLIR CMake targets, so static
toolchain builds do not require LLVM/MLIR shared libraries. The compiler
adapter is `libcim22_compiler.so` beside the runtime, or select it with
`CIM22_COMPILER_LIBRARY`.

Run applications with `uv run --locked --extra compiler python app.py` from
this directory, or use `uv run --project /path/to/mlir-cim22/python --locked
--extra compiler python app.py` elsewhere. Keep `--extra compiler` on `uv run`:
uv synchronizes the requested extras each time. Model-specific dependencies,
such as SpikingJelly, belong to the application environment.

## Model user workflow

Installing the package registers `cim22` through the `torch_dynamo_backends`
entry point, following the [PyTorch 2.14 custom backend contract](https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_custom_backends.html).
It receives an FX graph and example inputs, and returns a callable with that
graph's argument/result contract. MLIR owns CIM eligibility decisions.

```python
import torch

model = model.cpu().eval()
with torch.inference_mode():
    compiled = torch.compile(
        model, backend="cim22", fullgraph=True, dynamic=False
    )
    result = compiled(sample_input)  # captures, builds .so, loads, executes
    result2 = compiled(other_input)  # calls the loaded native program
```

Inputs must be static contiguous CPU FP32 tensors without gradients. Weights
must remain fixed. Normal in-place parameter updates are rejected; mutation
through `.data` after compilation is outside the frozen-parameter contract.
Changing shapes may cause Dynamo to build a new specialization. `fullgraph=True`
prevents graph breaks from silently splitting execution. There is no internal
eager fallback. Prefer `torch.inference_mode()` for every inference; `no_grad`
remains supported. Create/load the model outside the inference context so its
parameters retain the version counters used by the fixed-weight guards.
Keep input preparation consistent: switching between ordinary tensors and
tensors created inside `inference_mode` can cause a new Dynamo specialization.

For SpikingJelly **0.0.0.0.14**, set `PYTORCH_JIT=0` before importing it in this
wheel environment. Keep original neuron classes and reset before each complete
inference, including after any eager warm-up:

```python
from spikingjelly.activation_based import functional

with torch.inference_mode():
    functional.reset_net(model)
    compiled = torch.compile(
        model, backend="cim22", fullgraph=True, dynamic=False
    )
    functional.reset_net(compiled)
    result = compiled(sample_input)
    functional.reset_net(compiled)
    result2 = compiled(other_input)
```

For `step_mode="m"`, inputs remain `[T,B,C,H,W]`. T/B are merged and restored
only around stateless supported contractions. Neuron recurrence stays in the
compiled Host graph. Cross-call continuation without reset is unsupported.

The backend manages temporary shared libraries by default; loaded code lives
through its session. Optional `options={"artifact_dir": "build/models",
"debug_dir": "build/ir"}` retains one uniquely named graph `.so` per captured
specialization and corresponding diagnostic intermediates. Backend artifacts
use Dynamo's internal graph signature, which may include state outputs.
Dynamo handles those outputs and Python state assignments. Use explicit export
below for a standalone model signature.

## Save once, load without the compiler

```python
from pathlib import Path
from cim22 import compile_exported, load_compiled

# Export after resetting model state; use flat positional tensor inputs.
with torch.inference_mode():
    program = torch.export.export(model, (sample_input,), strict=True)
    with compile_exported(program, artifact_path=Path("model.so")) as runner:
        result = runner(sample_input)

# This can run in another process with only the execution runtime installed.
with torch.inference_mode(), load_compiled(Path("model.so")) as runner:
    result = runner(sample_input)
```

Only the model `.so` is a model-specific runtime artifact. It embeds CPU code,
compiled CIM transactions, weights, ordered static tensor signatures, and
Protobuf output metadata (`schema/cim22_model.proto`). Loading needs no JSON,
MLIR, `.pt2`, compiler subprocess or JIT.
Outputs support tensors, tuples, lists, and string-keyed dictionaries, within
the Protobuf model parser depth limit checked before compilation. The loader checks native ABI version **1**, Protobuf data format version **3** and
numeric contract version **2** separately. Existing custom binary/marshal models
are rejected with a recompile diagnostic. Recompile old model libraries.
The payload is parsed once at load into owned runtime data; inference continues
to use native tensor buffers. Scalar tensors and pure Host models are supported.
Load trusted models.

Deploy `model.so` with matching `libcim22_runtime.so`, the Python runtime
package, PyTorch/NumPy and
normal Linux C/C++ libraries. Python uses `protobuf==6.31.1`; inference needs
no `protoc`, and the runtime statically links Protobuf lite. Set
`CIM22_RUNTIME_LIBRARY`, or package the runtime
under `cim22/lib/`. The deployment process needs neither torch-mlir nor
`mlir-cim22-opt`, `libcim22_compiler.so`, or MLIR/LLVM compiler shared libraries.
Models use the build host's Linux architecture with generic CPU codegen.
Cross-compilation and arbitrary glibc/runtime compatibility are not promised.
USB execution is not implemented.

For a runtime-only environment using this checkout, run `uv sync --locked
--no-dev` from `python/`, then `uv run --locked --no-dev python app.py`.
Omit `--extra compiler`; torch-mlir and the development tools are not installed.

`artifact_path` must be a new `.so` filename. A successful build/load publishes
it atomically without overwriting files. Optional `debug_dir` saves
pt2/Linalg/packet/object intermediates; inference never reads them. The initial
AOT implementation uses O0 for bounded compilation of unrolled time graphs;
Host performance is not optimized.

An explicitly exported reset-state program begins each invocation from the
captured initial state and preserves recurrence within that invocation. It does
not synchronize an original Python model's membrane state. Calls on one session
are serialized. Outputs own their buffers and survive later calls or session
close. Native runner failures raise errors instead of returning partial outputs.

`cim22.Session(Path("model.so"))` provides the flat NumPy FP32 interface.
Raw packet loading through `Session` remains a diagnostic/test route requiring
the compiler adapter/JIT; it is not the deployment path.

## Verification

```sh
# From python/, with the native library paths above configured:
uv run --locked --extra compiler pytest
uv run --locked --extra compiler ruff check .
uv run --locked --extra compiler ruff format --check .
uv lock --check

# Fast schema rejection tests without building the native compiler:
uv run --locked pytest -m 'not integration'

# Apply formatting while editing:
uv run --locked --extra compiler ruff format .
```

Tests cover the registered backend, static T/B and reset, fixed-weight guards,
actual candidate-model CIM dispatch, Host/scalar/nested outputs, invalid ABI
and schema data, output lifetimes and atomic publication. Integration tests
fail explicitly if native dependencies are missing; they are not silently
skipped. Tests share compiled fixtures and use isolated temporary directories.
A fresh-process test relocates only model and runtime `.so` files, disables
frontend imports, compiler paths, subprocesses and
runtime JSON reads, and checks that compiler/MLIR DSOs were not loaded. Generic
regressions are separate from local model-specific accuracy investigations.
Ruff inherits the repository lint rules and 80-column formatting, selecting
Python 3.12 for this package and its tests.


### Protobuf code generation

Default CMake builds reuse ONNX's pinned Protobuf/protoc **31.1**. Generated C++
lives in the build directory; generated Python ships inside the runtime wheel.
Installation and inference do not regenerate bindings. After editing the schema:

```bash
cmake --build build --target update-cim22-protobuf
cmake --build build --target check-cim22-protobuf
```

Normal builds never overwrite packaged Python source. `check-mlir-cim22` checks
binding freshness. System ONNX mode requires a matching Protobuf **31.1.0** CMake
config exporting `protobuf::protoc` and static `protobuf::libprotobuf-lite`;
configure fails explicitly if either is missing or incompatible. C++ generation
and runtime must match exactly. The model `.so` only links the execution runtime,
not another Protobuf library.

New descriptive fields may be ignored by older readers. Changes to execution
semantics must upgrade the artifact version. Device handles, ports, timeouts and
live buffer pointers remain outside the persisted model.
