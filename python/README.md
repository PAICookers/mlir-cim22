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
the local **20260923.885** torch-mlir binary wheel, and LLVM/MLIR **22.1.8**.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.11+**
and supply an LLVM/MLIR CMake package. Place the matching daily torch-mlir
wheel in the repository root; its filename is recorded as a relative path in
`python/pyproject.toml`. From the repository root:

```sh
export MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir
export CC=clang CXX=clang++
uv sync --project python --locked --extra compiler
```

`python/pyproject.toml` and `python/uv.lock` define the single Python environment.
`.python-version` selects Python 3.12. scikit-build-core invokes CMake while
installing the package in editable mode; no separate native build is required.
Native assets are resolved through package resources so editable and regular
wheel installations use the same lookup.
The wheel includes `cim22/bin/mlir-cim22-opt` and
`cim22/lib/libcim22_{runtime,compiler}.so`. The package discovers them without
environment overrides. `CIM22_OPT`, `CIM22_RUNTIME_LIBRARY`, and
`CIM22_COMPILER_LIBRARY` remain available for a separate native development build.
The runtime library executes models; the compiler adapter writes and loads
native compilation artifacts. Both are required by the `torch.compile` workflow.

The default configuration is Release, with no ONNX importer, native tests, or
Python dev group. PyTorch uses an
[explicit CPU index](https://docs.astral.sh/uv/guides/integration/pytorch/), and
the compiler extra uses the daily local binary wheel in the repository root;
update its relative filename and version together when a new wheel is available. The lockfile
records resolved versions and local wheel metadata. The build backend does not
use setuptools; PyTorch still declares it as a transitive runtime dependency.
No CUDA packages or
torch-mlir source checkout are used. Compilation also needs a C linker (`cc`, or
`CIM22_CC`). Building requires LLVM/MLIR; a static toolchain build does not need
LLVM/MLIR shared libraries at runtime. Wheel portability depends on the supplied
toolchain and system libraries; a local build is not a manylinux certification.

To build a wheel rather than install the editable package:

```sh
uv build python --wheel
# Optional: retain a native build cache and limit parallelism.
uv build python --wheel \
  -Cbuild-dir=../build/wheel-release -Cbuild.tool-args=-j2
# Optional: include Debug symbols and the ONNX diagnostic importer.
uv build python --wheel \
  -Ccmake.build-type=Debug -Ccmake.define.MLIRCIM22_ENABLE_ONNX=ON
```

Editable installation reads Python edits directly, but native edits need a
rebuild and reinstall. After changing C++, CMake, TableGen or the schema, run
this from the repository root (with `MLIR_DIR`, `CC` and `CXX` set):

```sh
uv sync --project python --locked --extra compiler --group dev \
  --reinstall-package mlir-cim22-runtime \
  -Cbuild-dir=../build/editable -Cbuild.tool-args=-j2 \
  -Ccmake.build-type=Debug
```

Reuse the same build directory and options for subsequent rebuilds. Add
`-Ccmake.define.MLIRCIM22_ENABLE_ONNX=ON` only when that tool is needed. A plain
`uv run` does not rebuild native files; restart the application after reinstalling
so it loads the new shared libraries.
`uv --offline` only controls Python downloads; CMake still needs its native
dependency sources available locally or over the network.

The optional importer is installed as `cim22/bin/mlir-cim22-onnx-import` and is
unnecessary for ordinary PyTorch inference. Native tests are enabled separately
with `MLIRCIM22_INCLUDE_TESTS=ON`; see the [root build guide](../README.md).

Run applications with `uv run --locked --extra compiler python app.py` from
this directory, or use `uv run --project /path/to/mlir-cim22/python --locked
--extra compiler python app.py` elsewhere. Keep `--extra compiler` on `uv run`:
uv synchronizes the requested extras each time. Model-specific dependencies,
such as SpikingJelly, belong to the application environment.

For prebuilt-wheel prerequisites and the distinction between using and building
the compiler, see [Linux user prerequisites](../README.md#linux-user-prerequisites).

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
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    result = compiled(sample_input)  # captures, builds .so, loads, executes
    result2 = compiled(other_input)  # calls the loaded native program
```

Inputs must be static contiguous CPU FP32 tensors. Read-only, unaliased
parameters become frozen weights and must remain fixed. Changes to frozen
weights are not checked; reset Dynamo's compilation cache before recompiling
after a weight change.
Changing shapes may cause Dynamo to build a new specialization. `fullgraph=True`
prevents graph breaks from silently splitting execution. There is no internal
eager fallback. Run under `torch.inference_mode()` or `torch.no_grad()`;
backward computation is not supported.
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
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "software"},
    )
    functional.reset_net(compiled)
    result = compiled(sample_input)
    functional.reset_net(compiled)
    result2 = compiled(other_input)
```

For `step_mode="m"`, inputs remain `[T,B,C,H,W]`. T/B are merged and restored
only around stateless supported contractions. Neuron recurrence stays in the
compiled Host graph. Reset remains necessary when each inference should start
from the initial neuron state; tensor state updates otherwise persist normally.

The backend manages temporary shared libraries by default; loaded code lives
through its session. Optional `options={"artifact_dir": "build/models",
"debug_dir": "build/ir"}` retains one uniquely named graph `.so` per captured
specialization and corresponding diagnostic intermediates. Backend artifacts
use Dynamo's internal graph signature, which may include state outputs.
Dynamo restores the model's Python containers, constants and state assignments.
These graph artifacts are internal diagnostics, not a standalone deployment API.

PyTorch AOTAutograd functionalizes captured in-place operations and supplies
the runtime wrapper that writes updated inputs, buffers and parameters back.
Compilation runs in the standard backend callback; Dynamo manages caching and
wraps compilation failures in `BackendCompilerFailed`.
Mutable parameters and buffers remain runtime inputs. Parameters returned by
identity or view also stay dynamic so the wrapper preserves their aliases.
Only read-only parameters without output aliases are frozen as CIM weights.
Contractions with runtime weight inputs stay on Host, even with fixed shapes
and dtypes: CIM lowering needs the weight values at compile time.
Keep input storage overlap fixed between calls when inputs are mutated.
Changing independent inputs to shared-storage views can silently produce wrong
results due to [PyTorch #139789](https://github.com/pytorch/pytorch/issues/139789);
CIM22 does not add an alias guard for this upstream issue. Run each compiled
model sequentially. Dynamo guards shape, dtype and layout specialization;
the native Session still validates each array before passing its pointer to C++.
Support still depends on the functional operators produced by AOT and lowered
by torch-mlir: synthetic-base updates for distinct tensors sharing storage
currently fail in the local wheel. Compilation errors leave inputs unchanged;
there is no blanket rejection of input, buffer or parameter mutation.
The private compiler calls torch-mlir's `export_and_import()` directly once
on the functional graph, then runs native CIM lowering and linking. The runtime
returns an ordered flat Tensor sequence matching the FX graph. No Python
output-tree schema is stored or reconstructed by CIM22. Static tensor
signatures, compiled transactions and weights remain in the internal `.so`;
native ABI checks remain active. Pure Host graphs and scalar tensors are
supported.

Calls on one low-level Session are serialized. Computed outputs own their
buffers and survive later calls or Session close. PyTorch restores identity and views of
input tensors; distinct views of computed outputs do not preserve shared
storage. Native errors raise exceptions; partial outputs are never returned.
Diagnostic `.so` files are published atomically without overwriting an existing
file. The initial Host lowering uses O0 for bounded
compilation of unrolled time graphs; Host performance is not optimized.

## Linux USB board diagnostics

The compilation option `runtime="usb"` (default) fixes USB dispatch in the model
`.so`. Select `runtime="software"` explicitly for simulation. During development,
both backends write `data_format_version=1`; loaders do not enforce this field.
The schema defaults an absent backend to software (`SOFTWARE=0`, `USB=1`);
unknown backend values are rejected. Recompile artifacts from the previous
development enum numbering. Software models can omit input Cache addresses;
USB models require them regardless of the format version. Device parameters
stay in the live Session; they cannot change the compiled backend.

CMake enables USB by default and requires Linux, pkg-config, and `libusb-1.0`
development files; missing dependencies fail configuration. For a software-only
wheel, use `-Ccmake.define.CIM22_ENABLE_USB=OFF` with uv (or
`-DCIM22_ENABLE_USB=OFF` with CMake). Such builds need no libusb and can load USB
metadata, but reject device configuration explicitly.

The Session serializes device operations and claims the configured USB interface
exclusively through Linux/libusb. A second Session or process claiming that
interface fails with `LIBUSB_ERROR_BUSY`. Closing or reconfiguring the Session
releases its claim; failed reconfiguration leaves USB dispatch unavailable.
Other interfaces on a composite USB device are outside this claim.
Use one compiled specialization per device. Dynamo caches its Session: deleting
the compiled wrapper alone does not release the claim. Drop references and call
`torch.compiler.reset()` when retiring the compiled model before reusing the
device with another specialization.
Transport/protocol faults are sticky: reconfiguration and reopen cannot recover
them. Close the Session, follow the board's FPGA/chip reset procedure, then
create a new Session. Reopening a Host handle does not reset hardware.

```python
from cim22 import UsbConfig

config = UsbConfig(
    vendor_id=vendor_id,
    product_id=product_id,
    bulk_in_endpoint=bulk_in,
    bulk_out_endpoint=bulk_out,
)
compiled = torch.compile(
    model,
    backend="cim22",
    fullgraph=True,
    dynamic=False,
    options={"runtime": "usb", "usb_config": config},
)
# BF16 tensor inference raises until the response format is qualified.
```

For raw hardware diagnostics, retain the internal USB `.so` with `artifact_dir` and inspect it through
`Session`; packet loading is a compiler/JIT test route with software dispatch.

```python
from pathlib import Path
from cim22 import Session

with Session(Path("graph-identifier.so")) as session:
    session.configure_usb(config)
    raw_flits = session.capture_usb(0, transaction_input_bits)
```

`transaction_input_bits` is an aligned, contiguous NumPy `uint16` array shaped
`(dynamic_bindings, 64)`, containing already lowered BF16 input rows in the
embedded transaction's binding order. It is separate from the model's FP32
input signature: capture does not run the Host graph. The returned `uint64`
array has shape `(readbacks, 7)` in readback binding order, with one header and
six unchanged body flits per row.

Python compilation currently lowers CIM regions in BF16 mode. USB tensor
output remains disabled until BF16 response packing is confirmed
(CTQ-037/HWQ-LAYOUT-003); use raw capture for diagnostics. The C++ INT8 typed
runner is covered by fake-device tests. Neither those tests nor the capture API
establish physical-board correctness.

The transport follows framehandler's 12-byte records, 3072-byte TX chunks,
big-endian words and default link 2/channel 3. The runner uses the framehandler
20 ms startup/2 ms Once delays. BF16 Once-to-idle sequencing follows the supplier
testbench provisionally for raw diagnostics. These defaults were not validated
on a board in this change.

## Verification

```sh
# From the repository root, with MLIR_DIR and compiler variables configured:
uv sync --project python --locked --extra compiler --group dev
uv run --project python --locked --extra compiler --group dev pytest python/tests
uv lock --project python --check

# Fast USB configuration checks without running compilation tests:
uv run --project python --locked --extra compiler --group dev \
  pytest python/tests -m 'not integration'

# Use the host-installed Ruff; generated bindings are excluded by configuration.
ruff check --config python/pyproject.toml python test/python test/lit.cfg.py
ruff format --check --config python/pyproject.toml python test/python test/lit.cfg.py
# Apply formatting while editing:
ruff format --config python/pyproject.toml python test/python test/lit.cfg.py
```

To test a separate native build after installing the package, set
`CIM22_OPT` and `CIM22_RUNTIME_LIBRARY` to that build before running pytest.

Use host `clang-format` for project-owned C++ under `include/`, `lib/`, `tools/`,
and `test/Unit/`; leave generated and third-party sources unchanged.

Tests cover the registered backend, static T/B and reset, shape specialization,
actual candidate-model CIM dispatch, input/view mutation and buffer writeback,
Host/scalar/nested outputs, invalid ABI
and malformed model data, output lifetimes and atomic publication. Integration tests
fail explicitly if native dependencies are missing; they are not silently
skipped. Tests share compiled fixtures and use isolated temporary directories.
Generic regressions are separate from local model-specific accuracy investigations.
Ruff uses `python/pyproject.toml` for the shared lint rules and 80-column
formatting, selecting Python 3.12 for this package and its tests.


### Protobuf code generation

CMake builds Protobuf/protoc **31.1** directly, independently of ONNX.
Generated C++ lives in the build directory; generated Python lives in
`tests/_generated` for
model-format and interoperability tests. The runtime wheel needs neither these
bindings nor the Python `protobuf` dependency, which belongs to the `dev` group.
After editing the schema:

```bash
cmake --build build/debug --target update-cim22-protobuf
cmake --build build/debug --target check-cim22-protobuf
```

Normal builds never overwrite test bindings. `check-mlir-cim22` checks
binding freshness. The pinned protoc and C++ Protobuf runtime are built
together and must match exactly. The model `.so` only links the execution runtime,
not another Protobuf library.

New descriptive fields may be ignored by older readers. Once compatibility
checks are enabled, incompatible structural or numeric changes must upgrade
`data_format_version`. Device handles, ports, timeouts and live buffer pointers
remain outside the persisted model.
