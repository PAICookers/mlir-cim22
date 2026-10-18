# 源码构建

本指南面向 Linux x86-64 开发者。两条构建路径共用同一个 CMake 工程：

| 目标             | 方式                   | 结果                                             |
| ---------------- | ---------------------- | ------------------------------------------------ |
| 编译器本体       | CMake/Ninja            | 原生优化器、compiler/runtime 库，可选 ONNX 工具  |
| PyTorch 编译后端 | uv + scikit-build-core | 包含上述原生组件的 Python wheel 或 editable 安装 |

仅使用预编译包时，直接阅读 [Python 安装指南](python/README.md#安装)。以下命令除另有说明外，均在**仓库根目录**执行。

## 环境与依赖

需要 Git、C++17 编译器、**CMake >=3.28**、Ninja，以及默认 USB 构建需要的 `pkg-config` 和 libusb 开发文件。Python 后端和测试还需 Python 3.12 与 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

例如 Ubuntu 24.04：

```sh
sudo apt-get update
sudo apt-get install build-essential clang cmake ninja-build git \
  pkg-config libusb-1.0-0-dev python3.12 python3.12-venv
cmake --version
uv --version

git clone https://github.com/PAICookers/mlir-cim22.git
cd mlir-cim22
export CC=clang CXX=clang++
```

CMake 会下载并构建固定的 **Protobuf/protoc 31.1**；启用 ONNX 时还会构建 **ONNX 1.21.0**。无需安装系统 Protobuf 或 ONNX C++ 包。首次构建需要网络、足够的磁盘和内存；下文用两个并行任务限制资源占用，可按机器配置调整。

## 准备 LLVM/MLIR

已有兼容工具链时，只需设置其 `lib/cmake/mlir` 为 `MLIR_DIR`，可跳过本节构建。本项目要求 LLVM/MLIR 主版本 **22**；验证基线是 **22.1.8**，对应提交 `ca7933e47d3a3451d81e72ac174dcb5aa28b59d1`。

以下在仓库旁建立工具链源码与构建目录：

```sh
git init ../llvm-project
git -C ../llvm-project remote add origin https://github.com/llvm/llvm-project.git
git -C ../llvm-project fetch --depth=1 origin ca7933e47d3a3451d81e72ac174dcb5aa28b59d1
git -C ../llvm-project checkout --detach FETCH_HEAD

cmake -G Ninja -S ../llvm-project/llvm -B ../llvm-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=X86 \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF -DLLVM_ENABLE_PIC=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
  -DLLVM_INCLUDE_TESTS=ON -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF
cmake --build ../llvm-build --parallel 2
export MLIR_DIR="$(cd ../llvm-build && pwd)/lib/cmake/mlir"
```

静态 PIC 库供本项目链接；torch-mlir 使用自身的 Python 绑定，无需 LLVM Python 绑定。保留工具链源码和构建目录，后续打开新终端时重新设置 `MLIR_DIR`、`CC`、`CXX`。

## 编译器本体

此路径不需要安装 PyTorch 或 torch-mlir：

```sh
cmake -G Ninja -S . -B build/native \
  -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR="$MLIR_DIR"
cmake --build build/native --parallel 2
build/native/bin/mlir-cim22-opt --help
```

优化器位于 `build/native/bin/`，compiler/runtime 库位于 `build/native/lib/`。例如，将仓库中的混合计算测试转换为 packet 阶段的 MLIR：

```sh
build/native/bin/mlir-cim22-opt \
  test/Integration/ExecutionPlan/mixed-bf16-pipeline.mlir \
  --cim-bf16-pipeline='stage=packet max-kernel-columns=1' \
  -o build/native/model.packet.mlir
```

该命令用于检查编译 IR；PyTorch 后端还会生成目标文件、链接并加载模型。需要受限的 ONNX 导入器时，在配置命令中添加 `-DMLIRCIM22_ENABLE_ONNX=ON`，构建后查看参数：

```sh
build/native/bin/mlir-cim22-onnx-import --help
```

## PyTorch 编译后端

### 构建 wheel

发布包含 Web 界面的 wheel 前，先按 [Web 构建说明](web/README.md#从源码构建)
生成静态页面。该产物随 wheel 和 sdist 分发，终端用户无需 Node.js。

设置好上述工具链变量后执行：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 uv build --python 3.12 python \
  --wheel --out-dir wheelhouse
```

scikit-build-core 调用根 CMake 工程，默认构建 **Release、USB 开启、测试和 ONNX 关闭**。构建 wheel 本身不需要安装 PyTorch 或 torch-mlir。产物为 `wheelhouse/mlir_cim22-1.0.0-py3-none-linux_x86_64.whl`，包含：

```text
cim22/
  *.py
  bin/mlir-cim22-opt
  lib/libcim22_compiler.so
  lib/libcim22_runtime.so
  web/static/index.html
  web/static/assets/
```

按 [Python 安装指南](python/README.md#安装)安装该 wheel 和 torch-mlir，再运行自检脚本。普通用户仍需系统 `cc` 和链接开发环境来编译模型。本机构建只保证面向本机 ABI；可移植 manylinux 分发还需独立构建和验收。

### Editable 开发安装

从 [torch-mlir releases](https://github.com/llvm/torch-mlir/releases) 下载与当前后端兼容的 wheel，放在 **`python/` 目录**，与 `tool.uv.sources` 的路径一致：

```text
torch_mlir-<compatible-release>-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

然后执行：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 uv sync --project python --python 3.12 \
  --locked --extra compiler --no-dev
uv run --project python --no-sync python python/examples/torch_compile.py
```

该命令创建 `python/.venv`，安装 CPU PyTorch、torch-mlir 和本项目的 editable 版本，并构建原生组件。`compiler` extra 选择 torch-mlir；`--no-dev` 只跳过默认开发依赖。预编译用户的 `.venv` 与这里的 `python/.venv` 是不同环境。

元数据允许 Python `>=3.12`、Torch `>=2.14,<3` 和任意 torch-mlir 版本；当前回归使用与后端匹配的 Release wheel。更新依赖后需要重新验证后端。

### 构建目录与配置变体

`build-dir = "build"` 相对 `python/pyproject.toml`，所以 CMake 缓存、目标文件、原生库及依赖构建结果保存在 `python/build/`。最终 wheel 由 `--out-dir` 决定。重复构建可复用未变化的结果；默认目录同时用于 wheel 和 editable 构建。

不同工具链、配置或并行任务应使用不同目录。例如 Debug wheel：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 uv build --python 3.12 python --wheel \
  --out-dir wheelhouse -Cbuild-dir=build/debug -Ccmake.build-type=Debug
```

| 需求         | CMake 配置                     | uv 构建参数                                 |
| ------------ | ------------------------------ | ------------------------------------------- |
| 仅软件运行时 | `-DCIM22_ENABLE_USB=OFF`     | `-Ccmake.define.CIM22_ENABLE_USB=OFF`     |
| ONNX 导入器  | `-DMLIRCIM22_ENABLE_ONNX=ON` | `-Ccmake.define.MLIRCIM22_ENABLE_ONNX=ON` |

软件构建仍需显式选择 `runtime="software"`，不会将 USB 请求静默改为软件执行。

## 测试与开发

Python 代码修改在 editable 安装中直接生效。修改 C++、TableGen、schema 或 CMake 后，必须重新构建安装；单纯 `uv run` 不会检测这些文件的变化：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 uv sync --project python --python 3.12 \
  --locked --extra compiler --group dev --reinstall-package mlir-cim22
uv run --directory python --no-sync pytest
```

保留首次构建时选择的 `-Cbuild-dir`、Debug/USB/ONNX 等参数。完成重建后重启应用，使其加载新的原生库；已有模型需在新进程中重新编译。

原生 lit 测试另用一个目录，依赖锁定环境中的 NumPy、ONNX 和 Protobuf：

```sh
cmake --build ../llvm-build --target \
  FileCheck split-file mlir-runner mlir_c_runner_utils --parallel 2
cmake -G Ninja -S . -B build/test \
  -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR="$MLIR_DIR" \
  -DMLIRCIM22_INCLUDE_TESTS=ON -DMLIRCIM22_ENABLE_ONNX=ON \
  -DPython3_EXECUTABLE="$PWD/python/.venv/bin/python"
cmake --build build/test --target check-mlir-cim22 --parallel 2
```

自备工具链时应确认同样提供上述 LLVM 测试工具。测试说明见 [test/README](test/README.md)。Python 测试使用安装包内的原生组件；需要测试独立 CMake 构建时，可设置 `CIM22_OPT`、`CIM22_RUNTIME_LIBRARY` 和 `CIM22_COMPILER_LIBRARY` 为对应产物路径。

schema 修改后，用同一原生测试构建更新并检查 Python 测试绑定：

```sh
cmake --build build/test --target update-cim22-protobuf
cmake --build build/test --target check-cim22-protobuf
```

C++ protoc/运行库固定配对；Python `protobuf>=6.31.1` 仅用于测试，可使用更新运行库。正常构建不会覆盖仓库中的测试绑定。

使用本机 Ruff 和 clang-format 检查修改过的代码；它们不属于运行时依赖：

```sh
ruff check --config python/pyproject.toml python test/python test/lit.cfg.py
ruff format --check --config python/pyproject.toml python test/python test/lit.cfg.py
uv lock --project python --check
```

## 源码包与离线构建

在仓库根目录生成源码包，包含测试、示例和构建文件：

```sh
uv build --python 3.12 python --sdist --out-dir wheelhouse
```

解包到全新目录后，在解包目录构建 wheel；同样需要配置 `MLIR_DIR`。
重新验收修改过的 sdist 时不要复用旧解包/构建目录：归一化的文件时间戳
可能使增量构建错误地沿用旧目标文件。

```sh
uv build --python 3.12 --wheel
```

源码包内 C++ 源码位于 `native/`，由构建配置自动定位。

`uv --offline` 只限制 Python 下载，不能阻止 CMake FetchContent 访问网络。离线构建应预先准备固定版本及其传递依赖源码，并通过以下构建参数指定本地路径；仅启用 ONNX 时需要最后一项：

```text
-Ccmake.define.FETCHCONTENT_SOURCE_DIR_PROTOBUF=/path/to/protobuf-31.1
-Ccmake.define.FETCHCONTENT_SOURCE_DIR_ABSL=/path/to/abseil-cpp
-Ccmake.define.FETCHCONTENT_SOURCE_DIR_ONNX=/path/to/onnx
```

不要将 wheel、构建目录、日志或下载的依赖提交到 Git。
