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

CMake 会下载并构建固定的 **Protobuf/protoc 31.1**；启用 ONNX 时还会构建 **ONNX 1.21.0**。无需安装系统 Protobuf 或 ONNX C++ 包。首次构建需要网络、足够的磁盘和内存；示例中的并行任务数可按机器配置调整。

## 准备 LLVM/MLIR

已有兼容工具链时，只需设置其 `lib/cmake/mlir` 为 `MLIR_DIR`，可跳过本节构建。本项目要求 LLVM/MLIR **>=22**；验证版本是 **22.1.8**。

参考 [MLIR 官方入门指南](https://mlir.llvm.org/getting_started/)，以下在仓库旁拉取已验证的 `llvmorg-22.1.8` tag，并建立构建目录。只启用 MLIR 和本机代码生成目标，使用 Release 并保留断言检查：

```sh
git clone --depth 1 --branch llvmorg-22.1.8 \
  https://github.com/llvm/llvm-project.git ../llvm-project

cmake -G Ninja -S ../llvm-project/llvm -B ../llvm-build \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=Native \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build ../llvm-build --parallel 8
export MLIR_DIR="$(cd ../llvm-build && pwd)/lib/cmake/mlir"
```

LLVM 默认生成启用 PIC 的静态库。PIC 即位置无关代码，使这些库中的代码可以链接进本项目的共享库（如 `libcim22_compiler.so`）；这里的“静态”指 LLVM/MLIR 依赖库的形式，本项目仍会生成 `.so`。torch-mlir 使用自身的 Python 绑定，无需额外启用 LLVM/MLIR Python 绑定。保留工具链源码和构建目录。

后续源码构建使用以下工具链变量；打开新终端时需重新设置：

| 变量            | 含义                                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `MLIR_DIR`    | 必需：LLVM/MLIR 工具链中包含 `MLIRConfig.cmake` 的目录，例如上面的 `../llvm-build/lib/cmake/mlir`，不是本项目的 `build/`        |
| `CC`、`CXX` | 可选：首次配置时选择 C/C++ 编译器；上例使用 `clang`、`clang++`，未设置时由 CMake 选择系统默认编译器。已有构建目录会缓存编译器选择 |

只在 `cmake` 命令中传入 `-DMLIR_DIR=...` 会将路径保存到该构建目录的缓存，不会设置终端环境变量，也不会传给其他构建目录。

编辑器出现 `Could not find ... MLIRConfig.cmake` 时，为它使用的构建目录重新配置：

```sh
cmake -G Ninja -S . -B build -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
```

`MLIR_DIR` 必须指向包含 `MLIRConfig.cmake` 的目录。编辑器若使用独立构建目录，也需设置该目录的配置参数；终端中的环境变量未必传给已启动的编辑器。

## MLIR-CIM22 编译器

此路径不需要安装 PyTorch 或 torch-mlir：

```sh
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR="$MLIR_DIR"
cmake --build build --parallel 4
build/bin/mlir-cim22-opt --help
```

优化器位于 `build/bin/`，compiler/runtime 库位于 `build/lib/`。Python 打包另用 `python/build/`；wheel 内安装路径为 `cim22/bin/` 和 `cim22/lib/`，与这两个构建目录无关。

例如，将混合计算测试转换为 packet 阶段的 MLIR：

```sh
build/bin/mlir-cim22-opt \
  test/Integration/ExecutionPlan/mixed-bf16-pipeline.mlir \
  --cim-bf16-pipeline='stage=packet max-kernel-columns=1' \
  -o build/model.packet.mlir
```

该命令用于检查编译 IR；PyTorch 后端还会生成目标文件、链接并加载模型。需要受限的 ONNX 导入器时，在配置命令中添加 `-DMLIRCIM22_ENABLE_ONNX=ON`，构建后查看参数：

```sh
build/bin/mlir-cim22-onnx-import --help
```

### CIM 优化与流水线选项

`--cim-bf16-pipeline` 自动执行常量折叠、公共子表达式消除、Linear/卷积规范化、符合条件的 FP32 收缩运算转 BF16，以及 Host/CIM 划分。可在同一参数字符串中设置：

| 选项                   | 默认值     | 作用                                                                                                  |
| ---------------------- | ---------- | ----------------------------------------------------------------------------------------------------- |
| `max-kernel-columns` | `32`     | 正整数；矩阵列数超过阈值时，由 Host 按单列循环调用 CIM kernel                                         |
| `require-cim`        | `false`  | 设为 `true` 时，要求存在 CIM kernel，并拒绝仍留在 Host 的命名矩阵乘或卷积运算，用于检查卸载是否完成 |
| `stage`              | `packet` | 在 `split`、`vmm`、`packet` 或 `llvm` 阶段结束；`llvm` 还会将 Host 部分降到 LLVM 方言       |

例如，要求完成 CIM 卸载并按单列流式处理：

```sh
build/bin/mlir-cim22-opt \
  test/Integration/ExecutionPlan/native-bf16.mlir \
  --cim-bf16-pipeline='stage=packet max-kernel-columns=1 require-cim=true' \
  -o build/model.packet.mlir
```

单独运行规范化或 INT8 pass 时，还可使用以下选项；布尔值均可显式设置为 `true` 或 `false`：

| Pass                                                 | 选项与默认值                  | 作用                                                                                       |
| ---------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------ |
| `--normalize-cim-linear`、`--normalize-cim-conv` | `allow-f32=false`           | 允许规范化符合条件的 FP32 运算，本身不改变精度；BF16 流水线已设为 `true`                 |
| `--normalize-cim-conv`                             | `allow-bias-epilogue=false` | 将符合条件的 FP32 通道广播初值移到 Host bias 后处理；BF16 流水线已设为 `true`            |
| `--fold-cim-int8-bias`                             | `allow-extra-k-tile=true`   | 在满足条件的已标记 INT8 MatMul 中折叠 bias；设为 `false` 可避免因折叠额外增加一个 K tile |

这些是优化器 CLI 选项。当前 PyTorch 后端固定使用 `stage=packet max-kernel-columns=1`，未通过 `BackendOptions` 开放上述参数，也不自动运行 INT8 bias 折叠。FP32 转 BF16 会改变精度，调整后需重新做数值对照；阈值大小不直接代表性能优劣。

## PyTorch 编译后端

### 构建 wheel

发布包含 Web 界面的 wheel 前，先按 [Web 构建说明](web/README.md#从源码构建) 生成静态页面。

在当前终端设置 `MLIR_DIR`，并显式传给 wheel 构建。将下面的路径替换为实际 LLVM/MLIR 工具链路径；若按上文构建，使用 `$(cd ../llvm-build && pwd)/lib/cmake/mlir`：

```sh
export MLIR_DIR="/path/to/llvm-build/lib/cmake/mlir"
CMAKE_BUILD_PARALLEL_LEVEL=4 uv build --python 3.12 python \
  --wheel --out-dir python/dist -Ccmake.define.MLIR_DIR="$MLIR_DIR"
```

本体使用的 `build/` 与打包使用的 `python/build/` 各自保存 CMake 缓存；即使本体刚编译成功，wheel 也不会自动读取 `build/CMakeCache.txt`。

scikit-build-core 调用根 CMake 工程，默认构建 **Release、USB 开启、测试和 ONNX 关闭**。构建 wheel 本身不需要安装 PyTorch 或 torch-mlir。产物为 `python/dist/mlir_cim22-1.0.0-py3-none-linux_x86_64.whl`，包含：

```text
cim22/
  *.py
  bin/mlir-cim22-opt
  lib/libcim22_compiler.so
  lib/libcim22_runtime.so
  web/static/index.html
  web/static/assets/
```

按 [Python 安装指南](python/README.md#安装)安装该 wheel，再运行自检脚本。

## 测试与开发

### Python 开发

先在当前终端按[构建 wheel](#构建-wheel)设置 `MLIR_DIR`，并保持 `CC`、`CXX` 与首次配置一致。`uv sync` 创建 `python/.venv`，按锁文件安装开发依赖和本项目的 editable 版本，并构建原生组件。已有 Python 源码的修改在新进程中直接生效，无需重新打包：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 uv sync --project python --python 3.12 \
  --locked -Ccmake.define.MLIR_DIR="$MLIR_DIR"
uv run --directory python --no-sync pytest
```

修改 C++、TableGen、schema 或 CMake 后，显式重建安装，再启动测试或应用：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 uv sync --project python --locked \
  --reinstall-package mlir-cim22 -Ccmake.define.MLIR_DIR="$MLIR_DIR"
```

保留首次构建时使用的构建目录和 Debug/USB/ONNX 参数。editable 只省去 Python 源码改动后的打包，不替代原生重建或发行 wheel 验收。

### 原生测试与 schema

先准备上述 Python 环境。原生 lit 测试使用独立的 `build/test/` Debug 构建，避免覆盖 `build/` 中的 Release 配置：

```sh
cmake --build ../llvm-build --target \
  FileCheck split-file mlir-runner mlir_c_runner_utils --parallel 4
cmake -G Ninja -S . -B build/test \
  -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR="$MLIR_DIR" \
  -DMLIRCIM22_INCLUDE_TESTS=ON -DMLIRCIM22_ENABLE_ONNX=ON \
  -DPython3_EXECUTABLE="$PWD/python/.venv/bin/python"
cmake --build build/test --target check-mlir-cim22 --parallel 4
```

自备工具链时应确认同样提供上述 LLVM 测试工具。测试说明见 [test/README](test/README.md)。Python 测试使用安装包内的原生组件；需要测试独立 CMake 构建时，可设置 `CIM22_OPT`、`CIM22_RUNTIME_LIBRARY` 和 `CIM22_COMPILER_LIBRARY` 为对应产物路径。

schema 修改后，用同一原生测试构建更新并检查 Python 测试绑定：

```sh
cmake --build build/test --target update-cim22-protobuf
cmake --build build/test --target check-cim22-protobuf
```

C++ protoc/运行库固定配对；Python Protobuf 只用于测试。正常构建不会覆盖仓库中的测试绑定。

### 代码与锁文件检查

使用本机 Ruff 和 clang-format 检查修改过的代码；它们不属于运行时依赖：

```sh
ruff check --config python/pyproject.toml python test/python test/lit.cfg.py
ruff format --check --config python/pyproject.toml python test/python test/lit.cfg.py
uv lock --project python --check
```

## 源码包与离线构建

在仓库根目录生成源码包，包含测试、示例和构建文件：

```sh
uv build --python 3.12 python --sdist --out-dir python/dist
```

解包到全新目录后，在解包目录构建 wheel；同样需要配置 `MLIR_DIR`。重新验收修改过的 sdist 时不要复用旧解包/构建目录：归一化的文件时间戳可能使增量构建错误地沿用旧目标文件。

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
