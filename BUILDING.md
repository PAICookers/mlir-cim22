# 源码构建

默认命令面向 Linux x86-64；Windows 原生组件的构建见文末。两条构建路径共用同一个 CMake 工程：

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
cmake --build ../llvm-build --parallel 4
export MLIR_DIR="$(cd ../llvm-build && pwd)/lib/cmake/mlir"
```

静态 PIC 库供本项目链接；torch-mlir 使用自身的 Python 绑定，无需 LLVM Python 绑定。保留工具链源码和构建目录，后续打开新终端时重新设置 `MLIR_DIR`、`CC`、`CXX`。

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

设置好上述工具链变量后执行：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 uv build --python 3.12 python \
  --wheel --out-dir python/dist
```

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

按 [Python 安装指南](python/README.md#安装)安装该 wheel，再运行自检脚本。普通用户仍需系统 `cc` 和链接开发环境来编译模型。本机构建只保证面向本机 ABI；可移植 manylinux 分发还需独立构建和验收。

## 测试与开发

### Python 开发

`uv sync` 创建 `python/.venv`，按锁文件安装开发依赖和本项目的 editable 版本，并构建原生组件。已有 Python 源码的修改在新进程中直接生效，无需重新打包：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 uv sync --project python --python 3.12 \
  --locked
uv run --directory python --no-sync pytest
```

修改 C++、TableGen、schema 或 CMake 后，显式重建安装，再启动测试或应用：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=2 uv sync --project python --locked \
  --reinstall-package mlir-cim22
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

## Windows x64 原生编译器与 USB runtime

使用 **Visual Studio 2022 Build Tools（MSVC x64、Windows SDK）**、CMake 和 Ninja，在 “x64 Native Tools Command Prompt” 中启动 PowerShell。LLVM/MLIR 和本项目必须使用匹配的 Windows 原生工具链；Linux 的 LLVM 库不能用于此构建。

准备同版本 LLVM 22.1.8 源码，在仓库根目录执行：

```powershell
cmake -G Ninja -S ../llvm-project/llvm -B ../llvm-build-windows `
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=mlir `
  -DCMAKE_C_FLAGS=/utf-8 -DCMAKE_CXX_FLAGS=/utf-8 `
  -DLLVM_TARGETS_TO_BUILD=X86 -DBUILD_SHARED_LIBS=OFF `
  -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF `
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF -DMLIR_ENABLE_EXECUTION_ENGINE=ON `
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF `
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_DIA_SDK=OFF `
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF
cmake --build ../llvm-build-windows --parallel 2 --target mlir-opt `
  MLIRExecutionEngine MLIRBuiltinToLLVMIRTranslation `
  MLIRLLVMToLLVMIRTranslation MLIRTargetLLVMIRExport
```

Windows USB 使用供应方 `FrameHandler_USB_32b` 对应的 **Cypress CyAPI / CyUSB3** 驱动，不使用 libusb/WinUSB。准备匹配 x64 和 MSVC 运行库的 Cypress EZ-USB FX3 SDK；`CYAPI_ROOT` 指向 SDK 根目录（`library/cpp/inc`、`library/cpp/lib/x64/CyAPI.lib`），也可分别指定 `CYAPI_INCLUDE_DIR` 和 `CYAPI_LIBRARY`。

```powershell
cmake -G Ninja -S . -B build/windows -DCMAKE_BUILD_TYPE=Release `
  -DMLIR_DIR=../llvm-build-windows/lib/cmake/mlir `
  -DCIM22_ENABLE_USB=ON -DCYAPI_ROOT="C:/path/to/EZ-USB FX3 SDK/1.3"
cmake --build build/windows --parallel 2
```

供应方压缩包内的 `CyAPI.lib` 是 **x86**，不能链接到 x64 编译器；配置阶段会检查链接兼容性并报错，需要 SDK 中的 x64 版本，不能通过改文件名或 wheel 标签解决。SDK 及供应方源码保持外部依赖，不复制进仓库。

原生工具位于 `build/windows/bin`，runtime/compiler DLL 位于 `build/windows/lib`。CyAPI 静态链接，运行机器仍需安装匹配板卡的 Cypress CyUSB3 驱动和 MSVC x64 运行库。Windows 根据 VID/PID、可选 USB 地址选择唯一设备，核对接口及端点；不支持 `busNumber`。保留单个异步 IN 请求，轮询超时不取消请求；OUT 每次最多发送 512 字节，沿用现有 12 字节记录协议。关闭时如驱动未归还请求所有权，会保留相关资源至进程退出，避免释放仍在使用的缓冲区。

已验证 FX3 SDK **1.3.5** 的官方 x64 CyAPI 库与 MSVC 2022 可完成链接。wheel 会携带 SDK 的 `license/license.rtf`（可通过 `CYAPI_LICENSE_FILE` 指定位置），不打包 SDK 源码或安装板卡驱动。

Windows 驱动的无硬件测试使用项目自有 SDK 替身，可在启用原生测试、关闭 USB 构建时运行 `mlir-cim22-cyapi-driver-test`，不需要 CyAPI 库。它验证接口控制流程，不能替代真实 SDK 链接和板卡验收。Linux 继续使用 libusb。

可用现有打包入口生成包含 Windows 原生组件的 wheel：

```powershell
$env:CMAKE_GENERATOR = "Ninja"
uv build --python 3.12 python --wheel --out-dir wheelhouse `
  -C build-dir=build/windows-wheel `
  -C cmake.define.MLIR_DIR=C:/path/to/llvm-build-windows/lib/cmake/mlir `
  -C "cmake.define.CYAPI_ROOT=C:/path/to/EZ-USB FX3 SDK/1.3"
```

Windows wheel 包含 `mlir-cim22-opt.exe`、`cim22_compiler.dll`、`cim22_runtime.dll` 和模型链接所需的 `.lib`。Python 后端使用 MSVC 将 COFF `.obj` 链接为模型 `.dll`，通过与 Linux 相同的模型 ABI 加载。`torch.compile` 软件推理和 DLL 重新加载的用法见 [Windows 安装说明](python/README.md#windows-x64)。

开发环境的 `pyproject.toml` / `uv.lock` 按平台选择 torch-mlir：Linux 保留主分支的 `20261001` release wheel，Windows 从官方 release 安装 `20260923.595-cp312-abi3-win_amd64.whl`。不引入 torch-mlir 源码。在配置好 MSVC x64、`MLIR_DIR` 和 SDK 后，可执行：

```powershell
uv sync --project python --python 3.12 --locked --group dev `
  -C build-dir=build/windows `
  -C cmake.define.MLIR_DIR=C:/path/to/llvm-build-windows/lib/cmake/mlir `
  -C "cmake.define.CYAPI_ROOT=C:/path/to/EZ-USB FX3 SDK/1.3"
uv run --directory python --no-sync pytest tests/test_artifacts.py tests/test_backend.py tests/test_convolution.py tests/test_model_format.py tests/test_native_compiler.py tests/test_usb_runtime.py
```

这组测试覆盖原生模型编译、BF16 软件执行、Host 运算、状态更新、静态 T/B 维度、模型 ABI、USB 错误边界和 DLL 生命周期；POSIX 文件大小限制测试仅在 Linux 运行。全量 `pytest` 中的 `test_web.py` 会因 Linux 专用 `fcntl` 无法在 Windows 导入；Web UI 移植不在本次范围内。完整 DVS 数据集和真实 USB 板卡测试需要另行验收。
