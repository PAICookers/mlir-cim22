# 源码构建

默认命令面向 Linux x86-64；Windows 原生组件的构建见文末。两条构建路径共用同一个 CMake 工程：

| 目标             | 方式                   | 结果                                             |
| ---------------- | ---------------------- | ------------------------------------------------ |
| 编译器本体       | CMake/Ninja            | 原生优化器、compiler/runtime 库，可选 ONNX 工具  |
| PyTorch 编译后端 | uv + scikit-build-core | 包含上述原生组件的 Python wheel 或 editable 安装 |

仅使用预编译包时，直接阅读 [Python 安装指南](python/README.md#安装)。以下命令除另有说明外，均在**仓库根目录**执行。

## 环境与依赖

需要 Git、C++17 编译器、**CMake >=3.28**、Ninja，以及默认 USB 构建需要的 `pkg-config` 和 libusb 开发文件。Python 后端和测试还需 Python 3.12 与 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

Linux USB 后端直接使用系统 libusb，无需安装 Cypress `cyusb_linux` SDK。CMake 通过 `pkg-config` 获取 libusb 的头文件与链接配置。设备访问权限和板卡配套的 FX3 固件需另行准备；安装主机库不会下载设备固件。

PLL 自动串口控制仅支持 Linux，使用 POSIX 接口，不需要 PyUSB 或 pyserial。Linux 上未提供串口或指定设备不存在时，使用用户手动设置的时钟，不计为程序已配置或校验 PLL。Windows 保留 CyAPI USB 支持，省略串口选项并手动设钟；请求自动串口控制会报错。设备权限及可选串口的使用方法见 [USB 与 PLL 配置](python/README.md#usb-与-pll-配置)。

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

## GitHub Actions 预编译 wheel

[wheel 工作流](.github/workflows/wheels.yml) 在 `main` 更新或推送 `v*` 标签时自动构建，也支持从 Actions 页面手动运行；没有每日定时任务。Linux 使用 manylinux_2_28 x86-64，Windows 使用 Windows Server 2022 / MSVC x64；两者均为 Release、USB 开启、测试和 ONNX 导入器关闭。

源码 `python/pyproject.toml` 的版本是当前开发目标，例如 `1.0.0`：

- 分支构建在目标版本后加 `.dev`、提交的 UTC 日期和六位工作流序号（1–999999），例如 `1.0.0.dev20261028000042`。手动运行选择分支时遵循同一规则。
- 正式构建使用标签版本：源码版本为 `1.0.0` 时推送 `v1.0.0`，生成 `1.0.0`；标签与源码不一致则构建失败。手动运行选择标签时也会核对。

```text
mlir_cim22-1.0.0.dev20261028000042-py3-none-manylinux_2_28_x86_64.whl
mlir_cim22-1.0.0.dev20261028000042-py3-none-win_amd64.whl
mlir_cim22-1.0.0-py3-none-manylinux_2_28_x86_64.whl
mlir_cim22-1.0.0-py3-none-win_amd64.whl
```

发布时先确认源码版本和测试，再对对应提交创建并推送 `vX.Y.Z` 标签。正式版之后，将源码版本及锁文件中的本项目版本更新到下一个开发目标，再继续合入开发变更。`.dev` 是 Python 预发布版本，排在对应正式版之前；与正式版共享包索引时，通常需要显式选择版本或使用 `pip install --pre`。

### Windows 的最小 CyAPI 依赖

Actions 直接使用仓库中的 [third_party/cyapi](third_party/cyapi/README.md)，不下载或运行 FX3 SDK EXE，不登录英飞凌账号，也不需要私有依赖仓库或访问凭据。该目录保留官方 FX3 SDK 1.3.5 的两个必要头文件、x64 `CyAPI.lib`、组件 `license.rtf`、SDK `license.txt` 和 SHA256 校验清单；厂商文件内容与原始安装包一致。

CyAPI 使用其原始许可，不适用本项目的 Apache 许可。Windows wheel 的 `.dist-info/licenses/third-party/cyapi/` 保留两份协议及依赖说明，`cim22/share/licenses/cyapi/license.rtf` 保留现有运行库许可副本。保留声明不改变协议授予的用途与分发范围；发布含 CyAPI 的产物时仍须遵守两份协议。

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
`tools/` 仅打包其 CMake 文件和两个编译工具源码：`mlir-cim22-opt` 由正常的 `torch.compile` lowering 调用；`mlir-cim22-onnx-import` 随 `MLIRCIM22_ENABLE_ONNX` 启用，用于 ONNX 诊断和测试。源码包保留两者以支持可选构建；wheel 的安装内容由 CMake 安装目标决定。

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

Windows USB 使用 **Cypress CyAPI / CyUSB3** 驱动。仓库的 `third_party/cyapi` 已包含构建所需的官方 x64 依赖；也可自备匹配 x64 和 MSVC 运行库的 Cypress EZ-USB FX3 SDK。`CYAPI_ROOT` 指向该目录或 SDK 根目录（`library/cpp/inc`、`library/cpp/lib/x64/CyAPI.lib`），也可分别指定 `CYAPI_INCLUDE_DIR` 和 `CYAPI_LIBRARY`。

```powershell
cmake -G Ninja -S . -B build/windows -DCMAKE_BUILD_TYPE=Release `
  -DMLIR_DIR=../llvm-build-windows/lib/cmake/mlir `
  -DCIM22_ENABLE_USB=ON -DCYAPI_ROOT="$PWD/third_party/cyapi"
cmake --build build/windows --parallel 2
```

`CyAPI.lib` 必须为 x64，并与 MSVC 运行库配置匹配；CMake 配置阶段会检查链接兼容性。

原生工具位于 `build/windows/bin`，runtime/compiler DLL 位于 `build/windows/lib`。CyAPI 静态链接，运行机器仍需安装匹配板卡的 Cypress CyUSB3 驱动和 MSVC x64 运行库。Windows 根据 VID/PID、可选 USB 地址选择唯一设备，核对接口及端点；不支持 `busNumber`。保留单个异步 IN 请求，轮询超时不取消请求；OUT 每次最多发送 512 字节，沿用现有 12 字节记录协议。关闭时如驱动未归还请求所有权，会保留相关资源至进程退出，避免释放仍在使用的缓冲区。

已验证 FX3 SDK **1.3.5** 的官方 x64 CyAPI 库与 MSVC 2022 可完成链接。wheel 会携带 SDK 的 `license/license.rtf`（可通过 `CYAPI_LICENSE_FILE` 指定位置），不打包 SDK 源码或安装板卡驱动。

Windows 驱动的无硬件测试使用项目自有 SDK 替身，可在启用原生测试、关闭 USB 构建时运行 `mlir-cim22-cyapi-driver-test`，不需要 CyAPI 库。它验证接口控制流程，不能替代真实 SDK 链接和板卡验收。Linux 继续使用 libusb。

可用现有打包入口生成包含 Windows 原生组件的 wheel：

```powershell
$env:CMAKE_GENERATOR = "Ninja"
uv build --python 3.12 python --wheel --out-dir python/dist `
  -C build-dir=build/windows-wheel `
  -C cmake.define.MLIR_DIR=C:/path/to/llvm-build-windows/lib/cmake/mlir `
  -C "cmake.define.CYAPI_ROOT=$PWD/third_party/cyapi"
```

Windows wheel 包含 `mlir-cim22-opt.exe`、`cim22_compiler.dll`、`cim22_runtime.dll` 和模型链接所需的 `.lib`。Python 后端使用 MSVC 将 COFF `.obj` 链接为模型 `.dll`，通过与 Linux 相同的模型 ABI 加载。`torch.compile` 软件推理和 DLL 重新加载的用法见 [Windows 安装说明](python/README.md#windows-x64)。

开发环境的 `pyproject.toml` / `uv.lock` 按平台选择 torch-mlir：Linux 保留主分支的 `20261001` release wheel，Windows 从官方 release 安装 `20260923.595-cp312-abi3-win_amd64.whl`。不引入 torch-mlir 源码。在配置好 MSVC x64、`MLIR_DIR` 和 SDK 后，可执行：

```powershell
uv sync --project python --python 3.12 --locked --group dev `
  -C build-dir=build/windows `
  -C cmake.define.MLIR_DIR=C:/path/to/llvm-build-windows/lib/cmake/mlir `
  -C "cmake.define.CYAPI_ROOT=$PWD/third_party/cyapi"
uv run --directory python --no-sync pytest tests/test_artifacts.py tests/test_backend.py tests/test_convolution.py tests/test_model_format.py tests/test_native_compiler.py tests/test_usb_runtime.py
```

这组测试覆盖原生模型编译、BF16 软件执行、Host 运算、状态更新、静态 T/B 维度、模型 ABI、USB 错误边界和 DLL 生命周期；POSIX 文件大小限制测试仅在 Linux 运行。全量 `pytest` 中的 `test_web.py` 会因 Linux 专用 `fcntl` 无法在 Windows 导入；Web UI 移植不在本次范围内。完整 DVS 数据集和真实 USB 板卡测试需要另行验收。
