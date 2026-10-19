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
cmake --build ../llvm-build --parallel 2
export MLIR_DIR="$(cd ../llvm-build && pwd)/lib/cmake/mlir"
```

静态 PIC 库供本项目链接；torch-mlir 使用自身的 Python 绑定，无需 LLVM Python 绑定。保留工具链源码和构建目录，后续打开新终端时重新设置 `MLIR_DIR`、`CC`、`CXX`。

编辑器出现 `Could not find ... MLIRConfig.cmake` 时，为它使用的构建目录重新配置：

```sh
cmake -G Ninja -S . -B build -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir
```

`MLIR_DIR` 必须指向包含 `MLIRConfig.cmake` 的目录。编辑器若使用独立构建目录，
也需设置该目录的配置参数；终端中的环境变量未必传给已启动的编辑器。
可将个人路径放入不提交的 `CMakeUserPresets.json`，并在编辑器中选择相应 preset；
不要写进项目的 `CMakeLists.txt`。

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

按 [Python 安装指南](python/README.md#安装)安装该 wheel，再运行自检脚本；项目元数据会声明 torch-mlir 基础依赖。普通用户仍需系统 `cc` 和链接开发环境来编译模型。本机构建只保证面向本机 ABI；可移植 manylinux 分发还需独立构建和验收。

### Linux Release wheel

当前本地候选包（2026-10-18，`py3-none-linux_x86_64`）已通过全新 sdist 构建、
隔离安装和 124 项定向测试，包含优化器、两个原生库和 Web 静态页面。
但 compiler/优化器要求 `GLIBC_2.44`，compiler/runtime 要求
`GLIBCXX_3.4.32`；还依赖系统 zlib、zstd、libusb 及其传递依赖。
`auditwheel show` 也确认当前只能标记为 `linux_x86_64`。
它仅适合满足这些要求的已验证环境，不宜作为通用 Linux Release 附件。

面向更多发行版发布时，建议以 `manylinux_2_28_x86_64` 为目标评估以下流程：

1. 在该基线容器中构建 LLVM/MLIR SDK，再构建本项目；不能复用新宿主机编译的静态库。
2. 先生成前端，再从全新 sdist 构建 wheel。检查所有 ELF 的符号版本、
   指令集和运行库搜索路径，避免宿主机路径或 `-march=native`。
3. 在构建环境中运行 `auditwheel show`、
   `auditwheel repair --plat manylinux_2_28_x86_64 --wheel-dir repaired <wheel>`，
   处理非策略允许的共享库，并随附所需的第三方许可证。
4. 在干净的目标系统上安装修复后的 wheel、兼容的 CPU PyTorch 和 torch-mlir；
   在仓库外验证模型编译、软件执行、CLI 和 Web。USB 硬件验证单独记录。
5. Release 附件提供 wheel、SHA256、构建信息、依赖组合和适用平台说明。

`manylinux_2_28` 表示 glibc 2.28 基线，不涵盖 Alpine/musl、其他 CPU 架构或所有
Python/依赖组合；最终基线也受 PyTorch 和 torch-mlir wheel 限制。
`auditwheel repair` 能打包部分外部库并调整加载路径，不能降低已编译二进制要求的
glibc/libstdc++ 版本，不能仅修改 wheel 文件名。动态 `ctypes`/`dlopen` 加载及
安装后编译出的模型仍需实际测试。上述可移植构建尚未验收。
参见 [manylinux](https://github.com/pypa/manylinux) 和
[auditwheel 限制](https://github.com/pypa/auditwheel#limitations)。

### GitHub Actions 构建（开发者）

[Build Linux wheels](.github/workflows/wheels.yml) 手动触发，合并到默认分支后，
在 **Actions → Build Linux wheels → Run workflow** 选择待构建的分支或 tag。
所有原生编译均在固定镜像的 `manylinux_2_28_x86_64` 容器中执行：

1. 从官方 `llvm/llvm-project` 检出 22.1.8 的固定提交，构建 Release、静态 PIC、
   X86 SDK，包含 MLIR 执行引擎和 TableGen 工具；不添加源码 submodule。
2. 按源码 SHA、容器及构建脚本缓存安装后的 SDK。命中时跳过 LLVM 编译；
   另保留最多 2 GiB 的 ccache，在 SDK 需要重建时复用编译结果。
3. SDK 压缩包通过 artifact 传给一个 Ubuntu **24.04** 构建 job，校验 SHA256
   后构建前端、全新 sdist 和 manylinux wheel，执行 auditwheel 检查与修复。
4. 构建产物及 `SHA256SUMS` 通过 `actions/upload-artifact` 保存 14 天；Ubuntu
   **22.04、24.04、26.04** 只下载这一份 wheel，分别安装并验证 CPU 软件推理、
   定向 pytest、CLI 和 Web 静态资源/API，验证报告单独上传。

manylinux wheel 由同一构建 job 生成，不是三种 Ubuntu 专用包。Ubuntu 矩阵只用于
检查不同宿主机环境；GitHub Actions artifact 可手动下载，不会自动发布 Release。
首次 SDK 构建上限六小时、项目构建与验收上限两小时；均限制两个编译任务、
一个链接任务。SDK job 清理 runner 中未使用的 Android/.NET/GHC，以释放空间。
缓存可能被淘汰，不能视为永久工具链仓库；首次构建的耗时和磁盘仍需远程实测。
`uses` 使用当前最新 major tag。

安装验收使用 CPU PyTorch 2.14.1 和官方 torch-mlir 20261001 wheel，
不上传本机 torch-mlir wheel、模型或数据集。最终 wheel 内含 LLVM/MLIR 原生代码，
用户无需安装 SDK。源码允许 LLVM/MLIR >=22，发布构建固定 22.1.8；换版本后
必须重新构建 SDK 并验证。Ubuntu runner 的版本不决定容器中二进制的 glibc 基线。

设计参考 torch-mlir 的
[_build_wheels.yml](https://github.com/llvm/torch-mlir/blob/a5efa975854c04288217663ffa8e924693412be9/.github/workflows/_build_wheels.yml)：
固定 LLVM 源码、复用 ccache。本项目通过 artifact 传递安装后的 SDK，避免每次
wheel 构建重复编译 LLVM。

当前仅完成本地配置与脚本检查；首次远程运行成功前，不能声称已有可移植 Release
wheel。manylinux 审计及 Ubuntu 软件测试不代表所有发行版或真实 USB 板卡已验证。

### PR 代码检查

[pre-commit 配置](.pre-commit-config.yaml) 用于 PR 自动检查，无需在本机安装
Git hooks。远程环境通过 `cpp-linter-hooks`、`ruff-pre-commit` 和
`pre-commit-hooks` 安装工具，排除供应商、生成文件及包管理器锁文件的通用格式化。
Ruff 的 lint/format 均通过 `args` 指定 `python/pyproject.toml`，使目录外的
Python 脚本也使用相同规则。pre-commit.ci 需为仓库启用对应 GitHub App。
当前 cpp-linter 的版本解析仍依赖联网，
尚未验证其在 pre-commit.ci 中可用，不能仅凭配置文件声称 PR 已检查通过。
本机手动格式化仍使用已安装的 Ruff 和 clang-format，二者不作为项目运行时依赖。

### Editable 开发安装

`python/pyproject.toml` 的 `tool.uv.sources` 固定官方 torch-mlir 20261001
Release wheel URL，并选择 PyTorch CPU 索引；uv 自动下载和缓存，无需手动放置 wheel。
准备好前述 LLVM/MLIR 开发环境后执行：

```sh
CMAKE_BUILD_PARALLEL_LEVEL=4 uv sync --project python --python 3.12 \
  --locked --no-dev
uv run --project python --no-sync python python/examples/torch_compile.py
```

该命令创建 `python/.venv`，安装 CPU PyTorch、torch-mlir 和本项目的 editable 版本，并构建原生组件。`--no-dev` 只跳过默认开发依赖。预编译用户的 `.venv` 与这里的 `python/.venv` 是不同环境。

当前发布、CI 构建和回归仅支持 Python 3.12，因为固定的 torch-mlir wheel 只按该版本验收。
项目元数据暂时限制为 `>=3.12,<3.13`；未来支持其他 Python 版本时，需要先更新
torch-mlir 组合并重新验证后再放宽。Torch 要求为 `>=2.14,<3`，torch-mlir 版本需与后端匹配。

开发锁文件用 `uv lock --project python` 更新，保留上述来源。

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
  --locked --group dev --reinstall-package mlir-cim22
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
