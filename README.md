# MLIR-CIM22 Compiler

基于 MLIR 的 CIM22 编译器，以及可通过 `torch.compile` 使用的 PyTorch 推理后端。编译器将适合 CIM 的计算降低为目标指令，其余计算生成 CPU 代码；运行时负责加载模型并执行。

## 项目组成

| 部分            | 位置                                      | 用途                                       |
| --------------- | ----------------------------------------- | ------------------------------------------ |
| 编译器          | `include/`、`lib/`、`tools/`              | 方言、转换、映射、`mlir-cim22-opt` 和 `mlir-cim22-compile` 工具 |
| 运行时          | `runtime/`、`lib/Execution/`      | 模型加载、CPU/CIM 执行和 Linux/Windows USB 通信 |
| PyTorch 后端    | [python/](python/README.md)                | 将捕获的模型经 torch-mlir 交给本编译器     |
| 本地 Web 工作台 | [web/](web/README.md)                      | 固定 DVS 模型导入、编译与软件结果对比      |
| 模型格式        | [schema/](schema/README.md)                | 内嵌模型数据、字段约束与格式维护           |
| 测试            | [test/](test/README.md)、`python/tests/` | 编译器、数值、接口与软件执行回归           |

Python 发行包名为 `mlir-cim22`，导入包和后端名为 `cim22`。一个 wheel 包含 Python 封装、编译工具、优化器和 runtime 原生库，并声明 torch-mlir 为基础依赖。

## 快速上手

在自己的 Python 程序中使用编译器，从 [Python 用户快速开始](python/README.md#用户快速开始)
按顺序操作：准备本平台的项目和 torch-mlir 预编译 wheel，创建 Python 3.12
虚拟环境，保存模型程序，然后运行 `uv run python compile_model.py`。
该流程不需要构建 LLVM/MLIR 或安装本仓库源码。

支持 Linux x86-64（glibc >= 2.28）和 Windows x64。即使使用 wheel，模型链接
仍需要系统 C 工具链：Linux 的 `cc`（GCC/Clang），或 Windows 的 MSVC x64
开发者终端及 Windows SDK。完整安装命令、可复制的模型程序、预期输出和产物
统一见 [Python 使用说明](python/README.md#用户快速开始)。

- [编译 PyTorch 模型](python/README.md#编译模型)
- [TensorFlow 接口](python/README.md#tensorflow-接口)
- [仅编译模型部分层](python/README.md#按模块名选择-cim-执行)
- [USB 与 PLL 配置](python/README.md#usb-与-pll-配置)
- [运行仓库示例](python/README.md#可选tensorflow-与仓库示例)
- [启动 Web 工作台](web/README.md#启动)


## 源码构建与开发

- [MLIR-CIM22 编译器](docs/BUILDING.md#mlir-cim22-编译器)：准备 LLVM/MLIR，直接构建并使用原生工具。
- [PyTorch 编译后端](docs/BUILDING.md#pytorch-编译后端)：构建一体化 wheel。
- [测试与开发](docs/BUILDING.md#测试与开发)：运行 pytest、原生 lit 测试及增量重建。
- [Python 使用说明](python/README.md)：模型示例、后端选项和常见问题。
- [BF16 补偿原理与实现](docs/BF16.md)：sign-split 拆分、尺度恢复、代码入口与误差边界。

`mlir-cim22-opt` 也可单独用于 MLIR 检查和 pass 调试。
