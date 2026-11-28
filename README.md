# MLIR-CIM22 Compiler

基于 MLIR 的 CIM22 编译器，支持 PyTorch、TensorFlow 和 JAX 的静态推理。编译器将适合 CIM 的计算降低为目标指令，其余计算生成 CPU 代码；运行时负责加载模型并执行。

## 项目组成

| 部分            | 位置                                      | 用途                                                                |
| --------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| 编译器          | `include/`、`lib/`、`tools/`        | 方言、转换、映射、`mlir-cim22-opt` 和 `mlir-cim22-compile` 工具 |
| 运行时          | `runtime/`、`lib/Execution/`          | 模型加载、CPU/CIM 执行和 Linux/Windows USB 通信                     |
| Python 前端     | [python/](python/README.md)                | PyTorch 经 FX、TensorFlow 和 JAX 经 ONNX 接入 torch-mlir                              |
| 本地 Web 工作台 | [web/](web/README.md)                      | 框架验证、DVS 板卡执行与 MNIST 训练                                 |
| 模型格式        | [schema/](schema/README.md)                | 内嵌模型数据、字段约束与格式维护                                    |
| 测试            | [test/](test/README.md)、`python/tests/` | 编译器、数值、接口与软件执行回归                                    |

Python 发行包名为 `mlir-cim22`，导入包和后端名为 `cim22`。一个 wheel 包含 Python 封装、编译工具、优化器和 runtime 原生库，并声明 torch-mlir 为基础依赖。

## 快速上手

首次使用请从[安装 wheel 并编译第一个模型](.agents/skills/cim22-user-guide/references/install.md)开始。指南涵盖 Linux x86-64 和 Windows x64，可用软件仿真完成第一次推理。

- PyTorch 可在 `torch.compile` 中通过 `options["cim_layers"]` 指定由 CIM 执行的模块。见 [Python 选层示例](python/README.md#按模块选层)。
- JAX 模型通过 `cim22.jax.compile` 接入。安装 `jax` extra 后，可运行[固定权重的 Dense/ReLU 示例](python/examples/jax_compile.py)，对照 JAX、ONNX Runtime 和所选 CIM22 目标的输出。
- [启动 Web 工作台](.agents/skills/cim22-user-guide/references/web.md)可运行框架示例、[为固定 DVS 模型选层](web/README.md#dvs-选层)或进行 MNIST 训练。

可直接调用 [`$cim22-user-guide`](.agents/skills/cim22-user-guide/SKILL.md)，例如“用 $cim22-user-guide 帮我在没有板卡的 Linux 机器上编译并运行第一个 PyTorch 模型”。

## 源码构建与开发

- [MLIR-CIM22 编译器](docs/BUILDING.md#mlir-cim22-编译器)：准备 LLVM/MLIR，直接构建并使用原生工具。
- [PyTorch 编译后端](docs/BUILDING.md#pytorch-编译后端)：构建一体化 wheel。
- [测试与开发](docs/BUILDING.md#测试与开发)：运行 pytest、原生 lit 测试及增量重建。
- [Python 后端开发](python/README.md#源码开发)：源码入口、接口测试与编译诊断。
- [BF16 补偿原理与实现](docs/BF16.md)：sign-split 拆分、尺度恢复、代码入口与误差边界。

`mlir-cim22-opt` 也可单独用于 MLIR 检查和 pass 调试。
