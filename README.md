# mlir-cim22

基于 LLVM/MLIR 的 CIM22 编译器，以及可通过 `torch.compile` 使用的 PyTorch 推理后端。编译器将适合 CIM 的计算降低为目标指令，其余计算生成 CPU 代码；运行时负责加载模型并执行。目前可运行软件仿真，Linux USB 路径用于硬件诊断。

## 项目组成

| 部分         | 位置                                      | 用途                                       |
| ------------ | ----------------------------------------- | ------------------------------------------ |
| 编译器       | `include/`、`lib/`、`tools/`        | 方言、转换、映射和 `mlir-cim22-opt` 工具 |
| 运行时       | `lib/Runtime/`、`lib/Execution/`      | 模型加载、CPU/CIM 执行和 Linux USB 通信    |
| PyTorch 后端 | [python/](python/README.md)                | 将捕获的模型经 torch-mlir 交给本编译器     |
| 模型格式     | [schema/](schema/README.md)                 | 内嵌模型数据、字段约束与格式维护             |
| 测试         | [test/](test/README.md)、`python/tests/` | 编译器、数值、接口与软件执行回归           |

Python 发行包名为 `mlir-cim22`，导入包和后端名为 `cim22`。一个 wheel 包含 Python 封装、优化器以及 compiler/runtime 原生库；torch-mlir 是后端的必需外部依赖，需另外安装。

## 快速上手

使用 **Linux x86-64**。准备 Python 3.12、CPU PyTorch 和从官方 Release 页面下载的、与当前后端兼容的 torch-mlir wheel。项目元数据允许 Python `>=3.12`、PyTorch `>=2.14,<3` 和任意 torch-mlir 版本；具体组合仍需自行验证。

1. 按 [Python 安装指南](python/README.md#安装)准备虚拟环境和两个 wheel。
2. 从仓库根目录运行最小示例：

   ```sh
   .venv/bin/python python/examples/torch_compile.py
   ```

   成功时输出：

   ```text
   CIM22 torch.compile software execution passed (0, 65, 129).
   ```
3. 在自己的模型上使用同一后端：

   ```python
   import torch
   from torch import nn

   model = nn.Sequential(nn.Linear(64, 16), nn.ReLU()).cpu().eval()
   with torch.inference_mode():
       compiled = torch.compile(
           model, backend="cim22", fullgraph=True, dynamic=False,
           options={"runtime": "software"},
       )
       result = compiled(torch.ones(1, 64))
       print(result.shape)
   ```

   输出：

   ```text
   torch.Size([1, 16])
   ```

示例显式选择软件运行时；默认运行时是 USB。输入须为静态形状、连续的 CPU FP32 Tensor。首次调用完成编译和加载，相同特化的后续调用复用已加载代码。

拿到源码后需要先安装或构建，克隆仓库本身不会提供预编译文件。只有源码时，请从 [源码构建指南](BUILDING.md)开始。

## 源码构建与开发

- [编译器本体](BUILDING.md#编译器本体)：准备 LLVM/MLIR，直接构建和使用原生工具。
- [PyTorch 编译后端](BUILDING.md#pytorch-编译后端)：构建一体化 wheel 或 editable 安装。
- [测试与开发](BUILDING.md#测试与开发)：运行 pytest、原生 lit 测试及增量重建。
- [Python 使用说明](python/README.md)：模型示例、后端选项和常见问题。

`mlir-cim22-opt` 也可单独用于 MLIR 检查和 pass 调试。ONNX 导入器是可选的诊断工具，默认构建不包含它。
