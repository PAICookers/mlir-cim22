# CIM22 PyTorch Backend

本 Python 项目将 `cim22` 注册为 `torch.compile` 后端。模型经 torch-mlir 导入后，由原生编译器完成 CPU/CIM 划分、编译和链接，再由运行时执行。

发行包 `mlir-cim22` 已包含 Python 模块、`mlir-cim22-opt` 和两个原生库。安装 torch-mlir 后即可使用，无需单独安装本项目的 C++ 工具。

## 安装

以下命令均在**仓库根目录**执行，面向已有预编译 wheel 的用户。只有源码时，先阅读 [源码构建](../BUILDING.md#pytorch-编译后端)。

准备 Linux x86-64、Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。首次模型编译还需要系统 `cc`、链接器和 C 开发文件。包含 USB 支持的 wheel 还需要 libusb 动态库，即使选择软件运行时也一样。例如 Ubuntu：

```sh
sudo apt-get install build-essential libusb-1.0-0
```

预编译用户不需要 LLVM/MLIR、CMake 或 protoc；其他系统动态库要求以所用 wheel 的构建环境说明为准。

将维护者提供且适配本机系统 ABI 的 `mlir_cim22-*.whl`，以及从 [torch-mlir releases](https://github.com/llvm/torch-mlir-release/releases) 下载与当前后端兼容的 `torch_mlir-*.whl` 放入 `downloads/`。目录中每种 wheel 只保留一个版本，然后执行：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch>=2.14,<3'
uv pip install --python .venv/bin/python \
  ./downloads/torch_mlir-*.whl ./downloads/mlir_cim22-*.whl
.venv/bin/python python/examples/torch_compile.py
```

成功时示例输出：

```text
CIM22 torch.compile software execution passed (0, 65, 129).
```

它验证自定义 `nn.Module`、编译和三次软件执行。自己的应用也使用这个虚拟环境。

当前验证环境为 Python 3.12、CPU PyTorch 2.14.1 和 torch-mlir 20260923.885。包元数据允许 Python `>=3.12`、PyTorch `>=2.14,<3`，不代表所有组合均已测试。本机编译的 `linux_x86_64` wheel 可能依赖较新的 glibc 及系统动态库；应使用维护者明确标注适用环境的构建。

## 最小模型

将下面的代码保存为 `example.py`：

```python
import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 16)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(inputs))


model = SimpleModel().cpu().eval()
with torch.no_grad():
    model.linear.weight.fill_(1)
    model.linear.bias.fill_(1)

with torch.inference_mode():
    compiled = torch.compile(
        model, backend="cim22", fullgraph=True, dynamic=False,
        options={"runtime": "software"},
    )
    output = compiled(torch.ones(1, 64))
    torch.testing.assert_close(output, torch.full((1, 16), 65.0))
    print(output.shape)
```

运行：

```sh
.venv/bin/python example.py
```

输出：

```text
torch.Size([1, 16])
```

完整的自检脚本见 [examples/torch_compile.py](examples/torch_compile.py)。它采用可精确表示的固定权重；一般模型的 CIM BF16 软件计算存在近似误差。

## 使用约定

- 使用 `torch.inference_mode()` 或 `torch.no_grad()`，输入为连续 CPU FP32 Tensor，保持形状、布局及推理上下文一致。同一模型按顺序执行。
- 只读且无输出别名的参数会冻结为 CIM 权重。修改这些权重后须调用 `torch.compiler.reset()` 并重新编译；运行时权重保留在 CPU 路径。
- AOTAutograd 负责支持的原地更新和状态写回。带状态模型在每个独立序列前应按模型自身 API 重置；不会自动清零。保持各次调用的输入存储重叠关系一致。
- `fullgraph=True` 禁止图断裂，`dynamic=False` 使用静态特化；输入形状变化可能触发重新编译。算子支持仍受 torch-mlir 和原生 lowering 覆盖范围限制。
- 输入 identity/view 别名由 PyTorch 包装层处理；不同计算输出之间的共享存储不保证保留。不同 Tensor 共享底层存储的部分更新目前仍可能被上游拒绝。

使用 SpikingJelly 时，用 `functional.reset_net(model)` 在独立序列前重置神经元状态。

## 后端选项与排查

通过 `torch.compile(..., options={...})` 传入：

| 选项                            | 用途                                             |
| ------------------------------- | ------------------------------------------------ |
| `runtime="software"`          | 不连接设备，执行候选 BF16 软件模型               |
| `runtime="usb"`               | 默认值，使用 Linux USB；BF16 Tensor 输出尚不可用 |
| `usb_config=UsbConfig(...)`   | USB 设备参数，只能与 USB 运行时同时使用          |
| `artifact_dir="build/models"` | 保存每次特化的内部模型 `.so`                   |
| `debug_dir="build/ir"`        | 保存 Linalg、packet MLIR 和目标文件              |

首次调用包含编译、链接和加载；稳态性能应在预热后单独测量。`TORCH_LOGS=dynamo` 查看编译阶段耗时，`TORCH_LOGS=recompiles` 排查重复编译。特化缓存在进程内，保存 `.so` 不会提供跨进程自动缓存或稳定部署接口。

| 问题                       | 检查项                                                         |
| -------------------------- | -------------------------------------------------------------- |
| `Invalid backend: cim22` | 当前解释器是否安装了本项目 wheel，而非仅安装 Torch             |
| 找不到 torch-mlir          | 是否将匹配的 wheel 安装到同一虚拟环境                          |
| 缺少 `cc` 或链接失败     | 是否具备 C 编译/链接开发环境，以及 wheel 所需的系统库          |
| USB 报错或要求配置设备     | 无硬件时显式选择 `runtime="software"`                        |
| 编译失败或数值偏差         | 先运行自检脚本，再保存 `debug_dir`；检查输入约束和 BF16 误差 |

USB 原始诊断可通过 `cim22.Session`、`UsbConfig` 和 `capture_usb` 使用，具体数组与生命周期契约见 [runtime.py](cim22/runtime.py)。USB 传输故障后需关闭 Session 并按设备流程复位；重新打开 Host 句柄不会复位硬件。

## 从源码构建与测试

完整步骤见 [BUILDING](../BUILDING.md#pytorch-编译后端)：准备 LLVM/MLIR 后，可生成 wheel 或构建 editable 安装。Python 源码修改立即可见；C++、TableGen、schema 或 CMake 修改需要显式重建。

[开发与测试](../BUILDING.md#测试与开发)给出重建、pytest、lit 和 Protobuf 更新命令。应用特有的依赖应安装在自己的环境中；Python ONNX、Protobuf 和 pytest 是开发依赖，预编译 wheel 的普通运行无需安装这些包。
