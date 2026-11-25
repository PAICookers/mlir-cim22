# 安装 wheel 并编译第一个模型

在用户已有的 uv 应用环境中安装 CIM22 及其依赖。确认应用使用 Python 3.12，
并准备模型链接工具。

| 平台         | 环境要求                                                                        |
| ------------ | ------------------------------------------------------------------------------- |
| Linux x86-64 | glibc >= 2.28；`cc` 可执行文件及系统 C 运行库开发文件，GCC 或 Clang 均可      |
| Windows x64  | 在 MSVC x64 开发者 PowerShell 中运行，确保 `cl`、`link` 和 Windows SDK 可用 |

wheel 包含原生编译器和运行库；模型编译最后仍需调用系统 C 工具链链接模型动态库。
Linux 若没有 `cc` 但已安装 GCC/Clang，可设置 `CIM22_CC=gcc` 或
`CIM22_CC=clang`。

## 准备 wheel

1. 从 [Build wheels 的成功运行](https://github.com/PAICookers/mlir-cim22/actions/workflows/wheels.yml)
   下载本平台的 `mlir_cim22-*.whl`，记录运行提交。
   Linux 选择 `manylinux_2_28_x86_64`，Windows 选择 `win_amd64`。
2. 从 [torch-mlir v20261101](https://github.com/llvm/torch-mlir/releases/tag/v20261101)
   下载 CPython 3.12 对应的平台 wheel。

   | 平台 | torch-mlir wheel |
   | --- | --- |
   | Linux x86-64 | [torch_mlir-20261101-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl](https://github.com/llvm/torch-mlir/releases/download/v20261101/torch_mlir-20261101-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl) |
   | Windows x64 | [torch_mlir-20261101-cp312-abi3-win_amd64.whl](https://github.com/llvm/torch-mlir/releases/download/v20261101/torch_mlir-20261101-cp312-abi3-win_amd64.whl) |

保留实际文件名，将命令中的 `<torch-mlir-wheel>` 和 `<cim22-wheel>` 分别换成
文件名中 `.whl` 前的部分，并按实际存放位置替换路径。

### 可选：校验完整性

需要核对下载完整性时，从同一次构建下载 CIM22 wheel 对应的 `.whl.sha256`，
与 wheel 放在一起。在该目录下，Linux 用
`sha256sum --check "<cim22-wheel>.whl.sha256"` 校验。Windows PowerShell 用
`Get-FileHash -Algorithm SHA256 "<cim22-wheel>.whl"` 计算摘要，与校验文件中的
第一列比较。

## 安装到现有应用环境

在用户自己的应用工程目录执行以下命令。`uv add` 将两个预编译 wheel
登记为应用依赖，并安装到应用已有的工程环境。

已有符合 `torch>=2.14,<3` 的 PyTorch 可直接使用。若缺少 PyTorch 且应用需要
CPU 版本，可执行以下命令。已有不兼容版本时，先按应用自身的依赖约束调整。

```sh
uv add --index pytorch-cpu=https://download.pytorch.org/whl/cpu "torch>=2.14,<3"
```

在同一个环境中安装两个 wheel，然后检查依赖。

```sh
uv add "./wheels/<torch-mlir-wheel>.whl" "./wheels/<cim22-wheel>.whl"
uv run uv pip check
```

应用的 `pyproject.toml` 和 `uv.lock` 会记录 wheel 路径。将 wheel 保留在
相应位置，供后续 `uv run` 或 `uv sync` 使用。Web 或 TensorFlow 应用按
对应小节选择 extra。

安装 `tensorflow` extra 会由 TensorFlow 自动带入 `ml_dtypes`。仅使用
PyTorch 时，torch-mlir `20261101` 的基础 wheel 将该包作为可选支持。
如果模型含 BF16 常量，且 torch-mlir 提示缺少该包，可运行
`uv add ml_dtypes` 补装。CIM22 对 FP32 模型在编译器内部进行的
BF16 降低可直接使用。

本项目 wheel CI 使用 `torch==2.14.1+cpu`。Linux 最小模型编译也已实测
`torch==2.15.0+cpu`；其余版本和模型组合以实际编译结果为准。

## 检查 SDK 安装与用法

在应用环境中运行以下命令。

```sh
uv run cim22 --info
uv run cim22 --help
```

用 `--info` 确认解释器和模块路径指向应用环境。编译选项见
[Python 接口](python-api.md#编译选项)。

## 编译模型

将下面的完整程序保存为 `compile_model.py`：

```python
import torch
from torch import nn

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(64, 16), nn.ReLU()).cpu().eval()
inputs = torch.ones(1, 64)

with torch.inference_mode():
    reference = model(inputs)
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={
            "runtime": "simulator",
            "artifact_dir": "artifacts/linear",
            "debug_dir": "artifacts/linear-ir",
        },
    )
    output = compiled(inputs)
    torch.testing.assert_close(output, reference, rtol=0.05, atol=0.05)

print("Output shape:", tuple(output.shape))
print("Matches PyTorch reference")
```

运行：

```sh
uv run python compile_model.py
```

成功时打印：

```text
Output shape: (1, 16)
Matches PyTorch reference
```

首次调用 `compiled(inputs)` 才捕获、编译并加载模型；之后相同静态特化可复用。
`artifacts/linear/` 保存 `graph-<id>.so`（Linux）或 `graph-<id>.dll`（Windows）；
`artifacts/linear-ir/graph-<id>/` 保存 `model.linalg.mlir`、`model.packet.mlir`
和目标文件。

将 `model` 和 `inputs` 替换为自己的推理模型与样本即可沿用此入口。模型应处于
`eval()` 状态，输入为连续 CPU Tensor；使用 `fullgraph=True, dynamic=False`
捕获静态图。浮点 CIM 计算存在数值误差，比较容差应按模型要求选择。
更多约束见 [接口契约](python-api.md#接口契约)，部分层编译见 [按模块名选择 CIM 执行](python-api.md#按模块名选择-cim-执行)。

`runtime="simulator"` 在 CPU 上模拟 CIM 计算，不需要板卡。实际连接板卡时改为
`runtime="usb"`，并按 [USB 与 PLL 配置](python-api.md#usb-与-pll-配置)传入 `UsbConfig`。
未显式指定 `runtime` 时默认为 `usb`。

## 可选：TensorFlow 与仓库示例

TensorFlow 应用在同一个环境中追加安装项目 wheel 的 `tensorflow` extra：

```sh
uv add "./wheels/<cim22-wheel>.whl[tensorflow]"
```

[TensorFlow 接口](python-api.md#tensorflow-接口)给出了自己的程序中可用的完整示例。
仓库的示例脚本不随 wheel 安装，请选用项目 wheel 对应提交的脚本。在应用目录运行，并将下面的路径替换为实际路径：

```sh
uv run python /path/to/mlir-cim22/python/examples/torch_compile.py
uv run python /path/to/mlir-cim22/python/examples/tensorflow_compile.py
```
