# 安装 wheel 并编译第一个模型

以下流程在应用目录中执行，使用预编译 wheel；无需构建 LLVM/MLIR、
设置 `MLIR_DIR` 或安装本仓库源码。准备 Python 3.12 和 uv，以及模型链接工具：

| 平台         | 环境要求                                                                        |
| ------------ | ------------------------------------------------------------------------------- |
| Linux x86-64 | glibc >= 2.28；`cc` 可执行文件及系统 C 运行库开发文件，GCC 或 Clang 均可      |
| Windows x64  | 在 MSVC x64 开发者 PowerShell 中运行，确保 `cl`、`link` 和 Windows SDK 可用 |

wheel 包含原生编译器和运行库；模型编译最后仍需调用系统 C 工具链链接模型动态库。
Linux 若没有 `cc` 但已安装 GCC/Clang，可设置 `CIM22_CC=gcc` 或
`CIM22_CC=clang`。

## 准备 wheel

1. 从 [Build wheels 的成功运行](https://github.com/PAICookers/mlir-cim22/actions/workflows/wheels.yml)
   下载本平台的 `mlir_cim22-*.whl` 和对应的 `.sha256` 文件，记录运行提交。
   Linux 选择 `manylinux_2_28_x86_64`，Windows 选择 `win_amd64`。
2. 从 [torch-mlir v20261101](https://github.com/llvm/torch-mlir/releases/tag/v20261101)
   下载 CPython 3.12 对应的 wheel：
   | 平台         | torch-mlir wheel                                                                                                                                                                                                               |
   | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | Linux x86-64 | [torch_mlir-20261101-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl](https://github.com/llvm/torch-mlir/releases/download/v20261101/torch_mlir-20261101-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl) |
   | Windows x64  | [torch_mlir-20261101-cp312-abi3-win_amd64.whl](https://github.com/llvm/torch-mlir/releases/download/v20261101/torch_mlir-20261101-cp312-abi3-win_amd64.whl)                                                                     |

创建并进入应用目录，将两个 wheel 和项目 wheel 的 `.sha256` 放入其中的 `wheels/`：

```sh
mkdir my-cim22-app
cd my-cim22-app
mkdir wheels
```

后续命令均在此目录执行。下载的文件保留实际文件名；命令中的
`<torch-mlir-wheel>` 和 `<cim22-wheel>` 分别替换为文件名中 `.whl` 前的部分。安装前核对项目 wheel 的 SHA-256：在 Linux 的 `wheels/` 中运行 `sha256sum --check *.whl.sha256`；在 Windows PowerShell 中用 `Get-FileHash -Algorithm SHA256` 计算该 wheel 的摘要并与配套文件的首列比较。两者一致后回到应用目录。

## 创建环境并安装

```sh
uv venv --python 3.12
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.14,<3"
uv pip install "wheels/<torch-mlir-wheel>.whl" "wheels/<cim22-wheel>.whl" ml_dtypes
uv pip check
```

`ml_dtypes` 用于 torch-mlir 的 BF16 常量导入；所用上游 wheel 未将其声明为依赖，
因此在安装时一并补齐。Web 或 TensorFlow 用户还需按各自小节安装对应 extra。
`uv pip check` 应显示依赖兼容。后续 `uv run` 会使用应用目录中的 `.venv`，
无需手动激活环境。`torch>=2.14,<3` 是本项目声明的依赖范围；torch-mlir
`20261101` 的基础 wheel 没有声明 torch 版本约束，不代表任意组合都经过验证。
本项目 wheel CI 使用 `torch==2.14.1+cpu`；Linux 最小模型编译也已实测
`torch==2.15.0+cpu`。其他模型和版本组合以实际编译结果为准。

## 检查 SDK 安装与用法

使用包含 `cim22` 命令的 wheel 时，在应用目录中运行以下命令。

```sh
uv run cim22 --info
uv run cim22 --help
```

`--info` 显示 wheel 版本、当前 Python 解释器、虚拟环境和导入模块的路径。
使用本节创建的环境时，虚拟环境应指向应用目录中的 `.venv`，模块路径应位于其中的
`site-packages`。可用 `uv run cim22 --version` 单独查询版本，
也可通过 `uv run python -m cim22` 使用相同选项。
`--help` 给出应用调用示例及 `runtime`、`artifact_dir`、`debug_dir` 的用途。
完整模型示例见下一节，全部选项见 [Python 接口](python-api.md#编译选项)。

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
和目标文件。此程序不会额外生成示例的 `result.json` 或 `compile.log`。

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
uv pip install "wheels/<cim22-wheel>.whl[tensorflow]" ml_dtypes
```

[TensorFlow 接口](python-api.md#tensorflow-接口)给出了自己的程序中可用的完整示例。
仓库的示例脚本不随 wheel 安装，请选用项目 wheel 对应提交的脚本。在应用目录运行，并将下面的路径替换为实际路径：

```sh
uv run python /path/to/mlir-cim22/python/examples/torch_compile.py
uv run python /path/to/mlir-cim22/python/examples/tensorflow_compile.py
```

## 已有应用工程或在仓库中运行

上述 `uv run python` 面向不含 `pyproject.toml` 的独立应用目录。
已有 uv 工程时，将两个本地 wheel 和 CPU PyTorch 配置到该工程自己的依赖中，
由工程管理环境。不要在本仓库的 `python/` 中对 wheel 环境执行 `uv sync` 或
不带选项的 `uv run`，否则 uv 会安装当前源码项目并触发原生构建。

在仓库的 `python/` 目录运行示例时，先将 wheel 安装到此处的 `.venv`，再跳过项目同步：

```sh
uv run --no-project python examples/torch_compile.py
```

直接运行 `examples/` 中的脚本时，Python 优先查找脚本所在的目录。若在仓库的
`python/` 目录用 `python -m` 或 `python -c` 验证 wheel，则加上 `-P`，
避免当前目录下的源码包覆盖已安装的包。
