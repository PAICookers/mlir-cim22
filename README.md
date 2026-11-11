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

使用 **Linux x86-64 或 Windows x64**。项目要求 Python 3.12、CPU PyTorch
和与当前后端匹配的 [torch-mlir 20261101 wheel](https://github.com/llvm/torch-mlir/releases/tag/v20261101)。下面的流程把编译器安装到你自己的 uv 虚拟环境中，模型代码不需要放在本仓库内。

### 在自己的应用中安装并推理

先准备已经取得的两个 wheel 文件。`CIM22_WHEEL` 是本项目 wheel，
`TORCH_MLIR_WHEEL` 是对应平台的 torch-mlir wheel。

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.14,<3"
uv pip install --python .venv "$TORCH_MLIR_WHEEL" "$CIM22_WHEEL"
```

在应用中使用 `torch.compile`。第一次调用会捕获、编译并加载模型，后续相同静态特化会复用已生成的模型库。

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(64, 16), nn.ReLU()).cpu().eval()
with torch.inference_mode():
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
    result = compiled(torch.ones(1, 64))
    print(result.shape)
```

`runtime="simulator"` 在 CPU 上执行软件仿真。连接 CIM 板卡后改为
`runtime="usb"`，并按 [USB 与 PLL 配置](python/README.md#usb-与-pll-配置)传入
`UsbConfig`。输入应为静态形状的连续 CPU Tensor。

### 运行仓库示例

从仓库根目录进入 Python 环境后运行固定示例：

```sh
./.venv/bin/python python/examples/torch_compile.py
```

Windows 使用 `./.venv/Scripts/python.exe`。成功时会在
`outputs/torch/` 保存 `compile.log`、模型动态库、调试 IR 和 `result.json`。
TensorFlow 示例需要额外安装依赖：

```sh
uv pip install --python .venv "$CIM22_WHEEL[tensorflow]"
./.venv/bin/python python/examples/tensorflow_compile.py
```

### 运行测试

如果使用源码树，先按 [源码构建指南](docs/BUILDING.md) 配置 `MLIR_DIR`，然后在仓库根目录执行：

```sh
uv sync --project python --python 3.12 --locked \
  -Ccmake.define.MLIR_DIR="$MLIR_DIR"
uv run --project python --no-sync pytest
```

只检查编译器后端或 TensorFlow 接口时，可以缩小范围：

```sh
uv run --project python --no-sync pytest \
  python/tests/test_backend.py python/tests/test_tensorflow.py
```

### 启动 Web UI

Web UI 需要安装 `web` extra，并需要一个只读的 DVS 模型包。模型包至少包含
`weights/dvsgesture_mobilenet_snn_qat_int8.pt` 和 `data/test/`。

```sh
uv pip install --python .venv "$CIM22_WHEEL[web,tensorflow]"
./.venv/bin/cim22-web \
  --model-dir /path/to/model-package \
  --examples-dir /path/to/mlir-pku-cim22/python/examples
```

Windows 使用 `./.venv/Scripts/cim22-web.exe`。浏览器打开
<http://127.0.0.1:8000>。结果和调试文件默认写入
`~/.local/share/cim22-web`，可用 `--data-dir` 指定其他目录。Web 页面中的
PyTorch 和 TensorFlow 示例默认使用软件仿真，USB 模式需要已连接并有访问权限的板卡。

示例显式选择软件运行时；默认运行时是 USB。浮点模型使用 FP32，显式 INT8 乘法配合 INT32 累加可走整数路径。

拿到源码后需要先安装或构建，克隆仓库本身不会提供预编译文件。只有源码时，请从[源码构建指南](docs/BUILDING.md)开始。

## 源码构建与开发

- [MLIR-CIM22 编译器](docs/BUILDING.md#mlir-cim22-编译器)：准备 LLVM/MLIR，直接构建并使用原生工具。
- [PyTorch 编译后端](docs/BUILDING.md#pytorch-编译后端)：构建一体化 wheel。
- [测试与开发](docs/BUILDING.md#测试与开发)：运行 pytest、原生 lit 测试及增量重建。
- [Python 使用说明](python/README.md)：模型示例、后端选项和常见问题。
- [BF16 补偿原理与实现](docs/BF16.md)：sign-split 拆分、尺度恢复、代码入口与误差边界。

`mlir-cim22-opt` 也可单独用于 MLIR 检查和 pass 调试。
