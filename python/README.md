# CIM22 Python 开发指南

`mlir-cim22` 通过 Python entry point 注册 `torch.compile` 后端 `cim22`。PyTorch 是主要接入路径，TensorFlow 通过可选的 `tensorflow` extra 提供。两个前端共用 Linalg 之后的原生编译和 Session 运行时。

## 用户快速开始

下面的命令适合在你自己的应用目录中执行。准备本项目 wheel 和对应平台的
torch-mlir wheel 后，使用 uv 创建独立环境：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.14,<3"
uv pip install --python .venv \
  /path/to/torch_mlir-20261101-<platform>.whl \
  /path/to/mlir_cim22-<version>-<platform>.whl
```

将下面的代码保存为应用中的 Python 文件并运行。`simulator` 不需要连接板卡，
首次调用完成模型编译和加载：

```python
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(64, 16), torch.nn.ReLU()
).cpu().eval()
compiled = torch.compile(
    model,
    backend="cim22",
    fullgraph=True,
    dynamic=False,
    options={"runtime": "simulator", "artifact_dir": "artifacts/linear"},
)
with torch.inference_mode():
    output = compiled(torch.ones(1, 64))
print(output.shape)
```

连接 CIM 板卡时将运行时改为 `usb`，并传入 `UsbConfig`。完整的设备字段和
时序配置见 [USB 与 PLL 配置](#usb-与-pll-配置)。TensorFlow 应用在安装项目
wheel 时加上 extra：

```sh
uv pip install --python .venv \
  "/path/to/mlir_cim22-<version>-<platform>.whl[tensorflow]"
```

仓库提供的可复现实例可以直接运行：

```sh
./.venv/bin/python /path/to/mlir-pku-cim22/python/examples/torch_compile.py
./.venv/bin/python /path/to/mlir-pku-cim22/python/examples/tensorflow_compile.py
```

Windows 将解释器路径替换为 `./.venv/Scripts/python.exe`。示例会在输出目录
保存模型动态库、编译日志、调试 IR 和 `result.json`。

## 运行仓库测试

源码树测试需要 editable 环境和 LLVM/MLIR。先按 [源码构建说明](../docs/BUILDING.md)
准备 `MLIR_DIR`。以下命令在仓库的 `python/` 目录执行：

```sh
uv sync --project . --python 3.12 --locked \
  --extra web --extra tensorflow \
  -Ccmake.define.MLIR_DIR="$MLIR_DIR"
uv run --project . --no-sync pytest
```

只验证 PyTorch 后端和 TensorFlow 接口时，可以运行：

```sh
uv run --project . --no-sync pytest \
  tests/test_backend.py tests/test_tensorflow.py
```

`tensorflow` extra 会安装 TensorFlow、tf2onnx 和 ONNX。Linux 中如果同一进程
先加载 TensorFlow，再尝试首次加载 torch-mlir FX，CIM22 会返回可操作的错误，
建议将两个框架的编译放在独立 Python 进程中。

## PyTorch backend

### 最小示例

显式 INT8 矩阵乘使用只读权重，输入为 `torch.int8`，输出为 `torch.int32`。软件模式下可直接校验结果。

```python
import torch


class Int8Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones((64, 16), dtype=torch.int8), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch._int_mm(x, self.weight)


with torch.inference_mode():
    compiled = torch.compile(
        Int8Linear().eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"runtime": "simulator"},
    )
    x = torch.ones((1, 64), dtype=torch.int8)
    y = compiled(x)
    torch.testing.assert_close(y, torch.full((1, 16), 64, dtype=torch.int32))
```

### 接口契约

- 首次模型调用触发 Dynamo/AOTAutograd 捕获、torch-mlir 降低和原生编译。用 `fullgraph=True, dynamic=False` 检查整图静态捕获，稳态计时从预热后开始。
- 在 `torch.inference_mode()` 或 `torch.no_grad()` 下传入连续 CPU Tensor。输入按编译签名传递，支持 FP32、INT8、INT32。改变形状、布局或推理上下文可能触发重新特化。
- 只读且无输出别名的模块参数可冻结为 CIM 权重。更改冻结参数后调用 `torch.compiler.reset()` 并重新编译。运行时权重走 Host。
- 支持的状态写回与输入别名由 AOTAutograd 管理。带状态模型在独立序列前使用模型自身的重置接口，并保持输入存储重叠关系一致。

INT8 CIM 降低要求显式 `i8 × i8 → i32`、冻结权重和可证明的有符号 INT21 分块结果范围。FP32 收缩可走 CIM BF16，数值验证应使用对应的软件参考。Host 保留与 CIM dispatch 分别验证。

### 编译选项

两个前端接收同一份 `options` 字典。

| 键               | 行为                                                                                                              |
| ---------------- | ----------------------------------------------------------------------------------------------------------------- |
| `runtime` | 默认 `"usb"`，使用 USB 连接 CIM 板卡。`"simulator"` 在 CPU 上模拟 CIM 计算。PyTorch 也支持 `"reference"`，直接执行 CPU PyTorch 算子。 |
| `bf16_compensation` | 默认 `"sign-split"`，可选 `"legacy"` 或 `"none"` 建立对照。详见 [BF16 补偿](../docs/BF16.md)。 |
| `usb_config`   | 可选 `cim22.UsbConfig`，仅用于 USB 模式。由公共 Session 初始化流程应用。                                        |
| `artifact_dir` | 可选 `str` 或 `Path`。保存每次编译的 `graph-<id>.so` 或 `graph-<id>.dll`。                                |
| `debug_dir`    | 可选 `str` 或 `Path`。在同名 `graph-<id>/` 下保存 `model.linalg.mlir`、`model.packet.mlir` 和目标文件。 |

PyTorch 的 `reference` 忽略 BF16 补偿策略，且不生成原生制品。TensorFlow 可直接调用原 `tf.function` 取得参考结果。

## 按模块名选择 CIM 执行

`cim22.layers` 提供通用 PyTorch 分层执行接口，DVS Web 是它的一个应用。
对其他网络同样使用完整的 `named_modules()` 名称选择 `Conv2d`、`Linear`
叶子模块；其余模块继续由 PyTorch CPU 执行。

```python
import torch
from cim22.layers import LayerOffload, discover_cim_layers

model = torch.nn.Sequential(
    torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
).cpu().eval()
x = torch.ones(1, 4)
print(discover_cim_layers(model, (x,)))
compiled = LayerOffload(
    model, ["0", "2"], (x,),
    runtime="simulator", artifact_dir="artifacts/selected",
)
y = compiled(x)
print(compiled.layer_artifacts, compiled.layer_compile_seconds)
print(compiled.dispatch_count)
```

- `LayerOffload` 原地编译选中的叶子层，构造时完成预热；参考模型请事先单独
  构造或深拷贝。层名不可为空、重复、未知或指向共享模块实例。
- 模型使用 CPU、eval、固定权重及后端支持的静态输入签名，执行采用
  `torch.inference_mode()`；模块之外的控制流和输出结构由 PyTorch 保留。
  不提供通用模型导入器，也不保证所有网络算子均可在 CIM 上执行。
- 不传 `example_inputs` 的发现接口只枚举结构，尺寸列表为空。传入样本会
  执行一次模型并通过临时 hook 记录尺寸；有状态模型由应用负责重置。
- 有状态模型可传 `reset(model)` 回调，在每轮预热前和全部预热结束后重置；
  正式推理的序列边界仍由应用负责重置，不假定所有网络都使用脉冲状态。
- 每层独立保存产物及 IR，检查实际 CIM dispatch；后续调用若新增编译产物
  会报错。使用不同输入尺寸或改变冻结权重时，请重新构造模型并编译。
- 默认运行时为 `usb`，可传 `usb_config`；软件验证显式使用 `simulator`。
  这些参数复用现有后端，未增加精度或量化选项。

## TensorFlow 接口

在 editable 环境的安装命令上加 `--extra tensorflow`。安装 wheel 时使用同一版本的项目 wheel 并指定 `[tensorflow]` extra。两套最小示例分别在新 Python 进程中运行。

```python
import tensorflow as tf

from cim22.tensorflow import compile as compile_tf


@tf.function(input_signature=[tf.TensorSpec((1, 64), tf.int8)])
def infer(x):
    weight = tf.ones((64, 16), dtype=tf.int8)
    return tf.raw_ops.BatchMatMulV3(x=x, y=weight, Tout=tf.int32)


with compile_tf(infer, options={"runtime": "simulator"}) as compiled:
    y = compiled(tf.ones((1, 64), dtype=tf.int8))
    tf.debugging.assert_equal(y, tf.fill((1, 16), 64))
    assert compiled.session.dispatch_count == 0
```

- `compile_tf` 接收带平坦静态 `input_signature` 的无状态 `tf.function`，在返回前冻结权重并完成编译。Keras 调用放在 `tf.function` 内，并传入 `training=False`。
- `compiled(*inputs)` 接收 eager TF 张量，返回保持原容器结构的 TF 张量。使用 `with` 或 `close()` 释放 Session。底层 NumPy 接口和 dispatch 计数通过 `compiled.session` 获取。
- TF 主进程经 tf2onnx 导出 ONNX。编译 worker 使用安装的 torch-mlir wheel 将 ONNX 降低到 Linalg，再调用公共原生编译。应用进程加载模型产物并初始化 Session。
- `MLIRCIM22_ENABLE_ONNX` 控制原生 ONNX 诊断工具的构建。TF 编译入口使用上面的 worker 路径。
- 当前接口面向 eager 推理。在另一个 `tf.function` 中嵌入调用及训练不在此接口范围内。Linux 已加载 TensorFlow 时，CIM22 会拒绝首次在同一进程加载 PyTorch 路径的 torch-mlir FX 绑定。

完整的 [TF INT8 示例](examples/tensorflow_compile.py) 使用多维输入的两层 BatchMatMul 和截断 ReLU；[PyTorch 示例](examples/torch_compile.py) 使用 BF16 Conv2d、ReLU 和 Linear，并输出最终激活值。每次运行将 `compile.log`、`graph-<id>.so/.dll` 和 `ir/graph-<id>/` 调试文件保存在当前工作目录的示例输出目录，并打印 CIM 调用次数和输出误差。

## Session 与产物

用 `cim22.Session(Path(...))` 重新加载已保存的同平台模型动态库。按 `input_shapes` 和 `input_dtypes` 顺序传入连续、对齐的 NumPy 数组，结果为平坦 tuple。框架输出结构由各自的适配层恢复。模型所需的原生运行库应与产物配套。

当前公共 Session 边界支持 FP32、INT8 和 INT32；BF16 算子可在编译器内部执行，但原生 BF16 Tensor 作为直接输入或输出的公共 ABI 仍待后续工作树实现。BF16 示例因此将 BF16 计算放在模型内部，并输出当前 Session ABI 可承载的激活结果。

未设置 `artifact_dir` 时，临时模型随 Session 生命周期清理。PyTorch 编译后的 callable 持有 Session，TF callable 可显式关闭。Windows 上应在 Session 关闭后清理其加载的 DLL。

`num_transactions` 表示模型产物包含的 CIM transaction 数量，`dispatch_count` 表示最近一次调用实际发生的 dispatch 数量。使用 `debug_dir` 检查分区和降低结果，用实际 dispatch 验证执行路径。

### USB 与 PLL 配置

前端通过 `options={"runtime": "usb", "usb_config": config}` 交给公共运行时。直接加载 USB 模型动态库时，在首次 CIM dispatch 前调用 `session.configure_usb(config)`。

```python
from cim22 import CIMTimingConfig, UsbConfig

config = UsbConfig(
    vendor_id=0x04B4,
    product_id=0x00F1,
    bulk_in_endpoint=0x81,
    bulk_out_endpoint=0x01,
    timing=CIMTimingConfig(
        work_delay_ms=10,
        readback_select_delay_ms=40,
    ),
)
```

按实际 FPGA/FX3 固件设置 VID/PID、端点和接口。Linux 为对应设备配置 USB 访问权限，Windows 使用 CyAPI 与 CyUSB3 驱动。上述配置使用手动时钟。Linux 或 Windows 如需通过串口配置时钟，传入 `clock_frequency_mhz=24` 与实际 `pll_port`，表示写入、计算、回读三个阶段均使用 24 MHz；需要分阶段使用不同频率时，用 `PLLConfig` 指定三个阶段的命令。未列出的频率按实际频率选择最近的 `available` 表项，距离相同时选较低频率，并通过 `RuntimeWarning` 告知请求频率、选中频率及串口命令；`available` 是供应方表中的分类，不代表本次实板稳定性验证。

`session.configure_usb()` 打开 USB 设备并绑定时钟回调。Python 使用随 wheel 依赖自动安装的 pyserial，串口在首次执行阶段打开，随 Session 保持连接。C++ 在写入、计算、回读阶段的第一条 USB 命令前同步调用配置回调。首次命令要求匹配回显；后续变化的命令不读取或等待回显，相同的当前命令跳过发送。三个阶段使用相同频率时，首次只需发送一次。重新配置会恢复首次回显要求。

Windows 可直接使用 `COM10` 等端口名称，Linux 使用实际设备路径，无需独立配置脚本。未提供串口表示人工管理时钟；显式指定串口后，设备缺失、权限不足、写入或回显失败均会报错并停止阶段执行，必须显式重新配置后才能恢复。仅记录请求频率不能作为串口配置成功的证据。频率选择与参数校验见 [pll.py](cim22/pll.py) 和 [runtime.py](cim22/runtime.py)。

每条串口命令采用 200 ms 的写入及首次回显期限。后续无回显发送检查驱动输出队列为空，再保留三字节 9600/8N1 的 4 ms 发送等待。驱动队列不一定覆盖 USB 转串口适配器内部缓冲区；该等待是主机发送保障，不是实测校准、PLL 锁定或芯片已切换频率的证明。

原生 C/C++ 使用 `cim22_session_configure_usb_with_clock` 和 `CIM22ClockControl` 注册同步阶段回调；调用方保持函数指针和上下文有效，直到设备被替换或 Session 销毁，且不能在回调中重入同一 Session。旧的 PLL 串口配置 ABI 已移除，原生调用方需要迁移并重新编译，Python 包也必须与新运行库一起重新安装。

`UsbConfig.timing` 统一管理板卡时序：`startup_delay_ms`、
`cache_write_delay_ms`、`weight_write_delay_ms`、`work_delay_ms` 和
`readback_select_delay_ms`。默认值分别为 `20/1/2/10/40`，单位为毫秒，允许设为 `0`。

## 安装

Linux 和 Windows 共用 Python 3.12 与 uv 安装流程。按平台选择项目和
torch-mlir `20261101` wheel，匹配地址见 [pyproject.toml](pyproject.toml)。

| 平台         | 项目 wheel 平台标签                           | 虚拟环境解释器               | 模型编译工具                                            |
| ------------ | --------------------------------------------- | ---------------------------- | ------------------------------------------------------- |
| Linux x86-64 | `linux_x86_64` 或 `manylinux_2_28_x86_64` | `.venv/bin/python`         | `cc` 和 C 链接工具                                    |
| Windows x64  | `win_amd64`                                 | `.venv/Scripts/python.exe` | MSVC x64 开发者终端中的 `cl`、`link` 和 Windows SDK |

### 预编译 wheel

将项目与 torch-mlir 的 wheel 放入 `downloads/`，并将命令中的占位符替换为实际文件名。`--python .venv` 指向虚拟环境目录，两平台的安装命令相同。

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu "torch>=2.14,<3"
uv pip install --python .venv "downloads/<torch-mlir-wheel>.whl" "downloads/<cim22-wheel>.whl"
```

本机构建的 Linux wheel 依赖系统 libusb。发行包和平台依赖见 [wheel 构建说明](../docs/BUILDING.md#github-actions-预编译-wheel)。

### Editable 开发环境

按 [BUILDING](../docs/BUILDING.md) 准备原生工具链。在仓库根目录执行，将 `<MLIR_DIR>` 替换为包含 `MLIRConfig.cmake` 的实际目录。Windows USB 构建追加 `"-Ccmake.define.CYAPI_ROOT=<仓库根目录>/third_party/cyapi"`。

```sh
uv sync --project python --python 3.12 --locked "-Ccmake.define.MLIR_DIR=<MLIR_DIR>"
```

## 构建 wheel

在各自的目标平台从仓库根目录执行。准备 Python 3.12、CMake、Ninja 和本平台的 LLVM/MLIR 工具链。当前验证的 LLVM/MLIR 版本为 22.1.8，`MLIR_DIR` 指向含 `MLIRConfig.cmake` 的目录。两条命令使用独立的 CMake 缓存并将 wheel 写入 `python/dist/`。TensorFlow 依赖通过同一 wheel 的 `[tensorflow]` extra 在安装时选择。

### Linux x86-64

按 [Linux 构建依赖](../docs/BUILDING.md#环境与依赖) 准备 C/C++ 工具链及 libusb 开发库。

```sh
export MLIR_DIR="/path/to/llvm-build/lib/cmake/mlir"
export CMAKE_GENERATOR=Ninja
uv build --python 3.12 python --wheel --out-dir python/dist \
  "-Cbuild-dir=$PWD/build/wheel-linux" \
  "-Ccmake.build-type=Release" \
  "-Ccmake.define.MLIR_DIR=$MLIR_DIR"
```

本机构建产物为 `mlir_cim22-<version>-py3-none-linux_x86_64.whl`。发行用的 `manylinux_2_28_x86_64` wheel 由 [wheel 工作流](../docs/BUILDING.md#github-actions-预编译-wheel) 在对应容器中构建并经 `auditwheel repair` 处理。

### Windows x64

启动 PowerShell，按 [Windows 原生构建说明](../docs/BUILDING.md#windows-x64-原生编译器与-usb-runtime) 准备匹配的 LLVM/MLIR。USB 构建使用仓库的 x64 CyAPI。PowerShell 中将每个完整的 `-C...` 参数括在引号内。

```powershell
$env:MLIR_DIR = "C:/path/to/llvm-build-windows/lib/cmake/mlir"
$env:CMAKE_GENERATOR = "Ninja"
$cyapiRoot = (Resolve-Path "third_party/cyapi").Path
uv build --python 3.12 python --wheel --out-dir python/dist `
  "-Cbuild-dir=$PWD/build/wheel-windows" `
  '-Ccmake.build-type=Release' `
  "-Ccmake.define.MLIR_DIR=$env:MLIR_DIR" `
  "-Ccmake.define.CYAPI_ROOT=$cyapiRoot"
```

本机构建产物为 `mlir_cim22-<version>-py3-none-win_amd64.whl`。平台工具、编译器库、运行库及模型链接所需的导入库随包安装。

在各平台用干净的 Python 环境安装生成的 wheel 和匹配的 torch-mlir wheel，
从源码目录之外依次运行 `torch_compile.py`、`tensorflow_compile.py` 和
focused pytest；安装 `[web,tensorflow]` extra 后，再运行 `cim22-web --help`
并检查 Web UI 首页、静态资源和 `/api/state`。发行构建和源码构建见
[BUILDING](../docs/BUILDING.md#源码构建与离线构建)。

## 开发与验证

| 改动                            | 从哪里检查                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 捕获、冻结与 AOT 边界   | [backend.py](cim22/backend.py) 和 [test_backend.py](tests/test_backend.py)                                                                    |
| TF 签名、输出结构与编译进程隔离 | [tensorflow.py](cim22/tensorflow.py)、[_tensorflow_compile.py](cim22/_tensorflow_compile.py) 和 [test_tensorflow.py](tests/test_tensorflow.py) |
| 原生编译、链接及模型发布        | [compiler.py](cim22/compiler.py)、[test_compile_tool.py](tests/test_compile_tool.py) 和 [test_artifacts.py](tests/test_artifacts.py)     |
| Session 所有权、USB 与 PLL      | [runtime.py](cim22/runtime.py) 和 [test_usb_runtime.py](tests/test_usb_runtime.py)                                                            |

从仓库根目录运行，TF 用例需要安装 extra。

```sh
cd python
.venv/bin/python -m pytest tests/test_backend.py tests/test_usb_runtime.py
.venv/bin/python -m pytest tests/test_tensorflow.py
```

排查重复编译时使用 `TORCH_LOGS=recompiles`。定位 lowering 或链接问题时，先用 `debug_dir` 保留 IR 和目标文件，再检查安装包加载的原生组件。独立 CMake 构建可用 `CIM22_OPT`、`CIM22_COMPILE_TOOL`、`CIM22_RUNTIME_LIBRARY` 指向同一构建的产物，`CIM22_CC` 可指定模型链接工具。

修改 C++、TableGen、schema 或 CMake 后，复用原构建目录和配置，通过 `uv sync --reinstall-package mlir-cim22` 重装 editable 包。完整命令、Ruff、原生 lit 和 wheel 验收见 [BUILDING 的测试与开发](../docs/BUILDING.md#测试与开发)。软件执行、模拟 USB 配置及实板结果分开记录。

Web 工作台开发见 [web/README.md](../web/README.md)。
