# CIM22 Python 开发指南

`mlir-cim22` 通过 Python entry point 注册 `torch.compile` 后端 `cim22`。PyTorch 是主要接入路径，TensorFlow 通过可选的 `tensorflow` extra 提供。两个前端共用 Linalg 之后的原生编译和 Session 运行时。

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
    assert compiled.session.dispatch_count > 0
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

按实际 FPGA/FX3 固件设置 VID/PID、端点和接口。Linux 为对应设备配置 USB 访问权限，Windows 使用 CyAPI 与 CyUSB3 驱动。上述配置使用手动时钟。Linux 如需自动配置，传入 `clock_frequency_mhz` 与实际 `pll_port`，或使用 `PLLConfig` 指定三个阶段的命令。频率选择和设备字段校验见 [pll.py](cim22/pll.py) 和 [runtime.py](cim22/runtime.py)。

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
| 原生编译、链接及模型发布        | [compiler.py](cim22/compiler.py)、[test_native_compiler.py](tests/test_native_compiler.py) 和 [test_artifacts.py](tests/test_artifacts.py)     |
| Session 所有权、USB 与 PLL      | [runtime.py](cim22/runtime.py) 和 [test_usb_runtime.py](tests/test_usb_runtime.py)                                                            |

从仓库根目录运行，TF 用例需要安装 extra。

```sh
cd python
.venv/bin/python -m pytest tests/test_backend.py tests/test_usb_runtime.py
.venv/bin/python -m pytest tests/test_tensorflow.py
```

排查重复编译时使用 `TORCH_LOGS=recompiles`。定位 lowering 或链接问题时，先用 `debug_dir` 保留 IR 和目标文件，再检查安装包加载的原生组件。独立 CMake 构建可用 `CIM22_OPT`、`CIM22_RUNTIME_LIBRARY`、`CIM22_COMPILER_LIBRARY` 指向同一构建的产物，`CIM22_CC` 可指定模型链接工具。

修改 C++、TableGen、schema 或 CMake 后，复用原构建目录和配置，通过 `uv sync --reinstall-package mlir-cim22` 重装 editable 包。完整命令、Ruff、原生 lit 和 wheel 验收见 [BUILDING 的测试与开发](../docs/BUILDING.md#测试与开发)。软件执行、模拟 USB 配置及实板结果分开记录。

Web 工作台开发见 [web/README.md](../web/README.md)。
