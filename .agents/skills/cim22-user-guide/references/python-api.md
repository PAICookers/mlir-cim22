# CIM22 Python 接口

[PyTorch 后端](#pytorch-后端) · [编译选项](#编译选项) · [指定层执行](#按模块名选择-cim-执行) · [TensorFlow](#tensorflow-接口) · [加载产物](#session-与产物) · [USB 与 PLL](#usb-与-pll-配置)

## PyTorch 后端

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

- 通过 `torch.compile(model, backend="cim22", fullgraph=True, dynamic=False, options=...)`
  接入。在原生模式下，首次调用触发 Dynamo 捕获及 AOTAutograd 处理，再由
  torch-mlir 和原生编译器生成模型。后续调用可复用相同的静态特化，稳态计时从预热后开始。
- 在 `torch.inference_mode()` 或 `torch.no_grad()` 下传入连续 CPU Tensor。
  原生 Session 的输入和输出支持 FP32、INT8、INT32。改变形状、布局或推理
  上下文可能触发重新特化。
- 只读且无输出别名的模块参数可冻结为 CIM 权重。更改冻结参数后调用 `torch.compiler.reset()` 并重新编译。运行时权重走 Host。
- 原生模式下，支持的状态写回与输入别名由 AOTAutograd 管理。带状态模型在独立序列前使用模型自身的重置接口，并保持输入存储重叠关系一致。

INT8 CIM 降低要求显式 `i8 × i8 → i32`、冻结权重和可证明的有符号 INT21 分块结果范围。FP32 收缩可走 CIM BF16，数值验证应使用对应的软件参考。Host 保留与 CIM dispatch 分别验证。

### 编译选项

| `options` 键 | 行为 |
| --- | --- |
| `runtime` | 默认 `"usb"`，在 CIM 板卡上执行已映射的 CIM 运算。`"simulator"` 在 CPU 上模拟这些 CIM 运算。PyTorch 另支持 `"reference"`，用原始 CPU PyTorch 算子执行 Dynamo 捕获的图。 |
| `bf16_compensation` | 原生模式默认 `"sign-split"`，可选 `"legacy"` 或 `"none"`。详见仓库中的 `docs/BF16.md`。 |
| `usb_config` | `cim22.UsbConfig`，用于打开 USB 连接。仅用于 `"usb"`，与 `usb_session` 互斥。 |
| `usb_session` | 已配置的 `cim22.Session`，新模型共享其板卡连接、权重驻留与时钟控制。仅用于 `"usb"`，与 `usb_config` 互斥。 |
| `artifact_dir` | 可选 `str` 或 `Path`。整图编译保存 `graph-<id>.so` 或 `graph-<id>.dll`。 |
| `debug_dir` | 可选 `str` 或 `Path`。整图编译在 `graph-<id>/` 下保存 `model.linalg.mlir`、`model.packet.mlir` 和 `model.o`（Windows 为 `model.obj`）。 |
| `cim_layers` | 仅 PyTorch 原生模式。省略时整图自动映射。可传入一个模块路径、非空列表或非空元组。 |
| `compile_strictness` | 需同时传入 `cim_layers`。默认 `"strict"`，要求每个所选模块至少映射一个 CIM transaction。`"allow_host"` 允许零 transaction 的所选模块在 Host 运行。 |

使用板卡执行 CIM 时，按[设备配置](#usb-与-pll-配置)提供 `usb_config` 或
`usb_session`。整图自动映射时，编译器决定哪些算子部署到 CIM，编译成功的整图
可能只有 Host 计算。

`reference` 支持 CPU 输入，要求在无梯度的推理上下文中运行。它直接执行捕获的图，
跳过 AOTAutograd 与原生编译，忽略 `bf16_compensation`，并拒绝 `cim_layers`、
`artifact_dir` 和 `debug_dir`。TensorFlow 的参考结果可直接调用原 `tf.function` 获取。

## 按模块名选择 CIM 执行

在传给 `torch.compile` 的原模型上调用 `model.named_modules()`，查看名称非空的模块。
其中的名称就是相对该模型的完整路径。例如 `encoder.0` 表示 `encoder` 中的
第一个子模块，`head` 表示名为 `head` 的模块。单选时在 `options` 中传入
`"cim_layers": "encoder.0"`，多选时传入 `"cim_layers": ["encoder.0", "head"]`。
选择父模块也会覆盖后代和模块内直接调用的功能式算子，例如
`torch.nn.functional.linear`。后端按捕获到的模块归属拆分 AOT forward 图。
未选中的区域在 PyTorch CPU 上执行。影响跨分区别名、原地更新或读取存储布局
的操作也可保留在 PyTorch 中。每个选择是否能部署由首次调用时的编译结果确认。

Linux x86-64 和 Windows x64 使用相同的 Python 代码与选项。按
[安装指南](install.md)准备对应平台的 wheel 和链接工具，即使使用模拟器也需要
链接模型动态库。下面的示例从模型列出 `0`、`1`、`2`，再选择两个 `Linear`。

```python
import torch
from cim22.backend import compilation_report

torch.manual_seed(42)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
).cpu().eval()
for name, module in model.named_modules():
    if name:
        print(name, type(module).__name__)

x = torch.ones(1, 4)
with torch.inference_mode():
    expected = model(x)
    compiled = torch.compile(
        model,
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={
            "cim_layers": ["0", "2"],
            "compile_strictness": "strict",
            "runtime": "simulator",
            "artifact_dir": "artifacts/selected",
        },
    )
    actual = compiled(x)
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

report = compilation_report("artifacts/selected")
assert report is not None
assert all(count > 0 for count in report.layer_dispatch_counts.values())
print(report.layer_artifacts)
print("CIM dispatches:", report.layer_dispatch_counts)
```

- 将示例中的 `"cim_layers": ["0", "2"]` 改成 `"cim_layers": "2"` 即可单选。
  每次 `compiled(x)` 都执行完整模型，报告按所选模块列出实际 CIM 调用次数。
- 省略 `cim_layers` 时整图自动映射。显式名单拒绝空路径、重复、未知路径、
  父子重叠及共享模块导致的歧义。首次调用句柄时捕获并编译，未捕获的目标会报错。
  模型需要产生 Dynamo 可提交给后端的 FX 图。只返回 Python 常量的模型可能跳过后端。
- `strict` 按模块判断。一个模块拆成多个分区时，含 CIM transaction 的分区
  使用原生 Session，编译结果为零 transaction 的分区继续用 PyTorch 执行。
  `allow_host` 允许整个所选模块为零 transaction，并记录含模块路径的警告。
  非法名单、未捕获目标、编译、链接和设备错误在两种模式中都会报错。
  分区的编译或 USB 初始化失败时，整批编译失败并清理本次暂存的产物。
- 选层需要 CPU eval 模型、`fullgraph=True`、`dynamic=False` 和推理上下文。
  原模型的输出结构由 PyTorch 恢复。有状态模型由应用在样本或序列边界重置。
  仅供未选中区域使用的参数继续由 PyTorch 读取。
- 含 CIM transaction 的分区保存到
  `artifact_dir/graph-<id>/submod_<n>.so`（Windows 为 `.dll`）。
  调试文件通常位于 `debug_dir/graph-<id>/submod_<n>/`。两个输出目录相同时，
  调试文件放在 `artifact_dir/graph-<id>/ir/submod_<n>/`。
- 切换到板卡时，使用 `runtime="usb", usb_config=config`，参见
  [USB 与 PLL 配置](#usb-与-pll-配置)。所选模块共享板卡连接和 PLL，
  切换分区时按板卡上的权重驻留状态加载权重。

### 读取编译与执行报告

整图和选层的 PyTorch 原生模式在设置 `artifact_dir` 且成功推理后，都可用
`cim22.backend.compilation_report(artifact_dir)` 读取该目录最近一次成功执行的
特化快照。报告由该特化在进程内持有，当前没有可用报告时返回 `None`。

- `artifacts` 是已发布的模型动态库路径，`compile_seconds` 是该次特化的编译耗时。
- `dispatch_count` 是最近一次成功推理的 CIM 调用总数。选层时还有按原模块路径
  汇总的 `layer_artifacts`、`layer_compile_seconds`、`layer_dispatch_counts`。
  同一模块在一次推理中多次执行，调用次数相加。完全留在 Host 的所选模块
  对应空产物列表和零次 CIM 调用。

固定 DVS 应用的命令和输入资源见 [Web/DVS 使用说明](web.md#dvs-命令行执行)。

## TensorFlow 接口

在当前 wheel 环境按[安装指南](install.md#可选框架依赖与仓库示例)追加 `tensorflow` extra。两套最小示例分别在新 Python 进程中运行。

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
    print("CIM dispatches:", compiled.session.dispatch_count)
```

- `compile_tf` 接收带平坦静态 `input_signature` 的无状态 `tf.function`，在返回前冻结权重并完成编译。Keras 调用放在 `tf.function` 内，并传入 `training=False`。
- `compiled(*inputs)` 接收 eager TF 张量，返回保持原容器结构的 TF 张量。使用 `with` 或 `close()` 释放 Session。底层 NumPy 接口和 dispatch 计数通过 `compiled.session` 获取。
- TF 主进程经 tf2onnx 导出 ONNX。编译 worker 使用安装的 torch-mlir wheel 将 ONNX 降低到 Linalg，再调用公共原生编译。应用进程加载模型产物并初始化 Session。
- `MLIRCIM22_ENABLE_ONNX` 控制原生 ONNX 诊断工具的构建。TF 编译入口使用上面的 worker 路径。
- 当前接口面向 eager 推理。Linux 上，如果进程已导入 TensorFlow，而
  `torch_mlir.fx` 尚未加载，PyTorch 的 `cim22` 后端会报错，包括 `reference` 模式。
  可在独立进程中运行 PyTorch 编译，再按需加载模型产物。

仓库 JAX 和 TensorFlow 示例使用相同的 INT8 全连接模型和 ReLU，输入为
`(4,64)`，输出为 `(4,16)`，使用 INT32 累加、偏置和输出。
PyTorch 示例依次调用 `nn.Conv2d` 和 `nn.ReLU`，输入为 `(1,3,16,16)`，
输出为 `(1,4,16,16)`。输入、权重、偏置和输出为 FP32，卷积使用 BF16 CIM 运算。
每组输出与软件参考逐元素相等且 CIM 调度次数大于零时计为通过。
`compile.log` 显示三组英文结果和通过数，`result.json` 保存逐组明细。

## JAX 接口

安装 `jax` extra 后，将纯函数和位置输入的静态签名传给编译入口。闭包捕获的权重在导出时固定。返回的 callable 接受 JAX 数组，并按原函数的 pytree 结构返回 JAX 数组。

```python
import jax
import jax.numpy as jnp
from cim22.jax import compile

weight = jnp.full((64, 16), 0.5, dtype=jnp.float32)
bias = jnp.ones(16, dtype=jnp.float32)

def infer(x):
    return jax.nn.relu(x @ weight + bias)

signature = (jax.ShapeDtypeStruct((2, 64), jnp.float32),)
with compile(infer, signature, options={"runtime": "simulator"}) as model:
    result = model(jnp.ones((2, 64), dtype=jnp.float32))
    print(result, model.session.dispatch_count)
```

输入签名为平坦的位置参数序列，输入和输出支持 FP32、INT8 和 INT32。固定形状的模型须能由 jax2onnx 导出并被共用的 ONNX → torch-mlir → Linalg 链路接受。示例用同一份 ONNX 对照参考输出，检查 `model.packet.mlir` 中的设备 dispatch 和推理后的计数。

从仓库根目录运行 `python python/examples/jax_compile.py --runtime simulator`。板卡执行使用同一脚本，加上 `--runtime usb --usb-config <现场配置.json>`，配置方式见下方 USB 一节。选择 `--output` 可以保存模型、IR、日志与比较结果。

## Session 与产物

用 `cim22.Session(Path(...))` 重新加载已保存的同平台模型动态库。按 `input_shapes` 和 `input_dtypes` 顺序传入连续、对齐的 NumPy 数组，结果为平坦 tuple。框架输出结构由各自的适配层恢复。模型所需的原生运行库应与产物配套。

公共 Session 的输入和输出支持 FP32、INT8 和 INT32，CIM 算子可在编译器内部使用
BF16。上述 PyTorch 框架验证示例使用 FP32 输入与输出。

未设置 `artifact_dir` 时，临时模型随 Session 生命周期清理。PyTorch 编译句柄
持有 Session，TensorFlow 和 JAX 的编译句柄可显式关闭。Windows 上，PyTorch 编译
使用 Session 所有的临时 DLL 副本，`artifact_dir` 中可另存发布的模型 DLL。
用 Session 直接加载已保存的 DLL 时，关闭 Session 后再删除该文件。

`num_transactions` 表示模型产物包含的 CIM transaction 数量，`dispatch_count` 表示最近一次调用实际发生的 dispatch 数量。使用 `debug_dir` 检查分区和降低结果，用实际 dispatch 验证执行路径。

### USB 与 PLL 配置

前端通过 `options={"runtime": "usb", "usb_config": config}` 交给公共运行时。
直接加载 USB 模型动态库时，在首次 CIM 调用前调用 `session.configure_usb(config)`。
多个已加载的 USB 模型可通过 `other.share_usb(session)` 共享已配置的连接；
新编译模型也可使用 `options={"runtime": "usb", "usb_session": session}`。
共享双方必须使用同一个原生运行库，目标 Session 尚未配置 USB。
关闭一个模型不会关闭其他模型正在使用的连接；最后一个模型关闭时才释放
USB 与 PLL。

```python
from cim22 import UsbConfig

config = UsbConfig(
    vendor_id=0x04B4,
    product_id=0x00F1,
    bulk_in_endpoint=0x81,
    bulk_out_endpoint=0x01,
    pll_port="COM5",  # Linux 示例为 /dev/ttyUSB0，按实际设备替换
)
```

按实际情况设置 VID/PID、端点和接口。运行时通过配置的串口同步设钟。

USB 时延参数按平台分别配置：Linux 使用 `linux_async_in_timeout_ms`、
`linux_sync_out_timeout_ms` 和 `linux_event_poll_timeout_ms`；Windows 使用
`windows_async_in_timeout_ms`、`windows_sync_out_timeout_ms` 和
`windows_event_wait_timeout_ms`。接收线程启动期限由 `receive_ready_timeout_ms`
设置，回包期限由 `response_timeout_ms` 单独设置。

`session.configure_usb()` 打开 USB 设备并绑定时钟回调。Python 使用随 wheel 依赖自动安装的 pyserial，串口在首次执行阶段打开，随 Session 保持连接。C++ 在写入、计算、回读阶段的第一条 USB 命令前同步调用配置回调。首次命令要求匹配回显；后续变化的命令不读取或等待回显，相同的当前命令跳过发送。三个阶段使用相同频率时，首次只需发送一次。重新配置会恢复首次回显要求。

串口连接、配置写入或所需回显失败时，运行时停止该阶段。恢复后须重新调用 `session.configure_usb()`。

每条串口命令采用 200 ms 的写入及首次回显期限。后续无回显发送检查驱动输出队列为空，再保留三字节 9600/8N1 的 4 ms 发送等待。驱动队列不一定覆盖 USB 转串口适配器内部缓冲区。
