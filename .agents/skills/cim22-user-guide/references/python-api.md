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

- 首次模型调用触发 Dynamo/AOTAutograd 捕获、torch-mlir 降低和原生编译。用 `fullgraph=True, dynamic=False` 检查整图静态捕获，稳态计时从预热后开始。
- 在 `torch.inference_mode()` 或 `torch.no_grad()` 下传入连续 CPU Tensor。输入按编译签名传递，支持 FP32、INT8、INT32。改变形状、布局或推理上下文可能触发重新特化。
- 只读且无输出别名的模块参数可冻结为 CIM 权重。更改冻结参数后调用 `torch.compiler.reset()` 并重新编译。运行时权重走 Host。
- 支持的状态写回与输入别名由 AOTAutograd 管理。带状态模型在独立序列前使用模型自身的重置接口，并保持输入存储重叠关系一致。

INT8 CIM 降低要求显式 `i8 × i8 → i32`、冻结权重和可证明的有符号 INT21 分块结果范围。FP32 收缩可走 CIM BF16，数值验证应使用对应的软件参考。Host 保留与 CIM dispatch 分别验证。

### 编译选项

两个前端接收同一份 `options` 字典。

| 键                    | 行为                                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `runtime`           | 默认 `"usb"`，使用 USB 连接 CIM 板卡。`"simulator"` 在 CPU 上模拟 CIM 计算。PyTorch 也支持 `"reference"`，直接执行 CPU PyTorch 算子。 |
| `bf16_compensation` | 默认 `"sign-split"`，可选 `"legacy"` 或 `"none"` 建立对照。详见 仓库中的 `docs/BF16.md`。                                         |
| `usb_config`        | `cim22.UsbConfig`，用于打开 USB 连接；与 `usb_session` 互斥。                                                                           |
| `usb_session`       | 已配置的 `cim22.Session`，新模型共享其板卡连接、权重驻留与时钟控制；仅用于 USB。                                                          |
| `artifact_dir`      | 可选 `str` 或 `Path`。保存每次编译的 `graph-<id>.so` 或 `graph-<id>.dll`。                                                          |
| `debug_dir`         | 可选 `str` 或 `Path`。在同名 `graph-<id>/` 下保存 `model.linalg.mlir`、`model.packet.mlir` 和目标文件。                           |

PyTorch 的 `reference` 忽略 BF16 补偿策略，且不生成原生制品。TensorFlow 可直接调用原 `tf.function` 取得参考结果。

## 按模块名选择 CIM 执行

`cim22.layers` 提供通用 PyTorch 分层执行接口，DVS Web 和 CLI 复用它。
对其他网络同样使用完整的 `named_modules()` 名称选择 `Conv2d`、`Linear`
叶子模块；其余模块继续由 PyTorch CPU 执行。

```python
import torch
from cim22.layers import LayerOffload, discover_cim_layers

torch.manual_seed(42)
model = torch.nn.Sequential(
    torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
).cpu().eval()
x = torch.ones(1, 4)
print(discover_cim_layers(model, (x,)))
with torch.inference_mode():
    expected = model(x)  # LayerOffload 随后会原地编译选中的层
    compiled = LayerOffload(
        model, ["0", "2"], (x,),
        artifact_dir="artifacts/selected",
        runtime="simulator",
    )
    actual = compiled(x)
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)

print(compiled.layer_artifacts)
print("CIM dispatches:", compiled.dispatch_count)
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
- 每层独立保存产物及 IR。每次推理逐层检查是否实际执行了 CIM 计算，`dispatch_count` 给出该次调用的总数。
  跳过选中的层或触发额外编译会报错。若预热失败，选中的层会恢复为调用前的状态。
  使用不同输入尺寸或改变冻结权重时，请重新构造模型并编译。
- 默认运行时为 `usb`。板卡连接、设备权限、实际端点、时钟和等待参数需按
  [USB 与 PLL 配置](#usb-与-pll-配置) 准备。将示例切换到板卡时，改用 `runtime="usb", usb_config=config`。
- Linux 和 Windows 上，多个选定层共享一个板卡连接、接收线程、物理权重驻留
  和 PLL 控制器。每层仍有独立模型产物与调用计数；切换层时按实际驻留状态
  加载权重，板卡故障不会回退到软件仿真。

用户模型直接使用上述库接口；Web 的 DVS 页面与 `cim22-dvs` CLI 固定运行 DVS 应用。
固定应用代码位于 `cim22.apps`：`dvs` 提供 DVS 命令行流程，
`mnist_training` 提供 MNIST 训练流程。
DVS 命令和输入资源约定见 [Web/DVS 使用说明](web.md#dvs-命令行执行)。

## TensorFlow 接口

在当前 wheel 环境按[安装指南](install.md#可选tensorflow-与仓库示例)追加 `tensorflow` extra。两套最小示例分别在新 Python 进程中运行。

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
- 当前接口面向 eager 推理。在另一个 `tf.function` 中嵌入调用及训练不在此接口范围内。Linux 已加载 TensorFlow 时，CIM22 会拒绝首次在同一进程加载 PyTorch 路径的 torch-mlir FX 绑定。

仓库示例 `python/examples/tensorflow_compile.py` 使用两次二维 INT8 矩阵乘和截断 ReLU。`python/examples/torch_compile.py` 使用 Conv2d、ReLU 和 Linear，以 FP32 初始化权重并捕获模型，由编译器将符合条件的算子降为 BF16 CIM 运算。每次运行将 `compile.log`、`graph-<id>.so/.dll` 和 `ir/graph-<id>/` 调试文件保存在当前工作目录的示例输出目录，并打印 CIM 调用次数和输出误差。

## Session 与产物

用 `cim22.Session(Path(...))` 重新加载已保存的同平台模型动态库。按 `input_shapes` 和 `input_dtypes` 顺序传入连续、对齐的 NumPy 数组，结果为平坦 tuple。框架输出结构由各自的适配层恢复。模型所需的原生运行库应与产物配套。

当前公共 Session 边界支持 FP32、INT8 和 INT32；BF16 算子可在编译器内部执行，但原生 BF16 Tensor 作为直接输入或输出的公共 ABI 仍待后续工作树实现。PyTorch 示例在 FP32 边界传入输入并返回最终激活。

未设置 `artifact_dir` 时，临时模型随 Session 生命周期清理。PyTorch 编译后的 callable 持有 Session，TF callable 可显式关闭。Windows 上应在 Session 关闭后清理其加载的 DLL。

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
