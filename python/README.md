# CIM22 Python 后端

`mlir-cim22` 通过 Python entry point 注册 PyTorch 后端 `cim22`。TensorFlow 和 JAX 分别使用同一 wheel 的 `tensorflow` 和 `jax` extra。

## 用户快速开始

- [安装预编译 wheel、编译并运行第一个模型](../.agents/skills/cim22-user-guide/references/install.md)
- [PyTorch、TensorFlow、JAX、指定层执行和 USB 配置](../.agents/skills/cim22-user-guide/references/python-api.md)
- [PyTorch 示例](examples/torch_compile.py)、[TensorFlow 示例](examples/tensorflow_compile.py)和 [JAX 示例](examples/jax_compile.py)

## 按模块选层

在原模型上用 `model.named_modules()` 查看完整模块路径。假设 `model` 含有
`encoder.0` 和 `head`，`inputs` 是模型的静态 CPU 输入：

```python
import torch

with torch.inference_mode():
    compiled = torch.compile(
        model.eval(),
        backend="cim22",
        fullgraph=True,
        dynamic=False,
        options={"cim_layers": ["encoder.0", "head"], "runtime": "simulator"},
    )
    output = compiled(inputs)
```

单选可写 `"cim_layers": "head"`，选择父模块会覆盖其后代与内部的功能式算子。
首次调用时编译，返回完整模型的输出。未选中的区域由 PyTorch 在 CPU 上运行。
省略 `cim_layers` 时自动映射整图。默认 `compile_strictness="strict"` 要求每个
选中模块生成 CIM 运算。设为 `"allow_host"` 可让成功编译但没有 CIM 运算的
模块继续在 CPU 上执行，并记录警告。路径无效、目标未捕获或编译失败均会报错。

Linux x86-64 和 Windows x64 使用同一接口，分别需要 `cc` 和 MSVC x64 工具链。
使用板卡时选 `runtime="usb"` 并传入 USB 配置。产物和计数见
[选层与执行报告](../.agents/skills/cim22-user-guide/references/python-api.md#按模块名选择-cim-执行)。

## 源码开发

[源码构建指南](../docs/BUILDING.md)给出了 Linux、Windows 的工具链配置、editable 安装和 wheel 打包命令。
在 editable 环境中同时验证 Web、TensorFlow 和 JAX 时，给 `uv sync` 追加 `--extra web --extra tensorflow --extra jax`。安装完成后，在 `python/` 下运行：

```sh
uv run --no-sync pytest tests/test_backend.py tests/test_module_offload.py tests/test_tensorflow.py tests/test_jax.py tests/test_jax_example.py
```

| 改动                            | 从哪里检查                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 捕获、冻结与 AOT 边界   | [backend.py](cim22/backend.py) 和 [test_backend.py](tests/test_backend.py)                                                                    |
| TensorFlow 和 JAX 静态签名及输出结构 | [tensorflow.py](cim22/tensorflow.py)、[jax.py](cim22/jax.py) |
| 共用 ONNX 导入、编译与 Session 加载 | [_onnx_frontend.py](cim22/_onnx_frontend.py)、[_onnx_compile.py](cim22/_onnx_compile.py) |
| 原生编译、链接及模型发布        | [compiler.py](cim22/compiler.py)、[test_compile_tool.py](tests/test_compile_tool.py) 和 [test_artifacts.py](tests/test_artifacts.py)           |
| Session 所有权、USB 与 PLL      | [runtime.py](cim22/runtime.py) 和 [test_usb_runtime.py](tests/test_usb_runtime.py)                                                            |

排查重复编译时使用 `TORCH_LOGS=recompiles`。定位 lowering 或链接问题时，先用 `debug_dir` 保留 IR 和目标文件，再检查安装包加载的原生组件。独立 CMake 构建可用 `CIM22_OPT`、`CIM22_COMPILE_TOOL`、`CIM22_RUNTIME_LIBRARY` 指向同一构建的产物，`CIM22_CC` 可指定模型链接工具。

原生 C/C++ 调用方通过 `cim22_session_configure_usb_with_clock` 注册同步阶段回调。函数指针和上下文须保持有效，直到设备被替换或 Session 销毁；回调中不得重入同一 Session。板卡配置与时序复测见[实板测试指南](../.agents/skills/cim22-board-testing/SKILL.md)。
