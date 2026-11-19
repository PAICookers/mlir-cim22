# CIM22 Python 后端

`mlir-cim22` 通过 Python entry point 注册 PyTorch 后端 `cim22`。TensorFlow 接口使用同一 wheel 的 `tensorflow` extra。

## 用户快速开始

- [安装预编译 wheel、编译并运行第一个模型](../.agents/skills/cim22-user-guide/references/install.md)
- [PyTorch、TensorFlow、指定层执行和 USB 配置](../.agents/skills/cim22-user-guide/references/python-api.md)
- [PyTorch 示例](examples/torch_compile.py)和 [TensorFlow 示例](examples/tensorflow_compile.py)

## 源码开发

[源码构建指南](../docs/BUILDING.md)给出了 Linux、Windows 的工具链配置、editable 安装和 wheel 打包命令。
在 editable 环境中验证 Web 和 TensorFlow 时，给 `uv sync` 追加 `--extra web --extra tensorflow`。安装完成后，在 `python/` 下运行：

```sh
uv run --no-sync pytest tests/test_backend.py tests/test_tensorflow.py
```

| 改动                            | 从哪里检查                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| PyTorch 捕获、冻结与 AOT 边界   | [backend.py](cim22/backend.py) 和 [test_backend.py](tests/test_backend.py)                                                                    |
| TF 签名、输出结构与编译进程隔离 | [tensorflow.py](cim22/tensorflow.py)、[_tensorflow_compile.py](cim22/_tensorflow_compile.py) 和 [test_tensorflow.py](tests/test_tensorflow.py) |
| 原生编译、链接及模型发布        | [compiler.py](cim22/compiler.py)、[test_compile_tool.py](tests/test_compile_tool.py) 和 [test_artifacts.py](tests/test_artifacts.py)           |
| Session 所有权、USB 与 PLL      | [runtime.py](cim22/runtime.py) 和 [test_usb_runtime.py](tests/test_usb_runtime.py)                                                            |

排查重复编译时使用 `TORCH_LOGS=recompiles`。定位 lowering 或链接问题时，先用 `debug_dir` 保留 IR 和目标文件，再检查安装包加载的原生组件。独立 CMake 构建可用 `CIM22_OPT`、`CIM22_COMPILE_TOOL`、`CIM22_RUNTIME_LIBRARY` 指向同一构建的产物，`CIM22_CC` 可指定模型链接工具。

原生 C/C++ 调用方通过 `cim22_session_configure_usb_with_clock` 注册同步阶段回调。函数指针和上下文须保持有效，直到设备被替换或 Session 销毁；回调中不得重入同一 Session。板卡配置与时序复测见[实板测试指南](../.agents/skills/cim22-board-testing/SKILL.md)。
