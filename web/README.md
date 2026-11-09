# CIM22 编译验证工作台

这是一个面向 Linux x86-64 的本地 Web 界面，提供三个独立入口：

- **TensorFlow**：固定运行 `python/examples/tensorflow_compile.py`，验证
  `tf.function` 编译和执行。
- **PyTorch**：固定运行 `python/examples/torch_compile.py`，验证
  `torch.compile` 编译和执行。
- **DVS 应用**：导入固定 DVS 模型和测试集，对比纯软件参考与 CIM 软件结果。

TensorFlow 和 PyTorch 页面在开始编译前选择 `CIM 软件仿真` 或 `CIM USB
板卡`，默认是软件仿真。USB 模式复用 Python API 的 `UsbConfig` 参数，页面会
校验 VID、PID、端点、设备选择和时序；它需要已连接且有访问权限的 CIM 板卡。
仿真结果不能作为物理板卡执行证据。

## 启动

请先按 [Python 安装说明](../python/README.md#安装) 安装 Python 3.12、项目
wheel 及 `[web,tensorflow]` extra。启动服务：

```sh
cim22-web --model-dir /path/to/model-package
```

服务默认从源码树中的 `python/examples/` 查找框架示例。使用安装 wheel 或
其他目录时，可以显式指定：

```sh
cim22-web --model-dir /path/to/model-package \
  --examples-dir /path/to/mlir-pku-cim22/python/examples
```

打开 <http://127.0.0.1:8000>。结果、日志和调试产物写入 `--data-dir`，默认
为 `~/.local/share/cim22-web`。同一个结果目录只能由一个服务使用。

## 开发

维护者按 [源码构建说明](../docs/BUILDING.md) 准备依赖后，在 `web/` 执行：

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm test
```

构建结果写入 `python/cim22/web/static/`，Python 服务不需要常驻 Node.js。
