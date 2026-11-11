# CIM22 编译验证工作台

这是一个面向 Linux x86-64 和 Windows x64 的本地 Web 界面，提供三个独立入口：

- **TensorFlow**：固定运行 `python/examples/tensorflow_compile.py`，验证
  `tf.function` 编译和执行，使用多维 INT8 BatchMatMul 用例。
- **PyTorch**：固定运行 `python/examples/torch_compile.py`，验证
  `torch.compile` 编译和执行 BF16 Conv2d、ReLU、Linear 用例。
- **DVS 应用**：导入固定 DVS 模型和测试集，对比纯软件参考与 CIM 软件结果。

TensorFlow 和 PyTorch 页面在开始编译前选择 `CIM 软件仿真` 或 `CIM USB
板卡`，默认是软件仿真。USB 模式复用 Python API 的 `UsbConfig` 参数，页面会
校验 VID、PID、端点、设备选择和时序；它需要已连接且有访问权限的 CIM 板卡。
仿真结果不能作为物理板卡执行证据。

## 启动

请先按 [Python 安装说明](../python/README.md#用户快速开始) 使用 uv 创建
Python 3.12 环境，并安装项目 wheel 的 `[web,tensorflow]` extra：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.14,<3"
uv pip install --python .venv /path/to/torch_mlir-20261101-<platform>.whl
uv pip install --python .venv \
  "/path/to/mlir_cim22-<version>-<platform>.whl[web,tensorflow]"
```

`--model-dir` 指向只读的 DVS 模型包，目录至少包含以下路径：

```text
model-package/
├── weights/dvsgesture_mobilenet_snn_qat_int8.pt
└── data/test/
```

启动服务：

```sh
cim22-web --model-dir /path/to/model-package
```

服务默认从源码树中的 `python/examples/` 查找框架示例。使用安装 wheel 或
其他目录时，可以显式指定：

```sh
cim22-web --model-dir /path/to/model-package \
  --examples-dir /path/to/mlir-pku-cim22/python/examples
```

Windows 使用 `.venv\\Scripts\\cim22-web.exe`。打开
<http://127.0.0.1:8000>，先在页面中确认 `/api/state` 返回成功，再选择
PyTorch、TensorFlow 或 DVS 操作。结果、日志和调试产物写入 `--data-dir`，默认
为 `~/.local/share/cim22-web`。同一个结果目录只能由一个服务使用。

TensorFlow 和 PyTorch 页面默认使用 `simulator`。选择 USB 前，先确认板卡、
驱动和 `UsbConfig` 字段已经按 [Python 配置说明](../python/README.md#usb-与-pll-配置)
准备好。软件仿真结果不能替代物理板卡证据。

快速检查服务和静态页面：

```sh
curl -fsS http://127.0.0.1:8000/api/state
curl -fsS http://127.0.0.1:8000/ | head
```

## 开发

维护者按 [源码构建说明](../docs/BUILDING.md) 准备依赖后，在 `web/` 执行：

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm test
```

构建结果写入 `python/cim22/web/static/`，Python 服务不需要常驻 Node.js。
