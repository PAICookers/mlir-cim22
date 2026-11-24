---
name: cim22-user-guide
description: 帮助用户安装 mlir-cim22 预编译 wheel，接入 PyTorch 或 TensorFlow 推理模型，或使用 CIM22 本地 Web 工作台和固定 DVS 应用。适用于首次运行、选择执行后端和排查用户环境问题。
---

# 安装和使用 CIM22

按用户当前任务选一份指南。独立应用使用预编译 wheel；仅在用户需要修改源码或自行构建原生组件时转到仓库的 `docs/BUILDING.md`。

- **首次安装或编译失败**。按[安装与首次运行](references/install.md)选择平台 wheel，在应用自己的环境安装并检查依赖。wheel 自带编译器，模型链接仍需系统 C 工具链。用指南中的静态模型实际执行一次，记录输出对照及生成的模型文件；仅导入包或创建 callable 尚未触发 PyTorch 编译。
- **接入自己的模型**。按 [Python 接口](references/python-api.md)选择 PyTorch 整图或 TensorFlow 静态签名。PyTorch 选层时，[用 `named_modules()` 查找模块路径并传入 `options["cim_layers"]`](references/python-api.md#按模块名选择-cim-执行)。无板卡时显式选 `runtime="simulator"`，Python 编译默认使用 USB。用原框架输出核对数值，用 `compilation_report` 查看 PyTorch 最近一次推理的 CIM 调用次数。
- **启动工作台或运行固定 DVS 模型**。按 [Web 与 DVS 使用指南](references/web.md)准备对应的资源目录。PyTorch 和 TensorFlow 页面默认仿真，DVS 的编译预热会访问实板。用户自有网络的分层执行走 Python 接口。

用户选择实板时，使用其板卡实际的设备、端点、时钟和等待参数。报告验证结果时写明执行的是 CPU 参考、CIM 仿真还是 USB 板卡，并注明实际检查过的模型或样本。
