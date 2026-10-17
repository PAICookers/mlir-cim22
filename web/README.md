# DVS 本地模型工作台

Linux x86-64、本地单用户。React + TypeScript + Vite、Tailwind CSS 和
shadcn/ui 风格的本地组件；FastAPI 提供构建后的页面、HTTP 操作和 SSE。
只有一个独立 Python 工作进程执行计算，不需要数据库或常驻 Node 服务。

## 安装预编译 wheel

用户准备 Linux x86-64、Python 3.12 和 uv；系统依赖沿用
[Python 安装说明](../python/README.md#安装)。将维护者构建的
`mlir_cim22-1.0.0-*.whl` 和匹配的 torch-mlir wheel 放在 `downloads/`，
每种仅保留一个版本：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch>=2.14,<3'
uv pip install --python .venv/bin/python ./downloads/torch_mlir-*.whl
for wheel in ./downloads/mlir_cim22-*.whl; do
  uv pip install --python .venv/bin/python "${wheel}[web]"
done
.venv/bin/cim22-web --model-dir /path/to/DVSGesture_MobileNetSNN_QAT_INT8_C16_T32
```

打开 <http://127.0.0.1:8000>。后续只需最后一条启动命令。
模型、权重和测试集另行提供；前端页面已经在 wheel 中，用户不需要 Node.js。

## 源码开发与打包

先按 [源码构建说明](../BUILDING.md) 准备 Python 3.12、uv、
匹配的 torch-mlir wheel 和 native 工具链。在仓库根目录安装应用依赖：

```sh
uv sync --project python --locked --extra compiler --extra web --group dev
```

前端开发/构建使用 Node.js 22.12+ 和 pnpm 11.25：

```sh
cd web
pnpm install --frozen-lockfile
pnpm build
cd ..
```

构建产物写入 `python/cim22/web/static/`，由 FastAPI 提供。
发布时先完成上述前端构建，再在仓库根目录构建 Python 包：

```sh
uv build --python 3.12 python --out-dir wheelhouse
```

沿用主分支的 `scikit-build-core` 和 CMake 配置，包名为 `mlir-cim22`。
静态页面同时纳入 wheel 和 sdist；从 sdist 构建 wheel 不需要再次运行 Node.js。
GitHub Actions 也按「安装前端依赖 → 构建前端 → 构建 Python 包」执行。
原生依赖、LLVM 链接方式和系统兼容要求以 [构建说明](../BUILDING.md) 为准。

开发时可在 `web/` 运行 `pnpm dev`；其 `/api` 请求代理到本地 8000 端口。
组件遵循 [shadcn 手动安装结构](https://ui.shadcn.com/docs/installation/manual)，
可用 `components.json` 继续维护本地组件。

## 一个命令启动

从仓库根目录运行（模型包由用户提供，应用不会修改或重新分发它）：

```sh
uv run --project python --no-sync \
  cim22-web --model-dir /path/to/DVSGesture_MobileNetSNN_QAT_INT8_C16_T32
```

打开 <http://127.0.0.1:8000>。已安装的 wheel 也可直接运行 `cim22-web`。
只有构建页面时需要 Node.js。发布 wheel 前必须先构建前端；
页面静态文件会随 Python 包一起安装。

`--port 8000` 可调整端口。`--data-dir /path/to/results` 可调整结果目录；
默认是 `~/.local/share/cim22-web`。服务固定监听 `127.0.0.1`，只运行一个
Uvicorn worker，不使用 reload。一个结果目录同时只能被一个服务占用。

## 页面操作

1. **模型与编译**：填写本地权重文件和测试集目录路径，导入后点击编译。
   模型定义从启动时指定目录的 `model_mobilenet_snn_deploy.py` 读取。
   权重固定为 C16、expansion=2、LIF tau=2、T32 的部署 checkpoint；
   使用 `torch.load(weights_only=True)` 和严格权重加载。
2. **运行控制**：先运行一个样本，也可直接开始完整测试。重复点击单样本
   会运行下一个未完成样本。取消在当前完整样本对比保存后生效；编译阶段
   的取消会等待本次编译结束。浏览器关闭或刷新不会停止计算。
3. **结果对比**：显示两个模型在已完成样本上的准确率和推理总耗时，
   每行包含真实类别、两边预测、正误和调用耗时。可筛选、分页和导出 CSV。

测试集使用 `<test>/<class>/*.npz`，文件内有 `frames` 数组，尺寸必须为
`[32, 2, 128, 128]`。推理输入是连续 CPU FP32 `[32, 1, 2, 128, 128]`。
类别沿用验证过的 DatasetFolder 字典序：
`0, 1, 10, 2, 3, 4, 5, 6, 7, 8, 9`。
CSV 的 label/prediction 保存模型索引（0–10）；界面将索引映射回目录类别名。
可以使用测试集子集，但不会按缺失类别重新编号。未知类别目录会被拒绝。

## 计时、持久化和恢复约定

- 使用 `torch.inference_mode()`，每次完整调用前 `reset_net()`；两个模型
  独立持有神经元状态。输入和参数在 inference context 外准备，避免改变
  Dynamo 的输入特化条件。输出按时间维求和后取 argmax。
- 编译按钮实际调用固定尺寸样本，触发
  `torch.compile(..., backend="cim22", fullgraph=True, dynamic=False)`；
  明确传入 `runtime="software"`。界面的「编译与首次执行」时间包含捕获、
  编译、加载和首次执行，不加入任何样本的推理时间。
- 每个样本计时只包围同步 CPU 模型调用。数据读取、形状校验、神经元复位、
  预测汇总、结果保存和内存回收都在计时外。总耗时是已保存样本耗时之和。
- 单个工作进程在两次操作之间保持已编译的 callable；不导出新的独立加载
  接口。`.so` 保留在任务的 `artifacts/` 内，由现有 backend 生成。
- 每完成一对原始/CIM 推理，先以临时文件 + fsync + 原子替换提交 CSV，
  然后更新任务状态。未完成的一对不计入结果，续跑时重新执行该样本。
- 服务关闭/崩溃会停止它的计算子进程。重启或检测到异常退出后显示
  **已中断**，从 CSV 恢复完成数。续跑重新编译，既有 CSV 行保持不变。
- 恢复前核对模型、权重、样本内容和执行环境；变化后必须新建测试，
  不允许将不同环境的结果混合。已完成的任务不会被新导入覆盖。
  执行环境包含 Python 后端代码和原生库；升级后旧结果仍可查看，
  但需要重新导入再运行。

结果目录结构：

```text
current.json
<task-id>/
  config.json       # 输入路径、类别、样本顺序和内容哈希
  environment.json  # 软件与 native 库身份
  state.json        # 状态、编译记录、错误
  results.csv       # 已完成样本对比，进度恢复的依据
  artifacts/*.so
  worker.log        # 本地诊断日志
```

首版使用已验证的软件 runner；页面结果不代表物理板卡执行。

## 验证

```sh
cd python
uv run --locked --extra compiler --extra web --group dev pytest tests/test_web.py
uv run --locked --extra compiler --extra web --group dev ruff check cim22/web tests/test_web.py
uv run --locked --extra compiler --extra web --group dev ruff format --check cim22/web tests/test_web.py
cd ../web
pnpm build
pnpm exec playwright install chromium
pnpm test
```

浏览器回归测试模拟 API/SSE 的不同到达顺序，不需要模型或测试集。
已有受支持的浏览器时可设置 `PLAYWRIGHT_CHANNEL`（例如 `msedge`）。

任务测试覆盖取消/续跑、未完成样本重试、CSV 提交失败、服务重启、
互斥计算、类别映射和本地 API。真实模型检查需要外部 DVS 模型包和
完整 compiler/runtime 环境。CPU 软件 runner 的单样本需要数秒到数十秒，
整个测试集耗时取决于本机。
