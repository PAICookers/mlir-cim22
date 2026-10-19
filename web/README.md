# DVS 本地模型工作台

这是一个面向 Linux x86-64 的本地 Web 界面，用于比较固定 DVS 模型的原始
PyTorch 推理和 CIM22 软件运行结果。页面由 FastAPI 提供，不需要数据库或
常驻 Node.js 服务。

## 功能

- 导入本地模型权重和测试集目录。
- 编译模型，并运行单个样本或完整测试集。
- 对比两种运行路径的预测、准确率和推理耗时。
- 支持取消、续跑、筛选结果和导出 CSV。

页面结果来自软件运行器，不代表物理板卡执行结果。

## 使用已安装的 wheel

请先按 [Python 安装说明](../python/README.md#安装) 安装 Python 3.12、兼容的
项目 wheel 及其 `[web]` extra。torch-mlir 是项目的基础依赖，wheel 已包含前端静态页面，
用户不需要安装 Node.js。

启动服务：

```sh
cim22-web --model-dir /path/to/DVSGesture_MobileNetSNN_QAT_INT8_C16_T32
```

打开 <http://127.0.0.1:8000>，在页面中填写权重文件和测试集目录。模型包、
权重和测试集需要单独准备；`--model-dir` 指向固定模型包目录。

可选参数：

```text
--port PORT          HTTP 端口，默认 8000
--data-dir PATH      结果目录，默认 ~/.local/share/cim22-web
```

同一个结果目录只能由一个服务使用。刷新或关闭浏览器不会停止正在运行的
计算；服务重启后可以从已保存结果继续。

## 从源码构建

普通用户无需执行本节。维护者需要先按 [源码构建说明](../BUILDING.md) 准备
Python、uv、torch-mlir wheel 和 native 工具链，然后构建前端：

```sh
cd web
pnpm install --frozen-lockfile
pnpm build
```

也验证过等价的 Bun 和 npm 流程：

```sh
# Bun 1.3+
bun install --frozen-lockfile
bun run build

# npm 12；npm 12 默认禁止远程 tarball
npm install --package-lock-only --ignore-scripts --allow-remote=all
npm ci --ignore-scripts --allow-remote=all
npm run build
```

构建结果写入 `python/cim22/web/static/`。完成前端构建后，在仓库根目录构建
Python wheel：

```sh
uv build --python 3.12 python --out-dir python/dist
```

前端开发服务器可用 `pnpm dev` 启动；它会把 `/api` 请求代理到本地 8000
端口。pnpm 是项目默认包管理器，`pnpm-lock.yaml`、`bun.lock` 和
`package-lock.json` 分别对应三种已验证的安装方式。
