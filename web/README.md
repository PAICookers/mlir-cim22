# CIM22 本地编译验证工作台

[启动与使用指南](../.agents/skills/cim22-user-guide/references/web.md)包含环境安装、框架验证、DVS 指定层执行与命令行、MNIST 训练说明。

## 开发

前端开发工具需要 Node.js `^22.22.2 || ^24.15.0 || >=26`（CI 使用 Node.js 24），
并使用 `package.json` 声明的 pnpm 版本。维护者按
[源码构建说明](../docs/BUILDING.md) 准备依赖后，在 `web/` 执行：

```sh
pnpm install --frozen-lockfile
pnpm format:check
pnpm lint
pnpm build
pnpm test
```

使用 `pnpm format` 统一源码、测试和配置文件的排版；`pnpm format:check` 只检查。
Prettier 使用两空格缩进、双引号、分号和 LF 换行，按 80 列自动折行。
ESLint 检查 TypeScript、React Hooks 和配置文件，Stylelint 检查 CSS，并允许
Tailwind CSS 4 的 `@theme`。运行 `pnpm lint:fix` 自动修复可修复的问题，再执行
`pnpm format`。生成目录与锁文件不参与格式化。

前端依赖以 `pnpm-lock.yaml` 为准，安装与更新统一使用 pnpm。
构建结果写入 `python/cim22/web/static/`，Python 服务不需要常驻 Node.js。

### 分层执行的软件回归

依赖和原生工具链就绪后，在 `python/` 执行：

```sh
uv run --no-sync python -m pytest -q tests/test_layers.py
```

测试覆盖单层、连续两个 Linear 和中间保留 CPU Linear 的不连续选择，
逐层检查真实 simulator dispatch、独立产物、重复执行及 CPU 数值对比。
