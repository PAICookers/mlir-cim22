# CIM22 本地编译验证工作台

这是一个面向 Linux x86-64 和 Windows x64 的本地 Web 界面，提供四个独立入口：

- **TensorFlow**：验证 TensorFlow 框架支持。
- **PyTorch**：验证 PyTorch 框架支持。
- **DVS 应用**：在实板上运行固定 DVS 模型，与同一样本的 PyTorch CPU 结果对比。
- **MNIST 训练**：配置并训练两层全连接模型，查看训练日志与结果。

框架页面的参考结果由软件直接计算，CIM 结果由编译后的模型执行得到。
“CIM 调用次数”表示实际调用次数；接口未提供该信息时显示“—”。
框架运行日志和 MNIST 训练日志均显示完整已保存输出，可滚动查看和下载。

TensorFlow 和 PyTorch 页面在开始编译前选择 `软件仿真` 或 `CIM 板卡`，
默认是软件仿真。USB 模式复用 Python API 的 `UsbConfig` 参数，页面会
校验 VID、PID、端点、设备选择和时序；它需要已连接且有访问权限的 CIM 板卡。

## 启动

先完成 [Python 快速开始](install.md#创建环境并安装)，在自己的
应用目录中创建虚拟环境并安装两个 wheel。Web 需要额外的 CPU torchvision、服务端与
TensorFlow 依赖；在同一目录执行，将 `<cim22-wheel>` 替换为实际项目 wheel 文件名中 `.whl` 前的部分：

```sh
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.14,<3" "torchvision>=0.29,<0.30"
uv pip install "wheels/<cim22-wheel>.whl[web,tensorflow]" ml_dtypes
uv pip check
```

`--model-dir` 指向只读的 DVS 模型包，目录至少包含以下路径：

```text
model-package/
├── model_mobilenet_snn_deploy.py
├── weights/dvsgesture_mobilenet_snn_qat_int8.pt
└── data/test/
```

启动服务，使用实际模型包和仓库路径：

```sh
uv run cim22-web --model-dir /path/to/model-package --examples-dir /path/to/mlir-cim22/python/examples
```

wheel 不包含框架示例脚本；`--examples-dir` 应指向项目 wheel 对应提交的 `python/examples/`。

打开
[http://127.0.0.1:8000](http://127.0.0.1:8000)，先在页面中确认 `/api/state` 返回成功，再选择
PyTorch、TensorFlow、DVS 或 MNIST 操作。结果、日志和调试产物写入 `--data-dir`，默认
为启动时工作目录下的 `./cim22-web`。同一个结果目录只能由一个服务使用。

选择 USB 前，先确认板卡、
驱动和 `UsbConfig` 字段已经按 [Python 配置说明](python-api.md#usb-与-pll-配置)
准备好。

快速检查服务和静态页面：

```sh
curl -fsS http://127.0.0.1:8000/api/state
curl -fsS http://127.0.0.1:8000/ | head
```

## DVS 指定层执行

列出固定模型可选的 `Conv2d`、`Linear` 叶子层及实际多步输入/输出尺寸：

```sh
uv run cim22-web --model-dir /path/to/model-package --list-cim-layers
uv run cim22-web --model-dir /path/to/model-package --cim-layers classifier.head.5
```

`--cim-layers` 接受一个或多个不同的完整模块名，作为新任务的默认选择。
DVS 导入表单可以修改选择，或关闭指定层模式以使用原整模型路径。
选定层在 CIM 板卡上执行，其余层由 PyTorch CPU 执行。DVS 新任务固定使用
`usb`，整模型与指定层模式均使用导入表单中的板卡配置；板卡连接或计算失败
会报错，不会回退到软件仿真。编译时的预热也会访问板卡。

导入前确认板卡已连接、驱动及访问权限就绪，并核对 VID/PID、端点和等待参数。
页面使用预先配置的板卡时钟，不会自动设置 PLL；等待时间应与实际时钟及已验证
的板卡配置匹配。多个选定层共享同一板卡连接、物理权重状态及 PLL 控制器，
适用于 Linux 和 Windows。API 也可通过 `usb_config.pll_port` 和 `clock_frequency_mhz`
使用现有串口时钟控制接口，详见 [USB 与 PLL 配置](python-api.md#usb-与-pll-配置)。

`POST /api/import` 的 `cim_layers` 字段省略时使用服务默认值，`null` 表示整模型，
非空字符串列表表示指定层。空列表、重复、未知、非叶子、不支持的模块类型及
模块别名均会被拒绝。`usb_config` 为必填的 USB 参数对象，与框架页面使用同一
字段和校验规则。`GET /api/cim-layers` 返回候选层及尺寸。

任务目录的 `config.json` 和 `state.json` 保存实际层名、运行模式和后端。
每层产物及 IR 位于 `artifacts/<完整层名>/`（特殊路径字符使用 URL 编码）；
状态记录每层编译及预热耗时、最新执行的 CIM 调用总数和最新样本输出最大绝对误差。
编译与预热不计入正式样本推理耗时。选中任一层没有编译产物或 CIM 调用
会报错，正式样本触发新编译也会报错。整模型路径同样要求实际 CIM 调用。
每个样本先运行完整的 PyTorch CPU eager 模型作为软件参考，再运行编译后的板卡模型并保存两侧结果。
无效配置在启动时直接报错。硬件错误发生时不保存未完成的样本对；
排除故障后可续跑，改变输入或配置则需要新建任务。
启动框架页面的板卡任务时，服务会释放闲置的 DVS 板卡连接并保留已保存的样本结果。
切回 DVS 后，继续测试会重新编译和预热。

### DVS 命令行执行

`cim22-dvs` 在 Linux 和 Windows 上运行与页面相同的固定 DVS 模型、预热和
逐样本 PyTorch CPU 参考/板卡结果对比，默认执行一个样本。它不是用户模型的通用命令行导入器；
用户模型使用 [SDK 分层接口](python-api.md#按模块名选择-cim-执行)。

先通过 `--list-cim-layers` 查看实际层名，再传入一个或多个不同的层：

```sh
uv run cim22-dvs --model-dir ./model-package --list-cim-layers
uv run cim22-dvs --model-dir ./model-package --usb-config ./usb.json --cim-layers classifier.head.2 classifier.head.5 --action single --output ./dvs-board-results
```

`usb.json` 是 [UsbConfig 字段](python-api.md#usb-与-pll-配置) 对应的 JSON
对象。`timing` 使用毫秒整数；串口时钟可使用 `pll_port`（Linux 设备路径或
Windows COM 名称）和 `clock_frequency_mhz`。不指定串口则预先手动配置时钟。
`--weights` 和 `--dataset` 可覆盖模型包中的默认权重和 `data/test` 目录。
省略 `--cim-layers` 使用整模型模式。`--action compile` 仅编译与预热，
`single` 运行一个样本，`full` 运行测试集。编译预热同样会访问板卡。

输出目录可以是已有目录。重新运行会替换上次 CLI 生成的配置、状态、结果和
`artifacts/` 产物子目录，其他文件会保留。逐样本结果写入 `results.csv`，汇总写入
`summary.json`。
错误以非零退出码结束，状态记为 `interrupted`，未完成的软件/板卡结果对不写入
CSV。CLI 每次新建运行；需要取消后续跑时使用 Web UI。
两平台均可使用 `uv run python -m cim22.apps.dvs` 代替 `uv run cim22-dvs`。

## MNIST 模型训练

Web 工作台的“MNIST 训练”页面提供固定的 PyTorch 两层全连接模型：`small`
为 `784 -> 64 -> 10`，`standard` 为 `784 -> 128 -> 10`。页面可设置训练轮数、
batch size、学习率和数据缓存目录；数据不存在时会自动联网下载到缓存目录。训练
过程显示文本日志，每轮保存 Loss 和测试集准确率，并保留 `model.pt`、`config.json`、
`metrics.jsonl`、`summary.json` 和 `train.log`。该入口用于模型配置和训练验收，
不提供命令行训练选项、量化配置或任意拓扑编辑。
