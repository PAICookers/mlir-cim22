# CIM22 PyTorch Backend

本 Python 项目将 `cim22` 注册为 `torch.compile` 后端。模型经 torch-mlir 导入后，由原生编译器完成 CPU/CIM 划分、编译和链接，再由运行时执行。

固定 DVS 模型的本地浏览器界面见 [DVS 工作台](../web/README.md)，支持导入、编译、对比测试、取消、续跑和 CSV 结果。安装时增加 `web` extra；构建后的前端随 wheel 分发，用户不需要 Node.js。

发行包 `mlir-cim22` 已包含 Python 模块、`mlir-cim22-opt` 和两个原生库，并声明 torch-mlir 为基础依赖。

## 安装

以下安装命令可在任意工作目录执行；示例脚本路径相对仓库根目录。项目安装会解析 torch-mlir 基础依赖。

### 使用预编译 wheel

准备 Linux x86-64、Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。首次模型编译还需要系统 `cc`、链接器和 C 开发文件。普通 `linux_x86_64` wheel 直接链接系统 libusb，即使选择软件运行时也需要该动态库。例如 Ubuntu：

```sh
sudo apt-get install build-essential libusb-1.0-0
```

Linux 无需安装 Cypress `cyusb_linux` SDK；Linux 运行时直接使用 libusb。连接板卡时，还需为当前用户配置 USB 设备访问权限，并确保 FX3 已运行板卡配套固件。主机库安装不会下载或更新设备固件。

发行流程经 `auditwheel repair` 生成的 `manylinux` wheel 会携带可打包的 libusb 等动态依赖，通常不需要再安装系统 libusb；设备权限仍需单独配置。源码构建需要 libusb 开发文件，见 [环境与依赖](../BUILDING.md#环境与依赖)。可选 PLL 串口使用 POSIX 接口，无需 PyUSB 或 pyserial。

预编译用户不需要 LLVM/MLIR、CMake 或 protoc；其他系统动态库要求以所用 wheel 的构建环境说明为准。

将维护者提供的 `mlir_cim22-*.whl`，以及从 [torch-mlir releases](https://github.com/llvm/torch-mlir/releases) 下载与当前后端兼容的 `torch_mlir-*.whl` 放入 `downloads/`。目录中每种 wheel 只保留一个版本，然后执行：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch>=2.14,<3'
uv pip install --python .venv/bin/python \
  ./downloads/torch_mlir-*.whl ./downloads/mlir_cim22-*.whl
```

当前发布和验证环境为 Python 3.12.15、CPU PyTorch 2.14.1 和 torch-mlir 20261001。

使用 Web 工作台时，将上述安装命令中的项目 wheel 路径改为 `'./downloads/mlir_cim22-<version>-<tags>.whl[web]'`（替换为实际文件名），即可安装 Web 依赖；然后按 [Web 启动说明](../web/README.md#使用已安装的-wheel) 运行 `cim22-web`。不需要构建前端。没有源码时，可直接运行下文的最小模型。

### 从源码构建并安装 wheel

源码构建需要先按 [源码构建指南](../BUILDING.md#环境与依赖) 准备 LLVM/MLIR、CMake、Ninja 和本项目的 CMake 依赖。构建 wheel 本身不需要安装 PyTorch 或 torch-mlir：

若要提供 Web 界面，需先按 [Web 构建说明](../web/README.md#从源码构建) 生成前端静态页面，再在仓库根目录执行以下命令。将 `MLIR_DIR` 替换为实际工具链中包含 `MLIRConfig.cmake` 的目录；本体的 `build/` 缓存不会传给 Python 打包的 `python/build/`。

```sh
export MLIR_DIR="/path/to/llvm-build/lib/cmake/mlir"
CMAKE_BUILD_PARALLEL_LEVEL=4 uv build --python 3.12 python \
  --wheel --out-dir python/dist -Ccmake.define.MLIR_DIR="$MLIR_DIR"
```

然后创建一个自己的 Python 3.12 虚拟环境并安装刚构建的本项目 wheel：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch>=2.14,<3'
uv pip install --python .venv/bin/python \
  ./downloads/torch_mlir-*.whl ./python/dist/mlir_cim22-*.whl
```

如果已有自己的虚拟环境，将命令中的 `.venv/bin/python` 替换为该环境的 Python 路径即可。安装完成后，用该环境运行应用或本目录的自检脚本：

```sh
.venv/bin/python python/examples/torch_compile.py
```

预编译安装和源码构建安装都要求 `mlir-cim22`、torch-mlir 和应用使用同一个虚拟环境。当前发行目标是 Linux x86-64。

### Windows x64

Windows 使用 `mlir_cim22-1.0.0-py3-none-win_amd64.whl` 和
[torch_mlir-20260923.595-cp312-abi3-win_amd64.whl](https://github.com/llvm/torch-mlir-release/releases/download/dev-wheels/torch_mlir-20260923.595-cp312-abi3-win_amd64.whl)。
不能安装 Linux 的 torch-mlir wheel。已验证 Python 3.12、CPU PyTorch 2.14.1；前端使用现成 wheel，无需下载 torch-mlir 源码。

首次编译模型需要 Visual Studio 2022 Build Tools 的 MSVC x64 和 Windows SDK。
从 **x64 Native Tools Command Prompt** 启动 PowerShell，确保 `cl` 和 `link` 可用，再执行：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
uv pip install --python .venv/Scripts/python.exe ./downloads/torch_mlir-20260923.595-cp312-abi3-win_amd64.whl ./downloads/mlir_cim22-1.0.0-py3-none-win_amd64.whl
.venv/Scripts/python.exe python/examples/torch_compile.py
```

应用中的 `torch.compile` 调用相同。Windows 生成、加载 `.dll`，Linux 使用 `.so`；两者不能跨平台加载。已保存的模型可由 `Session(Path("model.dll"))` 加载，不再需要 C 链接器或 torch-mlir，仍需匹配的 CIM22 runtime 和 MSVC x64 运行库。`Session` 接收内部扁平 NumPy FP32 输入，形状可通过 `input_shapes` 查看，不能代替 PyTorch 包装层处理原始模型的状态和嵌套返回值。

Windows 临时模型 DLL 保留到 Session 关闭或编译缓存释放；加载期间不能删除该 DLL。USB 使用 Cypress CyUSB3 驱动，软件推理不要求连接板卡。当前验证覆盖软件推理及 USB 接口契约，不代表已完成真实板卡或完整 DVS 测试集验收。Web UI 仍依赖 Linux 的 `fcntl`，尚未移植到 Windows。

## 最小模型

将下面的代码保存为 `example.py`：

```python
import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 16)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(inputs))


model = SimpleModel().cpu().eval()
with torch.no_grad():
    model.linear.weight.fill_(1)
    model.linear.bias.fill_(1)

with torch.inference_mode():
    compiled = torch.compile(
        model, backend="cim22", fullgraph=True, dynamic=False,
        options={"runtime": "software"},
    )
    output = compiled(torch.ones(1, 64))
    torch.testing.assert_close(output, torch.full((1, 16), 65.0))
    print(output.shape)
```

运行：

```sh
.venv/bin/python example.py
```

输出：

```text
torch.Size([1, 16])
```

完整的自检脚本见 [examples/torch_compile.py](examples/torch_compile.py)。它采用可精确表示的固定权重；一般模型的 CIM BF16 软件计算存在近似误差。

## 使用约定

- 使用 `torch.inference_mode()` 或 `torch.no_grad()`，输入为连续 CPU FP32 Tensor，保持形状、布局及推理上下文一致。同一模型按顺序执行。
- 只读且无输出别名的参数会冻结为 CIM 权重。修改这些权重后须调用 `torch.compiler.reset()` 并重新编译；运行时权重保留在 CPU 路径。
- AOTAutograd 负责支持的原地更新和状态写回。带状态模型在每个独立序列前应按模型自身 API 重置；不会自动清零。保持各次调用的输入存储重叠关系一致。
- `fullgraph=True` 禁止图断裂，`dynamic=False` 使用静态特化；输入形状变化可能触发重新编译。算子支持仍受 torch-mlir 和原生 lowering 覆盖范围限制。
- 输入 identity/view 别名由 PyTorch 包装层处理；不同计算输出之间的共享存储不保证保留。不同 Tensor 共享底层存储的部分更新目前仍可能被上游拒绝。

使用 SpikingJelly 时，用 `functional.reset_net(model)` 在独立序列前重置神经元状态。

## 后端选项与排查

通过 `torch.compile(..., options={...})` 传入：

| 选项                            | 用途                                             |
| ------------------------------- | ------------------------------------------------ |
| `runtime="software"`          | 不连接设备，执行候选 BF16 软件模型               |
| `runtime="usb"`               | 默认值，Linux 用 libusb、Windows 用 CyAPI；BF16 回包按位模式解包 |
| `usb_config=UsbConfig(...)`   | USB 设备参数，只能与 USB 运行时同时使用          |
| `artifact_dir="build/models"` | 保存每次特化的内部模型，Linux `.so` / Windows `.dll` |
| `debug_dir="build/ir"`        | 保存 Linalg、packet MLIR 和目标文件              |

首次调用包含编译、链接和加载；稳态性能应在预热后单独测量。`TORCH_LOGS=dynamo` 查看编译阶段耗时，`TORCH_LOGS=recompiles` 排查重复编译。特化缓存在进程内，保存模型动态库不会提供跨进程自动缓存或稳定部署接口。

| 问题                       | 检查项                                                         |
| -------------------------- | -------------------------------------------------------------- |
| `Invalid backend: cim22` | 当前解释器是否安装了本项目 wheel，而非仅安装 Torch             |
| 找不到 torch-mlir          | 是否将匹配的 wheel 安装到同一虚拟环境                          |
| 缺少 `cc` / `cl` 或链接失败 | Linux 安装 C 开发环境；Windows 在 MSVC x64 开发者终端运行，并安装 Windows SDK |
| USB 报错或要求配置设备     | 无硬件时显式选择 `runtime="software"`                        |
| 编译失败或数值偏差         | 先运行自检脚本，再保存 `debug_dir`；检查输入约束和 BF16 误差 |

USB 原始诊断可通过 `cim22.Session`、`UsbConfig` 和 `capture_usb` 使用，具体数组与生命周期契约见 [runtime.py](cim22/runtime.py)。停止接收会保留成功收到的尾数据并检查真实接收错误；正常主动取消不计失败。故障后重新配置会先报告旧事务失败并关闭设备，随后允许显式重试 `configure_usb`。重新打开 Host 句柄不会复位硬件，也不会续跑失败的事务。

当前 CIM 芯片经供应方确认存在权重读电路缺陷。诊断测试可保留 type6 权重读取及协议、地址完整性检查，但跳过权重数值比较，明确标为 `SKIPPED`。生产推理与 `capture_usb` 只读取所需的 type5 计算结果，不需要额外读取权重。

## USB 与 PLL 配置

以下自动串口控制和设备权限示例用于 Linux。Windows 使用 CyAPI USB，支持同一频率表查询，但当前不支持 PLL 串口控制：省略 `pll` 和 `pll_port` 并手动设置时钟。Windows 上显式请求串口（包括不存在的端口）会返回不支持错误。

板卡须已上电并运行配套的 FPGA/FX3 固件。项目不自动安装厂商 SDK、修改系统权限或下载固件。

先为实际板卡的 VID/PID 配置 USB 访问权限。下例适用于已验证的 `04b4:00f1`；其他板卡应使用其实际标识。在 Ubuntu 上创建专用设备组和规则：

```sh
sudo groupadd -f cim22
sudo usermod -aG cim22 "$USER"
sudo tee /etc/udev/rules.d/70-cim22.rules >/dev/null <<'RULE'
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="04b4", ATTR{idProduct}=="00f1", GROUP="cim22", MODE="0660"
RULE
sudo udevadm control --reload-rules
```

重新登录以取得组成员身份，然后重新连接 FX3 USB。设备重新枚举后，旧 `/dev/bus/usb/<bus>/<address>` 路径可能改变；不将旧地址或一次性 ACL 当作持久配置。

串口是可选的。需要程序配置 PLL 时，使用 `/dev/serial/by-id/` 中的稳定路径，避免依赖可能变化的 `ttyUSB0` 编号。Ubuntu 的串口通常属于 `dialout` 组，可执行 `sudo usermod -aG dialout "$USER"` 后重新登录；其他发行版以串口实际所属组为准，例如 Arch 常用 `uucp`。使用前应关闭其他占用该串口的程序。

```python
import torch

from cim22 import UsbConfig

usb = UsbConfig(
    vendor_id=0x04B4,
    product_id=0x00F1,
    bulk_in_endpoint=0x81,
    bulk_out_endpoint=0x01,
    clock_frequency_mhz=24,
    pll_port="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
)
compiled = torch.compile(
    model, backend="cim22", fullgraph=True, dynamic=False,
    options={"usb_config": usb},
)
```

`clock_frequency_mhz` 按供应方 `PLL_clock_config.xlsx` 的 **目标频率**查表。代码内置全部 199 行的目标 MHz、实际 MHz、串口 HEX 和原表状态，不在运行时求解 PLL 参数，也不依赖 Excel 文件或读取库。表外目标（例如 26 MHz）直接报错；这只表示未收录该目标，不表示 PLL 无法生成。

按约定的使用策略，`可用`（`available`）不发警告，`最近可用`（`maybe available`）发出 `RuntimeWarning`，提示硬件运行可能不稳定，并列出目标、实际频率和串口数据。例如 24 MHz 对应 `2828A4`，无警告；目标 284 MHz 对应表中实际 283.2 MHz、`E813A4`，发出警告。标签按原表保留，并非本次运行的硬件实测结论。表中第 4 行还注明未校验 VCO 上限。

通过 `usb.clock_setting` 查看选中行的 `target_frequency_mhz`、`actual_frequency_mhz`、`serial_hex` 和 `status`。发送字节始终取表中原值，包括 20 MHz 的 `88B624`，不根据公式重新选择分频参数。

频率选项将该行的同一配置用于 Config、Work、Readback 三阶段，以 9600/8N1、高字节优先发送并检查匹配回显。需要三个阶段分别配置时，仍可使用 `pll=PLLConfig(port, config_hex, work_hex, readback_hex)`；原始 `pll` 与 `clock_frequency_mhz` 不能同时指定。

顶层导出 `Session`、`UsbConfig` 和 `PLLConfig`。常规推理使用 `torch.compile`；`Session` 用于原始回包等诊断。三阶段原始 HEX 配置需先从 `cim22` 导入 `PLLConfig`，并使用已确认适合板卡的参数。

省略串口选项时，两平台均使用用户手动设置的时钟；Linux 上指定串口设备不存在时也使用手动时钟。`clock_setting` 只表示选中表项，不证明时钟已发送、已生效或回显已通过。用户需在运行前手动完成所需时钟设置。Linux 串口存在但权限不足、通信失败或回显不匹配时仍返回错误。省略全部 PLL/频率选项保持原有手动时钟路径。

已验证板卡默认发送控制字为 `0x03`（link 0/channel 3）、每次 OUT 最多 512 字节，接收请求为 32768 字节。每个 Session 持有 USB 设备以服务顺序推理；传输故障中止当前事务，首次重新配置会报告旧故障并释放设备，随后可显式重试配置。重新打开句柄不表示硬件已复位，也不保证物理链路恢复。Linux 驱动只在当前接口设置不符时重新选择，避免重复选择触发的重开故障；当前修复仍需按实际板卡和用例验收，不能由主机测试推断长期稳定性。

## 开发与测试

完整步骤见 [BUILDING](../BUILDING.md#pytorch-编译后端)：准备 LLVM/MLIR 后，可生成 wheel 或构建 editable 安装。Python 源码修改立即可见；C++、TableGen、schema 或 CMake 修改需要显式重建。

[开发与测试](../BUILDING.md#测试与开发)给出重建、pytest、lit 和 Protobuf 更新命令。应用特有的依赖应安装在自己的环境中；Python ONNX、Protobuf 和 pytest 是开发依赖，预编译 wheel 的普通运行无需安装这些包。
