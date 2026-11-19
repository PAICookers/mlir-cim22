# 硬件排障与现场恢复

- [平台发现](#平台发现)
- [按观测定位与恢复](#按观测定位与恢复)
- [USB 收尾与重开](#usb-收尾与重开)
- [FPGA 与 FX3 回环](#fpga-与-fx3-回环)

Windows 的枚举、CyAPI 收发与回环诊断见 [Windows 实板与 USB](windows-board.md)。会话重开和 libusb 状态判读见 [USB FAQ](usb-faq.md)。

## 平台发现

| 检查 | Windows | Linux |
| --- | --- | --- |
| FX3 枚举及驱动 | `Get-PnpDevice -PresentOnly`，核对目标使用 CyUSB3，应用加载匹配位数的 CyAPI | `lsusb` 核对 FX3；检查实际设备节点的访问权限，以及应用加载的 libcyusb/libusb |
| PLL 串口 | `Get-PnpDevice -PresentOnly -Class Ports`，核对当前 COM 号 | 查 `/dev/serial/by-id` 或当前 tty 设备；核对用户权限和占用 |
| JTAG | 在设备管理器核对调试线，再从 Vivado Lab 列出目标 | 核对 USB 调试线、线缆驱动及权限，再从 Vivado Lab 列出目标 |

都需确认 VID/PID、实际接口、Bulk IN/OUT 及端点设置。重连后重新确认地址与串口。
关闭占用目标的终端、供应方 GUI 或测试进程前，先和其所有者协调。若 Linux 运行在
WSL 或虚拟机内，先确认需要的 USB/串口/JTAG 已实际透传，或由可访问 JTAG 的主机提供服务。
Linux 的 CyUSB 是 libusb 用户态封装。权限问题应按当前发行版及设备标识处理。

## 按观测定位与恢复

停止新 OUT，保存失败 API、完成前缀、原始 IN 和状态、串口记录及新鲜 ILA 采样。
优先根据最早出现异常的边界，向现场用户说明要操作的部件、理由、操作后的确认信号。

| 观测 | 指导现场用户 |
| --- | --- |
| FX3 或串口从主机消失，无法 claim 接口 | 确认目标供电，检查对应的数据线和主机端口，释放占用。重插后重新枚举，核对驱动、权限、接口和实际串口号；USB 重插是否使 FX3 断电取决于供电方式。 |
| JTAG 无目标、ILA 核消失或探针不匹配 | 检查 JTAG 线、FPGA 电源和调试时钟，确认板上镜像与 LTX，按 [Vivado Lab 步骤](vivado-lab.md#连接并辨认探针)刷新。确认 FPGA 掉电丢失镜像时，由授权人员使用板卡配套 bitstream 重新编程并重新绑定 LTX。 |
| FPGA idle，但同一会话空闲 IN 有意外数据；或收尾后仍有挂起请求 | 先按运行时取消、回收请求并关闭句柄。排查 FX3 端点残留和供电。如果证据指向 FX3 且普通重开未恢复，再由现场用户按板卡手册重置或对 FX3 断电重启。 |
| FIFO 有残留，FPGA↔CIM 的 req/ack 持续不等或计数停滞 | 检查 CIM 供电、PLL 时钟和板间连接。有必要时让现场用户按当前板卡手册执行 FPGA/CIM 对应的复位，按硬件版本确认按键和顺序。若需重插板间连接器，先按板卡要求断开相关电源。 |
| 仅数值异常，传输、协议和关闭正常 | 回到当前板卡通过的频率与等待组合，再逐项复测。若低频时仍稳定复现，按 [数值参考](runtime-validation.md#数值参考)和 ILA 检查最早的错误边界。 |

分别记录 FPGA 编程、CIM 复位、FX3 复位/断电和主机 USB 重枚举。一次改变一个环节能保留定位依据。
恢复后重新确认枚举、PLL 配置与空闲 ILA，在新会话先做空闲 IN，再跑供应方或当前可用的验证基线。
具体验收见[实板执行流程](workflow.md#4-基线配置和-work-门禁)。缺少硬件版本或供电信息时，
先让现场用户确认再操作。按故障范围选择恢复动作，避免自动复位、清端点或反复重发。

## USB 收尾与重开

- Windows 记录 CyAPI 失败步骤、`NTSTATUS`、`USBD_STATUS` 和完成长度。轮询超时可能
  仍在等待同一 IN；`FinishDataXfer` 失败属于传输故障，即使预期字节已经齐全。
- Linux 保留 libusb 状态、提交与完成长度；必要时用 usbmon 的 URB 完成记录辅助定位。
  取消返回硬错误时还要记录该错误，并确认运行时没有无限等待 transfer callback。
  libusb status 1 不能单独判定内核 EPROTO。若重开后首个 IN 报 EPROTO 或零字节，
  检查所加载 runtime 是否重复发送相同的 `SET_INTERFACE`。条件设置路径先用
  `GET_INTERFACE`，仅在实际 alternate setting 不符时切换。
- 两平台都要完成或取消挂起的 IN，保留有效尾数据并回收请求，再释放句柄。
  预期清理取消与运行中错误分别记录。正常关闭后可用新会话重验空闲，再继续测试。

## FPGA 与 FX3 回环

匹配回环镜像及 LTX。先用短记录与生产分块的已知字节流比较完整 TX/RX。定位满包边界时，
固定初始状态、载荷和 OUT 分块，对比精确长度与大 IN 请求，配合 ILA 看
`SLWR#`、`PKTEND#` 和计数。分别记录应用请求大小、端点包长与 burst、已知 FX3 DMA
缓冲设置和实际完成长度。32 位桥接常见 12 B 记录跨越 USB 请求边界，保持连续流。

回环验证当前 Host↔FX3↔FPGA 链路。恢复 CIM 桥接后重做空闲、送达和计算验证。
若 FPGA 回传和 GPIF 写计数均未增加，记录回包尚未进入该回传观测路径，继续查
CIM 握手及 FIFO。
