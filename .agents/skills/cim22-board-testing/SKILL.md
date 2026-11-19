---
name: cim22-board-testing
description: 在实体 CIM22 板卡上验证编译产物或供应方用例，使用 Vivado Lab 采集 ILA，调整 CIM 时钟与操作等待，或定位 FPGA、FX3 和 CIM 链路故障。适用于 Windows 和 Linux。
---
# CIM22 实板测试

## 按任务读取

| 任务                                               | 入口                                                                                                             |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 跑通用回归或供应方基线                             | [通用 pytest](references/workflow.md#运行通用-pytest) · [供应方基线](references/workflow.md#1-固定基准和比较范围) |
| 连接 JTAG、刷新 FPGA、采集 ILA、配置 pytest 观察器 | [Vivado Lab 与 ILA](references/vivado-lab.md)                                                                     |
| 查默认频率和等待，或调整配置                       | [时序配置](references/timing-tuning.md#配置默认值与覆盖方式) · [逐档验证](references/timing-tuning.md#逐档验证)   |
| 测量算子耗时或估算模型耗时                         | [实测耗时和模型估算](references/timing-tuning.md#实测耗时和模型估算)                                              |
| 排查 USB、串口、JTAG 或板间握手故障                | [硬件排障与现场恢复](references/hardware-recovery.md)                                                             |
| 增补 INT8/BF16、放置、事务或 Host 归约用例         | [编译产物与运行时验证](references/runtime-validation.md)                                                          |

## 执行要点

- 从当前检出和实际加载的运行库确定入口与能力。按现场环境取得匹配的板卡工程、探针和供应方材料。
- USB 时延配置按平台记录：Linux 使用 `linux_async_in_timeout_ms`、
  `linux_sync_out_timeout_ms` 和 `linux_event_poll_timeout_ms`；Windows 使用
  `windows_async_in_timeout_ms`、`windows_sync_out_timeout_ms` 和
  `windows_event_wait_timeout_ms`。`receive_ready_timeout_ms` 和
  `response_timeout_ms` 是公共生命周期/协议期限，不替代传输参数。
- Linux 的异步 Bulk IN 是一个持久请求；单次 transfer timeout 只结束当前空闲周期，
  正常情况下会重提交。停止时必须先取消并观察 callback 回收；若取消返回硬错误，
  记录该错误并确认关闭路径有界后再释放会话。Windows 记录 CyAPI 的事件等待与
  `Abort`/`FinishDataXfer` 状态，不把 Linux callback 语义套用到 Windows。
- USB、串口和 JTAG 各由一个执行者控制。复用用户已有的 Vivado 服务，按本次任务范围操作。
- 发送前确认空闲状态。以实际配置前缀和 FPGA 送达观测放行 Work。发现传输、协议或送达故障时，停止新 OUT 并保存证据。
- 报告实际执行的入口、协议、数值及关闭状态。离线比较或耗时估算直接查对应参考。
