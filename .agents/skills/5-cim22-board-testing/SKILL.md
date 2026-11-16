---
name: 5-cim22-board-testing
description: 在实体 CIM22 板卡上验证编译产物或供应方用例，使用 Vivado Lab 采集 ILA，调整 CIM 时钟与操作等待，或定位 FPGA、FX3 和 CIM 链路故障。适用于 Windows 和 Linux。
---

# CIM22 实板测试

## 按任务读取

| 任务 | 入口 |
| --- | --- |
| 跑通用回归或供应方基线 | [通用 pytest](references/workflow.md#运行通用-pytest) · [供应方基线](references/workflow.md#1-固定基准和比较范围) |
| 连接 JTAG、刷新 FPGA、采集 ILA、配置 pytest 观察器 | [Vivado Lab 与 ILA](references/vivado-lab.md) |
| 查默认频率和等待，或调整配置 | [时序配置](references/timing-tuning.md#配置默认值与覆盖方式) · [逐档验证](references/timing-tuning.md#逐档验证) |
| 测量算子耗时或估算模型耗时 | [实测耗时和模型估算](references/timing-tuning.md#实测耗时和模型估算) |
| 排查 USB、串口、JTAG 或板间握手故障 | [硬件排障与现场恢复](references/hardware-recovery.md) |
| 增补 INT8/BF16、放置、事务或 Host 归约用例 | [编译产物与运行时验证](references/runtime-validation.md) |

## 执行要点

- 从当前检出和实际加载的运行库确定入口与能力。按现场环境取得匹配的板卡工程、探针和供应方材料。
- USB、串口和 JTAG 各由一个执行者控制。复用用户已有的 Vivado 服务，按本次任务范围操作。
- 发送前确认空闲状态。以实际配置前缀和 FPGA 送达观测放行 Work。发现传输、协议或送达故障时，停止新 OUT 并保存证据。
- 报告实际执行的入口、协议、数值及关闭状态。离线比较或耗时估算直接查对应参考。
