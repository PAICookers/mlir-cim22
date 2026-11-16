# Vivado Lab 与 ILA

- [连接并辨认探针](#连接并辨认探针)
- [采集空闲快照](#采集空闲快照)
- [按需重新编程](#按需重新编程)
- [pytest 观察器](#pytest-观察器)
- [收尾与换档](#收尾与换档)

## 连接并辨认探针

Linux 使用安装提供的 `vivado_lab`，Windows 从 Vivado Lab 命令环境启动
`vivado_lab`，也可直接使用 GUI 的 Tcl Console。交互式启动用 `vivado_lab -mode tcl`。
批处理用 `vivado_lab -mode batch -source <Tcl脚本>`。路径含空格时由 shell 加引号。

确认主机识别目标 JTAG 适配器。复用已有 `hw_server`，与 GUI 操作者协调目标和 ILA 的控制权。
无现成服务时，在 JTAG 所在主机的 Vivado Lab 命令环境中启动 `hw_server`。提前把 `hw_server_url` 设为实际服务地址。需要复用 ChipScope 服务时，给
`connect_hw_server` 增加 `-cs_url` 与实际地址。按顺序执行下列 Tcl，查到标识后再继续。
LTX 使用与板上 bitstream 同一实现生成的配套文件，Windows 的 Tcl 路径可用正斜杠。

```tcl
open_hw_manager
connect_hw_server -url $hw_server_url
puts [get_hw_targets]
# 将 target_name 设为确认的完整目标标识。
set targets [get_hw_targets $target_name]
if {[llength $targets] != 1} {error "Expected one JTAG target"}
current_hw_target [lindex $targets 0]
open_hw_target
puts [get_hw_devices]
# 将 device_name 和 ltx 设为器件标识和配套探针文件。
set devices [get_hw_devices $device_name]
if {[llength $devices] != 1} {error "Expected one FPGA"}
set device [lindex $devices 0]
set_property PROBES.FILE $ltx $device
refresh_hw_device $device
foreach core [get_hw_ilas -of_objects $device] {
    puts "$core [get_property CORE_UUID $core]"
}
# 将 ila_uuid 设为本次观测使用的核 UUID。
set ilas [get_hw_ilas -of_objects $device -filter "CORE_UUID == $ila_uuid"]
if {[llength $ilas] != 1} {error "Expected one matching ILA"}
set ila [lindex $ilas 0]
puts [get_hw_probes -of_objects $ila]
```

核、UUID 或字段不匹配时，核对 JTAG 目标、调试时钟、板上镜像与 LTX。需要重新编程时，
按[硬件排障](hardware-recovery.md#按观测定位与恢复)确认范围和镜像。

## 采集空闲快照

确认 ILA 空闲后立即触发。将 `csv_path` 设为本轮未占用的输出文件。
需要捕获瞬时事件时，按现场探针语义设定触发条件，在发送前 arm，并记录条件。

```tcl
if {[get_property STATUS.CORE_STATUS $ila] ne "IDLE"} {error "ILA is busy"}
run_hw_ila -trigger_now $ila
# timeout 单位为分钟。
wait_on_hw_ila -timeout 0.2 $ila
set data [upload_hw_ila_data $ila]
write_hw_ila_data -csv_file $csv_path $data
```

空闲样本应显示输出 FIFO 无残留、计数稳定、req 与 ack 一致。0/0 和 1/1 均可表示空闲。
Work 放行还需核对本次 TX 和计数增量，见[实板执行流程](workflow.md#4-基线配置和-work-门禁)。
命令细节见 [AMD Vivado Tcl 命令手册 UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands)。

## 按需重新编程

在确认 FPGA 掉电或镜像确需更换、并获准编程时使用。确认目标器件，停止 USB 发送并结束
当前 ILA 采集。将 `bitstream` 和 `ltx` 设为现场确认的配套文件。

```tcl
set_property PROGRAM.FILE $bitstream $device
set_property PROBES.FILE $ltx $device
program_hw_devices $device
refresh_hw_device $device
```

重新发现 ILA、核对探针并采集新空闲快照。再按[恢复验收](hardware-recovery.md#按观测定位与恢复)
检查 FX3 和 CIM 状态。

## pytest 观察器

使用与当前源码匹配的 USB wheel。项目提供 `CIM22_BOARD_OBSERVER` 时，从仓库根目录单独
构建观察运行库，并使用与 wheel 匹配的 LLVM/MLIR SDK。将 `<MLIR_DIR>` 替换为实际目录。
Windows 使用 x64 MSVC 开发者终端和 x64 CyAPI。

```sh
cmake -G Ninja -S . -B build/board-observer -DCMAKE_BUILD_TYPE=Release "-DMLIR_DIR=<MLIR_DIR>" -DCIM22_ENABLE_USB=ON -DCIM22_BOARD_OBSERVER=ON
cmake --build build/board-observer --target cim22_runtime
```

- Windows 将 `$env:CIM22_RUNTIME_LIBRARY` 设为刚构建的 `cim22_runtime.dll` 完整路径。
- Linux 导出 `CIM22_RUNTIME_LIBRARY`，指向刚构建的 `libcim22_runtime.so`。
  动态加载器找不到依赖时，把实际 `libcyusb.so` 所在目录加入 `LD_LIBRARY_PATH`。

本地 ILA JSON 填写 `vivado_lab`（启动程序完整路径）、`hw_server`、`cs_server`、
`target`、`device`、`ltx`（完整路径）和 `ila_uuid`。当前观察器面向匹配的 32 位 CIM
桥接探针。核对实际字段、采样深度与核标识。pytest 启动 batch 客户端，发出编号采样请求
并读取完整 CSV。采集忙、超时或送达不符时停止放行。[pytest 命令](workflow.md#运行通用-pytest)

## 收尾与换档

完成或取消本次拥有的采集。批处理关闭自己的目标和连接，保留用户的 GUI 与服务器。

```tcl
close_hw_target
disconnect_hw_server
close_hw_manager
```

换档时等上一批处理客户端退出，再刷新当前 FPGA 并保留新空闲快照。
`refresh_hw_device` 更新 Vivado 中的设备与调试对象。用现场证据分别判断是否空闲、
是否需要复位或重新编程。
