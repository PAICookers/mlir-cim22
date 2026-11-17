# Batch client for the 32-bit bridge probes. Use the configured server.
lassign $argv root server cs_server target device_name probes uuid
open_hw_manager
connect_hw_server -url $server -cs_url $cs_server
set targets [get_hw_targets $target]
if {[llength $targets] != 1} {error "Expected one configured JTAG target"}
current_hw_target [lindex $targets 0]
open_hw_target
set device [get_hw_devices $device_name]
if {[llength $device] != 1} {error "Expected one configured FPGA"}
set_property PROBES.FILE $probes $device
refresh_hw_device $device
set ilas [get_hw_ilas -of_objects $device]
if {[llength $ilas] != 1} {error "Expected one ILA"}
set ila [lindex $ilas 0]
if {[get_property CORE_UUID $ila] ne $uuid} {error "ILA identity mismatch"}
set_property CONTROL.DATA_DEPTH 1024 $ila

for {set i 0} {![file exists [file join $root stop]]} {incr i} {
    set request [file join $root request-$i]
    set deadline [expr {[clock milliseconds] + 300000}]
    while {![file exists $request] && ![file exists [file join $root stop]]} {
        if {[clock milliseconds] > $deadline} {error "Capture request timeout"}
        after 10
    }
    if {[file exists [file join $root stop]]} {break}
    if {[get_property STATUS.CORE_STATUS $ila] ne "IDLE"} {error "ILA is busy"}
    run_hw_ila -trigger_now $ila
    # Vivado uses minutes. Leave time for upload within the host deadline.
    wait_on_hw_ila -timeout 0.2 $ila
    set captured [upload_hw_ila_data $ila]
    set temp [file join $root sample-$i.partial.csv]
    write_hw_ila_data -csv_file $temp $captured
    file rename $temp [file join $root sample-$i.csv]
}
close_hw_target
disconnect_hw_server
close_hw_manager
