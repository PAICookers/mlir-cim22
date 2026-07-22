// RUN: mlir-cim22-opt %s --lower-cimframe-commands-to-packets --verify-cimframe > %t
// RUN: mlir-cim22-opt %t --verify-cimframe > %t.2
// RUN: diff %t %t.2
// RUN: FileCheck %s < %t

// Software-only evidence: CF-P04, CF-P05, CF-P07.

// CHECK-LABEL: module @lower_commands
// CHECK-NOT: cimframe.start_int8_once
// CHECK: cimframe.control_int8_packet
// CHECK-SAME: macro = 0 : i32
// CHECK-SAME: route = array<i32: 0, 1, 2, 3, 4, 5>
// CHECK-NEXT: cimframe.work_once_packet
// CHECK-SAME: route = array<i32: 0, 1, 2, 3, 4, 5>
// CHECK-NEXT: cimframe.control_int8_packet
// CHECK-SAME: macro = 1 : i32
// CHECK-SAME: route = array<i32: -31, 31, -1, 1, -2, 2>
// CHECK-NEXT: cimframe.work_once_packet
// CHECK-SAME: route = array<i32: -31, 31, -1, 1, -2, 2>
// CHECK-NOT: cimframe.start_int8_once
module @lower_commands {
  cimframe.start_int8_once {
    route = array<i32: 0, 1, 2, 3, 4, 5>,
    macro = 0 : i32
  }
  cimframe.start_int8_once {
    route = array<i32: -31, 31, -1, 1, -2, 2>,
    macro = 1 : i32
  }
}
