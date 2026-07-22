// RUN: mlir-cim22-opt %s -split-input-file --verify-cimframe | mlir-cim22-opt -split-input-file --verify-cimframe | FileCheck %s

// Software-only evidence: CF-P01, CF-P02, CF-P03, CF-P06, CF-P08, CF-P09.

// CHECK-LABEL: module @zero_route
// CHECK: cimframe.start_int8_once
// CHECK-SAME: macro = 0 : i32
// CHECK-SAME: route = array<i32: 0, 0, 0, 0, 0, 0>
module @zero_route {
  cimframe.start_int8_once {
    route = array<i32: 0, 0, 0, 0, 0, 0>,
    macro = 0 : i32
  }
}

// -----

// CHECK-LABEL: module @representable_boundary
// CHECK: cimframe.start_int8_once
// CHECK-SAME: macro = 1 : i32
// CHECK-SAME: route = array<i32: -31, 31, 0, 0, 0, 0>
module @representable_boundary {
  cimframe.start_int8_once {
    route = array<i32: -31, 31, 0, 0, 0, 0>,
    macro = 1 : i32
  }
}

// -----

// CHECK-LABEL: module @canonical_packet_pair
// CHECK: cimframe.control_int8_packet
// CHECK-SAME: macro = 1 : i32
// CHECK-SAME: route = array<i32: 1, 2, 3, 4, 5, 6>
// CHECK-SAME: test.tag
// CHECK-NEXT: cimframe.work_once_packet
// CHECK-SAME: route = array<i32: 1, 2, 3, 4, 5, 6>
// CHECK-SAME: test.tag
module @canonical_packet_pair {
  cimframe.control_int8_packet {
    route = array<i32: 1, 2, 3, 4, 5, 6>,
    macro = 1 : i32,
    test.tag
  }
  cimframe.work_once_packet {
    route = array<i32: 1, 2, 3, 4, 5, 6>,
    test.tag
  }
}

// -----

// CHECK-LABEL: module @command_metadata
// CHECK: cimframe.start_int8_once
// CHECK-SAME: test.tag
module @command_metadata {
  cimframe.start_int8_once {
    route = array<i32: 0, 0, 0, 0, 0, 0>,
    macro = 0 : i32,
    test.tag
  }
}

// -----

// CHECK-LABEL: module @empty
module @empty {
}

// -----

// CHECK-LABEL: module @non_cimframe
// CHECK: func.func @noop
module @non_cimframe {
  func.func @noop() {
    return
  }
}
