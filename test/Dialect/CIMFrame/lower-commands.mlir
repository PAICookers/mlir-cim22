// RUN: mlir-cim22-opt %s --lower-cimframe-commands-to-packets --verify-cimframe > %t
// RUN: mlir-cim22-opt %t --verify-cimframe > %t.2
// RUN: diff %t %t.2
// RUN: mlir-cim22-opt %s | sed -n 's/.*words = \(dense<.*\).*/\1/p' > %t.words.before
// RUN: sed -n 's/.*words = \(dense<.*\).*/\1/p' %t > %t.words.after
// RUN: diff %t.words.before %t.words.after
// RUN: mlir-cim22-opt %s --lower-cimframe-commands-to-packets --verify-cimframe --mlir-elide-elementsattrs-if-larger=64 | FileCheck %s --check-prefix=ELIDE
// RUN: FileCheck %s < %t

// Software-only evidence: CF-P04, CF-P05, CF-P07 and M2.3 weight lowering.

#weight_words = dense<[
  -2147483648, 2147483647, -1, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
  96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
  128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
  144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
  176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
  192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
  224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
  240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
]> : tensor<256xi32>

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
// CHECK-NEXT: cimframe.cim_int8_weight_packet
// CHECK-SAME: route = array<i32: -31, 31, -1, 1, -2, 2>
// CHECK-SAME: words = dense<"0x
// CHECK-SAME: : tensor<256xi32>
// CHECK-NEXT: cimframe.control_int8_packet
// CHECK-SAME: macro = 1 : i32
// CHECK-SAME: route = array<i32: -31, 31, -1, 1, -2, 2>
// CHECK-NEXT: cimframe.cim_int8_weight_packet
// CHECK-SAME: route = array<i32: -31, 31, -1, 1, -2, 2>
// CHECK-SAME: words = dense<"0x
// CHECK-SAME: : tensor<256xi32>
// CHECK-NEXT: cimframe.control_int8_packet
// CHECK-SAME: macro = 0 : i32
// CHECK-SAME: route = array<i32: 0, 0, 0, 0, 0, 0>
// CHECK-NEXT: cimframe.cim_int8_weight_packet
// CHECK-SAME: route = array<i32: 0, 0, 0, 0, 0, 0>
// CHECK-SAME: words = dense<"0x
// CHECK-SAME: : tensor<256xi32>
// CHECK-NOT: cimframe.start_int8_once
// CHECK-NOT: cimframe.write_int8_weights

// ELIDE: cimframe.cim_int8_weight_packet
// ELIDE-SAME: words = dense_resource<__elided__> : tensor<256xi32>
module @lower_commands {
  cimframe.start_int8_once {
    route = array<i32: 0, 1, 2, 3, 4, 5>,
    macro = 0 : i32
  }
  cimframe.write_int8_weights {
    route = array<i32: -31, 31, -1, 1, -2, 2>,
    macro = 1 : i32,
    words = #weight_words
  }
  cimframe.write_int8_weights {
    route = array<i32: -31, 31, -1, 1, -2, 2>,
    macro = 1 : i32,
    words = #weight_words
  }
  cimframe.write_int8_weights {
    route = array<i32: 0, 0, 0, 0, 0, 0>,
    macro = 0 : i32,
    words = #weight_words
  }
}
