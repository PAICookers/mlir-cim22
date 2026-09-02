// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(partition-cim-program,form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule))' > %t.once
// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(partition-cim-program,form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule,map-cim-schedule))' > %t.twice
// RUN: diff %t.once %t.twice
// RUN: test "$(grep -c 'cim.vmm' %t.once)" -eq 40
// RUN: test "$(grep -c 'cim.mapping' %t.once)" -eq 40
// RUN: not grep -q 'cimframe\.' %t.once
// RUN: FileCheck %s < %t.once

// Software-only M4.0 coverage for every explicit profile slot.

// CHECK-LABEL: func.func @all_slots(
// CHECK-SAME: attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64}
// CHECK: cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, cim.transaction_idx = 0 : i64, core_idx = 0 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 0 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 0, 1>, destination = array<i64: 0, 1>{{.*}}route = array<i64: 0, 1, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 1 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 2 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 0, 2>, destination = array<i64: 0, 2>{{.*}}route = array<i64: 0, 2, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 2 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 4 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 0, 3>, destination = array<i64: 0, 3>{{.*}}route = array<i64: 0, 3, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 3 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 6 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 0, 4>, destination = array<i64: 0, 4>{{.*}}route = array<i64: 0, 4, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 4 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 8 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 1, 0>, destination = array<i64: 1, 0>{{.*}}route = array<i64: 0, 0, 1, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 5 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 10 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 1, 1>, destination = array<i64: 1, 1>{{.*}}route = array<i64: 1, 0, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 6 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 12 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 1, 2>, destination = array<i64: 1, 2>{{.*}}route = array<i64: 1, 1, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 7 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 14 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 1, 3>, destination = array<i64: 1, 3>{{.*}}route = array<i64: 1, 2, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 8 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 16 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 1, 4>, destination = array<i64: 1, 4>{{.*}}route = array<i64: 1, 3, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 9 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 18 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 2, 0>, destination = array<i64: 2, 0>{{.*}}route = array<i64: 0, 0, 2, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 10 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 20 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 2, 1>, destination = array<i64: 2, 1>{{.*}}route = array<i64: 1, 0, 1, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 11 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 22 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 2, 2>, destination = array<i64: 2, 2>{{.*}}route = array<i64: 2, 0, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 12 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 24 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 2, 3>, destination = array<i64: 2, 3>{{.*}}route = array<i64: 2, 1, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 13 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 26 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 2, 4>, destination = array<i64: 2, 4>{{.*}}route = array<i64: 2, 2, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 14 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 28 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 3, 0>, destination = array<i64: 3, 0>{{.*}}route = array<i64: 0, 0, 3, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 15 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 30 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 3, 1>, destination = array<i64: 3, 1>{{.*}}route = array<i64: 1, 0, 2, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 16 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 32 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 3, 2>, destination = array<i64: 3, 2>{{.*}}route = array<i64: 2, 0, 1, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 17 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 34 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 3, 3>, destination = array<i64: 3, 3>{{.*}}route = array<i64: 3, 0, 0, 0, 0, 0>{{.*}}}, cim.transaction_idx = 0 : i64, core_idx = 18 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 36 : i64
// CHECK: cim.mapping = {core_coord = array<i64: 3, 4>, destination = array<i64: 3, 4>, ingress = array<i64: 0, 0>, route = array<i64: 3, 1, 0, 0, 0, 0>, source = array<i64: 0, 0>}, cim.transaction_idx = 0 : i64, core_idx = 19 : i64{{.*}}macro_idx = 0 : i64{{.*}}work_id = 38 : i64
// CHECK: return {{.*}} : tensor<16x40xi32>
func.func @all_slots(%input: tensor<64x40xi8>) -> tensor<16x40xi32> {
  %weight = arith.constant dense<0> : tensor<16x64xi8>
  %zero = arith.constant dense<0> : tensor<16x40xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x40xi8>)
      outs(%zero : tensor<16x40xi32>) -> tensor<16x40xi32>
  return %result : tensor<16x40xi32>
}
