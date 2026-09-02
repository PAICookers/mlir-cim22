// RUN: mlir-cim22-opt %s -materialize-cim-execution-plan > %t.once
// RUN: mlir-cim22-opt %t.once -materialize-cim-execution-plan > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

module {
  func.func @two_work(%input0: tensor<64xi8>, %input1: tensor<64xi8>) -> tensor<16xi32>
      attributes {cim.placement_policy = "core-major-dual-macro-v1",
                  cim.route_policy = "lower-left-maximal-xy-v1",
                  cim.target_profile = "cim22-4x5-v1",
                  cim.target_profile_version = 1 : i64} {
    %weight0 = arith.constant dense<1> : tensor<16x64xi8>
    %weight1 = arith.constant dense<2> : tensor<16x64xi8>
    %partial0 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
    %extended0 = arith.extsi %partial0 : tensor<16xi21> to tensor<16xi32>
    %partial1 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_idx = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
    %extended1 = arith.extsi %partial1 : tensor<16xi21> to tensor<16xi32>
    %sum = arith.addi %extended0, %extended1 : tensor<16xi32>
    return %sum : tensor<16xi32>
  }
}

// CHECK: cim.static_weight @__cim_weight_two_work_s0_w0 = dense<1> : tensor<16x64xi8>
// CHECK-NEXT: cim.static_weight @__cim_weight_two_work_s0_w1 = dense<2> : tensor<16x64xi8>
// CHECK: func.func @two_work
// CHECK-SAME: cim.execution_plan_schema_version = 1 : i64
// CHECK: %[[TX:.*]]:2 = "cim.transaction"(%arg0, %arg1)
// CHECK: ^bb0(%[[INPUT0:.*]]: tensor<64xi8>, %[[INPUT1:.*]]: tensor<64xi8>):
// CHECK: cim.configure_input %[[INPUT0]] {{.*}}macro_idx = 0 : i64
// CHECK-NEXT: cim.configure_weight @__cim_weight_two_work_s0_w0 {{.*}}macro_idx = 0 : i64
// CHECK-NEXT: cim.configure_input %[[INPUT1]] {{.*}}macro_idx = 1 : i64
// CHECK-NEXT: cim.configure_weight @__cim_weight_two_work_s0_w1 {{.*}}macro_idx = 1 : i64
// CHECK-NEXT: cim.dispatch {{.*}}macro_idx = 0 : i64
// CHECK-NEXT: cim.dispatch {{.*}}macro_idx = 1 : i64
// CHECK-NEXT: cim.once {{.*}}core_idx = 0 : i64{{.*}}group_id = 0 : i64
// CHECK-NOT: macro_idx
// CHECK-NEXT: %[[READ0:.*]] = cim.readback {{.*}}macro_idx = 0 : i64{{.*}} : tensor<16xi21>
// CHECK-NEXT: %[[READ1:.*]] = cim.readback {{.*}}macro_idx = 1 : i64{{.*}} : tensor<16xi21>
// CHECK-NEXT: cim.group_barrier {group_id = 0 : i64}
// CHECK-NEXT: "cim.yield"(%[[READ0]], %[[READ1]])
// CHECK: %[[EXT0:.*]] = arith.extsi %[[TX]]#0
// CHECK: %[[EXT1:.*]] = arith.extsi %[[TX]]#1
// CHECK: arith.addi %[[EXT0]], %[[EXT1]]
// CHECK-NOT: cim.vmm
