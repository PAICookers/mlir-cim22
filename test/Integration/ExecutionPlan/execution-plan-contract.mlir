// RUN: mlir-cim22-opt %s > %t
// RUN: %python %S/../../python/CIM22/verify_execution_plan.py %t | FileCheck %s --check-prefix=VERIFY
// RUN: FileCheck %s --check-prefix=MLIR < %t

// VERIFY: PASS software-only profile=cim22-4x5-v1 schema=1 groups=1 works=2 configs=4

// MLIR: cim.static_weight @weight0 = dense<0> : tensor<16x64xi8>
// MLIR-LABEL: func.func @invoke
// MLIR-SAME: attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64}
// MLIR: cim.configure_input {{.*}}macro_slot = 0 : i64{{.*}}work_id = 0 : i64
// MLIR-NEXT: cim.configure_weight @weight0 {{.*}}macro_slot = 0 : i64{{.*}}work_id = 0 : i64
// MLIR-NEXT: cim.configure_input {{.*}}macro_slot = 1 : i64{{.*}}work_id = 1 : i64
// MLIR-NEXT: cim.configure_weight @weight0 {{.*}}macro_slot = 1 : i64{{.*}}work_id = 1 : i64
// MLIR-NEXT: cim.dispatch {{.*}}work_id = 0 : i64
// MLIR-NEXT: cim.dispatch {{.*}}work_id = 1 : i64
// MLIR-NEXT: cim.once {{.*}}group_id = 0 : i64
// MLIR-NEXT: %[[READ0:.*]] = cim.readback {{.*}}work_id = 0 : i64{{.*}} : tensor<16xi21>
// MLIR-NEXT: %[[READ1:.*]] = cim.readback {{.*}}work_id = 1 : i64{{.*}} : tensor<16xi21>
// MLIR-NEXT: cim.group_barrier {group_id = 0 : i64}
// MLIR: arith.extsi %[[READ0]] : tensor<16xi21> to tensor<16xi32>
// MLIR: arith.extsi %[[READ1]] : tensor<16xi21> to tensor<16xi32>
// MLIR: arith.addi
// MLIR-NOT: arith.trunci
// MLIR-NOT: cim.vmm
// MLIR-NOT: cimframe.

module {
  cim.static_weight @weight0 = dense<0> : tensor<16x64xi8>
  func.func @invoke(%input: tensor<64xi8>) -> tensor<16xi32> attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
    cim.configure_input %input {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>
    cim.configure_weight @weight0 {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    cim.configure_input %input {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64} : tensor<64xi8>
    cim.configure_weight @weight0 {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64}
    cim.dispatch {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    cim.dispatch {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64}
    cim.once {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64}
    %read0 = cim.readback {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<16xi21>
    %read1 = cim.readback {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64} : tensor<16xi21>
    cim.group_barrier {group_id = 0 : i64}
    %wide0 = arith.extsi %read0 : tensor<16xi21> to tensor<16xi32>
    %wide1 = arith.extsi %read1 : tensor<16xi21> to tensor<16xi32>
    %sum = arith.addi %wide0, %wide1 : tensor<16xi32>
    return %sum : tensor<16xi32>
  }
}
