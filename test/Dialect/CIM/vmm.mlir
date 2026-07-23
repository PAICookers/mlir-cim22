// RUN: mlir-cim22-opt %s | mlir-cim22-opt | FileCheck %s

// Software-only evidence: canonical parse -> print -> parse for VMM-P01/P02/P04/P05.

// CHECK-LABEL: func.func @vmm(
// CHECK-SAME: %[[INPUT:.*]]: tensor<64xi8>, %[[WEIGHT:.*]]: tensor<16x64xi8>
// CHECK: %[[RESULT:.*]] = cim.vmm %[[INPUT]], %[[WEIGHT]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK: return %[[RESULT]] : tensor<16xi21>
func.func @vmm(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) -> tensor<16xi21> {
  %result = cim.vmm %input, %weight : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// CHECK-LABEL: func.func @identified_vmm(
// CHECK: cim.vmm {{.*}} {k_tile = 2 : i64, m_tile = 0 : i64, n_tile = 1 : i64}
func.func @identified_vmm(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 1 : i64, k_tile = 2 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// CHECK-LABEL: func.func @scheduled_vmm(
// CHECK: cim.vmm {{.*}} {core_slot = 3 : i64, group_id = 3 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 1 : i64, work_id = 7 : i64}
func.func @scheduled_vmm(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 1 : i64, k_tile = 1 : i64, work_id = 7 : i64, group_id = 3 : i64, core_slot = 3 : i64, macro_slot = 1 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}
