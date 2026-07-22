// RUN: mlir-cim22-opt %s | mlir-cim22-opt | FileCheck %s

// Software-only evidence: canonical parse -> print -> parse for VMM-P01/P02.

// CHECK-LABEL: func.func @vmm(
// CHECK-SAME: %[[INPUT:.*]]: tensor<64xi8>, %[[WEIGHT:.*]]: tensor<16x64xi8>
// CHECK: %[[RESULT:.*]] = cim.vmm %[[INPUT]], %[[WEIGHT]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK: return %[[RESULT]] : tensor<16xi21>
func.func @vmm(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) -> tensor<16xi21> {
  %result = cim.vmm %input, %weight : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return %result : tensor<16xi21>
}
