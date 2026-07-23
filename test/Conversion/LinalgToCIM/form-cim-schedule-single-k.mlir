// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(form-cim-program,func.func(materialize-cim-schedule))' > %t.out
// RUN: test "$(grep -c 'cim.vmm' %t.out)" -eq 1
// RUN: test "$(grep -c 'arith.extsi' %t.out)" -eq 1
// RUN: test "$(grep -c 'arith.addi' %t.out)" -eq 0
// RUN: test "$(grep -c 'arith.trunci' %t.out)" -eq 0
// RUN: FileCheck %s < %t.out

// Software-only evidence for one M1.8 K tile and one schedule work item.

// CHECK-LABEL: func.func @single_k_tile
// CHECK: %[[PARTIAL:.*]] = cim.vmm {{.*}} {core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK-NEXT: %[[EXTENDED:.*]] = arith.extsi %[[PARTIAL]] : tensor<16xi21> to tensor<16xi32>
// CHECK: return {{.*}} : tensor<16x1xi32>
func.func @single_k_tile(%input: tensor<64x1xi8>) -> tensor<16x1xi32> {
  %weight = arith.constant dense<0> : tensor<16x64xi8>
  %zero = arith.constant dense<0> : tensor<16x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<16x1xi32>) -> tensor<16x1xi32>
  return %result : tensor<16x1xi32>
}
