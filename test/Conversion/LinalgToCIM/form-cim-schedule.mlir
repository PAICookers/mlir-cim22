// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(partition-cim-program,form-cim-program,func.func(materialize-cim-schedule))' > %t.once
// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(partition-cim-program,form-cim-program,func.func(materialize-cim-schedule,materialize-cim-schedule))' > %t.twice
// RUN: diff %t.once %t.twice
// RUN: test "$(grep -c 'cim.vmm' %t.once)" -eq 8
// RUN: test "$(grep -c 'arith.extsi' %t.once)" -eq 8
// RUN: test "$(grep -c 'arith.addi' %t.once)" -eq 4
// RUN: not grep -q 'arith.trunci' %t.once
// RUN: FileCheck %s < %t.once

// Software-only evidence for the provisional M5.1 schedule policy.

// CHECK-LABEL: func.func @matmul_integer
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, n_tile = 0 : i64, work_id = 1 : i64}
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 1 : i64, k_tile = 0 : i64, m_tile = 0 : i64, n_tile = 1 : i64, work_id = 2 : i64}
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 1 : i64, k_tile = 1 : i64, m_tile = 0 : i64, n_tile = 1 : i64, work_id = 3 : i64}
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 2 : i64, k_tile = 0 : i64, m_tile = 1 : i64, n_tile = 0 : i64, work_id = 4 : i64}
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 2 : i64, k_tile = 1 : i64, m_tile = 1 : i64, n_tile = 0 : i64, work_id = 5 : i64}
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 3 : i64, k_tile = 0 : i64, m_tile = 1 : i64, n_tile = 1 : i64, work_id = 6 : i64}
// CHECK: cim.vmm {{.*}} {cim.segment_id = 0 : i64, group_id = 3 : i64, k_tile = 1 : i64, m_tile = 1 : i64, n_tile = 1 : i64, work_id = 7 : i64}
// CHECK: return {{.*}} : tensor<17x2xi32>
func.func @matmul_integer(%input: tensor<65x2xi8>) -> tensor<17x2xi32> {
  %weight = arith.constant dense<0> : tensor<17x65xi8>
  %zero = arith.constant dense<0> : tensor<17x2xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<17x65xi8>, tensor<65x2xi8>)
      outs(%zero : tensor<17x2xi32>) -> tensor<17x2xi32>
  return %result : tensor<17x2xi32>
}

// CHECK-LABEL: func.func @no_vmm
// CHECK-NEXT: return
func.func @no_vmm() {
  return
}
