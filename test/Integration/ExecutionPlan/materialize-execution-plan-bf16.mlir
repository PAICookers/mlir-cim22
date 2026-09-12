// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan)' | FileCheck %s

// CHECK: cim.static_weight @{{.*}} = dense<1.000000e+00> : tensor<16x64xbf16>
// CHECK-LABEL: func.func @bf16_execution_plan(
// CHECK: cim.transaction
// CHECK: cim.configure_input {{.*}} : tensor<64xbf16>
// CHECK-NEXT: cim.configure_weight
// CHECK-NEXT: cim.dispatch
// CHECK-NEXT: cim.once
// CHECK-NEXT: %[[READ:.*]] = cim.readback {{.*}} : tensor<16xbf16>
// CHECK-NEXT: cim.group_barrier
// CHECK: "cim.yield"(%[[READ]]) : (tensor<16xbf16>) -> ()
// CHECK-NOT: cim.vmm
// CHECK: return %{{.*}} : tensor<16xbf16>
func.func @bf16_execution_plan(%input: tensor<64xbf16>) -> tensor<16xbf16> {
  %weight = arith.constant dense<1.0> : tensor<16x64xbf16>
  // Explicitly request the supplier software model, not standard BF16 math.
  %result = cim.vmm %input, %weight {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  return %result : tensor<16xbf16>
}
