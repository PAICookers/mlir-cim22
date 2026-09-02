// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(partition-cim-program,form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,func.func(verify-cim-execution-plan))' > %t
// RUN: FileCheck %s < %t
// RUN: sed 's/}) {cim.transaction_idx = 0 : i64/}) {cim.transaction_idx = 2 : i64/' %t > %t.gap
// RUN: not mlir-cim22-opt %t.gap -verify-cim-execution-plan 2>&1 | FileCheck %s --check-prefix=GAP
// RUN: sed 's/}) {cim.transaction_idx = 0 : i64} :/}) {} :/' %t > %t.missing
// RUN: not mlir-cim22-opt %t.missing 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: sed 's/cim.group_barrier {group_id = 0 : i64/cim.group_barrier {group_id = 1 : i64/' %t > %t.joined
// RUN: not mlir-cim22-opt %t.joined -verify-cim-execution-plan 2>&1 | FileCheck %s --check-prefix=JOINED

// CHECK-LABEL: func.func @host_handoff(
// CHECK: %{{.*}} = "cim.transaction"(%arg0)
// CHECK: cim.configure_input {{.*}}group_id = 0 : i64{{.*}}work_id = 0 : i64
// CHECK: cim.group_barrier {group_id = 1 : i64}
// CHECK: %[[HOST_INPUT:.*]] = arith.trunci {{.*}} : tensor<64xi21> to tensor<64xi8>
// CHECK: %{{.*}} = "cim.transaction"(%[[HOST_INPUT]])
// CHECK: cim.configure_input {{.*}}group_id = 0 : i64{{.*}}work_id = 0 : i64
// CHECK: cim.group_barrier {group_id = 0 : i64}
// CHECK-NOT: cim.vmm

// GAP: CIM execution plan expects cim.transaction_idx = 0
// MISSING: 'cim.transaction' op expects non-negative i64 cim.transaction_idx
// JOINED: CIM transaction expects 'group_id' to match its work/group identity

func.func @host_handoff(%input: tensor<64xi8>) -> tensor<16xi21> {
  %wideWeight = arith.constant dense<0> : tensor<64x64xi8>
  %wideZero = arith.constant dense<0> : tensor<64xi21>
  %wide = linalg.matvec
      ins(%wideWeight, %input : tensor<64x64xi8>, tensor<64xi8>)
      outs(%wideZero : tensor<64xi21>) -> tensor<64xi21>
  %hostInput = arith.trunci %wide : tensor<64xi21> to tensor<64xi8>
  %weight = arith.constant dense<0> : tensor<16x64xi8>
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %hostInput : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}
