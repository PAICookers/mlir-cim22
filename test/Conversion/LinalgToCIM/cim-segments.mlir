// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(partition-cim-program,form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,func.func(verify-cim-execution-plan))' > %t
// RUN: FileCheck %s < %t
// RUN: sed '0,/cim.segment_id = 1 : i64/s//cim.segment_id = 2 : i64/' %t > %t.gap
// RUN: not mlir-cim22-opt %t.gap -verify-cim-execution-plan 2>&1 | FileCheck %s --check-prefix=GAP
// RUN: sed '0,/cim.segment_id = 0 : i64, /s///' %t > %t.missing
// RUN: not mlir-cim22-opt %t.missing 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: sed '0,/cim.group_barrier {cim.segment_id = 0 : i64/s//cim.group_barrier {cim.segment_id = 1 : i64/' %t > %t.joined
// RUN: not mlir-cim22-opt %t.joined -verify-cim-execution-plan 2>&1 | FileCheck %s --check-prefix=JOINED

// CHECK-LABEL: func.func @host_handoff(
// CHECK: cim.configure_input {{.*}}cim.segment_id = 0 : i64{{.*}}group_id = 0 : i64{{.*}}work_id = 0 : i64
// CHECK: cim.group_barrier {cim.segment_id = 0 : i64, group_id = 1 : i64}
// CHECK: %[[HOST_INPUT:.*]] = arith.trunci {{.*}} : tensor<64xi21> to tensor<64xi8>
// CHECK: cim.configure_input %[[HOST_INPUT]] {{.*}}cim.segment_id = 1 : i64{{.*}}group_id = 0 : i64{{.*}}work_id = 0 : i64
// CHECK: cim.group_barrier {cim.segment_id = 1 : i64, group_id = 0 : i64}
// CHECK-NOT: cim.vmm

// GAP: CIM execution plan expects cim.segment_id = 1, but got 2
// MISSING: 'cim.configure_input' op requires complete execution-plan identity
// JOINED: CIM execution plan expects 'cim.segment_id' to match group identity

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
