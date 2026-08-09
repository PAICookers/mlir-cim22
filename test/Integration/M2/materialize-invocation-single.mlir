// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-invocation)' > %t
// RUN: %python %S/../../Reference/M2/invocation_oracle.py %t | FileCheck %s --check-prefix=ORACLE
// RUN: FileCheck %s --check-prefix=HOST < %t

// ORACLE: PASS software-only profile=cim22-4x5-v1 schema=1 groups=1 works=1 configs=2

// HOST: cim.configure_input {{.*}}macro_slot = 0 : i64{{.*}}work_id = 0 : i64
// HOST-NEXT: cim.configure_weight {{.*}}macro_slot = 0 : i64{{.*}}work_id = 0 : i64
// HOST-NEXT: cim.dispatch {{.*}}macro_slot = 0 : i64{{.*}}work_id = 0 : i64
// HOST-NEXT: cim.once
// HOST-NEXT: %[[READ:.*]] = cim.readback {{.*}}macro_slot = 0 : i64{{.*}}work_id = 0 : i64{{.*}} : tensor<16xi21>
// HOST-NEXT: cim.group_barrier {group_id = 0 : i64}
// HOST: arith.extsi %[[READ]] : tensor<16xi21> to tensor<16xi32>
// HOST-NOT: arith.addi
// HOST: return {{.*}} : tensor<16x1xi32>

func.func @matmul_integer_single(%input: tensor<64x1xi8>) -> tensor<16x1xi32> {
  %weight = arith.constant dense<0> : tensor<16x64xi8>
  %zero = arith.constant dense<0> : tensor<16x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<16x1xi32>) -> tensor<16x1xi32>
  return %result : tensor<16x1xi32>
}
