// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan)' > %t.once
// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan)' > %t.twice
// RUN: diff %t.once %t.twice
// RUN: %python %S/../../Reference/M2/execution_plan_oracle.py %t.once | FileCheck %s --check-prefix=ORACLE
// RUN: FileCheck %s --check-prefix=HOST < %t.once

// ORACLE: PASS software-only profile=cim22-4x5-v1 schema=1 groups=1 works=2 configs=4

// HOST: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// HOST: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// HOST: arith.addi {{.*}} : tensor<16xi32>
// HOST-NOT: arith.trunci
// HOST-NOT: cim.vmm
// HOST-NOT: cimframe.
// HOST: return {{.*}} : tensor<16x1xi32>

func.func @matmul_integer(%input: tensor<128x1xi8>) -> tensor<16x1xi32> {
  %weight = arith.constant dense<0> : tensor<16x128xi8>
  %zero = arith.constant dense<0> : tensor<16x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<16x128xi8>, tensor<128x1xi8>)
      outs(%zero : tensor<16x1xi32>) -> tensor<16x1xi32>
  return %result : tensor<16x1xi32>
}
