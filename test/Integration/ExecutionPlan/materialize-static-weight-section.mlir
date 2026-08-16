// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,materialize-cim-static-weight-section)' > %t.first
// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,materialize-cim-static-weight-section)' > %t.second
// RUN: cmp %t.first %t.second
// RUN: %python %S/../../python/CIM22/static_weight_section_oracle.py %t.first --expect-commands 2 | FileCheck %s --check-prefix=ORACLE
// RUN: sed '0,/macro = 0 : i32/s//macro = 1 : i32/' %t.first > %t.bad
// RUN: not %python %S/../../python/CIM22/static_weight_section_oracle.py %t.bad --expect-commands 2 2>&1 | FileCheck %s --check-prefix=FAULT
// RUN: FileCheck %s --check-prefix=HOST < %t.first

// ORACLE: PASS software-only commands=2 resources=2
// FAULT: OracleError: command 0: provenance/route/Macro mismatch
// HOST: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// HOST: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// HOST: arith.addi {{.*}} : tensor<16xi32>
// HOST-NOT: arith.trunci
// HOST: return {{.*}} : tensor<16x1xi32>

func.func @matmul_integer(%input: tensor<128x1xi8>) -> tensor<16x1xi32> {
  %weight = arith.constant dense<0> : tensor<16x128xi8>
  %zero = arith.constant dense<0> : tensor<16x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<16x128xi8>, tensor<128x1xi8>)
      outs(%zero : tensor<16x1xi32>) -> tensor<16x1xi32>
  return %result : tensor<16x1xi32>
}
