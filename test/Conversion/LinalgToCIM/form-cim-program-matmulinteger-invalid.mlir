// RUN: split-file %s %t
// RUN: mlir-cim22-opt %t/matvec.mlir -partition-cim-program -form-cim-program | FileCheck %s --check-prefix=MATVEC
// RUN: mlir-cim22-opt %t/matmul.mlir -partition-cim-program -form-cim-program | FileCheck %s --check-prefix=MATMUL

// MATVEC: linalg.matvec {cim.onnx.matmul_integer}
// MATVEC-NOT: cim.vmm
// MATMUL: linalg.matmul {cim.onnx.matmul_integer}
// MATMUL-NOT: cim.vmm

//--- matvec.mlir
func.func @marked_i21_matvec(%input: tensor<64xi8>) -> tensor<16xi21> {
  %weight = arith.constant dense<0> : tensor<16x64xi8>
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

//--- matmul.mlir
func.func @marked_i21_matmul(%input: tensor<64x1xi8>) -> tensor<16x1xi21> {
  %weight = arith.constant dense<0> : tensor<16x64xi8>
  %zero = arith.constant dense<0> : tensor<16x1xi21>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<16x1xi21>) -> tensor<16x1xi21>
  return %result : tensor<16x1xi21>
}
