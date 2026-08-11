// RUN: split-file %s %t
// RUN: not mlir-cim22-opt %t/partial.mlir -form-cim-program 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: not mlir-cim22-opt %t/result.mlir -form-cim-program 2>&1 | FileCheck %s --check-prefix=RESULT

// PARTIAL: ONNX MatMulInteger violates CTQ-013: a constant K<=64 partial is outside signed i21
// RESULT: ONNX MatMulInteger constant-weight range proof exceeds signed i32 result

//--- partial.mlir
func.func @partial_overflow(%input: tensor<64x1xi8>) -> tensor<1x1xi32> {
  %weight = arith.constant dense<-128> : tensor<1x64xi8>
  %zero = arith.constant dense<0> : tensor<1x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<1x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result : tensor<1x1xi32>
}

//--- result.mlir
func.func @result_overflow(%input: tensor<133145x1xi8>) -> tensor<1x1xi32> {
  %weight = arith.constant dense<127> : tensor<1x133145xi8>
  %zero = arith.constant dense<0> : tensor<1x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<1x133145xi8>, tensor<133145x1xi8>)
      outs(%zero : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result : tensor<1x1xi32>
}
