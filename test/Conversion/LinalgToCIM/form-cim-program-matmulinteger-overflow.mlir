// RUN: not mlir-cim22-opt %s -form-cim-program 2>&1 | FileCheck %s

// CHECK: ONNX MatMulInteger violates CTQ-013: a constant K<=64 partial is outside signed i21

func.func @partial_overflow(%input: tensor<64x1xi8>) -> tensor<1x1xi32> {
  %weight = arith.constant dense<-128> : tensor<1x64xi8>
  %zero = arith.constant dense<0> : tensor<1x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<1x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result : tensor<1x1xi32>
}
