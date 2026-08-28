// RUN: mlir-cim22-opt %s -partition-cim-program -form-cim-program | FileCheck %s

// CHECK: linalg.matmul {cim.onnx.matmul_integer}
// CHECK-NOT: cim.vmm

func.func @partial_overflow(%input: tensor<64x1xi8>) -> tensor<1x1xi32> {
  %weight = arith.constant dense<-128> : tensor<1x64xi8>
  %zero = arith.constant dense<0> : tensor<1x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<1x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %result : tensor<1x1xi32>
}
