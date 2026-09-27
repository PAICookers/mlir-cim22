// RUN: mlir-cim22-opt %s -normalize-cim-linear | FileCheck %s

// An activation-first Linear is transposed to the compiler's weight-first
// MatMul contract. The constant RHS is transposed at compile time.
func.func @linear(%activation: tensor<2x3xbf16>) -> tensor<2x2xbf16> {
  %weight = arith.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]> : tensor<3x2xbf16>
  %zero = arith.constant dense<0.0> : tensor<2x2xbf16>
  %result = linalg.matmul
      ins(%activation, %weight : tensor<2x3xbf16>, tensor<3x2xbf16>)
      outs(%zero : tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %result : tensor<2x2xbf16>
}

// CHECK-LABEL: func.func @linear(
// CHECK: arith.constant dense<{{.*}}1.000000e+00, 3.000000e+00, 5.000000e+00{{.*}}2.000000e+00, 4.000000e+00, 6.000000e+00{{.*}}> : tensor<2x3xbf16>
// CHECK: %[[INPUT:.*]] = linalg.transpose ins(%{{.*}} : tensor<2x3xbf16>)
// CHECK: %[[CORE:.*]] = linalg.matmul ins(%{{.*}}, %[[INPUT]] : tensor<2x3xbf16>, tensor<3x2xbf16>)
// CHECK: %[[OUTPUT:.*]] = linalg.transpose ins(%[[CORE]] : tensor<2x2xbf16>)
// CHECK: return %[[OUTPUT]] : tensor<2x2xbf16>

// A nonconstant RHS is not a deployable static CIM weight.
func.func @dynamic_weight(%activation: tensor<2x3xbf16>,
                          %weight: tensor<3x2xbf16>) -> tensor<2x2xbf16> {
  %zero = arith.constant dense<0.0> : tensor<2x2xbf16>
  %result = linalg.matmul
      ins(%activation, %weight : tensor<2x3xbf16>, tensor<3x2xbf16>)
      outs(%zero : tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %result : tensor<2x2xbf16>
}

// CHECK-LABEL: func.func @dynamic_weight(
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<2x3xbf16>, tensor<3x2xbf16>)
// CHECK-NOT: linalg.transpose

// Torch-style Linear can also arrive as a matmul with a transpose-B map.
func.func @transpose_b(%activation: tensor<2x3xbf16>) -> tensor<2x4xbf16> {
  %weight = arith.constant dense<1.0> : tensor<4x3xbf16>
  %zero = arith.constant dense<0.0> : tensor<2x4xbf16>
  %result = linalg.matmul indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (m, n)>]
      ins(%activation, %weight : tensor<2x3xbf16>, tensor<4x3xbf16>)
      outs(%zero : tensor<2x4xbf16>) -> tensor<2x4xbf16>
  return %result : tensor<2x4xbf16>
}

// CHECK-LABEL: func.func @transpose_b(
// CHECK: %[[INPUT:.*]] = linalg.transpose ins(%{{.*}} : tensor<2x3xbf16>)
// CHECK: %[[CORE:.*]] = linalg.matmul ins(%{{.*}}, %[[INPUT]] : tensor<4x3xbf16>, tensor<3x2xbf16>)
// CHECK: linalg.transpose ins(%[[CORE]] : tensor<4x2xbf16>)
