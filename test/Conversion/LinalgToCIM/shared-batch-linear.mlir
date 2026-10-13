// RUN: mlir-cim22-opt %s --cim-bf16-prepare --canonicalize --cse | FileCheck %s
// RUN: mlir-cim22-opt %s --normalize-cim-linear --canonicalize | FileCheck %s --check-prefix=OPTIN

// Distinct rows and batches catch transposition or T/B mixing. Resource storage
// is the default produced by torch-mlir; do not change the frontend to dense.
func.func @shared(%x: tensor<2x3x2xf32>) -> tensor<2x3x2xf32> {
  %w = arith.constant dense_resource<weights> : tensor<2x2xf32>
  %e = tensor.empty() : tensor<2x2xf32>
  %wt = linalg.transpose ins(%w : tensor<2x2xf32>) outs(%e : tensor<2x2xf32>) permutation = [1, 0]
  %eb = tensor.empty() : tensor<2x2x2xf32>
  %wb = linalg.generic {
    indexing_maps = [affine_map<(b,k,n)->(k,n)>, affine_map<(b,k,n)->(b,k,n)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%wt : tensor<2x2xf32>) outs(%eb : tensor<2x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<2x2x2xf32>
  %z = arith.constant dense<0.0> : tensor<2x3x2xf32>
  %r = linalg.batch_matmul ins(%x, %wb : tensor<2x3x2xf32>, tensor<2x2x2xf32>) outs(%z : tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
  return %r : tensor<2x3x2xf32>
}
// CHECK-LABEL: func.func @shared
// CHECK: dense<{{.*}}1.000000e+00, 2.000000e+00{{.*}}3.000000e+00, 4.000000e+00{{.*}}> : tensor<2x2xbf16>
// CHECK: tensor.collapse_shape %{{.*}} {{\[\[}}0, 1], [2]] : tensor<2x3x2xf32> into tensor<6x2xf32>
// CHECK: arith.truncf {{.*}} : tensor<2x6xf32> to tensor<2x6xbf16>
// CHECK: linalg.matmul {{.*}} tensor<2x2xbf16>, tensor<2x6xbf16>
// CHECK: arith.extf {{.*}} : tensor<2x6xbf16> to tensor<2x6xf32>
// CHECK: tensor.expand_shape {{.*}} into tensor<2x3x2xf32>
// CHECK-NOT: linalg.batch_matmul
// OPTIN-LABEL: func.func @shared
// OPTIN: linalg.batch_matmul

// A separate matrix per batch is not a shared Linear, even if constant.
func.func @per_batch(%x: tensor<2x3x2xf32>) -> tensor<2x3x2xf32> {
  %w = arith.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : tensor<2x2x2xf32>
  %z = arith.constant dense<0.0> : tensor<2x3x2xf32>
  %r = linalg.batch_matmul ins(%x, %w : tensor<2x3x2xf32>, tensor<2x2x2xf32>) outs(%z : tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
  return %r : tensor<2x3x2xf32>
}
// CHECK-LABEL: func.func @per_batch
// CHECK: linalg.batch_matmul
// CHECK-NOT: arith.truncf

// Nonzero initialization cannot move through the BF16 rounding boundary.
func.func @nonzero(%x: tensor<2x3x2xf32>) -> tensor<2x3x2xf32> {
  %w = arith.constant dense<1.0> : tensor<2x2xf32>
  %e = tensor.empty() : tensor<2x2x2xf32>
  %wb = linalg.broadcast ins(%w : tensor<2x2xf32>) outs(%e : tensor<2x2x2xf32>) dimensions = [0]
  %z = arith.constant dense<1.0> : tensor<2x3x2xf32>
  %r = linalg.batch_matmul ins(%x, %wb : tensor<2x3x2xf32>, tensor<2x2x2xf32>) outs(%z : tensor<2x3x2xf32>) -> tensor<2x3x2xf32>
  return %r : tensor<2x3x2xf32>
}
// CHECK-LABEL: func.func @nonzero
// CHECK: linalg.batch_matmul
// CHECK-NOT: arith.truncf

// Native BF16 and the named broadcast form do not require FP32 opt-in.
func.func @bf16_broadcast(%x: tensor<2x3x2xbf16>) -> tensor<2x3x2xbf16> {
  %w = arith.constant dense<1.0> : tensor<2x2xbf16>
  %e = tensor.empty() : tensor<2x2x2xbf16>
  %wb = linalg.broadcast ins(%w : tensor<2x2xbf16>) outs(%e : tensor<2x2x2xbf16>) dimensions = [0]
  %z = arith.constant dense<0.0> : tensor<2x3x2xbf16>
  %r = linalg.batch_matmul ins(%x, %wb : tensor<2x3x2xbf16>, tensor<2x2x2xbf16>) outs(%z : tensor<2x3x2xbf16>) -> tensor<2x3x2xbf16>
  return %r : tensor<2x3x2xbf16>
}
// CHECK-LABEL: func.func @bf16_broadcast
// CHECK: tensor.collapse_shape
// CHECK: linalg.matmul
// CHECK: tensor.expand_shape
// CHECK-NOT: linalg.batch_matmul
// OPTIN-LABEL: func.func @bf16_broadcast
// OPTIN: tensor.collapse_shape
// OPTIN: linalg.matmul
// OPTIN: tensor.expand_shape
// OPTIN-NOT: linalg.batch_matmul

{-#
  dialect_resources: {
    builtin: {
      weights: "0x040000000000803F000000400000404000008040"
    }
  }
#-}
