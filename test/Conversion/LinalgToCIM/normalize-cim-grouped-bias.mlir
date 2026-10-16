// RUN: mlir-cim22-opt %s --normalize-cim-conv='allow-f32=true allow-bias-epilogue=true' | FileCheck %s
// RUN: mlir-cim22-opt %s --normalize-cim-conv='allow-f32=true' | FileCheck %s --check-prefix=NOEPILOGUE --implicit-check-not=linalg.matmul

// The wheel reshapes an NCHW channel broadcast into grouped output axes.
// Keep that complete initializer as an FP32 epilogue, including shared users.
// CHECK-LABEL: func.func @grouped_channel_bias(
// CHECK: %[[BIAS:.*]] = linalg.broadcast
// CHECK: %[[INIT:.*]] = tensor.expand_shape %[[BIAS]]
// CHECK-COUNT-2: linalg.matmul
// CHECK: %[[RESULT:.*]] = arith.addf %{{.*}}, %[[INIT]]
// CHECK: return %[[RESULT]], %[[INIT]]
// NOEPILOGUE-LABEL: func.func @grouped_channel_bias(
// NOEPILOGUE: linalg.conv_2d_ngchw_gfchw
func.func @grouped_channel_bias(%input: tensor<1x2x1x5x5xf32>, %bias: tensor<4xf32>)
    -> (tensor<1x2x2x3x3xf32>, tensor<1x2x2x3x3xf32>) {
  %weight = arith.constant dense<1.0> : tensor<2x2x1x3x3xf32>
  %empty = tensor.empty() : tensor<1x4x3x3xf32>
  %broadcast = linalg.broadcast ins(%bias : tensor<4xf32>)
      outs(%empty : tensor<1x4x3x3xf32>) dimensions = [0, 2, 3]
  %init = tensor.expand_shape %broadcast [[0], [1, 2], [3], [4]]
      output_shape [1, 2, 2, 3, 3] : tensor<1x4x3x3xf32> into tensor<1x2x2x3x3xf32>
  %result = linalg.conv_2d_ngchw_gfchw
      ins(%input, %weight : tensor<1x2x1x5x5xf32>, tensor<2x2x1x3x3xf32>)
      outs(%init : tensor<1x2x2x3x3xf32>) -> tensor<1x2x2x3x3xf32>
  return %result, %init : tensor<1x2x2x3x3xf32>, tensor<1x2x2x3x3xf32>
}

// Splitting a spatial dimension is outside the channel-bias match.
// CHECK-LABEL: func.func @different_reassociation(
// CHECK-NOT: linalg.matmul
// CHECK: linalg.conv_2d_ngchw_gfchw
// NOEPILOGUE-LABEL: func.func @different_reassociation(
// NOEPILOGUE: linalg.conv_2d_ngchw_gfchw
func.func @different_reassociation(%input: tensor<1x2x1x5x5xf32>, %bias: tensor<2xf32>)
    -> tensor<1x2x2x3x3xf32> {
  %weight = arith.constant dense<1.0> : tensor<2x2x1x3x3xf32>
  %empty = tensor.empty() : tensor<1x2x6x3xf32>
  %broadcast = linalg.broadcast ins(%bias : tensor<2xf32>)
      outs(%empty : tensor<1x2x6x3xf32>) dimensions = [0, 2, 3]
  %init = tensor.expand_shape %broadcast [[0], [1], [2, 3], [4]]
      output_shape [1, 2, 2, 3, 3] : tensor<1x2x6x3xf32> into tensor<1x2x2x3x3xf32>
  %result = linalg.conv_2d_ngchw_gfchw
      ins(%input, %weight : tensor<1x2x1x5x5xf32>, tensor<2x2x1x3x3xf32>)
      outs(%init : tensor<1x2x2x3x3xf32>) -> tensor<1x2x2x3x3xf32>
  return %result : tensor<1x2x2x3x3xf32>
}
