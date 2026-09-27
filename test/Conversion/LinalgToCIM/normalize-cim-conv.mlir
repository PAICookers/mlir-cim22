// RUN: mlir-cim22-opt %s -normalize-cim-conv | FileCheck %s

func.func @odd(%input: tensor<1x1x5x5xi8>) -> tensor<1x1x3x3xi32> {
  %weight = arith.constant dense<1> : tensor<1x1x3x3xi8>
  %zero = arith.constant dense<0> : tensor<1x1x3x3xi32>
  %result = linalg.conv_2d_nchw_fchw {
      cim.onnx.conv_integer,
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %weight : tensor<1x1x5x5xi8>, tensor<1x1x3x3xi8>)
      outs(%zero : tensor<1x1x3x3xi32>) -> tensor<1x1x3x3xi32>
  return %result : tensor<1x1x3x3xi32>
}

// CHECK-LABEL: func.func @odd
// CHECK: %[[WEIGHT:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]] : tensor<1x1x3x3xi8> into tensor<1x9xi8>
// CHECK: %[[COL:.*]] = linalg.generic
// CHECK: } -> tensor<1x9x9xi8>
// CHECK: %[[COL2D:.*]] = tensor.collapse_shape %[[COL]] {{\[}}[0, 1], [2]] : tensor<1x9x9xi8> into tensor<9x9xi8>
// CHECK: %[[MATMUL:.*]] = linalg.matmul {cim.onnx.matmul_integer} ins(%[[WEIGHT]], %[[COL2D]] : tensor<1x9xi8>, tensor<9x9xi8>)
// CHECK: tensor.expand_shape %[[MATMUL]] {{\[}}[0, 1], [2]] output_shape [1, 1, 9] : tensor<1x9xi32> into tensor<1x1x9xi32>
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: return %{{.*}} : tensor<1x1x3x3xi32>

func.func @rectangular(%input: tensor<1x2x6x7xi8>) -> tensor<1x17x3x3xi32> {
  %weight = arith.constant dense<1> : tensor<17x2x2x3xi8>
  %zero = arith.constant dense<0> : tensor<1x17x3x3xi32>
  %result = linalg.conv_2d_nchw_fchw {
      cim.onnx.conv_integer,
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<2> : tensor<2xi64>
    } ins(%input, %weight : tensor<1x2x6x7xi8>, tensor<17x2x2x3xi8>)
      outs(%zero : tensor<1x17x3x3xi32>) -> tensor<1x17x3x3xi32>
  return %result : tensor<1x17x3x3xi32>
}

// CHECK-LABEL: func.func @rectangular
// CHECK: %[[RECT_WEIGHT:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]] : tensor<17x2x2x3xi8> into tensor<17x12xi8>
// CHECK: %[[RECT_COL:.*]] = linalg.generic
// CHECK: } -> tensor<1x12x9xi8>
// CHECK: %[[RECT_COL2D:.*]] = tensor.collapse_shape %[[RECT_COL]] {{\[}}[0, 1], [2]] : tensor<1x12x9xi8> into tensor<12x9xi8>
// CHECK: %[[RECT_MATMUL:.*]] = linalg.matmul {cim.onnx.matmul_integer} ins(%[[RECT_WEIGHT]], %[[RECT_COL2D]] : tensor<17x12xi8>, tensor<12x9xi8>)
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: return %{{.*}} : tensor<1x17x3x3xi32>

// BF16 spatial convolution preserves the named op's patch indexing and forms
// a weight-first matrix product without an integer marker.
func.func @bf16_conv(%input: tensor<1x2x5x5xbf16>)
    -> tensor<1x3x3x3xbf16> {
  %weight = arith.constant dense<1.0> : tensor<3x2x3x3xbf16>
  %zero = arith.constant dense<0.0> : tensor<1x3x3x3xbf16>
  %result = linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %weight : tensor<1x2x5x5xbf16>, tensor<3x2x3x3xbf16>)
      outs(%zero : tensor<1x3x3x3xbf16>) -> tensor<1x3x3x3xbf16>
  return %result : tensor<1x3x3x3xbf16>
}

// CHECK-LABEL: func.func @bf16_conv
// CHECK: %[[WEIGHT:.*]] = arith.constant dense<1.000000e+00> : tensor<3x18xbf16>
// CHECK: %[[COL:.*]] = linalg.generic
// CHECK: } -> tensor<18x9xbf16>
// CHECK: linalg.matmul ins(%[[WEIGHT]], %[[COL]] : tensor<3x18xbf16>, tensor<18x9xbf16>)
// CHECK-NOT: cim.onnx.matmul_integer
// CHECK: return %{{.*}} : tensor<1x3x3x3xbf16>

// The static batch and spatial axes become M=2*3*3 columns.
func.func @bf16_multi_batch(%input: tensor<2x1x5x5xbf16>)
    -> tensor<2x1x3x3xbf16> {
  %weight = arith.constant dense<1.0> : tensor<1x1x3x3xbf16>
  %zero = arith.constant dense<0.0> : tensor<2x1x3x3xbf16>
  %result = linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %weight : tensor<2x1x5x5xbf16>, tensor<1x1x3x3xbf16>)
      outs(%zero : tensor<2x1x3x3xbf16>) -> tensor<2x1x3x3xbf16>
  return %result : tensor<2x1x3x3xbf16>
}

// CHECK-LABEL: func.func @bf16_multi_batch
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x9xbf16>, tensor<9x18xbf16>)
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: return %{{.*}} : tensor<2x1x3x3xbf16>

// The common pointwise case has no input patch-copy operation.
func.func @bf16_pointwise(%input: tensor<1x2x3x4xbf16>)
    -> tensor<1x3x3x4xbf16> {
  %weight = arith.constant dense<1.0> : tensor<3x2x1x1xbf16>
  %zero = arith.constant dense<0.0> : tensor<1x3x3x4xbf16>
  %result = linalg.conv_2d_nchw_fchw
      ins(%input, %weight : tensor<1x2x3x4xbf16>, tensor<3x2x1x1xbf16>)
      outs(%zero : tensor<1x3x3x4xbf16>) -> tensor<1x3x3x4xbf16>
  return %result : tensor<1x3x3x4xbf16>
}

// CHECK-LABEL: func.func @bf16_pointwise
// CHECK-NOT: linalg.generic
// CHECK: %[[VIEW:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0, 1], [2, 3]] : tensor<1x2x3x4xbf16> into tensor<2x12xbf16>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul ins(%{{.*}}, %[[VIEW]] : tensor<3x2xbf16>, tensor<2x12xbf16>)
// CHECK: return %{{.*}} : tensor<1x3x3x4xbf16>
