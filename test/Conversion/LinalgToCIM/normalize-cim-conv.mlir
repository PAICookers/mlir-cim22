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
