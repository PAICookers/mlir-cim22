// RUN: mlir-cim22-opt %s -normalize-cim-conv | FileCheck %s --implicit-check-not=linalg.matmul

// Dynamic activation dimensions require a runtime convolution policy.
func.func @dynamic_shape(%input: tensor<?x2x5x5xbf16>,
                         %init: tensor<?x3x3x3xbf16>) -> tensor<?x3x3x3xbf16> {
  %weight = arith.constant dense<1.0> : tensor<3x2x3x3xbf16>
  %result = linalg.conv_2d_nchw_fchw
      ins(%input, %weight : tensor<?x2x5x5xbf16>, tensor<3x2x3x3xbf16>)
      outs(%init : tensor<?x3x3x3xbf16>) -> tensor<?x3x3x3xbf16>
  return %result : tensor<?x3x3x3xbf16>
}
// CHECK-LABEL: func.func @dynamic_shape
// CHECK: linalg.conv_2d_nchw_fchw

// Static shapes alone do not make a runtime weight deployable.
func.func @runtime_weight(%input: tensor<1x2x5x5xbf16>,
                          %weight: tensor<2x3x3xbf16>) -> tensor<1x2x3x3xbf16> {
  %zero = arith.constant dense<0.0> : tensor<1x2x3x3xbf16>
  %result = linalg.depthwise_conv_2d_nchw_chw
      ins(%input, %weight : tensor<1x2x5x5xbf16>, tensor<2x3x3xbf16>)
      outs(%zero : tensor<1x2x3x3xbf16>) -> tensor<1x2x3x3xbf16>
  return %result : tensor<1x2x3x3xbf16>
}
// CHECK-LABEL: func.func @runtime_weight
// CHECK: linalg.depthwise_conv_2d_nchw_chw

// Bias-filled accumulators cannot be silently replaced with zero.
func.func @nonzero_init(%input: tensor<1x2x5x5xbf16>) -> tensor<1x2x3x3xbf16> {
  %weight = arith.constant dense<1.0> : tensor<2x3x3xbf16>
  %bias = arith.constant dense<2.0> : tensor<1x2x3x3xbf16>
  %result = linalg.depthwise_conv_2d_nchw_chw
      ins(%input, %weight : tensor<1x2x5x5xbf16>, tensor<2x3x3xbf16>)
      outs(%bias : tensor<1x2x3x3xbf16>) -> tensor<1x2x3x3xbf16>
  return %result : tensor<1x2x3x3xbf16>
}
// CHECK-LABEL: func.func @nonzero_init
// CHECK: linalg.depthwise_conv_2d_nchw_chw

func.func @negative_zero(%input: tensor<1x2x5x5xbf16>) -> tensor<1x2x3x3xbf16> {
  %weight = arith.constant dense<1.0> : tensor<2x3x3xbf16>
  %zero = arith.constant dense<-0.0> : tensor<1x2x3x3xbf16>
  %result = linalg.depthwise_conv_2d_nchw_chw
      ins(%input, %weight : tensor<1x2x5x5xbf16>, tensor<2x3x3xbf16>)
      outs(%zero : tensor<1x2x3x3xbf16>) -> tensor<1x2x3x3xbf16>
  return %result : tensor<1x2x3x3xbf16>
}
// CHECK-LABEL: func.func @negative_zero
// CHECK: linalg.depthwise_conv_2d_nchw_chw

// This pass does not implicitly change FP32 arithmetic or accumulation types.
func.func @fp32(%input: tensor<1x2x5x5xf32>) -> tensor<1x3x3x3xf32> {
  %weight = arith.constant dense<1.0> : tensor<3x2x3x3xf32>
  %zero = arith.constant dense<0.0> : tensor<1x3x3x3xf32>
  %result = linalg.conv_2d_nchw_fchw
      ins(%input, %weight : tensor<1x2x5x5xf32>, tensor<3x2x3x3xf32>)
      outs(%zero : tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
  return %result : tensor<1x3x3x3xf32>
}
// CHECK-LABEL: func.func @fp32
// CHECK: linalg.conv_2d_nchw_fchw

func.func @fp32_accumulator(%input: tensor<1x2x5x5xbf16>) -> tensor<1x3x3x3xf32> {
  %weight = arith.constant dense<1.0> : tensor<3x2x3x3xbf16>
  %zero = arith.constant dense<0.0> : tensor<1x3x3x3xf32>
  %result = linalg.conv_2d_nchw_fchw
      ins(%input, %weight : tensor<1x2x5x5xbf16>, tensor<3x2x3x3xbf16>)
      outs(%zero : tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
  return %result : tensor<1x3x3x3xf32>
}
// CHECK-LABEL: func.func @fp32_accumulator
// CHECK: linalg.conv_2d_nchw_fchw

// RUN: mlir-cim22-opt %s --cim-bf16-prepare | FileCheck %s --check-prefix=BIAS
// BIAS-LABEL: func.func @nested
// BIAS: scf.if
// BIAS: linalg.conv_2d_nchw_fchw
// BIAS: scf.yield
// BIAS-NOT: arith.addf
func.func @nested(%cond: i1, %x: tensor<1x1x1x1xf32>, %bias: tensor<1xf32>) -> tensor<1x1x1x1xf32> {
      %w = arith.constant dense<0.5> : tensor<1x1x1x1xf32>
      %e = tensor.empty() : tensor<1x1x1x1xf32>
      %b = linalg.broadcast ins(%bias : tensor<1xf32>) outs(%e : tensor<1x1x1x1xf32>) dimensions = [0, 2, 3]
      %y = scf.if %cond -> tensor<1x1x1x1xf32> {
        %r = linalg.conv_2d_nchw_fchw ins(%x, %w : tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) outs(%b : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
        scf.yield %r : tensor<1x1x1x1xf32>
      } else { scf.yield %x : tensor<1x1x1x1xf32> }
      return %y : tensor<1x1x1x1xf32>
    }
