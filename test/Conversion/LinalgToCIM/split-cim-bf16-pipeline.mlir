// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=split require-cim=true' | FileCheck %s
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=packet require-cim=true' | FileCheck %s --check-prefix=PACKET
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=llvm require-cim=true' | FileCheck %s --check-prefix=LLVM
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=split bf16-compensation=none require-cim=true' | FileCheck %s --check-prefix=RAW

// Original f32 grouped convolution, including signed static weights, is
// normalized before correction. Every group has four branches; no application
// adapter is present. The two channels of a depthwise op remain separate.
func.func @grouped(%x: tensor<1x2x1x2x2xf32>) -> tensor<1x2x1x2x2xf32> {
  %w = arith.constant dense<[[[[[1.0]]]], [[[[-0.5]]]]]> : tensor<2x1x1x1x1xf32>
  %z = arith.constant dense<0.0> : tensor<1x2x1x2x2xf32>
  %r = linalg.conv_2d_ngchw_gfchw ins(%x, %w : tensor<1x2x1x2x2xf32>, tensor<2x1x1x1x1xf32>) outs(%z : tensor<1x2x1x2x2xf32>) -> tensor<1x2x1x2x2xf32>
  return %r : tensor<1x2x1x2x2xf32>
}
func.func @depthwise(%x: tensor<1x2x3x3xf32>) -> tensor<1x2x2x2xf32> {
  %w = arith.constant dense<[[[1.0, -0.5], [0.25, -0.125]], [[-1.0, 0.5], [-0.25, 0.125]]]> : tensor<2x2x2xf32>
  %z = arith.constant dense<0.0> : tensor<1x2x2x2xf32>
  %r = linalg.depthwise_conv_2d_nchw_chw ins(%x, %w : tensor<1x2x3x3xf32>, tensor<2x2x2xf32>) outs(%z : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %r : tensor<1x2x2x2xf32>
}
func.func @ordinary(%x: tensor<1x2x3x3xf32>) -> tensor<1x1x2x2xf32> {
  %w = arith.constant dense<-0.5> : tensor<1x2x2x2xf32>
  %z = arith.constant dense<0.0> : tensor<1x1x2x2xf32>
  %r = linalg.conv_2d_nchw_fchw ins(%x, %w : tensor<1x2x3x3xf32>, tensor<1x2x2x2xf32>) outs(%z : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
  return %r : tensor<1x1x2x2xf32>
}
// CHECK: module attributes {cim.bf16_compensation = "sign-split",
// CHECK: module @__cim_host
// CHECK-LABEL: func.func @grouped
// CHECK-COUNT-8: call @__cim_kernel_
// CHECK: return
// CHECK-LABEL: func.func @depthwise
// CHECK-COUNT-8: call @__cim_kernel_
// CHECK: return
// CHECK-LABEL: func.func @ordinary
// CHECK-COUNT-4: call @__cim_kernel_
// CHECK: return
// CHECK: module @__cim_device
// CHECK-COUNT-20: linalg.matmul
// CHECK-NOT: linalg.matmul
// RAW: module attributes {cim.bf16_compensation = "none",
// RAW: module @__cim_host
// RAW-LABEL: func.func @grouped
// RAW-COUNT-2: call @__cim_kernel_
// RAW: return
// RAW-LABEL: func.func @depthwise
// RAW-COUNT-2: call @__cim_kernel_
// RAW: return
// RAW-LABEL: func.func @ordinary
// RAW: call @__cim_kernel_
// RAW: return
// RAW: module @__cim_device
// RAW-COUNT-5: linalg.matmul
// RAW-NOT: linalg.matmul
// PACKET: module attributes {cim.bf16_compensation = "sign-split",
// PACKET: module @__cim_host
// PACKET: module @__cim_device
// PACKET: cimframe.control_bf16_packet
// PACKET-NOT: linalg.matmul
// LLVM: module attributes {cim.bf16_compensation = "sign-split",
// LLVM: llvm.func @grouped
// LLVM: llvm.func @depthwise
// LLVM: llvm.func @ordinary
