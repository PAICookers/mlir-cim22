// RUN: mlir-cim22-opt %s --cim-bf16-prepare | FileCheck %s --check-prefix=PREP
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=split' | FileCheck %s --check-prefix=SPLIT
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=vmm' | FileCheck %s --check-prefix=VMM
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline | FileCheck %s --check-prefix=PACKET
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=llvm' | FileCheck %s --check-prefix=LLVM
// RUN: not mlir-cim22-opt %s --cim-bf16-pipeline='stage=unknown' 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: mlir-cim22-opt %s --normalize-cim-conv='allow-f32=true' | FileCheck %s --check-prefix=NOEPILOGUE

// Conv -> Host bias -> depthwise -> Linear -> residual. Shared FP32 inputs
// and function signatures survive; each kernel receives one BF16 activation.
func.func @mixed(%input: tensor<1x2x3x3xf32>, %residual: tensor<1x2xf32>)
    -> tensor<1x2xf32> {
  %pw_weight = arith.constant dense<1.0> : tensor<3x2x1x1xf32>
  %z = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x3x3x3xf32>
  %zero = linalg.fill ins(%z : f32) outs(%empty : tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
  %bias_vec = arith.constant dense<0.5> : tensor<3xf32>
  %bias_init = linalg.broadcast ins(%bias_vec : tensor<3xf32>) outs(%empty : tensor<1x3x3x3xf32>) dimensions = [0, 2, 3]
  %pw = linalg.conv_2d_nchw_fchw ins(%input, %pw_weight : tensor<1x2x3x3xf32>, tensor<3x2x1x1xf32>)
      outs(%bias_init : tensor<1x3x3x3xf32>) -> tensor<1x3x3x3xf32>
  %dw_weight = arith.constant dense<1.0> : tensor<3x3x3xf32>
  %dw_zero = arith.constant dense<0.0> : tensor<1x3x1x1xf32>
  %dw_bias_vec = arith.constant dense<1.0> : tensor<3xf32>
  %dw_init = linalg.broadcast ins(%dw_bias_vec : tensor<3xf32>) outs(%dw_zero : tensor<1x3x1x1xf32>) dimensions = [0, 2, 3]
  %dw = linalg.depthwise_conv_2d_nchw_chw ins(%pw, %dw_weight : tensor<1x3x3x3xf32>, tensor<3x3x3xf32>)
      outs(%dw_init : tensor<1x3x1x1xf32>) -> tensor<1x3x1x1xf32>
  %flat = tensor.collapse_shape %dw [[0], [1, 2, 3]] : tensor<1x3x1x1xf32> into tensor<1x3xf32>
  %linear_weight = arith.constant dense<[[1.0, -1.0], [2.0, 1.0], [-1.0, 2.0]]> : tensor<3x2xf32>
  %linear_zero = arith.constant dense<0.0> : tensor<1x2xf32>
  %linear = linalg.matmul ins(%flat, %linear_weight : tensor<1x3xf32>, tensor<3x2xf32>)
      outs(%linear_zero : tensor<1x2xf32>) -> tensor<1x2xf32>
  %out = arith.addf %linear, %residual : tensor<1x2xf32>
  return %out : tensor<1x2xf32>
}

// A runtime weight stays entirely on Host, even with static shapes/zero init.
func.func @host_matmul(%a: tensor<2x3xf32>, %w: tensor<3x2xf32>) -> tensor<2x2xf32> {
  %zero = arith.constant dense<0.0> : tensor<2x2xf32>
  %out = linalg.matmul ins(%a, %w : tensor<2x3xf32>, tensor<3x2xf32>)
      outs(%zero : tensor<2x2xf32>) -> tensor<2x2xf32>
  return %out : tensor<2x2xf32>
}

// Cross-K arithmetic must return to Host after the transaction split.
func.func @cross_k(%x: tensor<65xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<1.0> : tensor<1x65xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x65xf32>, tensor<65xf32>)
      outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}

// PREP-LABEL: func.func @mixed
// PREP: arith.truncf {{.*}}tensor<2x9xf32> to tensor<2x9xbf16>
// PREP: linalg.matmul
// PREP: arith.extf
// PREP: arith.addf
// PREP-LABEL: func.func @host_matmul
// PREP-NOT: arith.truncf
// PREP: linalg.matmul
// PREP: return

// SPLIT: module @__cim_host
// SPLIT-LABEL: func.func @mixed
// SPLIT: call @__cim_kernel_0
// SPLIT: arith.extf
// SPLIT: arith.addf
// SPLIT: call @__cim_kernel_1
// SPLIT: call @__cim_kernel_2
// SPLIT: call @__cim_kernel_3
// SPLIT: call @__cim_kernel_4
// SPLIT: arith.addf
// SPLIT-LABEL: func.func @host_matmul
// SPLIT: linalg.matmul
// SPLIT: module @__cim_device
// SPLIT: func.func @__cim_kernel_0
// SPLIT: linalg.matmul
// SPLIT: func.func @__cim_kernel_4
// SPLIT: linalg.matmul

// VMM: module @__cim_host
// VMM: linalg.matmul
// VMM: module @__cim_device
// VMM: cim.vmm
// VMM-NOT: linalg.matmul

// PACKET: module @__cim_host
// PACKET: func.func private @__cim_kernel_5(
// PACKET: call @__cim_kernel_5_transaction_0
// PACKET: arith.extf
// PACKET: arith.extf
// PACKET: arith.addf
// PACKET: arith.truncf
// PACKET: module @__cim_device
// PACKET: cimframe.control_bf16_packet
// PACKET: cimframe.weight_exponent_packet
// PACKET: cim.transaction
// PACKET-NOT: cim.vmm
// PACKET-NOT: tensor.pad
// PACKET-NOT: tensor.extract_slice
// PACKET-NOT: arith.addf

// LLVM: module @__cim_host
// LLVM: llvm.func @mixed
// LLVM: llvm.call @__cim_kernel_0
// LLVM: llvm.func @host_matmul
// LLVM: llvm.func @__cim_kernel_0
// LLVM: module @__cim_device
// LLVM: cimframe.control_bf16_packet
// LLVM: cim.transaction

// INVALID: stage must be split, vmm, packet or llvm

// A default precision-preserving normalizer must retain bias-initialized Conv.
// NOEPILOGUE-LABEL: func.func @mixed
// NOEPILOGUE: linalg.conv_2d_nchw_fchw
// NOEPILOGUE: linalg.depthwise_conv_2d_nchw_chw

// Distinct column values in the executable test detect dropped/reordered
// slices or writing repeatedly to the same output column in a Host loop.
func.func @stream_order(%x: tensor<2x5xbf16>) -> tensor<2x5xbf16> {
  %w = arith.constant dense<[[1.0, 2.0], [-1.0, 1.0]]> : tensor<2x2xbf16>
  %z = arith.constant dense<0.0> : tensor<2x5xbf16>
  %r = linalg.matmul ins(%w, %x : tensor<2x2xbf16>, tensor<2x5xbf16>)
      outs(%z : tensor<2x5xbf16>) -> tensor<2x5xbf16>
  return %r : tensor<2x5xbf16>
}
