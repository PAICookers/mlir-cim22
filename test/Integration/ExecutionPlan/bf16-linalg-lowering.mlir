// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(normalize-cim-linear,normalize-cim-conv,partition-cim-program,form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,materialize-cim-static-weight-section,lower-cimframe-commands-to-packets,verify-cimframe)' | FileCheck %s

// A formed BF16 VMM must survive the existing schedule and packet pipeline.
// This checks compilation only; the supplier-model runner is not a standard
// BF16 numerical oracle for the conventional-BF16 hardware assumption.
// CHECK: cim.static_weight @__cim_weight_bf16_linalg_s0_w0
// CHECK: cimframe.control_bf16_packet
// CHECK: cimframe.weight_exponent_packet
// CHECK: cimframe.cim_bf16_weight_packet
// CHECK-LABEL: func.func @bf16_linalg(
// CHECK: cim.transaction
// CHECK-NOT: linalg.matvec
func.func @bf16_linalg(%input: tensor<64xbf16>) -> tensor<16xbf16> {
  %weight = arith.constant dense<1.0> : tensor<16x64xbf16>
  %zero = arith.constant dense<0.0> : tensor<16xbf16>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x64xbf16>, tensor<64xbf16>)
      outs(%zero : tensor<16xbf16>) -> tensor<16xbf16>
  return %result : tensor<16xbf16>
}

// A 65-element reduction crosses two K tiles; the resulting Host operations
// must remain after readback and before the function return.
// CHECK-LABEL: func.func @bf16_linalg_multi_k(
// CHECK: cim.transaction
// CHECK: arith.extf {{.*}} : tensor<16xbf16> to tensor<16xf32>
// CHECK: arith.extf {{.*}} : tensor<16xbf16> to tensor<16xf32>
// CHECK: arith.addf {{.*}} : tensor<16xf32>
// CHECK: arith.truncf {{.*}} : tensor<16xf32> to tensor<16xbf16>
// CHECK-NOT: linalg.matvec
func.func @bf16_linalg_multi_k(%input: tensor<65xbf16>) -> tensor<16xbf16> {
  %weight = arith.constant dense<1.0> : tensor<16x65xbf16>
  %zero = arith.constant dense<0.0> : tensor<16xbf16>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x65xbf16>, tensor<65xbf16>)
      outs(%zero : tensor<16xbf16>) -> tensor<16xbf16>
  return %result : tensor<16xbf16>
}

// An activation-first Linear with a constant weight reaches CIM through the
// BF16 transpose, tiled MatMul and packet stages.
// CHECK-LABEL: func.func @bf16_linear(
// CHECK: cim.transaction
// CHECK: arith.addf {{.*}} : tensor<16xf32>
// CHECK: linalg.transpose
// CHECK-NOT: linalg.matmul
func.func @bf16_linear(%input: tensor<2x65xbf16>) -> tensor<2x2xbf16> {
  %weight = arith.constant dense<1.0> : tensor<65x2xbf16>
  %zero = arith.constant dense<0.0> : tensor<2x2xbf16>
  %result = linalg.matmul
      ins(%input, %weight : tensor<2x65xbf16>, tensor<65x2xbf16>)
      outs(%zero : tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %result : tensor<2x2xbf16>
}


// A batch-one BF16 pointwise convolution reaches the packet pipeline through
// a reshape and the existing weight-first MatMul conversion.
// CHECK-LABEL: func.func @bf16_pointwise(
// CHECK: tensor.collapse_shape
// CHECK: cim.transaction
// CHECK-NOT: linalg.conv_2d_nchw_fchw
func.func @bf16_pointwise(%input: tensor<1x2x2x2xbf16>)
    -> tensor<1x16x2x2xbf16> {
  %weight = arith.constant dense<1.0> : tensor<16x2x1x1xbf16>
  %zero = arith.constant dense<0.0> : tensor<1x16x2x2xbf16>
  %result = linalg.conv_2d_nchw_fchw {
      dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } ins(%input, %weight : tensor<1x2x2x2xbf16>, tensor<16x2x1x1xbf16>)
      outs(%zero : tensor<1x16x2x2xbf16>) -> tensor<1x16x2x2xbf16>
  return %result : tensor<1x16x2x2xbf16>
}

// Two depthwise channels form separate contractions. Runtime activation
// conversions and bias remain Host operations around the CIM transactions.
// CHECK-LABEL: func.func @bf16_depthwise(
// CHECK: arith.truncf
// CHECK-COUNT-2: cim.transaction
// CHECK: arith.extf
// CHECK: arith.addf
// CHECK-NOT: linalg.depthwise_conv_2d_nchw_chw
func.func @bf16_depthwise(%input: tensor<1x2x3x3xf32>)
    -> tensor<1x2x1x1xf32> {
  %input_bf16 = arith.truncf %input : tensor<1x2x3x3xf32> to tensor<1x2x3x3xbf16>
  %weight = arith.constant dense<1.0> : tensor<2x3x3xbf16>
  %z = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<1x2x1x1xbf16>
  %zero = linalg.fill ins(%z : bf16) outs(%empty : tensor<1x2x1x1xbf16>) -> tensor<1x2x1x1xbf16>
  %conv = linalg.depthwise_conv_2d_nchw_chw
      ins(%input_bf16, %weight : tensor<1x2x3x3xbf16>, tensor<2x3x3xbf16>)
      outs(%zero : tensor<1x2x1x1xbf16>) -> tensor<1x2x1x1xbf16>
  %fp32 = arith.extf %conv : tensor<1x2x1x1xbf16> to tensor<1x2x1x1xf32>
  %bias = arith.constant dense<1.0> : tensor<1x2x1x1xf32>
  %result = arith.addf %fp32, %bias : tensor<1x2x1x1xf32>
  return %result : tensor<1x2x1x1xf32>
}
