// RUN: mlir-cim22-opt %s -normalize-cim-conv -normalize-cim-linear -canonicalize | FileCheck %s --check-prefix=NORMAL --implicit-check-not=cim.onnx
// RUN: mlir-cim22-opt %s -normalize-cim-conv -normalize-cim-linear -canonicalize -outline-cim-bf16='max-kernel-columns=1' | FileCheck %s --check-prefix=OUTLINE
// RUN: mlir-cim22-opt %s -normalize-cim-conv -normalize-cim-linear -canonicalize -partition-cim-program -form-cim-program | FileCheck %s --check-prefix=FORM --implicit-check-not=cim.onnx

// Source types, rather than value ranges or importer markers, select INT8.
func.func @linear(%x: tensor<2x65xi8>) -> tensor<2x17xi32> {
  %w = arith.constant dense<-3> : tensor<65x17xi8>
  %z = arith.constant dense<0> : tensor<2x17xi32>
  %r = linalg.matmul ins(%x, %w : tensor<2x65xi8>, tensor<65x17xi8>)
      outs(%z : tensor<2x17xi32>) -> tensor<2x17xi32>
  return %r : tensor<2x17xi32>
}
// NORMAL-LABEL: func.func @linear
// NORMAL: arith.constant dense<-3> : tensor<17x65xi8>
// NORMAL: linalg.matmul {cim.int8}
// NORMAL: linalg.transpose
// FORM-LABEL: func.func @linear
// FORM: cim.vmm {{.*}} -> tensor<16xi21>
// FORM: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// FORM: tensor.pad {{.*}} high[63]
// FORM: arith.addi {{.*}} : tensor<16xi32>
// FORM: tensor.concat dim(0)
// FORM: tensor.extract_slice {{.*}} : tensor<32xi32> to tensor<17xi32>
// FORM: return {{.*}} : tensor<2x17xi32>

func.func @transpose_b(%x: tensor<1x3xi8>) -> tensor<1x2xi32> {
  %w = arith.constant dense<[[1, -2, 3], [-4, 5, -6]]> : tensor<2x3xi8>
  %z = arith.constant dense<0> : tensor<1x2xi32>
  %r = linalg.matmul indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (n, k)>,
      affine_map<(m, n, k) -> (m, n)>]
      ins(%x, %w : tensor<1x3xi8>, tensor<2x3xi8>)
      outs(%z : tensor<1x2xi32>) -> tensor<1x2xi32>
  return %r : tensor<1x2xi32>
}
// NORMAL-LABEL: func.func @transpose_b
// NORMAL: arith.constant dense<{{.*}}1, -2, 3{{.*}}-4, 5, -6{{.*}}> : tensor<2x3xi8>
// NORMAL: linalg.matmul {cim.int8}
// FORM-LABEL: func.func @transpose_b
// FORM: cim.vmm {{.*}} -> tensor<16xi21>
// FORM: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// FORM: return {{.*}} : tensor<1x2xi32>

// K=72, N=17 and two patches exercise both tails and Host K reduction.
func.func @conv(%x: tensor<1x8x3x4xi8>) -> tensor<1x17x1x2xi32> {
  %w = arith.constant dense<2> : tensor<17x8x3x3xi8>
  %z = arith.constant dense<0> : tensor<1x17x1x2xi32>
  %r = linalg.conv_2d_nchw_fchw
      ins(%x, %w : tensor<1x8x3x4xi8>, tensor<17x8x3x3xi8>)
      outs(%z : tensor<1x17x1x2xi32>) -> tensor<1x17x1x2xi32>
  return %r : tensor<1x17x1x2xi32>
}
// NORMAL-LABEL: func.func @conv
// NORMAL: linalg.generic
// NORMAL: linalg.matmul {cim.int8}
// NORMAL: return {{.*}} : tensor<1x17x1x2xi32>
// FORM-LABEL: func.func @conv
// FORM: cim.vmm {{.*}} -> tensor<16xi21>
// FORM: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// FORM: arith.addi {{.*}} : tensor<16xi32>
// FORM: return {{.*}} : tensor<1x17x1x2xi32>

// Worst positive partial is 63*128*128=1032192, within signed21.
func.func @safe_k63(%x: tensor<1x63xi8>) -> tensor<1x1xi32> {
  %w = arith.constant dense<-128> : tensor<63x1xi8>
  %z = arith.constant dense<0> : tensor<1x1xi32>
  %r = linalg.matmul ins(%x, %w : tensor<1x63xi8>, tensor<63x1xi8>)
      outs(%z : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %r : tensor<1x1xi32>
}
// NORMAL-LABEL: func.func @safe_k63
// NORMAL: linalg.matmul {cim.int8}
// FORM-LABEL: func.func @safe_k63
// FORM: cim.vmm {{.*}} -> tensor<16xi21>
// FORM: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// FORM: return {{.*}} : tensor<1x1xi32>

// K64 may produce 1048576, one above signed21; preserve the Host operation.
func.func @unsafe_k64(%x: tensor<1x64xi8>) -> tensor<1x1xi32> {
  %w = arith.constant dense<-128> : tensor<64x1xi8>
  %z = arith.constant dense<0> : tensor<1x1xi32>
  %r = linalg.matmul ins(%x, %w : tensor<1x64xi8>, tensor<64x1xi8>)
      outs(%z : tensor<1x1xi32>) -> tensor<1x1xi32>
  return %r : tensor<1x1xi32>
}
// NORMAL-LABEL: func.func @unsafe_k64
// NORMAL-NOT: cim.int8
// NORMAL: linalg.matmul
// FORM-LABEL: func.func @unsafe_k64
// FORM-NOT: cim.vmm
// FORM: linalg.matmul
// FORM-NOT: cim.vmm
// FORM: return

// Outlining keeps the streamed columns and final output assembly on Host.
// OUTLINE: module @__cim_host
// OUTLINE: func.func @linear
// OUTLINE: scf.for
// OUTLINE: call @__cim_kernel_0
// OUTLINE: func.func @unsafe_k64
// OUTLINE: linalg.matmul
// OUTLINE: module @__cim_device
// OUTLINE: func.func @__cim_kernel_0
// OUTLINE-SAME: tensor<65x1xi8>) -> tensor<17x1xi32>
// OUTLINE: linalg.matmul {cim.int8}
// OUTLINE: func.func @__cim_kernel_1
// OUTLINE: func.func @__cim_kernel_2
// OUTLINE: func.func @__cim_kernel_3
// OUTLINE-NOT: func.func @__cim_kernel_4
