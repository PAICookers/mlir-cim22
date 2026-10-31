// RUN: mlir-cim22-opt %s -normalize-cim-conv -normalize-cim-linear -outline-cim-bf16 -mlir-print-op-generic | FileCheck %s --implicit-check-not=cim.int8 --implicit-check-not=cim.vmm --implicit-check-not=func.call

// Rejected candidates retain their original Host contracts.
// CHECK-LABEL: sym_name = "nonzero_init"
// CHECK: "linalg.matmul"
func.func @nonzero_init(%x: tensor<1x3xi8>) -> tensor<1x2xi32> {
  %w = arith.constant dense<1> : tensor<3x2xi8>
  %z = arith.constant dense<7> : tensor<1x2xi32>
  %r = linalg.matmul ins(%x, %w : tensor<1x3xi8>, tensor<3x2xi8>)
      outs(%z : tensor<1x2xi32>) -> tensor<1x2xi32>
  return %r : tensor<1x2xi32>
}

// CHECK-LABEL: sym_name = "wrong_result_width"
// CHECK: "linalg.matmul"
func.func @wrong_result_width(%x: tensor<1x3xi8>) -> tensor<1x2xi16> {
  %w = arith.constant dense<1> : tensor<3x2xi8>
  %z = arith.constant dense<0> : tensor<1x2xi16>
  %r = linalg.matmul ins(%x, %w : tensor<1x3xi8>, tensor<3x2xi8>)
      outs(%z : tensor<1x2xi16>) -> tensor<1x2xi16>
  return %r : tensor<1x2xi16>
}

// CHECK-LABEL: sym_name = "dynamic_weight"
// CHECK: "linalg.matmul"
func.func @dynamic_weight(%x: tensor<1x3xi8>, %w: tensor<3x2xi8>) -> tensor<1x2xi32> {
  %z = arith.constant dense<0> : tensor<1x2xi32>
  %r = linalg.matmul ins(%x, %w : tensor<1x3xi8>, tensor<3x2xi8>)
      outs(%z : tensor<1x2xi32>) -> tensor<1x2xi32>
  return %r : tensor<1x2xi32>
}

// CHECK-LABEL: sym_name = "unsigned"
// CHECK: "linalg.matmul"
// CHECK: "arith.extui"
func.func @unsigned(%x: tensor<1x3xi8>) -> tensor<1x2xi32> {
  %w = arith.constant dense<1> : tensor<3x2xi8>
  %z = arith.constant dense<0> : tensor<1x2xi32>
  %r = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
      ins(%x, %w : tensor<1x3xi8>, tensor<3x2xi8>)
      outs(%z : tensor<1x2xi32>) -> tensor<1x2xi32>
  return %r : tensor<1x2xi32>
}

// CHECK-LABEL: sym_name = "dynamic_shape"
// CHECK: "linalg.matmul"
func.func @dynamic_shape(%x: tensor<?x3xi8>, %z: tensor<?x2xi32>)
    -> tensor<?x2xi32> {
  %w = arith.constant dense<1> : tensor<3x2xi8>
  %r = linalg.matmul ins(%x, %w : tensor<?x3xi8>, tensor<3x2xi8>)
      outs(%z : tensor<?x2xi32>) -> tensor<?x2xi32>
  return %r : tensor<?x2xi32>
}

// A transpose-A layout is not the supported activation-first contract.
// CHECK-LABEL: sym_name = "transpose_a"
// CHECK: "linalg.matmul"
func.func @transpose_a(%x: tensor<3x1xi8>) -> tensor<1x2xi32> {
  %w = arith.constant dense<1> : tensor<3x2xi8>
  %z = arith.constant dense<0> : tensor<1x2xi32>
  %r = linalg.matmul indexing_maps = [
      affine_map<(m, n, k) -> (k, m)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>]
      ins(%x, %w : tensor<3x1xi8>, tensor<3x2xi8>)
      outs(%z : tensor<1x2xi32>) -> tensor<1x2xi32>
  return %r : tensor<1x2xi32>
}

// CHECK-LABEL: sym_name = "reversed_add"
// CHECK: "linalg.matmul"
// CHECK: "arith.addi"
func.func @reversed_add(%x: tensor<1x3xi8>) -> tensor<1x2xi32> {
  %w = arith.constant dense<1> : tensor<3x2xi8>
  %z = arith.constant dense<0> : tensor<1x2xi32>
  %r = "linalg.matmul"(%x, %w, %z) ({
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %v = arith.addi %p, %acc : i32
      linalg.yield %v : i32
  }) {operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>]}
      : (tensor<1x3xi8>, tensor<3x2xi8>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %r : tensor<1x2xi32>
}

// CHECK-LABEL: sym_name = "overflow_flags"
// CHECK: "linalg.matmul"
// CHECK: "arith.addi"
// CHECK-SAME: overflowFlags = #arith.overflow<nsw>
func.func @overflow_flags(%x: tensor<1x3xi8>) -> tensor<1x2xi32> {
  %w = arith.constant dense<1> : tensor<3x2xi8>
  %z = arith.constant dense<0> : tensor<1x2xi32>
  %r = "linalg.matmul"(%x, %w, %z) ({
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %v = arith.addi %acc, %p overflow<nsw> : i32
      linalg.yield %v : i32
  }) {operandSegmentSizes = array<i32: 2, 1>, indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>]}
      : (tensor<1x3xi8>, tensor<3x2xi8>, tensor<1x2xi32>) -> tensor<1x2xi32>
  return %r : tensor<1x2xi32>
}

// Unmarked Conv must retain an altered body rather than canonicalizing it.
// CHECK-LABEL: sym_name = "unsigned_conv"
// CHECK: "linalg.conv_2d_nchw_fchw"
// CHECK: "arith.extui"
func.func @unsigned_conv(%x: tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xi32> {
  %w = arith.constant dense<1> : tensor<1x1x1x1xi8>
  %z = arith.constant dense<0> : tensor<1x1x1x1xi32>
  %r = "linalg.conv_2d_nchw_fchw"(%x, %w, %z) ({
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extui %a : i8 to i32
      %be = arith.extui %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %v = arith.addi %acc, %p : i32
      linalg.yield %v : i32
  }) {dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<1x1x1x1xi8>, tensor<1x1x1x1xi8>, tensor<1x1x1x1xi32>)
      -> tensor<1x1x1x1xi32>
  return %r : tensor<1x1x1x1xi32>
}
