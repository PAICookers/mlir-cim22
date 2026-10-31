// RUN: mlir-cim22-opt %s -normalize-cim-conv -split-input-file -verify-diagnostics

// A trusted importer marker does not authorize replacing a modified region.
func.func @unsigned(%x: tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xi32> {
  %w = arith.constant dense<1> : tensor<1x1x1x1xi8>
  %z = arith.constant dense<0> : tensor<1x1x1x1xi32>
  // expected-error@+1 {{invalid INT8 Conv: does not match the frozen integer Conv profile}}
  %r = "linalg.conv_2d_nchw_fchw"(%x, %w, %z) ({
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extui %a : i8 to i32
      %be = arith.extui %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %v = arith.addi %acc, %p : i32
      linalg.yield %v : i32
  }) {cim.onnx.conv_integer, dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<1x1x1x1xi8>, tensor<1x1x1x1xi8>, tensor<1x1x1x1xi32>)
      -> tensor<1x1x1x1xi32>
  return %r : tensor<1x1x1x1xi32>
}

// -----

func.func @subtraction(%x: tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xi32> {
  %w = arith.constant dense<1> : tensor<1x1x1x1xi8>
  %z = arith.constant dense<0> : tensor<1x1x1x1xi32>
  // expected-error@+1 {{invalid INT8 Conv: does not match the frozen integer Conv profile}}
  %r = "linalg.conv_2d_nchw_fchw"(%x, %w, %z) ({
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %v = arith.subi %acc, %p : i32
      linalg.yield %v : i32
  }) {cim.onnx.conv_integer, dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<1x1x1x1xi8>, tensor<1x1x1x1xi8>, tensor<1x1x1x1xi32>)
      -> tensor<1x1x1x1xi32>
  return %r : tensor<1x1x1x1xi32>
}

// -----

func.func @poison_overflow(%x: tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xi32> {
  %w = arith.constant dense<1> : tensor<1x1x1x1xi8>
  %z = arith.constant dense<0> : tensor<1x1x1x1xi32>
  // expected-error@+1 {{invalid INT8 Conv: does not match the frozen integer Conv profile}}
  %r = "linalg.conv_2d_nchw_fchw"(%x, %w, %z) ({
    ^bb0(%a: i8, %b: i8, %acc: i32):
      %ae = arith.extsi %a : i8 to i32
      %be = arith.extsi %b : i8 to i32
      %p = arith.muli %ae, %be : i32
      %v = arith.addi %acc, %p overflow<nsw> : i32
      linalg.yield %v : i32
  }) {cim.onnx.conv_integer, dilations = dense<1> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>, operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<1x1x1x1xi8>, tensor<1x1x1x1xi8>, tensor<1x1x1x1xi32>)
      -> tensor<1x1x1x1xi32>
  return %r : tensor<1x1x1x1xi32>
}
