// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=packet max-kernel-columns=1' -o %t.packet
// RUN: %python %S/../../python/CIM22/verify_native_runtime.py %t.packet %cim22_runtime

// Supplier model: the helper uses small normal inputs so raw-exponent BF16
// reconstruction remains finite. Two columns exercise streaming, K=65 exercises
// Host reduction plus bias, and N=17 exercises lane tails.
// The private helper also checks readonly ownership across function calls.
func.func private @preprocess(%x: tensor<2x65xf32>) -> tensor<2x65xf32> {
  %doubled = arith.addf %x, %x : tensor<2x65xf32>
  return %doubled : tensor<2x65xf32>
}
func.func @main(%x: tensor<2x65xf32>) -> tensor<2x17xf32> {
  %input = func.call @preprocess(%x) : (tensor<2x65xf32>) -> tensor<2x65xf32>
  %w = arith.constant dense<1.0> : tensor<65x17xf32>
  %z = arith.constant dense<0.0> : tensor<2x17xf32>
  %r = linalg.matmul ins(%input, %w : tensor<2x65xf32>, tensor<65x17xf32>) outs(%z : tensor<2x17xf32>) -> tensor<2x17xf32>
  %one = arith.constant dense<1.0> : tensor<2x17xf32>
  %sum = arith.addf %r, %one : tensor<2x17xf32>
  return %sum : tensor<2x17xf32>
}
// REQUIRES: python312
