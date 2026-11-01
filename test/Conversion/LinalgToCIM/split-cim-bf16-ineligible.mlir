// RUN: mlir-cim22-opt %s --split-cim-bf16-signs | FileCheck %s --implicit-check-not=cim.bf16_compensation --implicit-check-not=cim.bf16_branch
// RUN: mlir-cim22-opt %s --split-cim-bf16-signs='bf16-compensation=legacy' | FileCheck %s --implicit-check-not=cim.bf16_compensation --implicit-check-not=cim.bf16_branch

// Pure INT8/Host modules do not acquire BF16 numerical semantics.
func.func @integer(%x: tensor<2xi8>) -> tensor<1xi32> {
  %w = arith.constant dense<-1> : tensor<1x2xi8>
  %z = arith.constant dense<0> : tensor<1xi32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xi8>, tensor<2xi8>) outs(%z : tensor<1xi32>) -> tensor<1xi32>
  return %r : tensor<1xi32>
}
// CHECK-LABEL: func.func @integer
// CHECK: linalg.matvec
// CHECK-NOT: linalg.matvec
// CHECK: return
func.func @bias_init(%x: tensor<2xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<-1.0> : tensor<1x2xf32>
  %bias = arith.constant dense<0.25> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%bias : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @bias_init
// CHECK: linalg.matvec
// CHECK-NOT: linalg.matvec
// CHECK: return
func.func @host(%x: tensor<2xf32>) -> tensor<2xf32> {
  %r = arith.addf %x, %x : tensor<2xf32>
  return %r : tensor<2xf32>
}
// CHECK-LABEL: func.func @host
// CHECK: arith.addf
// CHECK: return
