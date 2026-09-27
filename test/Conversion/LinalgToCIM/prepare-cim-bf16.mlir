// RUN: mlir-cim22-opt %s --prepare-cim-bf16 | FileCheck %s
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=llvm' | FileCheck %s --check-prefix=LLVM

// Convert the contraction only; the original FP32 activation is also returned.
func.func @shared(%x: tensor<2xf32>) -> (tensor<1xf32>, tensor<2xf32>) {
  %w = arith.constant dense<1.0009765625> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r, %x : tensor<1xf32>, tensor<2xf32>
}
// CHECK-LABEL: func.func @shared(
// CHECK-SAME: %[[X:.*]]: tensor<2xf32>)
// CHECK: arith.constant dense<1.000000e+00> : tensor<1x2xbf16>
// CHECK: arith.truncf %[[X]] : tensor<2xf32> to tensor<2xbf16>
// CHECK: linalg.matvec
// CHECK: %[[OUT:.*]] = arith.extf {{.*}} : tensor<1xbf16> to tensor<1xf32>
// CHECK: return %[[OUT]], %[[X]]

func.func @runtime_weight(%w: tensor<1x2xf32>, %x: tensor<2xf32>) -> tensor<1xf32> {
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @runtime_weight
// CHECK-NOT: arith.truncf
// CHECK: linalg.matvec
// CHECK-NOT: arith.extf
// CHECK: return

// Generated names must not collide with preexisting user declarations.
func.func private @__cim_kernel_0()
func.func private @__cim_kernel_1_transaction_0()

// LLVM: module @__cim_host
// LLVM: llvm.func @shared
// LLVM: llvm.call @__cim_kernel_1
// LLVM: llvm.func @conditional
// LLVM: llvm.func @__cim_kernel_1_transaction_1
// LLVM: module @__cim_device
// LLVM: func.func @__cim_kernel_1_transaction_1

func.func @bias_init(%x: tensor<2xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<1.0> : tensor<1x2xf32>
  %z = arith.constant dense<2.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @bias_init
// CHECK-NOT: arith.truncf
// CHECK: linalg.matvec
// CHECK-NOT: arith.extf
// CHECK: return

func.func @negative_zero(%x: tensor<2xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<1.0> : tensor<1x2xf32>
  %z = arith.constant dense<-0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @negative_zero
// CHECK-NOT: arith.truncf
// CHECK: linalg.matvec
// CHECK-NOT: arith.extf
// CHECK: return

func.func @dynamic(%x: tensor<?xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<1.0> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<?xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @dynamic
// CHECK-NOT: arith.truncf
// CHECK: linalg.matvec
// CHECK-NOT: arith.extf
// CHECK: return

func.func @conditional(%condition: i1, %x: tensor<2xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<1.0> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %result = scf.if %condition -> tensor<1xf32> {
    %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
    scf.yield %r : tensor<1xf32>
  } else {
    scf.yield %z : tensor<1xf32>
  }
  return %result : tensor<1xf32>
}
// CHECK-LABEL: func.func @conditional
// CHECK: scf.if
// CHECK-NOT: arith.truncf
// CHECK: linalg.matvec
// CHECK-NOT: arith.extf
// CHECK: return
