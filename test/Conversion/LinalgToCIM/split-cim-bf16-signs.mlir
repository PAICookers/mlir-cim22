// RUN: mlir-cim22-opt %s --split-cim-bf16-signs --split-cim-bf16-signs | FileCheck %s
// RUN: mlir-cim22-opt %s --split-cim-bf16-signs --prepare-cim-bf16 --split-cim-bf16-signs | FileCheck %s --check-prefix=PREPARED

// General signed input: four immutable nonnegative contractions. The original
// input and one bias addition remain outside the BF16 branch boundaries.
func.func @signed(%x: tensor<2xf32>) -> (tensor<1xf32>, tensor<2xf32>) {
  %w = arith.constant dense<[[1.0, -0.5]]> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %bias = arith.constant dense<0.25> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  %y = arith.addf %r, %bias : tensor<1xf32>
  return %y, %x : tensor<1xf32>, tensor<2xf32>
}
// CHECK: module attributes {cim.bf16_compensation = "sign-split"}
// CHECK-LABEL: func.func @signed(
// CHECK-SAME: %[[X:.*]]: tensor<2xf32>)
// CHECK: arith.constant dense<{{.*}}> : tensor<1x2xf32>
// CHECK: arith.cmpf oeq, %[[X]],
// CHECK: arith.cmpf olt, %[[X]],
// CHECK: arith.select
// CHECK: arith.negf %[[X]]
// CHECK: arith.select
// CHECK: linalg.matvec
// CHECK: linalg.matvec
// CHECK: arith.subf
// CHECK: linalg.matvec
// CHECK: linalg.matvec
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK: return {{.*}}, %[[X]]
// PREPARED-LABEL: func.func @signed
// PREPARED-COUNT-4: linalg.matvec
// PREPARED-NOT: linalg.matvec
// PREPARED: return

// A nonnegative spike slice and zero-initialized destination prove the
// entire updated activation nonnegative.
func.func @nonnegative(%mask: tensor<1xi1>) -> tensor<1xf32> {
  %empty = tensor.empty() : tensor<1xf32>
  %x = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%mask : tensor<1xi1>) outs(%empty : tensor<1xf32>) {
    ^bb0(%bit: i1, %out: f32):
      %value = arith.uitofp %bit : i1 to f32
      linalg.yield %value : f32
  } -> tensor<1xf32>
  %initial = arith.constant dense<0.0> : tensor<2xf32>
  %updated = tensor.insert_slice %x into %initial[1] [1] [1] : tensor<1xf32> into tensor<2xf32>
  %w = arith.constant dense<[[1.0, -0.5]]> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %updated : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @nonnegative
// CHECK-NOT: arith.cmpf olt
// CHECK-NOT: arith.negf
// CHECK-COUNT-2: linalg.matvec
// CHECK-NOT: linalg.matvec
// CHECK: arith.subf
// CHECK: return
// PREPARED-LABEL: func.func @nonnegative
// PREPARED-COUNT-2: linalg.matvec
// PREPARED-NOT: linalg.matvec
// PREPARED: return

// The source can introduce negative elements even when the destination is zero.
func.func @unknown_slice_source(%source: tensor<1xf32>) -> tensor<1xf32> {
  %initial = arith.constant dense<0.0> : tensor<2xf32>
  %updated = tensor.insert_slice %source into %initial[1] [1] [1] : tensor<1xf32> into tensor<2xf32>
  %w = arith.constant dense<[[1.0, -0.5]]> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %updated : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @unknown_slice_source
// CHECK: arith.cmpf olt
// CHECK-COUNT-4: linalg.matvec
// CHECK-NOT: linalg.matvec
// CHECK: return
// PREPARED-LABEL: func.func @unknown_slice_source
// PREPARED-COUNT-4: linalg.matvec
// PREPARED-NOT: linalg.matvec
// PREPARED: return

// Elements outside the slice retain the destination's original values.
func.func @negative_slice_dest() -> tensor<1xf32> {
  %initial = arith.constant dense<[-1.0, 0.0]> : tensor<2xf32>
  %source = arith.constant dense<1.0> : tensor<1xf32>
  %updated = tensor.insert_slice %source into %initial[1] [1] [1] : tensor<1xf32> into tensor<2xf32>
  %w = arith.constant dense<[[1.0, -0.5]]> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %updated : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @negative_slice_dest
// CHECK: arith.cmpf olt
// CHECK-COUNT-4: linalg.matvec
// CHECK-NOT: linalg.matvec
// CHECK: return
// PREPARED-LABEL: func.func @negative_slice_dest
// PREPARED-COUNT-4: linalg.matvec
// PREPARED-NOT: linalg.matvec
// PREPARED: return

// Explicit BF16 source contractions widen each branch for the Host combine
// and round only the final signed value back to the original result dtype.
func.func @explicit_bf16(%x: tensor<2xbf16>) -> tensor<1xbf16> {
  %w = arith.constant dense<[[1.0, -0.5]]> : tensor<1x2xbf16>
  %z = arith.constant dense<0.0> : tensor<1xbf16>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xbf16>, tensor<2xbf16>) outs(%z : tensor<1xbf16>) -> tensor<1xbf16>
  return %r : tensor<1xbf16>
}
// CHECK-LABEL: func.func @explicit_bf16
// CHECK-COUNT-4: arith.extf {{.*}} : tensor<1xbf16> to tensor<1xf32>
// CHECK: arith.subf {{.*}} : tensor<1xf32>
// CHECK: arith.addf {{.*}} : tensor<1xf32>
// CHECK: arith.truncf {{.*}} : tensor<1xf32> to tensor<1xbf16>
// CHECK: return
// PREPARED-LABEL: func.func @explicit_bf16
// PREPARED-COUNT-4: linalg.matvec
// PREPARED-NOT: linalg.matvec
// PREPARED: return

// No dynamic-weight or accumulator semantic is invented by correction.
func.func @runtime_weight(%w: tensor<1x2xf32>, %x: tensor<2xf32>) -> tensor<1xf32> {
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @runtime_weight
// CHECK-NOT: arith.cmpf
// CHECK: linalg.matvec
// CHECK-NOT: linalg.matvec
// CHECK: return
// PREPARED-LABEL: func.func @runtime_weight
// PREPARED-NOT: arith.truncf
// PREPARED: linalg.matvec
// PREPARED: return

// Keep zero-weight branches: otherwise illegal non-finite activation could be
// erased before the native finite-input validation. Signed zero becomes +0.
func.func @zero_weight(%x: tensor<2xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<-0.0> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @zero_weight(
// CHECK-SAME: %[[ZERO_X:.*]]: tensor<2xf32>)
// CHECK: arith.constant dense<0.000000e+00> : tensor<1x2xf32>
// CHECK: arith.constant dense<0.000000e+00> : tensor<1x2xf32>
// CHECK: %[[NEG:.*]] = arith.cmpf olt, %[[ZERO_X]],
// CHECK: arith.select %[[NEG]], %{{.*}}, %[[ZERO_X]]
// CHECK-COUNT-4: linalg.matvec
// CHECK: return
// PREPARED-LABEL: func.func @zero_weight
// PREPARED-COUNT-4: linalg.matvec
// PREPARED: return

// maxnum ignores NaN, so a NaN constant cannot prove the result nonnegative:
// maxnum(NaN, -1) is -1 and still needs the negative activation branches.
func.func @maxnum_nan(%x: tensor<2xf32>) -> tensor<1xf32> {
  %nan = arith.constant dense<0x7FC00000> : tensor<2xf32>
  %input = arith.maxnumf %nan, %x : tensor<2xf32>
  %w = arith.constant dense<[[1.0, -0.5]]> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %input : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// CHECK-LABEL: func.func @maxnum_nan
// CHECK: %[[MAXNUM:.*]] = arith.maxnumf
// CHECK: arith.cmpf olt, %[[MAXNUM]],
// CHECK: arith.negf %[[MAXNUM]]
// CHECK-COUNT-4: linalg.matvec
// CHECK: return
// PREPARED-LABEL: func.func @maxnum_nan
// PREPARED-COUNT-4: linalg.matvec
// PREPARED: return
