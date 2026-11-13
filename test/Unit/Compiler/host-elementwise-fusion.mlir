// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=llvm' | FileCheck %s
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=llvm' --mlir-print-ir-after=linalg-fuse-elementwise-ops -o %t 2>&1 | FileCheck %s --check-prefix=FUSED
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=llvm' --mlir-print-ir-after=linalg-fold-unit-extent-dims -o %t 2>&1 | FileCheck %s --check-prefix=FOLDED

// Unit iterators are folded before fusing Host elementwise chains. The public
// shape and the scalar operation order must survive lowering to LLVM.
module {
  func.func @main(%arg0: tensor<1x4x1xf32>) -> tensor<1x4x1xf32> {
    %init = tensor.empty() : tensor<1x4x1xf32>
    %one = arith.constant 1.0 : f32
    %two = arith.constant 2.0 : f32
    %a = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<1x4x1xf32>) outs(%init : tensor<1x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %x = arith.addf %in, %one : f32
      linalg.yield %x : f32
    } -> tensor<1x4x1xf32>
    %b = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a : tensor<1x4x1xf32>) outs(%init : tensor<1x4x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %x = arith.mulf %in, %two : f32
      linalg.yield %x : f32
    } -> tensor<1x4x1xf32>
    return %b : tensor<1x4x1xf32>
  }
}

// CHECK-NOT: linalg.generic
// CHECK: llvm.func @main
// CHECK-NOT: linalg.generic

// FUSED-LABEL: func.func @main
// FUSED: linalg.generic
// FUSED: %[[SUM:.*]] = arith.addf
// FUSED-NEXT: %[[PRODUCT:.*]] = arith.mulf %[[SUM]],
// FUSED-NEXT: linalg.yield %[[PRODUCT]]
// FUSED-NOT: linalg.generic
// FUSED: return

// FOLDED-LABEL: func.func @main
// FOLDED-SAME: tensor<1x4x1xf32>
// FOLDED: tensor.collapse_shape
// FOLDED-SAME: into tensor<4xf32>
// FOLDED: linalg.generic
// FOLDED-SAME: iterator_types = ["parallel"]
// FOLDED: tensor.expand_shape
// FOLDED-SAME: into tensor<1x4x1xf32>
// FOLDED: return
