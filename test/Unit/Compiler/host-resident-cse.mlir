// RUN: mlir-cim22-opt %s --cim-host-to-llvm --mlir-print-ir-before=one-shot-bufferize -o %t 2>&1 | FileCheck %s

// Resident helpers can contain repeated tensor views after branch cloning.
// The Host pipeline must CSE those pure views before bufferization.
module {
  func.func private @__cim_resident_linear(%arg0: tensor<4x2xf32>) -> tensor<4xf32> {
    %init = tensor.empty() : tensor<4xf32>
    %first = tensor.extract_slice %arg0[0, 0] [4, 1] [1, 1]
        : tensor<4x2xf32> to tensor<4xf32>
    %second = tensor.extract_slice %arg0[0, 0] [4, 1] [1, 1]
        : tensor<4x2xf32> to tensor<4xf32>
    %out = tensor.insert_slice %first into %init[0] [4] [1]
        : tensor<4xf32> into tensor<4xf32>
    %final = tensor.insert_slice %second into %out[0] [4] [1]
        : tensor<4xf32> into tensor<4xf32>
    return %final : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func private @__cim_resident_linear
// CHECK: tensor.extract_slice
// CHECK-NOT: tensor.extract_slice
// CHECK: return %{{.*}} : tensor<4xf32>
