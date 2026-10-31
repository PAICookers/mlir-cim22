// RUN: mlir-cim22-opt %s --normalize-cim-linear | FileCheck %s

// Only exact signed-i8, zero-point-zero contractions may be narrowed.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0)>

// CHECK-LABEL: func.func @symmetric(
// CHECK: linalg.matmul
// CHECK-SAME: cim.int8
  func.func @symmetric(%arg0: tensor<1x64xi8>) -> tensor<1x16xi32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<1> : tensor<64x16xi32>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tensor.empty() : tensor<1x64xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x64xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i8, %out: i32):
      %8 = arith.extsi %in : i8 to i32
      linalg.yield %8 : i32
    } -> tensor<1x64xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_0 : tensor<1x64xi32>, tensor<1xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i32, %in_1: i8, %out: i32):
      %8 = arith.extsi %in_1 : i8 to i32
      %9 = arith.subi %in, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<1x64xi32>
    %3 = tensor.empty() : tensor<1x16xi64>
    %4 = linalg.fill ins(%c0_i64 : i64) outs(%3 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %5 = linalg.matmul ins(%2, %cst : tensor<1x64xi32>, tensor<64x16xi32>) outs(%4 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %6 = tensor.empty() : tensor<1x16xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x16xi64>) outs(%6 : tensor<1x16xi32>) {
    ^bb0(%in: i64, %out: i32):
      %8 = arith.trunci %in : i64 to i32
      linalg.yield %8 : i32
    } -> tensor<1x16xi32>
    return %7 : tensor<1x16xi32>
  }

// CHECK-LABEL: func.func @nonzero_point(
// CHECK-NOT: cim.int8
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x64xi32>, tensor<64x16xi32>)
// CHECK-NOT: cim.int8
// CHECK: return
  func.func @nonzero_point(%arg0: tensor<1x64xi8>) -> tensor<1x16xi32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<1> : tensor<64x16xi32>
    %cst_0 = arith.constant dense<1> : tensor<1xi8>
    %0 = tensor.empty() : tensor<1x64xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x64xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i8, %out: i32):
      %8 = arith.extsi %in : i8 to i32
      linalg.yield %8 : i32
    } -> tensor<1x64xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_0 : tensor<1x64xi32>, tensor<1xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i32, %in_1: i8, %out: i32):
      %8 = arith.extsi %in_1 : i8 to i32
      %9 = arith.subi %in, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<1x64xi32>
    %3 = tensor.empty() : tensor<1x16xi64>
    %4 = linalg.fill ins(%c0_i64 : i64) outs(%3 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %5 = linalg.matmul ins(%2, %cst : tensor<1x64xi32>, tensor<64x16xi32>) outs(%4 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %6 = tensor.empty() : tensor<1x16xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x16xi64>) outs(%6 : tensor<1x16xi32>) {
    ^bb0(%in: i64, %out: i32):
      %8 = arith.trunci %in : i64 to i32
      linalg.yield %8 : i32
    } -> tensor<1x16xi32>
    return %7 : tensor<1x16xi32>
  }

// CHECK-LABEL: func.func @unsigned_extension(
// CHECK-NOT: cim.int8
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x64xi32>, tensor<64x16xi32>)
// CHECK-NOT: cim.int8
// CHECK: return
  func.func @unsigned_extension(%arg0: tensor<1x64xi8>) -> tensor<1x16xi32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<1> : tensor<64x16xi32>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tensor.empty() : tensor<1x64xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x64xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i8, %out: i32):
      %8 = arith.extui %in : i8 to i32
      linalg.yield %8 : i32
    } -> tensor<1x64xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_0 : tensor<1x64xi32>, tensor<1xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i32, %in_1: i8, %out: i32):
      %8 = arith.extui %in_1 : i8 to i32
      %9 = arith.subi %in, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<1x64xi32>
    %3 = tensor.empty() : tensor<1x16xi64>
    %4 = linalg.fill ins(%c0_i64 : i64) outs(%3 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %5 = linalg.matmul ins(%2, %cst : tensor<1x64xi32>, tensor<64x16xi32>) outs(%4 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %6 = tensor.empty() : tensor<1x16xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x16xi64>) outs(%6 : tensor<1x16xi32>) {
    ^bb0(%in: i64, %out: i32):
      %8 = arith.trunci %in : i64 to i32
      linalg.yield %8 : i32
    } -> tensor<1x16xi32>
    return %7 : tensor<1x16xi32>
  }

// CHECK-LABEL: func.func @wide_weight(
// CHECK-NOT: cim.int8
// CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<1x64xi32>, tensor<64x16xi32>)
// CHECK-NOT: cim.int8
// CHECK: return
  func.func @wide_weight(%arg0: tensor<1x64xi8>) -> tensor<1x16xi32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<128> : tensor<64x16xi32>
    %cst_0 = arith.constant dense<0> : tensor<1xi8>
    %0 = tensor.empty() : tensor<1x64xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<1x64xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i8, %out: i32):
      %8 = arith.extsi %in : i8 to i32
      linalg.yield %8 : i32
    } -> tensor<1x64xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %cst_0 : tensor<1x64xi32>, tensor<1xi8>) outs(%0 : tensor<1x64xi32>) {
    ^bb0(%in: i32, %in_1: i8, %out: i32):
      %8 = arith.extsi %in_1 : i8 to i32
      %9 = arith.subi %in, %8 : i32
      linalg.yield %9 : i32
    } -> tensor<1x64xi32>
    %3 = tensor.empty() : tensor<1x16xi64>
    %4 = linalg.fill ins(%c0_i64 : i64) outs(%3 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %5 = linalg.matmul ins(%2, %cst : tensor<1x64xi32>, tensor<64x16xi32>) outs(%4 : tensor<1x16xi64>) -> tensor<1x16xi64>
    %6 = tensor.empty() : tensor<1x16xi32>
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<1x16xi64>) outs(%6 : tensor<1x16xi32>) {
    ^bb0(%in: i64, %out: i32):
      %8 = arith.trunci %in : i64 to i32
      linalg.yield %8 : i32
    } -> tensor<1x16xi32>
    return %7 : tensor<1x16xi32>
  }
