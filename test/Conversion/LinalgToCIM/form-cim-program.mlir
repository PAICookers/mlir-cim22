// RUN: split-file %s %t
// RUN: mlir-cim22-opt %t/canonical.mlir -partition-cim-program -form-cim-program > %t/once.mlir
// RUN: mlir-cim22-opt %t/once.mlir -partition-cim-program -form-cim-program > %t/twice.mlir
// RUN: diff %t/once.mlir %t/twice.mlir
// RUN: FileCheck %s --check-prefix=CHECK < %t/once.mlir
// RUN: mlir-cim22-opt %t/noncanonical.mlir -partition-cim-program -form-cim-program --mlir-print-op-generic --mlir-print-local-scope | FileCheck %s --check-prefix=GENERIC

//--- canonical.mlir

// CHECK-LABEL: func.func @exact_candidate(
// CHECK-SAME: %[[WEIGHT:.*]]: tensor<16x64xi8>, %[[INPUT:.*]]: tensor<64xi8>
// CHECK: %[[RESULT:.*]] = cim.vmm %[[INPUT]], %[[WEIGHT]] {cim.transaction_idx = 0 : i64{{.*}}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK-NOT: linalg.matvec
// CHECK: return %[[RESULT]] : tensor<16xi21>
func.func @exact_candidate(%weight: tensor<16x64xi8>, %input: tensor<64xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// CHECK-LABEL: func.func @multiple_candidates(
// CHECK-SAME: %[[WEIGHT0:.*]]: tensor<16x64xi8>, %[[INPUT0:.*]]: tensor<64xi8>, %[[WEIGHT1:.*]]: tensor<16x64xi8>, %[[INPUT1:.*]]: tensor<64xi8>
// CHECK: %[[RESULT0:.*]] = cim.vmm %[[INPUT0]], %[[WEIGHT0]] {cim.transaction_idx = 0 : i64{{.*}}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK-NEXT: %[[RESULT1:.*]] = cim.vmm %[[INPUT1]], %[[WEIGHT1]] {cim.transaction_idx = 1 : i64{{.*}}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK-NOT: linalg.matvec
// CHECK: return %[[RESULT0]], %[[RESULT1]] : tensor<16xi21>, tensor<16xi21>
func.func @multiple_candidates(
    %weight0: tensor<16x64xi8>, %input0: tensor<64xi8>,
    %weight1: tensor<16x64xi8>, %input1: tensor<64xi8>)
    -> (tensor<16xi21>, tensor<16xi21>) {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result0 = linalg.matvec
      ins(%weight0, %input0 : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  %result1 = linalg.matvec
      ins(%weight1, %input1 : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result0, %result1 : tensor<16xi21>, tensor<16xi21>
}

// Keep the supported and unsupported operations in one function so checks
// cannot accidentally attribute the surviving matvec to the wrong candidate.
// CHECK-LABEL: func.func @mixed_initializers(
// CHECK-SAME: %[[WEIGHT:.*]]: tensor<16x64xi8>, %[[INPUT:.*]]: tensor<64xi8>
// CHECK: %[[ONE:.*]] = arith.constant dense<1> : tensor<16xi21>
// CHECK: %[[CONVERTED:.*]] = cim.vmm %[[INPUT]], %[[WEIGHT]] {cim.transaction_idx = 0 : i64{{.*}}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK-NEXT: %[[KEPT:.*]] = linalg.matvec ins(%[[WEIGHT]], %[[INPUT]] : tensor<16x64xi8>, tensor<64xi8>) outs(%[[ONE]] : tensor<16xi21>) -> tensor<16xi21>
// CHECK: return %[[CONVERTED]], %[[KEPT]] : tensor<16xi21>, tensor<16xi21>
func.func @mixed_initializers(%weight: tensor<16x64xi8>,
                              %input: tensor<64xi8>)
    -> (tensor<16xi21>, tensor<16xi21>) {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %one = arith.constant dense<1> : tensor<16xi21>
  %converted = linalg.matvec
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  %kept = linalg.matvec
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%one : tensor<16xi21>) -> tensor<16xi21>
  return %converted, %kept : tensor<16xi21>, tensor<16xi21>
}

// CHECK-LABEL: func.func @wrong_input_type(
// CHECK: %[[RESULT:.*]] = linalg.matvec
// CHECK-NOT: cim.vmm
// CHECK: return %[[RESULT]] : tensor<16xi21>
func.func @wrong_input_type(%weight: tensor<16x64xi16>,
                            %input: tensor<64xi16>) -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x64xi16>, tensor<64xi16>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// CHECK-LABEL: func.func @wrong_accumulator_type(
// CHECK: %[[RESULT:.*]] = linalg.matvec
// CHECK-NOT: cim.vmm
// CHECK: return %[[RESULT]] : tensor<16xi32>
func.func @wrong_accumulator_type(%weight: tensor<16x64xi8>,
                                  %input: tensor<64xi8>) -> tensor<16xi32> {
  %zero = arith.constant dense<0> : tensor<16xi32>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}

// CHECK-LABEL: func.func @dynamic_weight_shape(
// CHECK: %[[RESULT:.*]] = linalg.matvec
// CHECK-NOT: cim.vmm
// CHECK: return %[[RESULT]] : tensor<16xi21>
func.func @dynamic_weight_shape(%weight: tensor<?x64xi8>,
                                %input: tensor<64xi8>) -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<?x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// CHECK-LABEL: func.func @buffer_semantics(
// CHECK: linalg.matvec
// CHECK-NOT: cim.vmm
// CHECK: return
func.func @buffer_semantics(%weight: memref<16x64xi8>,
                            %input: memref<64xi8>,
                            %output: memref<16xi21>) {
  linalg.matvec
      ins(%weight, %input : memref<16x64xi8>, memref<64xi8>)
      outs(%output : memref<16xi21>)
  return
}

#weight = affine_map<(n, k) -> (n, k)>
#input = affine_map<(n, k) -> (k)>
#output = affine_map<(n, k) -> (n)>

// CHECK-LABEL: func.func @generic_matvec(
// CHECK: %[[RESULT:.*]] = linalg.generic
// CHECK-NOT: cim.vmm
// CHECK: return %[[RESULT]] : tensor<16xi21>
func.func @generic_matvec(%weight: tensor<16x64xi8>,
                          %input: tensor<64xi8>) -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.generic {
      indexing_maps = [#weight, #input, #output],
      iterator_types = ["parallel", "reduction"]}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) {
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput : i21
      %sum = arith.addi %accumulator, %product : i21
      linalg.yield %sum : i21
  } -> tensor<16xi21>
  return %result : tensor<16xi21>
}

//--- noncanonical.mlir

// Generic syntax can construct a verified MatvecOp with unsigned extension.
// It has the candidate shape and zero init, but not the signed VMM semantics.
func.func @noncanonical_unsigned_region(%weight: tensor<16x64xi8>,
                                        %input: tensor<64xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = "linalg.matvec"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extui %weightElement : i8 to i21
      %extendedInput = arith.extui %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput : i21
      %sum = arith.addi %accumulator, %product : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<16x64xi8>, tensor<64xi8>, tensor<16xi21>)
      -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// Generic syntax can also override the memoized indexing maps while retaining
// the candidate types, zero init, and canonical scalar body.
func.func @noncanonical_indexing_maps(%weight: tensor<16x64xi8>,
                                      %input: tensor<64xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = "linalg.matvec"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput : i21
      %sum = arith.addi %accumulator, %product : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {
    operandSegmentSizes = array<i32: 2, 1>,
    linalg.memoized_indexing_maps = [
      affine_map<(d0, d1) -> (d1, d0)>,
      affine_map<(d0, d1) -> (d0)>,
      affine_map<(d0, d1) -> (d1)>
    ]
  } : (tensor<16x64xi8>, tensor<64xi8>, tensor<16xi21>)
      -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// Non-default overflow flags change the scalar operation contract and must
// not be discarded even when types, maps, and data flow are otherwise exact.
func.func @noncanonical_overflow_flags(%weight: tensor<16x64xi8>,
                                       %input: tensor<64xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = "linalg.matvec"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput overflow<nsw> : i21
      %sum = arith.addi %accumulator, %product : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<16x64xi8>, tensor<64xi8>, tensor<16xi21>)
      -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// Test the accumulator flag independently: multiply keeps its default flags,
// while only the reduction add carries a no-signed-wrap promise.
func.func @noncanonical_add_overflow_flags(%weight: tensor<16x64xi8>,
                                           %input: tensor<64xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = "linalg.matvec"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput : i21
      %sum = arith.addi %accumulator, %product overflow<nsw> : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<16x64xi8>, tensor<64xi8>, tensor<16xi21>)
      -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// GENERIC-LABEL: sym_name = "noncanonical_unsigned_region"
// GENERIC: "arith.extui"
// GENERIC: "arith.extui"
// GENERIC-NOT: "cim.vmm"
// GENERIC-LABEL: sym_name = "noncanonical_indexing_maps"
// GENERIC: linalg.memoized_indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>]
// GENERIC-NOT: "cim.vmm"
// GENERIC-LABEL: sym_name = "noncanonical_overflow_flags"
// GENERIC: "arith.muli"
// GENERIC-SAME: overflowFlags = #arith.overflow<nsw>
// GENERIC-NOT: "cim.vmm"
// GENERIC-LABEL: sym_name = "noncanonical_add_overflow_flags"
// GENERIC: "arith.muli"
// GENERIC-SAME: overflowFlags = #arith.overflow<none>
// GENERIC: "arith.addi"
// GENERIC-SAME: overflowFlags = #arith.overflow<nsw>
// GENERIC-NOT: "cim.vmm"
