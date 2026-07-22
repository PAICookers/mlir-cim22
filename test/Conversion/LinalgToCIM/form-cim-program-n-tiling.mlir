// RUN: split-file %s %t
// RUN: mlir-cim22-opt %t/positive.mlir -form-cim-program > %t/once.mlir
// RUN: mlir-cim22-opt %t/once.mlir -form-cim-program > %t/twice.mlir
// RUN: diff %t/once.mlir %t/twice.mlir
// RUN: FileCheck %s --check-prefix=POS < %t/once.mlir
// RUN: mlir-cim22-opt %t/negative.mlir -form-cim-program --mlir-print-op-generic --mlir-print-local-scope | FileCheck %s --check-prefix=NEG

//--- positive.mlir

// POS-LABEL: func.func @n32_two_tiles(
// POS-SAME: %[[W32:.*]]: tensor<32x64xi8>, %[[I32:.*]]: tensor<64xi8>
// POS: %[[S320:.*]] = tensor.extract_slice %[[W32]][0, 0] [16, 64] [1, 1] : tensor<32x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V320:.*]] = cim.vmm %[[I32]], %[[S320]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S321:.*]] = tensor.extract_slice %[[W32]][16, 0] [16, 64] [1, 1] : tensor<32x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V321:.*]] = cim.vmm %[[I32]], %[[S321]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[C32:.*]] = tensor.concat dim(0) %[[V320]], %[[V321]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NOT: tensor.concat
// POS-NOT: linalg.matvec
// POS: return %[[C32]] : tensor<32xi21>
func.func @n32_two_tiles(%weight: tensor<32x64xi8>, %input: tensor<64xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<32x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<32xi21>) -> tensor<32xi21>
  return %result : tensor<32xi21>
}

// POS-LABEL: func.func @n48_three_tiles(
// POS-SAME: %[[W48:.*]]: tensor<48x64xi8>, %[[I48:.*]]: tensor<64xi8>
// POS: %[[S480:.*]] = tensor.extract_slice %[[W48]][0, 0] [16, 64] [1, 1] : tensor<48x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V480:.*]] = cim.vmm %[[I48]], %[[S480]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S481:.*]] = tensor.extract_slice %[[W48]][16, 0] [16, 64] [1, 1] : tensor<48x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V481:.*]] = cim.vmm %[[I48]], %[[S481]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S482:.*]] = tensor.extract_slice %[[W48]][32, 0] [16, 64] [1, 1] : tensor<48x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V482:.*]] = cim.vmm %[[I48]], %[[S482]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[C48:.*]] = tensor.concat dim(0) %[[V480]], %[[V481]], %[[V482]] : (tensor<16xi21>, tensor<16xi21>, tensor<16xi21>) -> tensor<48xi21>
// POS-NOT: tensor.concat
// POS-NOT: linalg.matvec
// POS: return %[[C48]] : tensor<48xi21>
func.func @n48_three_tiles(%weight: tensor<48x64xi8>, %input: tensor<64xi8>)
    -> tensor<48xi21> {
  %zero = arith.constant dense<0> : tensor<48xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<48x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<48xi21>) -> tensor<48xi21>
  return %result : tensor<48xi21>
}

// N=16 must retain the original one-VMM behavior: no slice and no concat.
// POS-LABEL: func.func @n16_original(
// POS-SAME: %[[W16:.*]]: tensor<16x64xi8>, %[[I16:.*]]: tensor<64xi8>
// POS-NOT: tensor.extract_slice
// POS: %[[V16:.*]] = cim.vmm %[[I16]], %[[W16]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: tensor.concat
// POS-NOT: linalg.matvec
// POS: return %[[V16]] : tensor<16xi21>
func.func @n16_original(%weight: tensor<16x64xi8>, %input: tensor<64xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// POS-LABEL: func.func @mixed_n_and_result_user(
// POS: %[[V16M:.*]] = cim.vmm %{{.*}}, %{{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S0M:.*]] = tensor.extract_slice %{{.*}}[0, 0] [16, 64] [1, 1] : tensor<32x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V0M:.*]] = cim.vmm %{{.*}}, %[[S0M]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S1M:.*]] = tensor.extract_slice %{{.*}}[16, 0] [16, 64] [1, 1] : tensor<32x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V1M:.*]] = cim.vmm %{{.*}}, %[[S1M]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[CM:.*]] = tensor.concat dim(0) %[[V0M]], %[[V1M]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS: %[[ELEM:.*]] = tensor.extract %[[CM]][%{{.*}}] : tensor<32xi21>
// POS-NOT: linalg.matvec
// POS: return %[[V16M]], %[[ELEM]] : tensor<16xi21>, i21
func.func @mixed_n_and_result_user(
    %weight16: tensor<16x64xi8>, %input16: tensor<64xi8>,
    %weight32: tensor<32x64xi8>, %input32: tensor<64xi8>)
    -> (tensor<16xi21>, i21) {
  %zero16 = arith.constant dense<0> : tensor<16xi21>
  %zero32 = arith.constant dense<0> : tensor<32xi21>
  %index = arith.constant 17 : index
  %result16 = linalg.matvec
      ins(%weight16, %input16 : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero16 : tensor<16xi21>) -> tensor<16xi21>
  %result32 = linalg.matvec
      ins(%weight32, %input32 : tensor<32x64xi8>, tensor<64xi8>)
      outs(%zero32 : tensor<32xi21>) -> tensor<32xi21>
  %element = tensor.extract %result32[%index] : tensor<32xi21>
  return %result16, %element : tensor<16xi21>, i21
}

//--- negative.mlir

func.func @dynamic_n(%weight: tensor<?x64xi8>, %input: tensor<64xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<?x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<32xi21>) -> tensor<32xi21>
  return %result : tensor<32xi21>
}

func.func @zero_n(%weight: tensor<0x64xi8>, %input: tensor<64xi8>)
    -> tensor<0xi21> {
  %zero = arith.constant dense<> : tensor<0xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<0x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<0xi21>) -> tensor<0xi21>
  return %result : tensor<0xi21>
}

func.func @n_not_multiple_of_16(%weight: tensor<33x64xi8>,
                               %input: tensor<64xi8>) -> tensor<33xi21> {
  %zero = arith.constant dense<0> : tensor<33xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<33x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<33xi21>) -> tensor<33xi21>
  return %result : tensor<33xi21>
}

func.func @wrong_k(%weight: tensor<32x63xi8>, %input: tensor<63xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<32x63xi8>, tensor<63xi8>)
      outs(%zero : tensor<32xi21>) -> tensor<32xi21>
  return %result : tensor<32xi21>
}

func.func @nonzero_init(%weight: tensor<32x64xi8>, %input: tensor<64xi8>)
    -> tensor<32xi21> {
  %one = arith.constant dense<1> : tensor<32xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<32x64xi8>, tensor<64xi8>)
      outs(%one : tensor<32xi21>) -> tensor<32xi21>
  return %result : tensor<32xi21>
}

func.func @wrong_types(%weight: tensor<32x64xi16>, %input: tensor<64xi16>)
    -> tensor<32xi32> {
  %zero = arith.constant dense<0> : tensor<32xi32>
  %result = linalg.matvec
      ins(%weight, %input : tensor<32x64xi16>, tensor<64xi16>)
      outs(%zero : tensor<32xi32>) -> tensor<32xi32>
  return %result : tensor<32xi32>
}

func.func @buffer_semantics(%weight: memref<32x64xi8>,
                            %input: memref<64xi8>,
                            %output: memref<32xi21>) {
  linalg.matvec
      ins(%weight, %input : memref<32x64xi8>, memref<64xi8>)
      outs(%output : memref<32xi21>)
  return
}

func.func @noncanonical_unsigned_region(%weight: tensor<32x64xi8>,
                                        %input: tensor<64xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
  %result = "linalg.matvec"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extui %weightElement : i8 to i21
      %extendedInput = arith.extui %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput : i21
      %sum = arith.addi %accumulator, %product : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<32x64xi8>, tensor<64xi8>, tensor<32xi21>)
      -> tensor<32xi21>
  return %result : tensor<32xi21>
}

func.func @noncanonical_indexing_maps(%weight: tensor<32x64xi8>,
                                      %input: tensor<64xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
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
  } : (tensor<32x64xi8>, tensor<64xi8>, tensor<32xi21>)
      -> tensor<32xi21>
  return %result : tensor<32xi21>
}

func.func @noncanonical_muli_flags(%weight: tensor<32x64xi8>,
                                   %input: tensor<64xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
  %result = "linalg.matvec"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput overflow<nsw> : i21
      %sum = arith.addi %accumulator, %product : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<32x64xi8>, tensor<64xi8>, tensor<32xi21>)
      -> tensor<32xi21>
  return %result : tensor<32xi21>
}

func.func @noncanonical_addi_flags(%weight: tensor<32x64xi8>,
                                   %input: tensor<64xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
  %result = "linalg.matvec"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput : i21
      %sum = arith.addi %accumulator, %product overflow<nsw> : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<32x64xi8>, tensor<64xi8>, tensor<32xi21>)
      -> tensor<32xi21>
  return %result : tensor<32xi21>
}

// NEG-LABEL: sym_name = "dynamic_n"
// NEG: "linalg.matvec"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "zero_n"
// NEG: "linalg.matvec"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "n_not_multiple_of_16"
// NEG: "linalg.matvec"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "wrong_k"
// NEG: "linalg.matvec"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "nonzero_init"
// NEG: "linalg.matvec"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "wrong_types"
// NEG: "linalg.matvec"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "buffer_semantics"
// NEG: "linalg.matvec"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "noncanonical_unsigned_region"
// NEG: "arith.extui"
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "noncanonical_indexing_maps"
// NEG: linalg.memoized_indexing_maps
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "noncanonical_muli_flags"
// NEG: "arith.muli"
// NEG-SAME: overflowFlags = #arith.overflow<nsw>
// NEG-NOT: "cim.vmm"
// NEG-LABEL: sym_name = "noncanonical_addi_flags"
// NEG: "arith.addi"
// NEG-SAME: overflowFlags = #arith.overflow<nsw>
// NEG-NOT: "cim.vmm"
