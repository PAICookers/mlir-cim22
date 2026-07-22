// RUN: split-file %s %t
// RUN: mlir-cim22-opt %t/positive.mlir -form-cim-program > %t/once.mlir
// RUN: mlir-cim22-opt %t/once.mlir -form-cim-program > %t/twice.mlir
// RUN: diff %t/once.mlir %t/twice.mlir
// RUN: FileCheck %s --check-prefix=POS < %t/once.mlir
// RUN: mlir-cim22-opt %t/negative.mlir -form-cim-program --mlir-print-op-generic --mlir-print-local-scope | FileCheck %s --check-prefix=NEG --implicit-check-not='"cim.vmm"'

//--- positive.mlir

// POS-LABEL: func.func @n1_tail_only(
// POS-SAME: %[[W1:.*]]: tensor<1x64xi8>, %[[I1:.*]]: tensor<64xi8>
// POS: %[[S1:.*]] = tensor.extract_slice %[[W1]][0, 0] [1, 64] [1, 1] : tensor<1x64xi8> to tensor<1x64xi8>
// POS-NEXT: %[[Z1:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[P1:.*]] = tensor.pad %[[S1]] low[0, 0] high[15, 0] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z1]] : i8
// POS-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V1:.*]] = cim.vmm %[[I1]], %[[P1]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: tensor.concat
// POS-NEXT: %[[R1:.*]] = tensor.extract_slice %[[V1]][0] [1] [1] : tensor<16xi21> to tensor<1xi21>
// POS-NOT: tensor.pad
// POS-NOT: tensor.extract_slice
// POS-NOT: linalg.matvec
// POS: return %[[R1]] : tensor<1xi21>
func.func @n1_tail_only(%weight: tensor<1x64xi8>, %input: tensor<64xi8>)
    -> tensor<1xi21> {
  %zero = arith.constant dense<0> : tensor<1xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<1x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<1xi21>) -> tensor<1xi21>
  return %result : tensor<1xi21>
}

// POS-LABEL: func.func @n15_tail_only(
// POS-SAME: %[[W15:.*]]: tensor<15x64xi8>, %[[I15:.*]]: tensor<64xi8>
// POS: %[[S15:.*]] = tensor.extract_slice %[[W15]][0, 0] [15, 64] [1, 1] : tensor<15x64xi8> to tensor<15x64xi8>
// POS-NEXT: %[[Z15:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[P15:.*]] = tensor.pad %[[S15]] low[0, 0] high[1, 0] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z15]] : i8
// POS-NEXT: } : tensor<15x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V15:.*]] = cim.vmm %[[I15]], %[[P15]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: tensor.concat
// POS-NEXT: %[[R15:.*]] = tensor.extract_slice %[[V15]][0] [15] [1] : tensor<16xi21> to tensor<15xi21>
// POS-NOT: tensor.pad
// POS-NOT: tensor.extract_slice
// POS-NOT: linalg.matvec
// POS: return %[[R15]] : tensor<15xi21>
func.func @n15_tail_only(%weight: tensor<15x64xi8>, %input: tensor<64xi8>)
    -> tensor<15xi21> {
  %zero = arith.constant dense<0> : tensor<15xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<15x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<15xi21>) -> tensor<15xi21>
  return %result : tensor<15xi21>
}

// POS-LABEL: func.func @n17_tail_and_result_user(
// POS-SAME: %[[W17:.*]]: tensor<17x64xi8>, %[[I17:.*]]: tensor<64xi8>
// POS: %[[S170:.*]] = tensor.extract_slice %[[W17]][0, 0] [16, 64] [1, 1] : tensor<17x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V170:.*]] = cim.vmm %[[I17]], %[[S170]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S171:.*]] = tensor.extract_slice %[[W17]][16, 0] [1, 64] [1, 1] : tensor<17x64xi8> to tensor<1x64xi8>
// POS-NEXT: %[[Z17:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[P17:.*]] = tensor.pad %[[S171]] low[0, 0] high[15, 0] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z17]] : i8
// POS-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V171:.*]] = cim.vmm %[[I17]], %[[P17]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[C17:.*]] = tensor.concat dim(0) %[[V170]], %[[V171]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NEXT: %[[R17:.*]] = tensor.extract_slice %[[C17]][0] [17] [1] : tensor<32xi21> to tensor<17xi21>
// POS: %[[E17:.*]] = tensor.extract %[[R17]][%{{.*}}] : tensor<17xi21>
// POS-NOT: tensor.pad
// POS-NOT: tensor.concat
// POS-NOT: linalg.matvec
// POS: return %[[E17]] : i21
func.func @n17_tail_and_result_user(
    %weight: tensor<17x64xi8>, %input: tensor<64xi8>, %index: index) -> i21 {
  %zero = arith.constant dense<0> : tensor<17xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<17x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<17xi21>) -> tensor<17xi21>
  %element = tensor.extract %result[%index] : tensor<17xi21>
  return %element : i21
}

// POS-LABEL: func.func @n31_full_and_tail(
// POS-SAME: %[[W31:.*]]: tensor<31x64xi8>, %[[I31:.*]]: tensor<64xi8>
// POS: %[[S310:.*]] = tensor.extract_slice %[[W31]][0, 0] [16, 64] [1, 1] : tensor<31x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V310:.*]] = cim.vmm %[[I31]], %[[S310]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S311:.*]] = tensor.extract_slice %[[W31]][16, 0] [15, 64] [1, 1] : tensor<31x64xi8> to tensor<15x64xi8>
// POS-NEXT: %[[Z31:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[P31:.*]] = tensor.pad %[[S311]] low[0, 0] high[1, 0] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z31]] : i8
// POS-NEXT: } : tensor<15x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V311:.*]] = cim.vmm %[[I31]], %[[P31]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[C31:.*]] = tensor.concat dim(0) %[[V310]], %[[V311]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NEXT: %[[R31:.*]] = tensor.extract_slice %[[C31]][0] [31] [1] : tensor<32xi21> to tensor<31xi21>
// POS-NOT: tensor.pad
// POS-NOT: tensor.concat
// POS-NOT: tensor.extract_slice
// POS-NOT: linalg.matvec
// POS: return %[[R31]] : tensor<31xi21>
func.func @n31_full_and_tail(%weight: tensor<31x64xi8>, %input: tensor<64xi8>)
    -> tensor<31xi21> {
  %zero = arith.constant dense<0> : tensor<31xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<31x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<31xi21>) -> tensor<31xi21>
  return %result : tensor<31xi21>
}

// POS-LABEL: func.func @n33_two_full_and_tail(
// POS-SAME: %[[W33:.*]]: tensor<33x64xi8>, %[[I33:.*]]: tensor<64xi8>
// POS: %[[S330:.*]] = tensor.extract_slice %[[W33]][0, 0] [16, 64] [1, 1] : tensor<33x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V330:.*]] = cim.vmm %[[I33]], %[[S330]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S331:.*]] = tensor.extract_slice %[[W33]][16, 0] [16, 64] [1, 1] : tensor<33x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V331:.*]] = cim.vmm %[[I33]], %[[S331]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S332:.*]] = tensor.extract_slice %[[W33]][32, 0] [1, 64] [1, 1] : tensor<33x64xi8> to tensor<1x64xi8>
// POS-NEXT: %[[Z33:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[P33:.*]] = tensor.pad %[[S332]] low[0, 0] high[15, 0] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z33]] : i8
// POS-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V332:.*]] = cim.vmm %[[I33]], %[[P33]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[C33:.*]] = tensor.concat dim(0) %[[V330]], %[[V331]], %[[V332]] : (tensor<16xi21>, tensor<16xi21>, tensor<16xi21>) -> tensor<48xi21>
// POS-NEXT: %[[R33:.*]] = tensor.extract_slice %[[C33]][0] [33] [1] : tensor<48xi21> to tensor<33xi21>
// POS-NOT: tensor.pad
// POS-NOT: tensor.concat
// POS-NOT: tensor.extract_slice
// POS-NOT: linalg.matvec
// POS: return %[[R33]] : tensor<33xi21>
func.func @n33_two_full_and_tail(%weight: tensor<33x64xi8>, %input: tensor<64xi8>)
    -> tensor<33xi21> {
  %zero = arith.constant dense<0> : tensor<33xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<33x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<33xi21>) -> tensor<33xi21>
  return %result : tensor<33xi21>
}

// POS-LABEL: func.func @n32_two_tiles(
// POS-SAME: %[[W32:.*]]: tensor<32x64xi8>, %[[I32:.*]]: tensor<64xi8>
// POS-NOT: tensor.pad
// POS: %[[S320:.*]] = tensor.extract_slice %[[W32]][0, 0] [16, 64] [1, 1] : tensor<32x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V320:.*]] = cim.vmm %[[I32]], %[[S320]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S321:.*]] = tensor.extract_slice %[[W32]][16, 0] [16, 64] [1, 1] : tensor<32x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V321:.*]] = cim.vmm %[[I32]], %[[S321]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[C32:.*]] = tensor.concat dim(0) %[[V320]], %[[V321]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NOT: tensor.pad
// POS-NOT: tensor.extract_slice
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
// POS-NOT: tensor.pad
// POS: %[[S480:.*]] = tensor.extract_slice %[[W48]][0, 0] [16, 64] [1, 1] : tensor<48x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V480:.*]] = cim.vmm %[[I48]], %[[S480]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S481:.*]] = tensor.extract_slice %[[W48]][16, 0] [16, 64] [1, 1] : tensor<48x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V481:.*]] = cim.vmm %[[I48]], %[[S481]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[S482:.*]] = tensor.extract_slice %[[W48]][32, 0] [16, 64] [1, 1] : tensor<48x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V482:.*]] = cim.vmm %[[I48]], %[[S482]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[C48:.*]] = tensor.concat dim(0) %[[V480]], %[[V481]], %[[V482]] : (tensor<16xi21>, tensor<16xi21>, tensor<16xi21>) -> tensor<48xi21>
// POS-NOT: tensor.pad
// POS-NOT: tensor.extract_slice
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

// N=16 must retain the original one-VMM behavior: no slice, pad, or concat.
// POS-LABEL: func.func @n16_original(
// POS-SAME: %[[W16:.*]]: tensor<16x64xi8>, %[[I16:.*]]: tensor<64xi8>
// POS-NOT: tensor.extract_slice
// POS-NOT: tensor.pad
// POS: %[[V16:.*]] = cim.vmm %[[I16]], %[[W16]] : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: tensor.extract_slice
// POS-NOT: tensor.pad
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
// NEG-LABEL: sym_name = "zero_n"
// NEG: "linalg.matvec"
// NEG-LABEL: sym_name = "nonzero_init"
// NEG: "linalg.matvec"
// NEG-LABEL: sym_name = "wrong_types"
// NEG: "linalg.matvec"
// NEG-LABEL: sym_name = "buffer_semantics"
// NEG: "linalg.matvec"
// NEG-LABEL: sym_name = "noncanonical_unsigned_region"
// NEG: "arith.extui"
// NEG-LABEL: sym_name = "noncanonical_indexing_maps"
// NEG: linalg.memoized_indexing_maps
// NEG-LABEL: sym_name = "noncanonical_muli_flags"
// NEG: "arith.muli"
// NEG-SAME: overflowFlags = #arith.overflow<nsw>
// NEG-LABEL: sym_name = "noncanonical_addi_flags"
// NEG: "arith.addi"
// NEG-SAME: overflowFlags = #arith.overflow<nsw>
