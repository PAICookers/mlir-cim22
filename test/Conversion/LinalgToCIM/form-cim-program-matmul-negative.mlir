// RUN: mlir-cim22-opt %s -partition-cim-program -form-cim-program --mlir-print-op-generic --mlir-print-local-scope > %t.out
// RUN: FileCheck %s --check-prefix=NEG < %t.out
// RUN: not grep -q '"cim.vmm"' %t.out

// NEG-LABEL: sym_name = "dynamic_n"
// NEG: "linalg.matmul"
func.func @dynamic_n(%weight: tensor<?x64xi8>, %input: tensor<64x2xi8>,
                     %init: tensor<?x2xi21>) -> tensor<?x2xi21> {
  %result = linalg.matmul
      ins(%weight, %input : tensor<?x64xi8>, tensor<64x2xi8>)
      outs(%init : tensor<?x2xi21>) -> tensor<?x2xi21>
  return %result : tensor<?x2xi21>
}

// NEG-LABEL: sym_name = "dynamic_k"
// NEG: "linalg.matmul"
func.func @dynamic_k(%weight: tensor<16x?xi8>, %input: tensor<?x2xi8>)
    -> tensor<16x2xi21> {
  %zero = arith.constant dense<0> : tensor<16x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x?xi8>, tensor<?x2xi8>)
      outs(%zero : tensor<16x2xi21>) -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// NEG-LABEL: sym_name = "dynamic_m"
// NEG: "linalg.matmul"
func.func @dynamic_m(%weight: tensor<16x64xi8>, %input: tensor<64x?xi8>,
                     %init: tensor<16x?xi21>) -> tensor<16x?xi21> {
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x?xi8>)
      outs(%init : tensor<16x?xi21>) -> tensor<16x?xi21>
  return %result : tensor<16x?xi21>
}

// NEG-LABEL: sym_name = "zero_n"
// NEG: "linalg.matmul"
func.func @zero_n(%weight: tensor<0x64xi8>, %input: tensor<64x2xi8>)
    -> tensor<0x2xi21> {
  %zero = arith.constant dense<0> : tensor<0x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<0x64xi8>, tensor<64x2xi8>)
      outs(%zero : tensor<0x2xi21>) -> tensor<0x2xi21>
  return %result : tensor<0x2xi21>
}

// NEG-LABEL: sym_name = "zero_k"
// NEG: "linalg.matmul"
func.func @zero_k(%weight: tensor<16x0xi8>, %input: tensor<0x2xi8>)
    -> tensor<16x2xi21> {
  %zero = arith.constant dense<0> : tensor<16x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x0xi8>, tensor<0x2xi8>)
      outs(%zero : tensor<16x2xi21>) -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// NEG-LABEL: sym_name = "zero_m"
// NEG: "linalg.matmul"
func.func @zero_m(%weight: tensor<16x64xi8>, %input: tensor<64x0xi8>)
    -> tensor<16x0xi21> {
  %zero = arith.constant dense<0> : tensor<16x0xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x0xi8>)
      outs(%zero : tensor<16x0xi21>) -> tensor<16x0xi21>
  return %result : tensor<16x0xi21>
}

// NEG-LABEL: sym_name = "nonzero_init"
// NEG: "linalg.matmul"
func.func @nonzero_init(%weight: tensor<16x64xi8>, %input: tensor<64x2xi8>)
    -> tensor<16x2xi21> {
  %one = arith.constant dense<1> : tensor<16x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x2xi8>)
      outs(%one : tensor<16x2xi21>) -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// All shape, type, map, and body conditions are canonical, but the zero init
// is not defined directly by arith.constant.
// NEG-LABEL: sym_name = "indirect_zero_init"
// NEG: "tensor.extract_slice"
// NEG: "linalg.matmul"
func.func @indirect_zero_init(%weight: tensor<16x64xi8>,
                              %input: tensor<64x2xi8>)
    -> tensor<16x2xi21> {
  %larger_zero = arith.constant dense<0> : tensor<16x3xi21>
  %zero = tensor.extract_slice %larger_zero[0, 0] [16, 2] [1, 1]
      : tensor<16x3xi21> to tensor<16x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x2xi8>)
      outs(%zero : tensor<16x2xi21>) -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// NEG-LABEL: sym_name = "wrong_input_type"
// NEG: "linalg.matmul"
func.func @wrong_input_type(%weight: tensor<16x64xi16>,
                            %input: tensor<64x2xi16>) -> tensor<16x2xi21> {
  %zero = arith.constant dense<0> : tensor<16x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xi16>, tensor<64x2xi16>)
      outs(%zero : tensor<16x2xi21>) -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// NEG-LABEL: sym_name = "wrong_accumulator_type"
// NEG: "linalg.matmul"
func.func @wrong_accumulator_type(%weight: tensor<16x64xi8>,
                                  %input: tensor<64x2xi8>)
    -> tensor<16x2xi32> {
  %zero = arith.constant dense<0> : tensor<16x2xi32>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x2xi8>)
      outs(%zero : tensor<16x2xi32>) -> tensor<16x2xi32>
  return %result : tensor<16x2xi32>
}

// NEG-LABEL: sym_name = "buffer_semantics"
// NEG: "linalg.matmul"
func.func @buffer_semantics(%weight: memref<16x64xi8>,
                            %input: memref<64x2xi8>,
                            %output: memref<16x2xi21>) {
  linalg.matmul
      ins(%weight, %input : memref<16x64xi8>, memref<64x2xi8>)
      outs(%output : memref<16x2xi21>)
  return
}

// NEG-LABEL: sym_name = "alternate_maps"
// NEG: "linalg.matmul"
func.func @alternate_maps(%weight_transposed: tensor<64x16xi8>,
                          %input: tensor<64x2xi8>) -> tensor<16x2xi21> {
  %zero = arith.constant dense<0> : tensor<16x2xi21>
  %result = linalg.matmul indexing_maps = [
      affine_map<(n, m, k) -> (k, n)>,
      affine_map<(n, m, k) -> (k, m)>,
      affine_map<(n, m, k) -> (n, m)>]
      ins(%weight_transposed, %input : tensor<64x16xi8>, tensor<64x2xi8>)
      outs(%zero : tensor<16x2xi21>) -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// NEG-LABEL: sym_name = "unsigned_region"
// NEG: "linalg.matmul"
// NEG: "arith.extui"
func.func @unsigned_region(%weight: tensor<16x64xi8>,
                           %input: tensor<64x2xi8>) -> tensor<16x2xi21> {
  %zero = arith.constant dense<0> : tensor<16x2xi21>
  %result = linalg.matmul {cast = #linalg.type_fn<cast_unsigned>}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x2xi8>)
      outs(%zero : tensor<16x2xi21>) -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// NEG-LABEL: sym_name = "overflow_flags"
// NEG: "linalg.matmul"
// NEG: "arith.muli"
// NEG-SAME: overflowFlags = #arith.overflow<nsw>
func.func @overflow_flags(%weight: tensor<16x64xi8>,
                          %input: tensor<64x2xi8>) -> tensor<16x2xi21> {
  %zero = arith.constant dense<0> : tensor<16x2xi21>
  %result = "linalg.matmul"(%weight, %input, %zero) ({
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput overflow<nsw> : i21
      %sum = arith.addi %accumulator, %product : i21
      "linalg.yield"(%sum) : (i21) -> ()
  }) {operandSegmentSizes = array<i32: 2, 1>}
      : (tensor<16x64xi8>, tensor<64x2xi8>, tensor<16x2xi21>)
      -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

#weight = affine_map<(n, m, k) -> (n, k)>
#input = affine_map<(n, m, k) -> (k, m)>
#output = affine_map<(n, m, k) -> (n, m)>

// NEG-LABEL: sym_name = "generic_matmul"
// NEG: "linalg.generic"
func.func @generic_matmul(%weight: tensor<16x64xi8>,
                          %input: tensor<64x2xi8>) -> tensor<16x2xi21> {
  %zero = arith.constant dense<0> : tensor<16x2xi21>
  %result = linalg.generic {
      indexing_maps = [#weight, #input, #output],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x2xi8>)
      outs(%zero : tensor<16x2xi21>) {
    ^bb0(%weightElement: i8, %inputElement: i8, %accumulator: i21):
      %extendedWeight = arith.extsi %weightElement : i8 to i21
      %extendedInput = arith.extsi %inputElement : i8 to i21
      %product = arith.muli %extendedWeight, %extendedInput : i21
      %sum = arith.addi %accumulator, %product : i21
      linalg.yield %sum : i21
  } -> tensor<16x2xi21>
  return %result : tensor<16x2xi21>
}

// NEG-LABEL: sym_name = "batch_matmul"
// NEG: "linalg.batch_matmul"
func.func @batch_matmul(%weight: tensor<1x16x64xi8>,
                        %input: tensor<1x64x2xi8>) -> tensor<1x16x2xi21> {
  %zero = arith.constant dense<0> : tensor<1x16x2xi21>
  %result = linalg.batch_matmul
      ins(%weight, %input : tensor<1x16x64xi8>, tensor<1x64x2xi8>)
      outs(%zero : tensor<1x16x2xi21>) -> tensor<1x16x2xi21>
  return %result : tensor<1x16x2xi21>
}
