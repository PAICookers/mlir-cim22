// RUN: split-file %s %t
// RUN: mlir-cim22-opt %t/positive.mlir -partition-cim-program -form-cim-program > %t/once.mlir
// RUN: mlir-cim22-opt %t/once.mlir -partition-cim-program -form-cim-program > %t/twice.mlir
// RUN: diff %t/once.mlir %t/twice.mlir
// RUN: FileCheck %s --check-prefix=POS < %t/once.mlir
// RUN: not grep -q 'linalg.matvec' %t/once.mlir
// RUN: mlir-cim22-opt %t/negative.mlir -partition-cim-program -form-cim-program --mlir-print-op-generic --mlir-print-local-scope > %t/negative.out
// RUN: FileCheck %s --check-prefix=NEG < %t/negative.out
// RUN: not grep -q '"cim.vmm"' %t/negative.out

//--- positive.mlir

// One function covers result replacement, multiple candidates, and program order.
// POS-LABEL: func.func @full_k_tiles(
// POS-SAME: %[[W128:.*]]: tensor<16x128xi8>, %[[I128:.*]]: tensor<128xi8>, %[[W192:.*]]: tensor<16x192xi8>, %[[I192:.*]]: tensor<192xi8>, %[[INDEX:.*]]: index
// POS-NOT: tensor.extract_slice
// POS-NOT: tensor.pad
// POS-NOT: tensor.concat
// POS-NOT: cim.vmm
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS: %[[I128_0:.*]] = tensor.extract_slice %[[I128]][0] [64] [1] : tensor<128xi8> to tensor<64xi8>
// POS-NEXT: %[[W128_0:.*]] = tensor.extract_slice %[[W128]][0, 0] [16, 64] [1, 1] : tensor<16x128xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V128_0:.*]] = cim.vmm %[[I128_0]], %[[W128_0]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E128_0:.*]] = arith.extsi %[[V128_0]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[I128_1:.*]] = tensor.extract_slice %[[I128]][64] [64] [1] : tensor<128xi8> to tensor<64xi8>
// POS-NEXT: %[[W128_1:.*]] = tensor.extract_slice %[[W128]][0, 64] [16, 64] [1, 1] : tensor<16x128xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V128_1:.*]] = cim.vmm %[[I128_1]], %[[W128_1]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E128_1:.*]] = arith.extsi %[[V128_1]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[A128:.*]] = arith.addi %[[E128_0]], %[[E128_1]] : tensor<16xi22>
// POS-NEXT: %[[R128:.*]] = arith.trunci %[[A128]] : tensor<16xi22> to tensor<16xi21>
// POS-NEXT: %[[USER:.*]] = tensor.extract %[[R128]][%[[INDEX]]] : tensor<16xi21>
// POS-NEXT: %[[I192_0:.*]] = tensor.extract_slice %[[I192]][0] [64] [1] : tensor<192xi8> to tensor<64xi8>
// POS-NEXT: %[[W192_0:.*]] = tensor.extract_slice %[[W192]][0, 0] [16, 64] [1, 1] : tensor<16x192xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V192_0:.*]] = cim.vmm %[[I192_0]], %[[W192_0]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E192_0:.*]] = arith.extsi %[[V192_0]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[I192_1:.*]] = tensor.extract_slice %[[I192]][64] [64] [1] : tensor<192xi8> to tensor<64xi8>
// POS-NEXT: %[[W192_1:.*]] = tensor.extract_slice %[[W192]][0, 64] [16, 64] [1, 1] : tensor<16x192xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V192_1:.*]] = cim.vmm %[[I192_1]], %[[W192_1]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E192_1:.*]] = arith.extsi %[[V192_1]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[A192_0:.*]] = arith.addi %[[E192_0]], %[[E192_1]] : tensor<16xi23>
// POS-NEXT: %[[I192_2:.*]] = tensor.extract_slice %[[I192]][128] [64] [1] : tensor<192xi8> to tensor<64xi8>
// POS-NEXT: %[[W192_2:.*]] = tensor.extract_slice %[[W192]][0, 128] [16, 64] [1, 1] : tensor<16x192xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V192_2:.*]] = cim.vmm %[[I192_2]], %[[W192_2]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E192_2:.*]] = arith.extsi %[[V192_2]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[A192_1:.*]] = arith.addi %[[A192_0]], %[[E192_2]] : tensor<16xi23>
// POS-NEXT: %[[R192:.*]] = arith.trunci %[[A192_1]] : tensor<16xi23> to tensor<16xi21>
// POS-NEXT: return %[[USER]], %[[R192]] : i21, tensor<16xi21>
func.func @full_k_tiles(
    %weight128: tensor<16x128xi8>, %input128: tensor<128xi8>,
    %weight192: tensor<16x192xi8>, %input192: tensor<192xi8>,
    %index: index) -> (i21, tensor<16xi21>) {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result128 = linalg.matvec
      ins(%weight128, %input128 : tensor<16x128xi8>, tensor<128xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  %element = tensor.extract %result128[%index] : tensor<16xi21>
  %result192 = linalg.matvec
      ins(%weight192, %input192 : tensor<16x192xi8>, tensor<192xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %element, %result192 : i21, tensor<16xi21>
}

// POS-LABEL: func.func @k64_legacy(
// POS-SAME: %[[W64:.*]]: tensor<16x64xi8>, %[[I64:.*]]: tensor<64xi8>
// POS-NOT: tensor.extract_slice
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS: %[[R64:.*]] = cim.vmm %[[I64]], %[[W64]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: tensor.extract_slice
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS: return %[[R64]] : tensor<16xi21>
func.func @k64_legacy(%weight: tensor<16x64xi8>, %input: tensor<64xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x64xi8>, tensor<64xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

//--- negative.mlir

func.func @dynamic_k(%weight: tensor<16x?xi8>, %input: tensor<?xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec ins(%weight, %input : tensor<16x?xi8>, tensor<?xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

func.func @zero_k(%weight: tensor<16x0xi8>, %input: tensor<0xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec ins(%weight, %input : tensor<16x0xi8>, tensor<0xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

func.func @nonzero_init(%weight: tensor<16x128xi8>, %input: tensor<128xi8>)
    -> tensor<16xi21> {
  %one = arith.constant dense<1> : tensor<16xi21>
  %result = linalg.matvec ins(%weight, %input : tensor<16x128xi8>, tensor<128xi8>)
      outs(%one : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

func.func @i32_accumulator(%weight: tensor<16x128xi8>, %input: tensor<128xi8>)
    -> tensor<16xi32> {
  %zero = arith.constant dense<0> : tensor<16xi32>
  %result = linalg.matvec ins(%weight, %input : tensor<16x128xi8>, tensor<128xi8>)
      outs(%zero : tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}

// NEG-LABEL: sym_name = "dynamic_k"
// NEG: "linalg.matvec"
// NEG-LABEL: sym_name = "zero_k"
// NEG: "linalg.matvec"
// NEG-LABEL: sym_name = "nonzero_init"
// NEG: "linalg.matvec"
// NEG-LABEL: sym_name = "i32_accumulator"
// NEG: "linalg.matvec"
