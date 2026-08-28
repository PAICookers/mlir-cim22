// RUN: mlir-cim22-opt %s -partition-cim-program -form-cim-program > %t.once.mlir
// RUN: mlir-cim22-opt %t.once.mlir -partition-cim-program -form-cim-program > %t.twice.mlir
// RUN: diff %t.once.mlir %t.twice.mlir
// RUN: not grep -Eq 'linalg\.(matmul|matvec)' %t.once.mlir
// RUN: FileCheck %s --check-prefix=POS < %t.once.mlir
// RUN: FileCheck %s --check-prefix=NOINIT < %t.once.mlir

// Check the complete body of each function independently from the ordered
// structural checks below. No generated column may derive its MatVec init by
// slicing the source matrix initializer.
// NOINIT-LABEL: func.func @native_m1_result_user(
// NOINIT: %[[INIT0:.*]] = arith.constant dense<0> : tensor<16x1xi21>
// NOINIT-NOT: tensor.extract_slice %[[INIT0]]
// NOINIT-LABEL: func.func @max_tail_m2(
// NOINIT: %[[INIT1:.*]] = arith.constant dense<0> : tensor<1x2xi21>
// NOINIT-NOT: tensor.extract_slice %[[INIT1]]
// NOINIT-LABEL: func.func @combined_m2_n17_k65(
// NOINIT: %[[INIT2:.*]] = arith.constant dense<0> : tensor<17x2xi21>
// NOINIT-NOT: tensor.extract_slice %[[INIT2]]

// POS-LABEL: func.func @native_m1_result_user(
// POS-SAME: %[[W:.*]]: tensor<16x64xi8>, %[[X:.*]]: tensor<64x1xi8>, %[[ROW:.*]]: index
// POS: %[[INIT:.*]] = arith.constant dense<0> : tensor<16x1xi21>
// POS-NOT: arith.constant dense<0> : tensor<16xi21>
// POS: %[[MVZERO:.*]] = arith.constant dense<0> : tensor<16xi21>
// POS-NOT: arith.constant dense<0> : tensor<16xi21>
// POS: %[[COL:.*]] = tensor.extract_slice %[[X]][0, 0] [64, 1] [1, 1] : tensor<64x1xi8> to tensor<64xi8>
// POS-NEXT: %[[VMM:.*]] = cim.vmm %[[COL]], %[[W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[EXPANDED:.*]] = tensor.expand_shape %[[VMM]] {{.*}}output_shape [16, 1] : tensor<16xi21> into tensor<16x1xi21>
// POS-NOT: tensor.concat dim(1)
// POS-NEXT: %[[ELEMENT:.*]] = tensor.extract %[[EXPANDED]][%[[ROW]], %{{.*}}] : tensor<16x1xi21>
// POS-NEXT: return %[[ELEMENT]] : i21
func.func @native_m1_result_user(%weight: tensor<16x64xi8>,
                                 %input: tensor<64x1xi8>,
                                 %row: index) -> i21 {
  %zero_index = arith.constant 0 : index
  %zero = arith.constant dense<0> : tensor<16x1xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xi8>, tensor<64x1xi8>)
      outs(%zero : tensor<16x1xi21>) -> tensor<16x1xi21>
  %element = tensor.extract %result[%row, %zero_index] : tensor<16x1xi21>
  return %element : i21
}

// POS-LABEL: func.func @max_tail_m2(
// POS-SAME: %[[W:.*]]: tensor<1x1xi8>, %[[X:.*]]: tensor<1x2xi8>
// POS: %[[INIT:.*]] = arith.constant dense<0> : tensor<1x2xi21>
// POS-NOT: arith.constant dense<0> : tensor<1xi21>
// POS: %[[MVZERO:.*]] = arith.constant dense<0> : tensor<1xi21>
// POS-NOT: arith.constant dense<0> : tensor<1xi21>
// POS: %[[COL0:.*]] = tensor.extract_slice %[[X]][0, 0] [1, 1] [1, 1] : tensor<1x2xi8> to tensor<1xi8>
// POS: %[[PAD0:.*]] = tensor.pad %[[COL0]] low[0] high[63]
// POS: %[[VMM0:.*]] = cim.vmm %[[PAD0]], %{{.*}} {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS: %[[N0M0:.*]] = tensor.extract_slice %[[VMM0]][0] [1] [1] : tensor<16xi21> to tensor<1xi21>
// POS-NEXT: %[[EXPANDED0:.*]] = tensor.expand_shape %[[N0M0]] {{.*}}output_shape [1, 1] : tensor<1xi21> into tensor<1x1xi21>
// POS-NEXT: %[[COL1:.*]] = tensor.extract_slice %[[X]][0, 1] [1, 1] [1, 1] : tensor<1x2xi8> to tensor<1xi8>
// POS: %[[PAD1:.*]] = tensor.pad %[[COL1]] low[0] high[63]
// POS: %[[VMM1:.*]] = cim.vmm %[[PAD1]], %{{.*}} {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS: %[[N0M1:.*]] = tensor.extract_slice %[[VMM1]][0] [1] [1] : tensor<16xi21> to tensor<1xi21>
// POS-NEXT: %[[EXPANDED1:.*]] = tensor.expand_shape %[[N0M1]] {{.*}}output_shape [1, 1] : tensor<1xi21> into tensor<1x1xi21>
// POS-NEXT: %[[RESULT:.*]] = tensor.concat dim(1) %[[EXPANDED0]], %[[EXPANDED1]] : (tensor<1x1xi21>, tensor<1x1xi21>) -> tensor<1x2xi21>
// POS-NOT: tensor.concat dim(1)
// POS-NEXT: return %[[RESULT]] : tensor<1x2xi21>
func.func @max_tail_m2(%weight: tensor<1x1xi8>, %input: tensor<1x2xi8>)
    -> tensor<1x2xi21> {
  %zero = arith.constant dense<0> : tensor<1x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<1x1xi8>, tensor<1x2xi8>)
      outs(%zero : tensor<1x2xi21>) -> tensor<1x2xi21>
  return %result : tensor<1x2xi21>
}

// Bind every M/N/K path in order. Existing M1.6 tests own pad-region details;
// this test owns column identity and the MatMul-to-MatVec composition.
// POS-LABEL: func.func @combined_m2_n17_k65(
// POS-SAME: %[[W:.*]]: tensor<17x65xi8>, %[[X:.*]]: tensor<65x2xi8>
// POS: %[[INIT:.*]] = arith.constant dense<0> : tensor<17x2xi21>
// POS-NOT: arith.constant dense<0> : tensor<17xi21>
// POS: %[[MVZERO:.*]] = arith.constant dense<0> : tensor<17xi21>
// POS-NOT: arith.constant dense<0> : tensor<17xi21>
// POS: %[[COL0:.*]] = tensor.extract_slice %[[X]][0, 0] [65, 1] [1, 1] : tensor<65x2xi8> to tensor<65xi8>
// POS-NEXT: %[[M0N0K0I:.*]] = tensor.extract_slice %[[COL0]][0] [64] [1] : tensor<65xi8> to tensor<64xi8>
// POS-NEXT: %[[M0N0K0W:.*]] = tensor.extract_slice %[[W]][0, 0] [16, 64] [1, 1] : tensor<17x65xi8> to tensor<16x64xi8>
// POS-NEXT: %[[M0N0K0V:.*]] = cim.vmm %[[M0N0K0I]], %[[M0N0K0W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M0N0K0E:.*]] = arith.extsi %[[M0N0K0V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M0N0K1IS:.*]] = tensor.extract_slice %[[COL0]][64] [1] [1] : tensor<65xi8> to tensor<1xi8>
// POS-NEXT: %[[M0N0K1WS:.*]] = tensor.extract_slice %[[W]][0, 64] [16, 1] [1, 1] : tensor<17x65xi8> to tensor<16x1xi8>
// POS: %[[M0N0K1I:.*]] = tensor.pad %[[M0N0K1IS]] low[0] high[63]
// POS: %[[M0N0K1W:.*]] = tensor.pad %[[M0N0K1WS]] low[0, 0] high[0, 63]
// POS: %[[M0N0K1V:.*]] = cim.vmm %[[M0N0K1I]], %[[M0N0K1W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M0N0K1E:.*]] = arith.extsi %[[M0N0K1V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M0N0A:.*]] = arith.addi %[[M0N0K0E]], %[[M0N0K1E]] : tensor<16xi22>
// POS-NEXT: %[[M0N0R:.*]] = arith.trunci %[[M0N0A]] : tensor<16xi22> to tensor<16xi21>
// POS-NEXT: %[[M0N1K0I:.*]] = tensor.extract_slice %[[COL0]][0] [64] [1] : tensor<65xi8> to tensor<64xi8>
// POS-NEXT: %[[M0N1K0WS:.*]] = tensor.extract_slice %[[W]][16, 0] [1, 64] [1, 1] : tensor<17x65xi8> to tensor<1x64xi8>
// POS: %[[M0N1K0W:.*]] = tensor.pad %[[M0N1K0WS]] low[0, 0] high[15, 0]
// POS: %[[M0N1K0V:.*]] = cim.vmm %[[M0N1K0I]], %[[M0N1K0W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M0N1K0E:.*]] = arith.extsi %[[M0N1K0V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M0N1K1IS:.*]] = tensor.extract_slice %[[COL0]][64] [1] [1] : tensor<65xi8> to tensor<1xi8>
// POS-NEXT: %[[M0N1K1WS:.*]] = tensor.extract_slice %[[W]][16, 64] [1, 1] [1, 1] : tensor<17x65xi8> to tensor<1x1xi8>
// POS: %[[M0N1K1I:.*]] = tensor.pad %[[M0N1K1IS]] low[0] high[63]
// POS: %[[M0N1K1W:.*]] = tensor.pad %[[M0N1K1WS]] low[0, 0] high[15, 63]
// POS: %[[M0N1K1V:.*]] = cim.vmm %[[M0N1K1I]], %[[M0N1K1W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M0N1K1E:.*]] = arith.extsi %[[M0N1K1V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M0N1A:.*]] = arith.addi %[[M0N1K0E]], %[[M0N1K1E]] : tensor<16xi22>
// POS-NEXT: %[[M0N1R:.*]] = arith.trunci %[[M0N1A]] : tensor<16xi22> to tensor<16xi21>
// POS-NEXT: %[[M0NC:.*]] = tensor.concat dim(0) %[[M0N0R]], %[[M0N1R]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NEXT: %[[M0CROP:.*]] = tensor.extract_slice %[[M0NC]][0] [17] [1] : tensor<32xi21> to tensor<17xi21>
// POS-NEXT: %[[EXPANDED0:.*]] = tensor.expand_shape %[[M0CROP]] {{.*}}output_shape [17, 1] : tensor<17xi21> into tensor<17x1xi21>
// POS-NEXT: %[[COL1:.*]] = tensor.extract_slice %[[X]][0, 1] [65, 1] [1, 1] : tensor<65x2xi8> to tensor<65xi8>
// POS-NEXT: %[[M1N0K0I:.*]] = tensor.extract_slice %[[COL1]][0] [64] [1] : tensor<65xi8> to tensor<64xi8>
// POS-NEXT: %[[M1N0K0W:.*]] = tensor.extract_slice %[[W]][0, 0] [16, 64] [1, 1] : tensor<17x65xi8> to tensor<16x64xi8>
// POS-NEXT: %[[M1N0K0V:.*]] = cim.vmm %[[M1N0K0I]], %[[M1N0K0W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M1N0K0E:.*]] = arith.extsi %[[M1N0K0V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M1N0K1IS:.*]] = tensor.extract_slice %[[COL1]][64] [1] [1] : tensor<65xi8> to tensor<1xi8>
// POS-NEXT: %[[M1N0K1WS:.*]] = tensor.extract_slice %[[W]][0, 64] [16, 1] [1, 1] : tensor<17x65xi8> to tensor<16x1xi8>
// POS: %[[M1N0K1I:.*]] = tensor.pad %[[M1N0K1IS]] low[0] high[63]
// POS: %[[M1N0K1W:.*]] = tensor.pad %[[M1N0K1WS]] low[0, 0] high[0, 63]
// POS: %[[M1N0K1V:.*]] = cim.vmm %[[M1N0K1I]], %[[M1N0K1W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M1N0K1E:.*]] = arith.extsi %[[M1N0K1V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M1N0A:.*]] = arith.addi %[[M1N0K0E]], %[[M1N0K1E]] : tensor<16xi22>
// POS-NEXT: %[[M1N0R:.*]] = arith.trunci %[[M1N0A]] : tensor<16xi22> to tensor<16xi21>
// POS-NEXT: %[[M1N1K0I:.*]] = tensor.extract_slice %[[COL1]][0] [64] [1] : tensor<65xi8> to tensor<64xi8>
// POS-NEXT: %[[M1N1K0WS:.*]] = tensor.extract_slice %[[W]][16, 0] [1, 64] [1, 1] : tensor<17x65xi8> to tensor<1x64xi8>
// POS: %[[M1N1K0W:.*]] = tensor.pad %[[M1N1K0WS]] low[0, 0] high[15, 0]
// POS: %[[M1N1K0V:.*]] = cim.vmm %[[M1N1K0I]], %[[M1N1K0W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M1N1K0E:.*]] = arith.extsi %[[M1N1K0V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M1N1K1IS:.*]] = tensor.extract_slice %[[COL1]][64] [1] [1] : tensor<65xi8> to tensor<1xi8>
// POS-NEXT: %[[M1N1K1WS:.*]] = tensor.extract_slice %[[W]][16, 64] [1, 1] [1, 1] : tensor<17x65xi8> to tensor<1x1xi8>
// POS: %[[M1N1K1I:.*]] = tensor.pad %[[M1N1K1IS]] low[0] high[63]
// POS: %[[M1N1K1W:.*]] = tensor.pad %[[M1N1K1WS]] low[0, 0] high[15, 63]
// POS: %[[M1N1K1V:.*]] = cim.vmm %[[M1N1K1I]], %[[M1N1K1W]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[M1N1K1E:.*]] = arith.extsi %[[M1N1K1V]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[M1N1A:.*]] = arith.addi %[[M1N1K0E]], %[[M1N1K1E]] : tensor<16xi22>
// POS-NEXT: %[[M1N1R:.*]] = arith.trunci %[[M1N1A]] : tensor<16xi22> to tensor<16xi21>
// POS-NEXT: %[[M1NC:.*]] = tensor.concat dim(0) %[[M1N0R]], %[[M1N1R]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NEXT: %[[M1CROP:.*]] = tensor.extract_slice %[[M1NC]][0] [17] [1] : tensor<32xi21> to tensor<17xi21>
// POS-NEXT: %[[EXPANDED1:.*]] = tensor.expand_shape %[[M1CROP]] {{.*}}output_shape [17, 1] : tensor<17xi21> into tensor<17x1xi21>
// POS-NEXT: %[[RESULT:.*]] = tensor.concat dim(1) %[[EXPANDED0]], %[[EXPANDED1]] : (tensor<17x1xi21>, tensor<17x1xi21>) -> tensor<17x2xi21>
// POS-NOT: tensor.concat dim(1)
// POS-NEXT: return %[[RESULT]] : tensor<17x2xi21>
func.func @combined_m2_n17_k65(%weight: tensor<17x65xi8>,
                               %input: tensor<65x2xi8>)
    -> tensor<17x2xi21> {
  %zero = arith.constant dense<0> : tensor<17x2xi21>
  %result = linalg.matmul
      ins(%weight, %input : tensor<17x65xi8>, tensor<65x2xi8>)
      outs(%zero : tensor<17x2xi21>) -> tensor<17x2xi21>
  return %result : tensor<17x2xi21>
}
