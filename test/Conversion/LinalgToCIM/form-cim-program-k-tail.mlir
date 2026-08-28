// RUN: split-file %s %t
// RUN: mlir-cim22-opt %t/positive.mlir -partition-cim-program -form-cim-program > %t/once.mlir
// RUN: mlir-cim22-opt %t/once.mlir -partition-cim-program -form-cim-program > %t/twice.mlir
// RUN: diff %t/once.mlir %t/twice.mlir
// RUN: not grep -q 'linalg.matvec' %t/once.mlir
// RUN: FileCheck %s --check-prefix=POS < %t/once.mlir
// RUN: mlir-cim22-opt %t/negative.mlir -partition-cim-program -form-cim-program --mlir-print-op-generic --mlir-print-local-scope > %t/negative.out
// RUN: FileCheck %s --check-prefix=NEG < %t/negative.out
// RUN: not grep -q '"cim.vmm"' %t/negative.out

//--- positive.mlir

// POS-LABEL: func.func @n16_k1(
// POS-SAME: %[[W1:.*]]: tensor<16x1xi8>, %[[I1:.*]]: tensor<1xi8>
// POS-NOT: tensor.extract_slice
// POS-NOT: tensor.pad
// POS-NOT: cim.vmm
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS-NOT: tensor.concat
// POS: %[[Z1:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[IP1:.*]] = tensor.pad %[[I1]] low[0] high[63] {
// POS-NEXT: ^bb0(%{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z1]] : i8
// POS-NEXT: } : tensor<1xi8> to tensor<64xi8>
// POS-NEXT: %[[WP1:.*]] = tensor.pad %[[W1]] low[0, 0] high[0, 63] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z1]] : i8
// POS-NEXT: } : tensor<16x1xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V1:.*]] = cim.vmm %[[IP1]], %[[WP1]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS-NOT: tensor.concat
// POS-NOT: tensor.extract_slice
// POS-NEXT: return %[[V1]] : tensor<16xi21>
func.func @n16_k1(%weight: tensor<16x1xi8>, %input: tensor<1xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x1xi8>, tensor<1xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// POS-LABEL: func.func @n1_k63(
// POS-SAME: %[[W63:.*]]: tensor<1x63xi8>, %[[I63:.*]]: tensor<63xi8>
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS-NOT: tensor.concat
// POS: %[[Z63:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[IP63:.*]] = tensor.pad %[[I63]] low[0] high[1] {
// POS-NEXT: ^bb0(%{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z63]] : i8
// POS-NEXT: } : tensor<63xi8> to tensor<64xi8>
// POS-NEXT: %[[WP63:.*]] = tensor.pad %[[W63]] low[0, 0] high[15, 1] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z63]] : i8
// POS-NEXT: } : tensor<1x63xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V63:.*]] = cim.vmm %[[IP63]], %[[WP63]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS-NOT: tensor.concat
// POS-NEXT: %[[R63:.*]] = tensor.extract_slice %[[V63]][0] [1] [1] : tensor<16xi21> to tensor<1xi21>
// POS-NEXT: return %[[R63]] : tensor<1xi21>
func.func @n1_k63(%weight: tensor<1x63xi8>, %input: tensor<63xi8>)
    -> tensor<1xi21> {
  %zero = arith.constant dense<0> : tensor<1xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<1x63xi8>, tensor<63xi8>)
      outs(%zero : tensor<1xi21>) -> tensor<1xi21>
  return %result : tensor<1xi21>
}

// POS-LABEL: func.func @n17_k1(
// POS-SAME: %[[W17K1:.*]]: tensor<17x1xi8>, %[[I17K1:.*]]: tensor<1xi8>
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS: %[[W17K10S:.*]] = tensor.extract_slice %[[W17K1]][0, 0] [16, 1] [1, 1] : tensor<17x1xi8> to tensor<16x1xi8>
// POS-NEXT: %[[Z17K10:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[I17K10:.*]] = tensor.pad %[[I17K1]] low[0] high[63] {
// POS-NEXT: ^bb0(%{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z17K10]] : i8
// POS-NEXT: } : tensor<1xi8> to tensor<64xi8>
// POS-NEXT: %[[W17K10:.*]] = tensor.pad %[[W17K10S]] low[0, 0] high[0, 63] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z17K10]] : i8
// POS-NEXT: } : tensor<16x1xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V17K10:.*]] = cim.vmm %[[I17K10]], %[[W17K10]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[W17K11S:.*]] = tensor.extract_slice %[[W17K1]][16, 0] [1, 1] [1, 1] : tensor<17x1xi8> to tensor<1x1xi8>
// POS-NEXT: %[[Z17K11:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[I17K11:.*]] = tensor.pad %[[I17K1]] low[0] high[63] {
// POS-NEXT: ^bb0(%{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z17K11]] : i8
// POS-NEXT: } : tensor<1xi8> to tensor<64xi8>
// POS-NEXT: %[[W17K11:.*]] = tensor.pad %[[W17K11S]] low[0, 0] high[15, 63] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z17K11]] : i8
// POS-NEXT: } : tensor<1x1xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V17K11:.*]] = cim.vmm %[[I17K11]], %[[W17K11]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NOT: arith.extsi
// POS-NOT: arith.addi
// POS-NOT: arith.trunci
// POS-NEXT: %[[C17K1:.*]] = tensor.concat dim(0) %[[V17K10]], %[[V17K11]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NEXT: %[[R17K1:.*]] = tensor.extract_slice %[[C17K1]][0] [17] [1] : tensor<32xi21> to tensor<17xi21>
// POS-NEXT: return %[[R17K1]] : tensor<17xi21>
func.func @n17_k1(%weight: tensor<17x1xi8>, %input: tensor<1xi8>)
    -> tensor<17xi21> {
  %zero = arith.constant dense<0> : tensor<17xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<17x1xi8>, tensor<1xi8>)
      outs(%zero : tensor<17xi21>) -> tensor<17xi21>
  return %result : tensor<17xi21>
}

// POS-LABEL: func.func @n16_k65(
// POS-SAME: %[[W65:.*]]: tensor<16x65xi8>, %[[I65:.*]]: tensor<65xi8>
// POS-NOT: tensor.concat
// POS: %[[IS650:.*]] = tensor.extract_slice %[[I65]][0] [64] [1] : tensor<65xi8> to tensor<64xi8>
// POS-NEXT: %[[WS650:.*]] = tensor.extract_slice %[[W65]][0, 0] [16, 64] [1, 1] : tensor<16x65xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V650:.*]] = cim.vmm %[[IS650]], %[[WS650]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E650:.*]] = arith.extsi %[[V650]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[IS651:.*]] = tensor.extract_slice %[[I65]][64] [1] [1] : tensor<65xi8> to tensor<1xi8>
// POS-NEXT: %[[WS651:.*]] = tensor.extract_slice %[[W65]][0, 64] [16, 1] [1, 1] : tensor<16x65xi8> to tensor<16x1xi8>
// POS-NEXT: %[[Z65:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[IP651:.*]] = tensor.pad %[[IS651]] low[0] high[63] {
// POS-NEXT: ^bb0(%{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z65]] : i8
// POS-NEXT: } : tensor<1xi8> to tensor<64xi8>
// POS-NEXT: %[[WP651:.*]] = tensor.pad %[[WS651]] low[0, 0] high[0, 63] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z65]] : i8
// POS-NEXT: } : tensor<16x1xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V651:.*]] = cim.vmm %[[IP651]], %[[WP651]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E651:.*]] = arith.extsi %[[V651]] : tensor<16xi21> to tensor<16xi22>
// POS-NEXT: %[[A65:.*]] = arith.addi %[[E650]], %[[E651]] : tensor<16xi22>
// POS-NEXT: %[[R65:.*]] = arith.trunci %[[A65]] : tensor<16xi22> to tensor<16xi21>
// POS-NOT: tensor.concat
// POS-NEXT: return %[[R65]] : tensor<16xi21>
func.func @n16_k65(%weight: tensor<16x65xi8>, %input: tensor<65xi8>)
    -> tensor<16xi21> {
  %zero = arith.constant dense<0> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x65xi8>, tensor<65xi8>)
      outs(%zero : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// POS-LABEL: func.func @n17_k191_result_user(
// POS-SAME: %[[W191:.*]]: tensor<17x191xi8>, %[[I191:.*]]: tensor<191xi8>, %[[INDEX:.*]]: index
// First N tile: two full K tiles and one 63-element tail, reduced at i23.
// POS: %[[I000:.*]] = tensor.extract_slice %[[I191]][0] [64] [1] : tensor<191xi8> to tensor<64xi8>
// POS-NEXT: %[[W000:.*]] = tensor.extract_slice %[[W191]][0, 0] [16, 64] [1, 1] : tensor<17x191xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V000:.*]] = cim.vmm %[[I000]], %[[W000]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E000:.*]] = arith.extsi %[[V000]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[I001:.*]] = tensor.extract_slice %[[I191]][64] [64] [1] : tensor<191xi8> to tensor<64xi8>
// POS-NEXT: %[[W001:.*]] = tensor.extract_slice %[[W191]][0, 64] [16, 64] [1, 1] : tensor<17x191xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V001:.*]] = cim.vmm %[[I001]], %[[W001]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E001:.*]] = arith.extsi %[[V001]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[A000:.*]] = arith.addi %[[E000]], %[[E001]] : tensor<16xi23>
// POS-NEXT: %[[I002S:.*]] = tensor.extract_slice %[[I191]][128] [63] [1] : tensor<191xi8> to tensor<63xi8>
// POS-NEXT: %[[W002S:.*]] = tensor.extract_slice %[[W191]][0, 128] [16, 63] [1, 1] : tensor<17x191xi8> to tensor<16x63xi8>
// POS-NEXT: %[[Z0:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[I002:.*]] = tensor.pad %[[I002S]] low[0] high[1] {
// POS-NEXT: ^bb0(%{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z0]] : i8
// POS-NEXT: } : tensor<63xi8> to tensor<64xi8>
// POS-NEXT: %[[W002:.*]] = tensor.pad %[[W002S]] low[0, 0] high[0, 1] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z0]] : i8
// POS-NEXT: } : tensor<16x63xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V002:.*]] = cim.vmm %[[I002]], %[[W002]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E002:.*]] = arith.extsi %[[V002]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[A001:.*]] = arith.addi %[[A000]], %[[E002]] : tensor<16xi23>
// POS-NEXT: %[[R0:.*]] = arith.trunci %[[A001]] : tensor<16xi23> to tensor<16xi21>
// Second N tile starts from its own first partial and pads N high by 15.
// POS-NEXT: %[[I100:.*]] = tensor.extract_slice %[[I191]][0] [64] [1] : tensor<191xi8> to tensor<64xi8>
// POS-NEXT: %[[W100S:.*]] = tensor.extract_slice %[[W191]][16, 0] [1, 64] [1, 1] : tensor<17x191xi8> to tensor<1x64xi8>
// POS-NEXT: %[[Z1:.*]] = arith.constant 0 : i8
// POS-NEXT: %[[W100:.*]] = tensor.pad %[[W100S]] low[0, 0] high[15, 0] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z1]] : i8
// POS-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V100:.*]] = cim.vmm %[[I100]], %[[W100]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E100:.*]] = arith.extsi %[[V100]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[I101:.*]] = tensor.extract_slice %[[I191]][64] [64] [1] : tensor<191xi8> to tensor<64xi8>
// POS-NEXT: %[[W101S:.*]] = tensor.extract_slice %[[W191]][16, 64] [1, 64] [1, 1] : tensor<17x191xi8> to tensor<1x64xi8>
// POS-NEXT: %[[W101:.*]] = tensor.pad %[[W101S]] low[0, 0] high[15, 0] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z1]] : i8
// POS-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V101:.*]] = cim.vmm %[[I101]], %[[W101]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E101:.*]] = arith.extsi %[[V101]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[A100:.*]] = arith.addi %[[E100]], %[[E101]] : tensor<16xi23>
// POS-NEXT: %[[I102S:.*]] = tensor.extract_slice %[[I191]][128] [63] [1] : tensor<191xi8> to tensor<63xi8>
// POS-NEXT: %[[W102S:.*]] = tensor.extract_slice %[[W191]][16, 128] [1, 63] [1, 1] : tensor<17x191xi8> to tensor<1x63xi8>
// POS-NEXT: %[[I102:.*]] = tensor.pad %[[I102S]] low[0] high[1] {
// POS-NEXT: ^bb0(%{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z1]] : i8
// POS-NEXT: } : tensor<63xi8> to tensor<64xi8>
// POS-NEXT: %[[W102:.*]] = tensor.pad %[[W102S]] low[0, 0] high[15, 1] {
// POS-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// POS-NEXT: tensor.yield %[[Z1]] : i8
// POS-NEXT: } : tensor<1x63xi8> to tensor<16x64xi8>
// POS-NEXT: %[[V102:.*]] = cim.vmm %[[I102]], %[[W102]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// POS-NEXT: %[[E102:.*]] = arith.extsi %[[V102]] : tensor<16xi21> to tensor<16xi23>
// POS-NEXT: %[[A101:.*]] = arith.addi %[[A100]], %[[E102]] : tensor<16xi23>
// POS-NEXT: %[[R1:.*]] = arith.trunci %[[A101]] : tensor<16xi23> to tensor<16xi21>
// POS-NEXT: %[[C:.*]] = tensor.concat dim(0) %[[R0]], %[[R1]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// POS-NEXT: %[[R:.*]] = tensor.extract_slice %[[C]][0] [17] [1] : tensor<32xi21> to tensor<17xi21>
// POS-NEXT: %[[E:.*]] = tensor.extract %[[R]][%[[INDEX]]] : tensor<17xi21>
// POS-NEXT: return %[[E]] : i21
func.func @n17_k191_result_user(
    %weight: tensor<17x191xi8>, %input: tensor<191xi8>, %index: index) -> i21 {
  %zero = arith.constant dense<0> : tensor<17xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<17x191xi8>, tensor<191xi8>)
      outs(%zero : tensor<17xi21>) -> tensor<17xi21>
  %element = tensor.extract %result[%index] : tensor<17xi21>
  return %element : i21
}

//--- negative.mlir

func.func @tail_nonzero_init(%weight: tensor<16x65xi8>,
                             %input: tensor<65xi8>) -> tensor<16xi21> {
  %one = arith.constant dense<1> : tensor<16xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x65xi8>, tensor<65xi8>)
      outs(%one : tensor<16xi21>) -> tensor<16xi21>
  return %result : tensor<16xi21>
}

// NEG-LABEL: sym_name = "tail_nonzero_init"
// NEG: "linalg.matvec"
