// RUN: mlir-cim22-opt %s -partition-cim-program -form-cim-program > %t.once.mlir
// RUN: mlir-cim22-opt %t.once.mlir -partition-cim-program -form-cim-program > %t.twice.mlir
// RUN: diff %t.once.mlir %t.twice.mlir
// RUN: not grep -q 'linalg.matvec' %t.once.mlir
// RUN: FileCheck %s --check-prefix=N32 < %t.once.mlir
// RUN: FileCheck %s --check-prefix=N17 < %t.once.mlir

// N32-LABEL: func.func @n32_k128(
// N32-SAME: %[[W:.*]]: tensor<32x128xi8>, %[[I:.*]]: tensor<128xi8>
// N32-NOT: tensor.extract_slice
// N32-NOT: cim.vmm
// N32-NOT: arith.extsi
// N32-NOT: arith.addi
// N32-NOT: arith.trunci
// N32-NOT: tensor.pad
// N32-NOT: tensor.concat
// N32: %[[I00:.*]] = tensor.extract_slice %[[I]][0] [64] [1] : tensor<128xi8> to tensor<64xi8>
// N32-NEXT: %[[W00:.*]] = tensor.extract_slice %[[W]][0, 0] [16, 64] [1, 1] : tensor<32x128xi8> to tensor<16x64xi8>
// N32-NEXT: %[[V00:.*]] = cim.vmm %[[I00]], %[[W00]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N32-NEXT: %[[E00:.*]] = arith.extsi %[[V00]] : tensor<16xi21> to tensor<16xi22>
// N32-NEXT: %[[I01:.*]] = tensor.extract_slice %[[I]][64] [64] [1] : tensor<128xi8> to tensor<64xi8>
// N32-NEXT: %[[W01:.*]] = tensor.extract_slice %[[W]][0, 64] [16, 64] [1, 1] : tensor<32x128xi8> to tensor<16x64xi8>
// N32-NEXT: %[[V01:.*]] = cim.vmm %[[I01]], %[[W01]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N32-NEXT: %[[E01:.*]] = arith.extsi %[[V01]] : tensor<16xi21> to tensor<16xi22>
// N32-NEXT: %[[A0:.*]] = arith.addi %[[E00]], %[[E01]] : tensor<16xi22>
// N32-NEXT: %[[R0:.*]] = arith.trunci %[[A0]] : tensor<16xi22> to tensor<16xi21>
// N32-NEXT: %[[I10:.*]] = tensor.extract_slice %[[I]][0] [64] [1] : tensor<128xi8> to tensor<64xi8>
// N32-NEXT: %[[W10:.*]] = tensor.extract_slice %[[W]][16, 0] [16, 64] [1, 1] : tensor<32x128xi8> to tensor<16x64xi8>
// N32-NEXT: %[[V10:.*]] = cim.vmm %[[I10]], %[[W10]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N32-NEXT: %[[E10:.*]] = arith.extsi %[[V10]] : tensor<16xi21> to tensor<16xi22>
// N32-NEXT: %[[I11:.*]] = tensor.extract_slice %[[I]][64] [64] [1] : tensor<128xi8> to tensor<64xi8>
// N32-NEXT: %[[W11:.*]] = tensor.extract_slice %[[W]][16, 64] [16, 64] [1, 1] : tensor<32x128xi8> to tensor<16x64xi8>
// N32-NEXT: %[[V11:.*]] = cim.vmm %[[I11]], %[[W11]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N32-NEXT: %[[E11:.*]] = arith.extsi %[[V11]] : tensor<16xi21> to tensor<16xi22>
// N32-NEXT: %[[A1:.*]] = arith.addi %[[E10]], %[[E11]] : tensor<16xi22>
// N32-NEXT: %[[R1:.*]] = arith.trunci %[[A1]] : tensor<16xi22> to tensor<16xi21>
// N32-NEXT: %[[C:.*]] = tensor.concat dim(0) %[[R0]], %[[R1]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// N32-NOT: tensor.extract_slice
// N32-NOT: cim.vmm
// N32-NOT: arith.extsi
// N32-NOT: arith.addi
// N32-NOT: arith.trunci
// N32-NOT: tensor.pad
// N32-NOT: tensor.concat
// N32: return %[[C]] : tensor<32xi21>
func.func @n32_k128(%weight: tensor<32x128xi8>, %input: tensor<128xi8>)
    -> tensor<32xi21> {
  %zero = arith.constant dense<0> : tensor<32xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<32x128xi8>, tensor<128xi8>)
      outs(%zero : tensor<32xi21>) -> tensor<32xi21>
  return %result : tensor<32xi21>
}

// N17-LABEL: func.func @n17_k192_result_user(
// N17-SAME: %[[W:.*]]: tensor<17x192xi8>, %[[I:.*]]: tensor<192xi8>, %[[INDEX:.*]]: index
// N17-NOT: tensor.extract_slice
// N17-NOT: cim.vmm
// N17-NOT: arith.extsi
// N17-NOT: arith.addi
// N17-NOT: arith.trunci
// N17-NOT: tensor.pad
// N17-NOT: tensor.concat
// N17: %[[I00:.*]] = tensor.extract_slice %[[I]][0] [64] [1] : tensor<192xi8> to tensor<64xi8>
// N17-NEXT: %[[W00:.*]] = tensor.extract_slice %[[W]][0, 0] [16, 64] [1, 1] : tensor<17x192xi8> to tensor<16x64xi8>
// N17-NEXT: %[[V00:.*]] = cim.vmm %[[I00]], %[[W00]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N17-NEXT: %[[E00:.*]] = arith.extsi %[[V00]] : tensor<16xi21> to tensor<16xi23>
// N17-NEXT: %[[I01:.*]] = tensor.extract_slice %[[I]][64] [64] [1] : tensor<192xi8> to tensor<64xi8>
// N17-NEXT: %[[W01:.*]] = tensor.extract_slice %[[W]][0, 64] [16, 64] [1, 1] : tensor<17x192xi8> to tensor<16x64xi8>
// N17-NEXT: %[[V01:.*]] = cim.vmm %[[I01]], %[[W01]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N17-NEXT: %[[E01:.*]] = arith.extsi %[[V01]] : tensor<16xi21> to tensor<16xi23>
// N17-NEXT: %[[A00:.*]] = arith.addi %[[E00]], %[[E01]] : tensor<16xi23>
// N17-NEXT: %[[I02:.*]] = tensor.extract_slice %[[I]][128] [64] [1] : tensor<192xi8> to tensor<64xi8>
// N17-NEXT: %[[W02:.*]] = tensor.extract_slice %[[W]][0, 128] [16, 64] [1, 1] : tensor<17x192xi8> to tensor<16x64xi8>
// N17-NEXT: %[[V02:.*]] = cim.vmm %[[I02]], %[[W02]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N17-NEXT: %[[E02:.*]] = arith.extsi %[[V02]] : tensor<16xi21> to tensor<16xi23>
// N17-NEXT: %[[A01:.*]] = arith.addi %[[A00]], %[[E02]] : tensor<16xi23>
// N17-NEXT: %[[R0:.*]] = arith.trunci %[[A01]] : tensor<16xi23> to tensor<16xi21>
// N17-NEXT: %[[I10:.*]] = tensor.extract_slice %[[I]][0] [64] [1] : tensor<192xi8> to tensor<64xi8>
// N17-NEXT: %[[W10:.*]] = tensor.extract_slice %[[W]][16, 0] [1, 64] [1, 1] : tensor<17x192xi8> to tensor<1x64xi8>
// N17-NEXT: %[[ZERO:.*]] = arith.constant 0 : i8
// N17-NEXT: %[[P10:.*]] = tensor.pad %[[W10]] low[0, 0] high[15, 0] {
// N17-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// N17-NEXT: tensor.yield %[[ZERO]] : i8
// N17-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// N17-NEXT: %[[V10:.*]] = cim.vmm %[[I10]], %[[P10]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N17-NEXT: %[[E10:.*]] = arith.extsi %[[V10]] : tensor<16xi21> to tensor<16xi23>
// N17-NEXT: %[[I11:.*]] = tensor.extract_slice %[[I]][64] [64] [1] : tensor<192xi8> to tensor<64xi8>
// N17-NEXT: %[[W11:.*]] = tensor.extract_slice %[[W]][16, 64] [1, 64] [1, 1] : tensor<17x192xi8> to tensor<1x64xi8>
// N17-NEXT: %[[P11:.*]] = tensor.pad %[[W11]] low[0, 0] high[15, 0] {
// N17-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// N17-NEXT: tensor.yield %[[ZERO]] : i8
// N17-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// N17-NEXT: %[[V11:.*]] = cim.vmm %[[I11]], %[[P11]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N17-NEXT: %[[E11:.*]] = arith.extsi %[[V11]] : tensor<16xi21> to tensor<16xi23>
// N17-NEXT: %[[A10:.*]] = arith.addi %[[E10]], %[[E11]] : tensor<16xi23>
// N17-NEXT: %[[I12:.*]] = tensor.extract_slice %[[I]][128] [64] [1] : tensor<192xi8> to tensor<64xi8>
// N17-NEXT: %[[W12:.*]] = tensor.extract_slice %[[W]][16, 128] [1, 64] [1, 1] : tensor<17x192xi8> to tensor<1x64xi8>
// N17-NEXT: %[[P12:.*]] = tensor.pad %[[W12]] low[0, 0] high[15, 0] {
// N17-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index):
// N17-NEXT: tensor.yield %[[ZERO]] : i8
// N17-NEXT: } : tensor<1x64xi8> to tensor<16x64xi8>
// N17-NEXT: %[[V12:.*]] = cim.vmm %[[I12]], %[[P12]] {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// N17-NEXT: %[[E12:.*]] = arith.extsi %[[V12]] : tensor<16xi21> to tensor<16xi23>
// N17-NEXT: %[[A11:.*]] = arith.addi %[[A10]], %[[E12]] : tensor<16xi23>
// N17-NEXT: %[[R1:.*]] = arith.trunci %[[A11]] : tensor<16xi23> to tensor<16xi21>
// N17-NEXT: %[[C:.*]] = tensor.concat dim(0) %[[R0]], %[[R1]] : (tensor<16xi21>, tensor<16xi21>) -> tensor<32xi21>
// N17-NEXT: %[[CROP:.*]] = tensor.extract_slice %[[C]][0] [17] [1] : tensor<32xi21> to tensor<17xi21>
// N17-NEXT: %[[ELEMENT:.*]] = tensor.extract %[[CROP]][%[[INDEX]]] : tensor<17xi21>
// N17-NOT: tensor.extract_slice
// N17-NOT: cim.vmm
// N17-NOT: arith.extsi
// N17-NOT: arith.addi
// N17-NOT: arith.trunci
// N17-NOT: tensor.pad
// N17-NOT: tensor.concat
// N17: return %[[ELEMENT]] : i21
func.func @n17_k192_result_user(
    %weight: tensor<17x192xi8>, %input: tensor<192xi8>, %index: index) -> i21 {
  %zero = arith.constant dense<0> : tensor<17xi21>
  %result = linalg.matvec
      ins(%weight, %input : tensor<17x192xi8>, tensor<192xi8>)
      outs(%zero : tensor<17xi21>) -> tensor<17xi21>
  %element = tensor.extract %result[%index] : tensor<17xi21>
  return %element : i21
}
