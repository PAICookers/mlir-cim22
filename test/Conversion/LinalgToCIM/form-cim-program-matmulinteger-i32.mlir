// RUN: mlir-cim22-opt %s -partition-cim-program -form-cim-program > %t.out
// RUN: test "$(grep -c 'cim.vmm' %t.out)" -eq 8
// RUN: test "$(grep -c 'arith.extsi' %t.out)" -eq 8
// RUN: test "$(grep -c 'arith.addi' %t.out)" -eq 4
// RUN: not grep -q 'arith.trunci' %t.out
// RUN: FileCheck %s < %t.out

// CHECK-LABEL: func.func @matmul_integer
// CHECK: tensor.extract_slice {{.*}}[0] [64] [1] : tensor<65xi8> to tensor<64xi8>
// CHECK: tensor.extract_slice {{.*}}[0, 0] [16, 64] [1, 1] : tensor<17x65xi8> to tensor<16x64xi8>
// CHECK: cim.vmm {{.*}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
// CHECK: arith.extsi {{.*}} : tensor<16xi21> to tensor<16xi32>
// CHECK: arith.addi {{.*}} : tensor<16xi32>
// CHECK: tensor.pad {{.*}} low[0] high[63]
// CHECK: tensor.pad {{.*}} low[0, 0] high[15, 63]
// CHECK: tensor.concat dim(0)
// CHECK: tensor.extract_slice {{.*}}[0] [17] [1] : tensor<32xi32> to tensor<17xi32>
// CHECK: tensor.concat dim(1)
// CHECK: return {{.*}} : tensor<17x2xi32>
func.func @matmul_integer(%input: tensor<65x2xi8>) -> tensor<17x2xi32> {
  %weight = arith.constant dense<0> : tensor<17x65xi8>
  %zero = arith.constant dense<0> : tensor<17x2xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<17x65xi8>, tensor<65x2xi8>)
      outs(%zero : tensor<17x2xi32>) -> tensor<17x2xi32>
  return %result : tensor<17x2xi32>
}

// CHECK-LABEL: func.func @unmarked_i32
// CHECK: linalg.matmul
// CHECK-NOT: cim.vmm
// CHECK: return {{.*}} : tensor<17x2xi32>
func.func @unmarked_i32(%input: tensor<65x2xi8>) -> tensor<17x2xi32> {
  %weight = arith.constant dense<0> : tensor<17x65xi8>
  %zero = arith.constant dense<0> : tensor<17x2xi32>
  %result = linalg.matmul
      ins(%weight, %input : tensor<17x65xi8>, tensor<65x2xi8>)
      outs(%zero : tensor<17x2xi32>) -> tensor<17x2xi32>
  return %result : tensor<17x2xi32>
}
