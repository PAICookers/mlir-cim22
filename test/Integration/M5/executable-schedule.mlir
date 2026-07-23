// RUN: mlir-cim22-opt %s -form-cim-program -materialize-cim-schedule > %t
// RUN: %python %S/../../Reference/M5/schedule_oracle.py %t --m 2 --k 65 --n 17 | FileCheck %s

// CHECK: PASS software-only M=2 K=65 N=17 work=8 groups=4 dtype=int32 shape=(2, 17) seed=2205
// CHECK: boundaries first=0:(0,0,0)/g0/c0/m0 core19=[NA] wrap=NA last=[6:(1,1,0)/g3/c3/m0,7:(1,1,1)/g3/c3/m1]

func.func @matmul_integer(%input: tensor<65x2xi8>) -> tensor<17x2xi32> {
  %weight = arith.constant dense<0> : tensor<17x65xi8>
  %zero = arith.constant dense<0> : tensor<17x2xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<17x65xi8>, tensor<65x2xi8>)
      outs(%zero : tensor<17x2xi32>) -> tensor<17x2xi32>
  return %result : tensor<17x2xi32>
}
