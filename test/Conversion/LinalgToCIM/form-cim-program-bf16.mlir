// RUN: mlir-cim22-opt %s -partition-cim-program -form-cim-program | FileCheck %s

// BF16 tiles form VMMs; multiple K tiles are combined in f32 on the Host.
// CHECK-LABEL: func.func @bf16_matvec(
// CHECK: %[[RESULT:.*]] = cim.vmm {{.*}} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
// CHECK: return %[[RESULT]] : tensor<16xbf16>
func.func @bf16_matvec(%weight: tensor<16x64xbf16>, %input: tensor<64xbf16>)
    -> tensor<16xbf16> {
  %zero = arith.constant dense<0.0> : tensor<16xbf16>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x64xbf16>, tensor<64xbf16>)
      outs(%zero : tensor<16xbf16>) -> tensor<16xbf16>
  return %result : tensor<16xbf16>
}

// CHECK-LABEL: func.func @bf16_short_k(
// CHECK: tensor.pad {{.*}} high[61]
// CHECK: tensor.pad {{.*}} high[14, 61]
// CHECK: cim.vmm {{.*}} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
// CHECK: %[[RESULT:.*]] = tensor.extract_slice
// CHECK: return %[[RESULT]] : tensor<2xbf16>
func.func @bf16_short_k(%weight: tensor<2x3xbf16>, %input: tensor<3xbf16>)
    -> tensor<2xbf16> {
  %zero = arith.constant dense<0.0> : tensor<2xbf16>
  %result = linalg.matvec
      ins(%weight, %input : tensor<2x3xbf16>, tensor<3xbf16>)
      outs(%zero : tensor<2xbf16>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}

// CHECK-LABEL: func.func @bf16_matmul(
// CHECK: cim.vmm {{.*}} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
// CHECK: cim.vmm {{.*}} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
// CHECK: %[[RESULT:.*]] = tensor.concat dim(1)
// CHECK: return %[[RESULT]] : tensor<16x2xbf16>
func.func @bf16_matmul(%weight: tensor<16x64xbf16>,
                       %input: tensor<64x2xbf16>) -> tensor<16x2xbf16> {
  %zero = arith.constant dense<0.0> : tensor<16x2xbf16>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xbf16>, tensor<64x2xbf16>)
      outs(%zero : tensor<16x2xbf16>) -> tensor<16x2xbf16>
  return %result : tensor<16x2xbf16>
}

// CHECK-LABEL: func.func @bf16_k_tail(
// CHECK: %[[P0:.*]] = cim.vmm {{.*}}k_tile = 0 : i64{{.*}} -> tensor<16xbf16>
// CHECK: %[[F0:.*]] = arith.extf %[[P0]] : tensor<16xbf16> to tensor<16xf32>
// CHECK: %[[P1:.*]] = cim.vmm {{.*}}k_tile = 1 : i64{{.*}} -> tensor<16xbf16>
// CHECK: %[[F1:.*]] = arith.extf %[[P1]] : tensor<16xbf16> to tensor<16xf32>
// CHECK: %[[SUM:.*]] = arith.addf %[[F0]], %[[F1]] : tensor<16xf32>
// CHECK: %[[RESULT:.*]] = arith.truncf %[[SUM]] : tensor<16xf32> to tensor<16xbf16>
// CHECK: return %[[RESULT]] : tensor<16xbf16>
func.func @bf16_k_tail(%weight: tensor<16x65xbf16>, %input: tensor<65xbf16>)
    -> tensor<16xbf16> {
  %zero = arith.constant dense<0.0> : tensor<16xbf16>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x65xbf16>, tensor<65xbf16>)
      outs(%zero : tensor<16xbf16>) -> tensor<16xbf16>
  return %result : tensor<16xbf16>
}

// The model's fully connected layers reduce 256 and 512 inputs; this small
// output exercises the same multi-tile BF16 path.
// CHECK-LABEL: func.func @bf16_linear_k256(
// CHECK: cim.vmm {{.*}}k_tile = 0 : i64
// CHECK: cim.vmm {{.*}}k_tile = 1 : i64
// CHECK: cim.vmm {{.*}}k_tile = 2 : i64
// CHECK: cim.vmm {{.*}}k_tile = 3 : i64
// CHECK: arith.truncf {{.*}} : tensor<16xf32> to tensor<16xbf16>
func.func @bf16_linear_k256(%weight: tensor<2x256xbf16>,
                            %input: tensor<256xbf16>) -> tensor<2xbf16> {
  %zero = arith.constant dense<0.0> : tensor<2xbf16>
  %result = linalg.matvec
      ins(%weight, %input : tensor<2x256xbf16>, tensor<256xbf16>)
      outs(%zero : tensor<2xbf16>) -> tensor<2xbf16>
  return %result : tensor<2xbf16>
}
