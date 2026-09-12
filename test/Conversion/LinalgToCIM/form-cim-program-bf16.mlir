// RUN: mlir-cim22-opt %s -partition-cim-program -form-cim-program | FileCheck %s --implicit-check-not=cim.vmm --implicit-check-not=cim.transaction_idx

// Standard BF16 contractions must stay on the Host: the supplier bit-level
// model does not preserve their arithmetic, including zero and unit inputs.
// CHECK-LABEL: func.func @bf16_matvec(
// CHECK: %[[RESULT:.*]] = linalg.matvec
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
// CHECK: %[[RESULT:.*]] = linalg.matvec
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
// CHECK: %[[RESULT:.*]] = linalg.matmul
// CHECK: return %[[RESULT]] : tensor<16x2xbf16>
func.func @bf16_matmul(%weight: tensor<16x64xbf16>,
                       %input: tensor<64x2xbf16>) -> tensor<16x2xbf16> {
  %zero = arith.constant dense<0.0> : tensor<16x2xbf16>
  %result = linalg.matmul
      ins(%weight, %input : tensor<16x64xbf16>, tensor<64x2xbf16>)
      outs(%zero : tensor<16x2xbf16>) -> tensor<16x2xbf16>
  return %result : tensor<16x2xbf16>
}

// Larger reductions must also remain untouched.
// CHECK-LABEL: func.func @bf16_k_tail(
// CHECK: linalg.matvec
// CHECK-NOT: cim.vmm
func.func @bf16_k_tail(%weight: tensor<16x65xbf16>, %input: tensor<65xbf16>)
    -> tensor<16xbf16> {
  %zero = arith.constant dense<0.0> : tensor<16xbf16>
  %result = linalg.matvec
      ins(%weight, %input : tensor<16x65xbf16>, tensor<65xbf16>)
      outs(%zero : tensor<16xbf16>) -> tensor<16xbf16>
  return %result : tensor<16xbf16>
}
