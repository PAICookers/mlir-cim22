// RUN: mlir-cim22-opt %s --split-input-file --split-cim-bf16-signs -verify-diagnostics

func.func @nan_weight(%x: tensor<2xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<0x7FC00000> : tensor<1x2xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  // expected-error@+1 {{BF16 correction requires finite fixed weights}}
  %r = linalg.matvec ins(%w, %x : tensor<1x2xf32>, tensor<2xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}

// -----

func.func @inf_weight(%x: tensor<2xbf16>) -> tensor<1xbf16> {
  %w = arith.constant dense<0x7F80> : tensor<1x2xbf16>
  %z = arith.constant dense<0.0> : tensor<1xbf16>
  // expected-error@+1 {{BF16 correction requires finite fixed weights}}
  %r = linalg.matvec ins(%w, %x : tensor<1x2xbf16>, tensor<2xbf16>) outs(%z : tensor<1xbf16>) -> tensor<1xbf16>
  return %r : tensor<1xbf16>
}
