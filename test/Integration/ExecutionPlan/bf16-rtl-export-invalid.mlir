// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,materialize-cim-static-weight-section,lower-cimframe-commands-to-packets,verify-cimframe)' -o %t.packet
// RUN: not mlir-cim22-opt %t.packet --export-cim-rtl-artifacts='output-dir=%t.artifacts input-cache-file=%t.missing' -o /dev/null 2>&1 | FileCheck %s
// RUN: test ! -e %t.artifacts

// Reject the mode before reading input bytes or writing any INT8 artifacts.
// CHECK: error: RTL artifact export does not support BF16 transactions
func.func @bf16_dual_macro(%input: tensor<64xbf16>) -> (tensor<16xbf16>, tensor<16xbf16>) {
  %weight = arith.constant dense<1.0> : tensor<16x64xbf16>
  %first = cim.vmm %input, %weight {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %second = cim.vmm %input, %weight {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  return %first, %second : tensor<16xbf16>, tensor<16xbf16>
}
