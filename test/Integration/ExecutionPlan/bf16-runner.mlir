// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,materialize-cim-static-weight-section,lower-cimframe-commands-to-packets,verify-cimframe)' -o %t.packet
// RUN: mlir-cim22-cim-executable-runner-test %t.packet bf16 | FileCheck %s

// Candidate shared-exponent arithmetic, with fixed independently derived
// scalar products. Macro 0 lane 0: 1.5 * 2^-126 = 0x00c0;
// Macro 1 lane 0: 2.25 * 2^-110 = 0x0910. Each lane doubles the
// magnitude and alternates the sign. Historical board demo goldens are separate.
// CHECK: BF16 run=0 macro=0 00c0 8140 01c0 8240 02c0 8340 03c0 8440 04c0 8540 05c0 8640 06c0 8740 07c0 8840
// CHECK-NEXT: BF16 run=0 macro=1 0910 8990 0a10 8a90 0b10 8b90 0c10 8c90 0d10 8d90 0e10 8e90 0f10 8f90 1010 9090
// CHECK-NEXT: BF16 run=1 macro=0 00c0 8140 01c0 8240 02c0 8340 03c0 8440 04c0 8540 05c0 8640 06c0 8740 07c0 8840
// CHECK-NEXT: BF16 run=1 macro=1 0910 8990 0a10 8a90 0b10 8b90 0c10 8c90 0d10 8d90 0e10 8e90 0f10 8f90 1010 9090
func.func @bf16_run(%input0: tensor<64xbf16>, %input1: tensor<64xbf16>) -> (tensor<16xbf16>, tensor<16xbf16>) {
  %small0 = arith.constant dense<[
    [0x0080, 0x8080], [0x8100, 0x0100],
    [0x0180, 0x8180], [0x8200, 0x0200],
    [0x0280, 0x8280], [0x8300, 0x0300],
    [0x0380, 0x8380], [0x8400, 0x0400],
    [0x0480, 0x8480], [0x8500, 0x0500],
    [0x0580, 0x8580], [0x8600, 0x0600],
    [0x0680, 0x8680], [0x8700, 0x0700],
    [0x0780, 0x8780], [0x8800, 0x0800]
  ]> : tensor<16x2xbf16>
  %small1 = arith.constant dense<[
    [0x08c0, 0x88c0], [0x8940, 0x0940],
    [0x09c0, 0x89c0], [0x8a40, 0x0a40],
    [0x0ac0, 0x8ac0], [0x8b40, 0x0b40],
    [0x0bc0, 0x8bc0], [0x8c40, 0x0c40],
    [0x0cc0, 0x8cc0], [0x8d40, 0x0d40],
    [0x0dc0, 0x8dc0], [0x8e40, 0x0e40],
    [0x0ec0, 0x8ec0], [0x8f40, 0x0f40],
    [0x0fc0, 0x8fc0], [0x9040, 0x1040]
  ]> : tensor<16x2xbf16>
  %zero = arith.constant 0.0 : bf16
  %weight0 = tensor.pad %small0 low[0, 0] high[0, 62] {
  ^bb0(%i: index, %j: index):
    tensor.yield %zero : bf16
  } : tensor<16x2xbf16> to tensor<16x64xbf16>
  %weight1 = tensor.pad %small1 low[0, 0] high[0, 62] {
  ^bb0(%i: index, %j: index):
    tensor.yield %zero : bf16
  } : tensor<16x2xbf16> to tensor<16x64xbf16>
  %first = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %second = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  return %first, %second : tensor<16xbf16>, tensor<16xbf16>
}
