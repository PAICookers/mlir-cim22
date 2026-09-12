// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(func.func(materialize-cim-schedule,map-cim-schedule),materialize-cim-execution-plan,materialize-cim-static-weight-section,lower-cimframe-commands-to-packets,verify-cimframe)' -o %t.packet
// RUN: mlir-cim22-cim-executable-runner-test %t.packet bf16 | FileCheck %s

// NUM-018/019 supplier oracle, not IEEE BF16 multiplication:
// Macro 0 has lane exponents 1..16 and mantissas alternating [64,-64]/[-64,64].
// With input [64,-96], lane sums are +/-10240, shift=6, fraction=0x20.
// Macro 1 has exponents 17..32 and mantissas [96,-96]/[-96,96].
// With input [32,-64], lane sums are +/-9216, shift=6, fraction=0x10.
// Output exponent is weight_exp + input_exp(127) + 1 + shift.
// Fixed expectations do not call the production prealignment/layout helpers.
// CHECK: BF16 run=0 macro=0 43a0 c420 44a0 c520 45a0 c620 46a0 c720 47a0 c820 48a0 c920 49a0 ca20 4aa0 cb20
// CHECK-NEXT: BF16 run=0 macro=1 4b90 cc10 4c90 cd10 4d90 ce10 4e90 cf10 4f90 d010 5090 d110 5190 d210 5290 d310
// CHECK-NEXT: BF16 run=1 macro=0 43a0 c420 44a0 c520 45a0 c620 46a0 c720 47a0 c820 48a0 c920 49a0 ca20 4aa0 cb20
// CHECK-NEXT: BF16 run=1 macro=1 4b90 cc10 4c90 cd10 4d90 ce10 4e90 cf10 4f90 d010 5090 d110 5190 d210 5290 d310
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
