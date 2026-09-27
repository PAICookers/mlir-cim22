// Explicit supplier-model BF16 VMM, not standard BF16 matrix multiplication.
// The two weight templates reuse Integration/ExecutionPlan/bf16-runner.mlir.
// One transaction: cores 0-2 use both Macros, core 3 uses only Macro 0.
// The reference exporter supplies all eight Cache rows to each listed Macro.
func.func @bf16_run(%input0: tensor<64xbf16>, %input1: tensor<64xbf16>) -> tensor<112xbf16> {
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
  %empty = tensor.empty() : tensor<112xbf16>
  %v0 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v1 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v2 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 2 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v3 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 3 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v4 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 4 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v5 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 5 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v6 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 6 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %r0 = tensor.insert_slice %v0 into %empty[0] [16] [1] : tensor<16xbf16> into tensor<112xbf16>
  %r1 = tensor.insert_slice %v1 into %r0[16] [16] [1] : tensor<16xbf16> into tensor<112xbf16>
  %r2 = tensor.insert_slice %v2 into %r1[32] [16] [1] : tensor<16xbf16> into tensor<112xbf16>
  %r3 = tensor.insert_slice %v3 into %r2[48] [16] [1] : tensor<16xbf16> into tensor<112xbf16>
  %r4 = tensor.insert_slice %v4 into %r3[64] [16] [1] : tensor<16xbf16> into tensor<112xbf16>
  %r5 = tensor.insert_slice %v5 into %r4[80] [16] [1] : tensor<16xbf16> into tensor<112xbf16>
  %r6 = tensor.insert_slice %v6 into %r5[96] [16] [1] : tensor<16xbf16> into tensor<112xbf16>
  return %r6 : tensor<112xbf16>
}
