// Explicit supplier-model BF16 VMM, not standard BF16 matrix multiplication.
// The two Macro weight tiles reuse test/Integration/ExecutionPlan/bf16-runner.mlir.
// Optional full-array case: 40 independent VMM tiles, two Macros per core.
// The reference exporter supplies all eight Cache rows to every active Macro.
func.func @bf16_run(%input0: tensor<64xbf16>, %input1: tensor<64xbf16>) -> tensor<20x32xbf16> {
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
  %empty = tensor.empty() : tensor<20x32xbf16>
  %v0 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v1 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 0 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v2 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 1 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v3 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 1 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v4 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 2 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v5 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 2 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v6 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 3 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v7 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 3 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v8 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 4 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v9 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 4 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v10 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 5 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v11 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 5 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v12 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 6 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v13 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 6 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v14 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 7 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v15 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 7 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v16 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 8 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v17 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 8 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v18 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 9 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v19 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 9 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v20 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 10 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v21 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 10 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v22 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 11 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v23 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 11 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v24 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 12 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v25 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 12 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v26 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 13 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v27 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 13 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v28 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 14 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v29 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 14 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v30 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 15 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v31 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 15 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v32 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 16 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v33 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 16 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v34 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 17 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v35 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 17 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v36 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 18 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v37 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 18 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v38 = cim.vmm %input0, %weight0 {cim.transaction_idx = 0 : i64, m_tile = 19 : i64, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %v39 = cim.vmm %input1, %weight1 {cim.transaction_idx = 0 : i64, m_tile = 19 : i64, n_tile = 1 : i64, k_tile = 0 : i64} : tensor<64xbf16>, tensor<16x64xbf16> -> tensor<16xbf16>
  %r0 = tensor.insert_slice %v0 into %empty[0, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r1 = tensor.insert_slice %v1 into %r0[0, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r2 = tensor.insert_slice %v2 into %r1[1, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r3 = tensor.insert_slice %v3 into %r2[1, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r4 = tensor.insert_slice %v4 into %r3[2, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r5 = tensor.insert_slice %v5 into %r4[2, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r6 = tensor.insert_slice %v6 into %r5[3, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r7 = tensor.insert_slice %v7 into %r6[3, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r8 = tensor.insert_slice %v8 into %r7[4, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r9 = tensor.insert_slice %v9 into %r8[4, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r10 = tensor.insert_slice %v10 into %r9[5, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r11 = tensor.insert_slice %v11 into %r10[5, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r12 = tensor.insert_slice %v12 into %r11[6, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r13 = tensor.insert_slice %v13 into %r12[6, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r14 = tensor.insert_slice %v14 into %r13[7, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r15 = tensor.insert_slice %v15 into %r14[7, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r16 = tensor.insert_slice %v16 into %r15[8, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r17 = tensor.insert_slice %v17 into %r16[8, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r18 = tensor.insert_slice %v18 into %r17[9, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r19 = tensor.insert_slice %v19 into %r18[9, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r20 = tensor.insert_slice %v20 into %r19[10, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r21 = tensor.insert_slice %v21 into %r20[10, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r22 = tensor.insert_slice %v22 into %r21[11, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r23 = tensor.insert_slice %v23 into %r22[11, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r24 = tensor.insert_slice %v24 into %r23[12, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r25 = tensor.insert_slice %v25 into %r24[12, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r26 = tensor.insert_slice %v26 into %r25[13, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r27 = tensor.insert_slice %v27 into %r26[13, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r28 = tensor.insert_slice %v28 into %r27[14, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r29 = tensor.insert_slice %v29 into %r28[14, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r30 = tensor.insert_slice %v30 into %r29[15, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r31 = tensor.insert_slice %v31 into %r30[15, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r32 = tensor.insert_slice %v32 into %r31[16, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r33 = tensor.insert_slice %v33 into %r32[16, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r34 = tensor.insert_slice %v34 into %r33[17, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r35 = tensor.insert_slice %v35 into %r34[17, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r36 = tensor.insert_slice %v36 into %r35[18, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r37 = tensor.insert_slice %v37 into %r36[18, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r38 = tensor.insert_slice %v38 into %r37[19, 0] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  %r39 = tensor.insert_slice %v39 into %r38[19, 16] [1, 16] [1, 1] : tensor<16xbf16> into tensor<20x32xbf16>
  return %r39 : tensor<20x32xbf16>
}
