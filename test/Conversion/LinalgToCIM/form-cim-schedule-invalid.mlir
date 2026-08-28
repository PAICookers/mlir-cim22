// RUN: mlir-cim22-opt %s -split-input-file -verify-diagnostics --pass-pipeline='builtin.module(func.func(materialize-cim-schedule))'

// Software-only rejection coverage for the provisional M5.1 schedule policy.

// expected-error@+1 {{materialize-cim-schedule rejects mixed scheduled and unscheduled cim.vmm operations}}
func.func @mixed_schedule(%input: tensor<64xi8>,
                          %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64, work_id = 0 : i64, group_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  %1 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 1 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @wrong_formula(%input: tensor<64xi8>,
                         %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64, work_id = 0 : i64, group_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  // expected-error@+1 {{expects 'group_id' = 0, but got 1}}
  %1 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 1 : i64, work_id = 1 : i64, group_id = 1 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @duplicate_identity(%input: tensor<64xi8>,
                              %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  // expected-error@+1 {{rejects duplicate tile identity}}
  %1 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+1 {{requires a complete rectangular tile identity space}}
func.func @incomplete_rectangle(%input: tensor<64xi8>,
                                %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  %1 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 1 : i64,
      k_tile = 1 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @noncontiguous_order(%input: tensor<64xi8>,
                               %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{requires contiguous M-major/N-major/K-minor tile order}}
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 1 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  %1 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+1 {{cannot represent nested control flow}}
func.func @nested_vmm(%input: tensor<64xi8>,
                      %weight: tensor<16x64xi8>) {
  %generated = tensor.generate {
    ^bb0(%index: index):
      %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
          k_tile = 0 : i64}
          : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
      %zero = arith.constant 0 : i8
      tensor.yield %zero : i8
  } : tensor<1xi8>
  return
}

// -----

func.func @dependent_pair(%input: tensor<64xi8>,
                          %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  %narrow = arith.trunci %0 : tensor<16xi21> to tensor<16xi8>
  %dependent_input = tensor.concat dim(0) %narrow, %narrow, %narrow, %narrow
      : (tensor<16xi8>, tensor<16xi8>, tensor<16xi8>, tensor<16xi8>)
      -> tensor<64xi8>
  // expected-error@+1 {{cannot pair SSA-dependent VMM work}}
  %1 = cim.vmm %dependent_input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64,
      n_tile = 0 : i64, k_tile = 1 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}
