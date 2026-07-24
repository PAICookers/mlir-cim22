// RUN: mlir-cim22-opt %s -split-input-file -verify-diagnostics

// Software-only evidence: VMM-N01 through VMM-N15 verifier/parser coverage.

// VMM-N01: input rank.
func.func @input_rank(%input: tensor<1x64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects input shape [64]}}
  %0 = cim.vmm %input, %weight : tensor<1x64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N02: input K, including dynamic K through one representative case.
func.func @input_dynamic_k(%input: tensor<?xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects input shape [64]}}
  %0 = cim.vmm %input, %weight : tensor<?xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N03: input dtype.
func.func @input_dtype(%input: tensor<64xi16>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{operand #0 must be ranked tensor of 8-bit signless integer values}}
  %0 = "cim.vmm"(%input, %weight) : (tensor<64xi16>, tensor<16x64xi8>) -> tensor<16xi21>
  return
}

// -----

// VMM-N04: weight rank.
func.func @weight_rank(%input: tensor<64xi8>, %weight: tensor<1024xi8>) {
  // expected-error@+1 {{expects weight shape [16, 64] in [N, K] order}}
  %0 = cim.vmm %input, %weight : tensor<64xi8>, tensor<1024xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N05: transposed weight axes.
func.func @weight_axes(%input: tensor<64xi8>, %weight: tensor<64x16xi8>) {
  // expected-error@+1 {{expects weight shape [16, 64] in [N, K] order}}
  %0 = cim.vmm %input, %weight : tensor<64xi8>, tensor<64x16xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N06: weight dtype.
func.func @weight_dtype(%input: tensor<64xi8>, %weight: tensor<16x64xi16>) {
  // expected-error@+1 {{operand #1 must be ranked tensor of 8-bit signless integer values}}
  %0 = "cim.vmm"(%input, %weight) : (tensor<64xi8>, tensor<16x64xi16>) -> tensor<16xi21>
  return
}

// -----

// VMM-N07: result shape, including static/dynamic/rank errors through one case.
func.func @result_shape(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects result shape [16]}}
  %0 = cim.vmm %input, %weight : tensor<64xi8>, tensor<16x64xi8> -> tensor<?xi21>
  return
}

// -----

// VMM-N08: result dtype.
func.func @result_dtype(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{result #0 must be ranked tensor of 21-bit signless integer values}}
  %0 = "cim.vmm"(%input, %weight) : (tensor<64xi8>, tensor<16x64xi8>) -> tensor<16xi32>
  return
}

// -----

// VMM-N09: arity.
func.func @operand_arity(%input: tensor<64xi8>) {
  // expected-error@+1 {{expected 2 operands, but found 1}}
  %0 = "cim.vmm"(%input) : (tensor<64xi8>) -> tensor<16xi21>
  return
}

// -----

// VMM-N10: tile identity is all-or-none.
func.func @partial_tile_identity(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{requires m_tile, n_tile, and k_tile together}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N11: schedule attributes are all-or-none.
func.func @partial_schedule(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{requires work_id and group_id together}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N12: a schedule requires logical tile identity.
func.func @schedule_without_tile(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{requires tile identity before schedule attributes}}
  %0 = cim.vmm %input, %weight {work_id = 0 : i64, group_id = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N13: logical identifiers are non-negative.
func.func @negative_tile(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects 'k_tile' to be non-negative}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 0 : i64, k_tile = -1 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @negative_schedule(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects 'group_id' to be non-negative}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64, work_id = 0 : i64, group_id = -1 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N14: identity and schedule attributes use i64.
func.func @wrong_tile_attribute_type(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects 'm_tile' to be an i64 attribute}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i32, n_tile = 0 : i64, k_tile = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @wrong_schedule_attribute_type(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects 'work_id' to be an i64 attribute}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64, work_id = 0 : i32, group_id = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// VMM-N15: target mapping attributes are all-or-none and schedule-owned.
func.func @partial_mapping(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{requires core_slot, macro_slot, and cim.mapping together}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64, work_id = 0 : i64, group_id = 0 : i64, core_slot = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @mapping_without_schedule(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{requires logical schedule before target mapping}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64, core_slot = 0 : i64, macro_slot = 0 : i64, cim.mapping = {}} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @wrong_mapping_type(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{expects 'cim.mapping' to be a dictionary attribute}}
  %0 = cim.vmm %input, %weight {m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64, work_id = 0 : i64, group_id = 0 : i64, core_slot = 0 : i64, macro_slot = 0 : i64, cim.mapping = 0 : i64} : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}
