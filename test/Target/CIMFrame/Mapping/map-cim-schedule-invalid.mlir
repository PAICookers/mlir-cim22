// RUN: mlir-cim22-opt %s -split-input-file -verify-diagnostics --pass-pipeline='builtin.module(func.func(map-cim-schedule))'

// Software-only M4.0 rejection coverage.

// expected-error@+2 {{map-cim-schedule requires complete tile and logical schedule identity}}
func.func @missing_schedule(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64, k_tile = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

func.func @wrong_group(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  // expected-error@+1 {{map-cim-schedule expects group_id = 0 for work_id = 0}}
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64, work_id = 0 : i64, group_id = 1 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+1 {{map-cim-schedule rejects stale function target attributes}}
func.func @stale_function_attrs(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) attributes {cim.target_profile = "cim22-4x5-v1"} {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, m_tile = 0 : i64, n_tile = 0 : i64,
      k_tile = 0 : i64, work_id = 0 : i64, group_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+1 {{map-cim-schedule requires cim.target_profile = 'cim22-4x5-v1'}}
func.func @unsupported_profile(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "unknown", cim.target_profile_version = 1 : i64} {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+1 {{map-cim-schedule requires cim.target_profile_version = 1 : i64}}
func.func @unsupported_version(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 2 : i64} {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+1 {{map-cim-schedule requires all target profile and policy attributes together}}
func.func @partial_mapped_function_attrs(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) attributes {cim.target_profile = "cim22-4x5-v1"} {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+2 {{map-cim-schedule rejects stale target resource mapping}}
func.func @stale_resource(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 1 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+2 {{map-cim-schedule rejects invalid or stale cim.mapping}}
func.func @route_mismatch(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 1, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+2 {{map-cim-schedule rejects invalid or stale cim.mapping}}
func.func @copy_route(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 1, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}

// -----

// expected-error@+1 {{map-cim-schedule rejects mixed mapped and unmapped cim.vmm work}}
func.func @mixed_mapping(%input: tensor<64xi8>, %weight: tensor<16x64xi8>) {
  %0 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  %1 = cim.vmm %input, %weight {cim.segment_id = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64,
      m_tile = 0 : i64, n_tile = 0 : i64, work_id = 1 : i64}
      : tensor<64xi8>, tensor<16x64xi8> -> tensor<16xi21>
  return
}
