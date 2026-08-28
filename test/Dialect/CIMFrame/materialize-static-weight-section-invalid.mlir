// RUN: mlir-cim22-opt %s -split-input-file --materialize-cim-static-weight-section -verify-diagnostics

// Software-only M2.4b diagnostics.

#mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}

// expected-error@+1 {{materialize-cim-static-weight-section rejects pre-existing top-level cimframe operations}}
module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>
  cimframe.start_int8_once {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  func.func @preexisting(%input: tensor<64xi8>) attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
    cim.configure_weight @weight {cim.segment_id = 0 : i64, cim.mapping = #mapping, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    return
  }
}

// -----

#mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}
module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>
  // expected-error@+1 {{materialize-cim-static-weight-section requires cim.execution_plan_schema_version = 1 : i64}}
  func.func @missing_schema() attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
    cim.configure_weight @weight {cim.segment_id = 0 : i64, cim.mapping = #mapping, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    return
  }
}

// -----

#mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}
module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>
  // expected-error@+1 {{materialize-cim-static-weight-section requires cim.target_profile = 'cim22-4x5-v1'}}
  func.func @bad_profile() attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "other", cim.target_profile_version = 1 : i64} {
    cim.configure_weight @weight {cim.segment_id = 0 : i64, cim.mapping = #mapping, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    return
  }
}

// -----

#mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}
module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>
  func.func @extra_provenance() attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
    // expected-error@+1 {{materialize-cim-static-weight-section rejects unexpected configure_weight attribute 'extra'}}
    cim.configure_weight @weight {cim.segment_id = 0 : i64, cim.mapping = #mapping, core_slot = 0 : i64, extra = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    return
  }
}

// -----

#copy_mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 1, 0, 0>, source = array<i64: 0, 0>}
module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>
  func.func @copy_route() attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
    // expected-error@+1 {{expects onecast route with zero Copy fields}}
    cim.configure_weight @weight {cim.segment_id = 0 : i64, cim.mapping = #copy_mapping, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    return
  }
}
