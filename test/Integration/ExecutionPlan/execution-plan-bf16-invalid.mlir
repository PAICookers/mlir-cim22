// RUN: mlir-cim22-opt %s -split-input-file -verify-cim-execution-plan -verify-diagnostics

module {
  cim.static_weight @weight = dense<0.0> : tensor<16x64xbf16>
  func.func @mixed_work_mode(%input: tensor<64xi8>) attributes {cim.execution_plan_schema_version = 1 : i64} {
    %0 = "cim.transaction"(%input) ({
    ^bb0(%arg0: tensor<64xi8>):
      cim.configure_input %arg0 {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>
      // expected-error@+1 {{expects static weight element type to match configured input}}
      cim.configure_weight @weight {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.dispatch {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.once {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64}
      %read = cim.readback {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<16xi21>
      cim.group_barrier {group_id = 0 : i64}
      "cim.yield"(%read) : (tensor<16xi21>) -> ()
    }) {cim.transaction_idx = 0 : i64} : (tensor<64xi8>) -> tensor<16xi21>
    return
  }
}

// -----

module {
  cim.static_weight @weight = dense<0.0> : tensor<16x64xbf16>
  func.func @bf16_i21_readback(%input: tensor<64xbf16>) attributes {cim.execution_plan_schema_version = 1 : i64} {
    %0 = "cim.transaction"(%input) ({
    ^bb0(%arg0: tensor<64xbf16>):
      cim.configure_input %arg0 {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xbf16>
      cim.configure_weight @weight {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.dispatch {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
      cim.once {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64}
      // expected-error@+1 {{result element type does not match configured data mode}}
      %read = cim.readback {cim.mapping = {route = array<i64: 0, 0, 0, 0, 0, 0>}, core_idx = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_idx = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<16xi21>
      cim.group_barrier {group_id = 0 : i64}
      "cim.yield"(%read) : (tensor<16xi21>) -> ()
    }) {cim.transaction_idx = 0 : i64} : (tensor<64xbf16>) -> tensor<16xi21>
    return
  }
}
