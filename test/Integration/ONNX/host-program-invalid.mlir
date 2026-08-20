// RUN: mlir-cim22-opt %s -split-input-file -verify-diagnostics -verify-cim-execution-plan -o /dev/null

module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>

  func.func @consume_before_barrier(%input: tensor<64xi8>)
      attributes {cim.execution_plan_schema_version = 1 : i64} {
    cim.configure_input %input {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>
    cim.configure_weight @weight {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    cim.dispatch {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    cim.once {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64}
    %readback = cim.readback {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<16xi21>
    // expected-error@+1 {{CIM execution plan found an interleaved Host or out-of-order CIM operation; expected cim.group_barrier}}
    %extended = arith.extsi %readback : tensor<16xi21> to tensor<16xi32>
    cim.group_barrier {group_id = 0 : i64}
    return
  }
}

// -----

module {
  cim.static_weight @weight = dense<0> : tensor<16x64xi8>

  func.func @host_inside_segment(%input: tensor<64xi8>)
      attributes {cim.execution_plan_schema_version = 1 : i64} {
    cim.configure_input %input {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>
    cim.configure_weight @weight {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    // expected-error@+1 {{CIM execution plan found an interleaved Host or out-of-order CIM operation; expected cim.dispatch}}
    %zero = arith.constant dense<0> : tensor<16xi32>
    cim.dispatch {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    cim.once {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64}
    %readback = cim.readback {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<16xi21>
    cim.group_barrier {group_id = 0 : i64}
    return
  }
}
