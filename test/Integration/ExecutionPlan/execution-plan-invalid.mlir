// RUN: mlir-cim22-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error@+1 {{expects value type tensor<16x64xi8>}}
  cim.static_weight @bad = dense<0> : tensor<8x64xi8>
}

// -----

module {
  // expected-error@+1 {{must not carry per-work execution-plan provenance}}
  cim.static_weight @extra_provenance = dense<0> : tensor<16x64xi8> {work_id = 0 : i64}
}

// -----

module {
  func.func @missing_provenance(%input: tensor<64xi8>) {
    // expected-error@+1 {{requires complete execution-plan provenance}}
    cim.configure_input %input {work_id = 0 : i64} : tensor<64xi8>
    return
  }
}

// -----

module {
  func.func @copy_route(%input: tensor<64xi8>) {
    // expected-error@+1 {{expects onecast route with zero Copy fields}}
    cim.configure_input %input {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 1, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>
    return
  }
}

// -----

module {
  func.func @macro_once() {
    // expected-error@+1 {{must carry only group_id, core_slot, and cim.mapping provenance}}
    cim.once {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, macro_slot = 0 : i64}
    return
  }
}

// -----

module {
  func.func @work_once() {
    // expected-error@+1 {{must carry only group_id, core_slot, and cim.mapping provenance}}
    cim.once {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, work_id = 0 : i64}
    return
  }
}

// -----

module {
  func.func @wrong_readback() {
    // expected-error@+1 {{must be ranked tensor of 21-bit signless integer values}}
    %0 = cim.readback {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<16xi32>
    return
  }
}

// -----

module {
  func.func @missing_weight() {
    // expected-error@+1 {{expects resource to reference cim.static_weight}}
    cim.configure_weight @missing {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    return
  }
}

// -----

module {
  func.func @negative_barrier() {
    // expected-error@+1 {{expects non-negative i64 group_id}}
    cim.group_barrier {group_id = -1 : i64}
    return
  }
}

// -----

module {
  func.func @work_barrier() {
    // expected-error@+1 {{must carry only group_id provenance}}
    cim.group_barrier {group_id = 0 : i64, work_id = 0 : i64}
    return
  }
}
