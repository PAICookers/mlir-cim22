// RUN: mlir-cim22-opt %s --materialize-cim-static-weight-section > %t.first
// RUN: mlir-cim22-opt %s --materialize-cim-static-weight-section > %t.second
// RUN: diff %t.first %t.second
// RUN: FileCheck %s --check-prefix=COMMAND < %t.first
// RUN: mlir-cim22-opt %s --materialize-cim-static-weight-section --lower-cimframe-commands-to-packets --verify-cimframe | FileCheck %s --check-prefix=PACKET

// Software-only M2.4b evidence. HWSRC-046 accepts the layout; CTQ-020 still
// owns execution validation.

#mapping0 = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}
#mapping19 = {core_coord = array<i64: 3, 4>, destination = array<i64: 3, 4>, ingress = array<i64: 0, 0>, route = array<i64: 3, 1, 0, 0, 0, 0>, source = array<i64: 0, 0>}

// COMMAND-LABEL: module @static_weights
// COMMAND: cim.static_weight @ones = dense<-1> : tensor<16x64xi8>
// COMMAND-NEXT: cim.static_weight @zero = dense<0> : tensor<16x64xi8>
// COMMAND-NEXT: cimframe.write_int8_weights
// COMMAND-SAME: cim.provenance = {core_slot = 0 : i64, function = @first, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, mapping = {{.*}}, n_tile = 0 : i64, resource = @ones, work_id = 0 : i64}
// COMMAND-SAME: macro = 0 : i32
// COMMAND-SAME: route = array<i32: 0, 0, 0, 0, 0, 0>
// COMMAND-SAME: words = dense<-1> : tensor<256xi32>
// COMMAND-NEXT: cimframe.write_int8_weights
// COMMAND-SAME: cim.provenance = {core_slot = 0 : i64, function = @first, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, mapping = {{.*}}, n_tile = 0 : i64, resource = @ones, work_id = 1 : i64}
// COMMAND-SAME: macro = 1 : i32
// COMMAND-SAME: route = array<i32: 0, 0, 0, 0, 0, 0>
// COMMAND-SAME: words = dense<-1> : tensor<256xi32>
// COMMAND-NEXT: cimframe.write_int8_weights
// COMMAND-SAME: cim.provenance = {core_slot = 19 : i64, function = @second, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 1 : i64, macro_slot = 0 : i64, mapping = {{.*}}, n_tile = 2 : i64, resource = @zero, work_id = 0 : i64}
// COMMAND-SAME: macro = 0 : i32
// COMMAND-SAME: route = array<i32: 3, 1, 0, 0, 0, 0>
// COMMAND-SAME: words = dense<0> : tensor<256xi32>
// COMMAND-NEXT: func.func @first
// COMMAND-SAME: cim.execution_plan_schema_version = 1 : i64
// COMMAND: cim.configure_weight @ones
// COMMAND: cim.configure_weight @ones
// COMMAND: %[[READ0:.*]] = cim.readback
// COMMAND: %[[READ1:.*]] = cim.readback
// COMMAND: %[[WIDE0:.*]] = arith.extsi %[[READ0]]
// COMMAND: %[[WIDE1:.*]] = arith.extsi %[[READ1]]
// COMMAND: arith.addi %[[WIDE0]], %[[WIDE1]]
// COMMAND-LABEL: func.func @second
// COMMAND: cim.configure_weight @zero
// COMMAND-NOT: cimframe.start_int8_once

// PACKET-LABEL: module @static_weights
// PACKET: cimframe.control_int8_packet
// PACKET-SAME: cim.provenance = {core_slot = 0 : i64, function = @first, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, mapping = {{.*}}, n_tile = 0 : i64, resource = @ones, work_id = 0 : i64}
// PACKET-SAME: macro = 0 : i32
// PACKET-SAME: route = array<i32: 0, 0, 0, 0, 0, 0>
// PACKET-NEXT: cimframe.cim_int8_weight_packet
// PACKET-SAME: cim.provenance = {core_slot = 0 : i64, function = @first, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, mapping = {{.*}}, n_tile = 0 : i64, resource = @ones, work_id = 0 : i64}
// PACKET-SAME: route = array<i32: 0, 0, 0, 0, 0, 0>
// PACKET-NEXT: cimframe.control_int8_packet
// PACKET-SAME: macro = 1 : i32
// PACKET-NEXT: cimframe.cim_int8_weight_packet
// PACKET-NEXT: cimframe.control_int8_packet
// PACKET-SAME: cim.provenance = {core_slot = 19 : i64, function = @second, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 1 : i64, macro_slot = 0 : i64, mapping = {{.*}}, n_tile = 2 : i64, resource = @zero, work_id = 0 : i64}
// PACKET-SAME: macro = 0 : i32
// PACKET-SAME: route = array<i32: 3, 1, 0, 0, 0, 0>
// PACKET-NEXT: cimframe.cim_int8_weight_packet
// PACKET-SAME: cim.provenance = {core_slot = 19 : i64, function = @second, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 1 : i64, macro_slot = 0 : i64, mapping = {{.*}}, n_tile = 2 : i64, resource = @zero, work_id = 0 : i64}
// PACKET-NOT: cimframe.write_int8_weights

module @static_weights {
  cim.static_weight @ones = dense<-1> : tensor<16x64xi8>
  cim.static_weight @zero = dense<0> : tensor<16x64xi8>
  func.func @first(%input0: tensor<64xi8>, %input1: tensor<64xi8>) -> tensor<16xi32> attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
    cim.configure_input %input0 {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<64xi8>
    cim.configure_weight @ones {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    cim.configure_input %input1 {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64} : tensor<64xi8>
    cim.configure_weight @ones {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64}
    cim.dispatch {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
    cim.dispatch {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64}
    cim.once {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64}
    %read0 = cim.readback {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64} : tensor<16xi21>
    %read1 = cim.readback {cim.mapping = #mapping0, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 0 : i64, work_id = 1 : i64} : tensor<16xi21>
    cim.group_barrier {group_id = 0 : i64}
    %wide0 = arith.extsi %read0 : tensor<16xi21> to tensor<16xi32>
    %wide1 = arith.extsi %read1 : tensor<16xi21> to tensor<16xi32>
    %sum = arith.addi %wide0, %wide1 : tensor<16xi32>
    return %sum : tensor<16xi32>
  }
  func.func @second(%input: tensor<64xi8>) attributes {cim.execution_plan_schema_version = 1 : i64, cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64} {
    cim.configure_input %input {cim.mapping = #mapping19, core_slot = 19 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 1 : i64, macro_slot = 0 : i64, n_tile = 2 : i64, work_id = 0 : i64} : tensor<64xi8>
    cim.configure_weight @zero {cim.mapping = #mapping19, core_slot = 19 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 1 : i64, macro_slot = 0 : i64, n_tile = 2 : i64, work_id = 0 : i64}
    cim.dispatch {cim.mapping = #mapping19, core_slot = 19 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 1 : i64, macro_slot = 0 : i64, n_tile = 2 : i64, work_id = 0 : i64}
    cim.once {cim.mapping = #mapping19, core_slot = 19 : i64, group_id = 0 : i64}
    %read = cim.readback {cim.mapping = #mapping19, core_slot = 19 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 1 : i64, macro_slot = 0 : i64, n_tile = 2 : i64, work_id = 0 : i64} : tensor<16xi21>
    cim.group_barrier {group_id = 0 : i64}
    return
  }
}
