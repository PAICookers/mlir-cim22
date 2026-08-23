// RUN: mlir-cim22-opt %s -split-input-file --verify-cimframe -verify-diagnostics

// Software-only evidence: CF-N01 through CF-N08 and CF-N11.

module {
  // CF-N01: route arity below six.
  // expected-error@+1 {{expects route to contain 6 signed distances}}
  cimframe.start_int8_once {route = array<i32: 0, 0, 0, 0, 0>, macro = 0 : i32}
}

// -----

module {
  // CF-N01: route arity above six.
  // expected-error@+1 {{expects route to contain 6 signed distances}}
  cimframe.start_int8_once {route = array<i32: 0, 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
}

// -----

module {
  // CF-N02: lower representability bound.
  // expected-error@+1 {{expects each route distance in [-31, 31]}}
  cimframe.start_int8_once {route = array<i32: -32, 0, 0, 0, 0, 0>, macro = 0 : i32}
}

// -----

module {
  // CF-N02: upper representability bound.
  // expected-error@+1 {{expects each route distance in [-31, 31]}}
  cimframe.start_int8_once {route = array<i32: 32, 0, 0, 0, 0, 0>, macro = 0 : i32}
}

// -----

module {
  // CF-N03: macro below zero.
  // expected-error@+1 {{expects macro to be 0 or 1}}
  cimframe.start_int8_once {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = -1 : i32}
}

// -----

module {
  // CF-N03: macro above one.
  // expected-error@+1 {{expects macro to be 0 or 1}}
  cimframe.start_int8_once {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 2 : i32}
}

// -----

module {
  // CF-N04: required route is absent.
  // expected-error@+1 {{requires attribute 'route'}}
  "cimframe.start_int8_once"() {macro = 0 : i32} : () -> ()
}

// -----

module {
  // CF-N04: mode is not a command protocol field.
  // expected-error@+1 {{unexpected protocol attribute 'mode'}}
  cimframe.start_int8_once {
    route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32, mode = 0 : i32
  }
}

// -----

module {
  // CF-N04: raw type is not a control-packet protocol field.
  // expected-error@+1 {{unexpected protocol attribute 'type'}}
  cimframe.control_int8_packet {
    route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32, type = 8 : i32
  }
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>}
}

// -----

module {
  // CF-N04: raw run state is not a work-packet protocol field.
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  // expected-error@+1 {{unexpected protocol attribute 'run'}}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, run = 1 : i32}
}

// -----

module {
  // CF-N04: work packet does not own macro selection.
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  // expected-error@+1 {{unexpected protocol attribute 'macro'}}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
}

// -----

// CF-N05: command and packet stages are module-global and cannot mix.
// expected-error@+1 {{cannot mix cimframe command and packet stages in one module}}
module {
  cimframe.start_int8_once {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>}
}

// -----

module {
  // CF-N06: orphan work packet.
  // expected-error@+1 {{expects work_once_packet immediately preceded by control_int8_packet}}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>}
}

// -----

module {
  // CF-N06: reversed pair.
  // expected-error@+1 {{expects work_once_packet immediately preceded by control_int8_packet}}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>}
  // expected-error@+1 {{expects control_int8_packet immediately followed by work_once_packet or cim_int8_weight_packet}}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
}

// -----

module {
  // CF-N07: every direct operation participates in adjacency.
  // expected-error@+1 {{expects control_int8_packet immediately followed by work_once_packet or cim_int8_weight_packet}}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  func.func private @separator()
  // expected-error@+1 {{expects work_once_packet immediately preceded by control_int8_packet}}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>}
}

// -----

module {
  // CF-N08: paired routes must be identical.
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  // expected-error@+1 {{expects control/work routes to match}}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 1>}
}

// -----

module {
  func.func @nested_command() {
    // CF-N11: command must be a direct ModuleOp child.
    // expected-error@+1 {{expects parent op 'builtin.module'}}
    cimframe.start_int8_once {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
    return
  }
}

// -----

module {
  func.func @nested_control_packet() {
    // CF-N11: control packet must be a direct ModuleOp child.
    // expected-error@+1 {{expects parent op 'builtin.module'}}
    cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
    return
  }
}

// -----

module {
  func.func @nested_work_packet() {
    // CF-N11: work packet must be a direct ModuleOp child.
    // expected-error@+1 {{expects parent op 'builtin.module'}}
    cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>}
    return
  }
}


// -----

#words255 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]> : tensor<255xi32>
module {
  // M2.3: weight payload below the frozen semantic address count.
  // expected-error@+1 {{expects words to contain 256 i32 values}}
  cimframe.write_int8_weights {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32, words = #words255}
}

// -----

#words256 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]> : tensor<256xi32>

module {
  // M2.3: write command directly rejects an unrepresentable route.
  // expected-error@+1 {{expects each route distance in [-31, 31]}}
  cimframe.write_int8_weights {route = array<i32: 32, 0, 0, 0, 0, 0>, macro = 0 : i32, words = #words256}
}

// -----

#words256 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]> : tensor<256xi32>

module {
  // M2.3: write command directly rejects an unknown Macro selector.
  // expected-error@+1 {{expects macro to be 0 or 1}}
  cimframe.write_int8_weights {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 2 : i32, words = #words256}
}

// -----

#words257 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0
]> : tensor<257xi32>

module {
  // M2.3: weight payload above the frozen semantic address count.
  // expected-error@+1 {{expects words to contain 256 i32 values}}
  cimframe.write_int8_weights {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32, words = #words257}
}

// -----

module {
  // M2.3: words are required.
  // expected-error@+1 {{requires attribute 'words'}}
  "cimframe.write_int8_weights"() {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32} : () -> ()
}

// -----

#words256 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]> : tensor<256xi32>

module {
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  // M2.3: the compound packet does not own Macro selection.
  // expected-error@+1 {{unexpected protocol attribute 'macro'}}
  cimframe.cim_int8_weight_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, words = #words256, macro = 0 : i32}
}

// -----

#words256 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]> : tensor<256xi32>

// M2.3: command and packet operations cannot share one module stage.
// expected-error@+1 {{cannot mix cimframe command and packet stages in one module}}
module {
  cimframe.write_int8_weights {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32, words = #words256}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  cimframe.cim_int8_weight_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, words = #words256}
}

// -----

#words256 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]> : tensor<256xi32>

module {
  // M2.3: orphan/reversed weight consumer.
  // expected-error@+1 {{expects cim_int8_weight_packet immediately preceded by control_int8_packet}}
  cimframe.cim_int8_weight_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, words = #words256}

  // M2.3: any direct operation between control and consumer is illegal.
  // expected-error@+1 {{expects control_int8_packet immediately followed by work_once_packet or cim_int8_weight_packet}}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  func.func private @separator()
  // expected-error@+1 {{expects cim_int8_weight_packet immediately preceded by control_int8_packet}}
  cimframe.cim_int8_weight_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, words = #words256}

  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  // M2.3: paired routes must be identical.
  // expected-error@+1 {{expects control/weight routes to match}}
  cimframe.cim_int8_weight_packet {route = array<i32: 0, 0, 0, 0, 0, 1>, words = #words256}

  // M2.3: controls cannot be consecutive or shared.
  // expected-error@+1 {{expects control_int8_packet immediately followed by work_once_packet or cim_int8_weight_packet}}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 1 : i32}
  cimframe.work_once_packet {route = array<i32: 0, 0, 0, 0, 0, 0>}
}

// -----

#words256 = dense<[
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]> : tensor<256xi32>

module {
  func.func @nested_weight_command() {
    // M2.3: command must be a direct ModuleOp child.
    // expected-error@+1 {{expects parent op 'builtin.module'}}
    cimframe.write_int8_weights {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32, words = #words256}
    return
  }
  func.func @nested_weight_packet() {
    // M2.3: packet must be a direct ModuleOp child.
    // expected-error@+1 {{expects parent op 'builtin.module'}}
    cimframe.cim_int8_weight_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, words = #words256}
    return
  }
}
