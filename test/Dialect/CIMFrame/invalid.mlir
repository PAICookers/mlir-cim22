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
  // expected-error@+1 {{expects control_int8_packet immediately followed by work_once_packet}}
  cimframe.control_int8_packet {route = array<i32: 0, 0, 0, 0, 0, 0>, macro = 0 : i32}
}

// -----

module {
  // CF-N07: every direct operation participates in adjacency.
  // expected-error@+1 {{expects control_int8_packet immediately followed by work_once_packet}}
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
