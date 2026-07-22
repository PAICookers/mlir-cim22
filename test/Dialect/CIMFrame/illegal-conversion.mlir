// RUN: not mlir-cim22-opt %s --allow-unregistered-dialect 2>&1 | FileCheck %s

// Software-only evidence: CF-N09. The registered dialect is closed-world.

// CHECK: unregistered operation 'cimframe.unhandled' found in dialect ('cimframe') that does not allow unknown operations
module {
  "cimframe.unhandled"() : () -> ()
}
