// RUN: echo "" | mlir-cim22-opt --show-dialects | FileCheck %s

// Software-only evidence: confirms the tool registers the logical CIM dialect.
// CHECK: Available Dialects:
// CHECK-SAME: cim
