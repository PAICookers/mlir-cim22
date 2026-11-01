// RUN: mlir-cim22-opt %s --split-cim-bf16-signs='bf16-compensation=legacy' | FileCheck %s --check-prefix=LEGACY
// RUN: mlir-cim22-opt %s --split-cim-bf16-signs='bf16-compensation=none' | FileCheck %s --check-prefix=NONE
// RUN: not mlir-cim22-opt %s --split-cim-bf16-signs='bf16-compensation=unknown' 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not mlir-cim22-opt %s --cim-bf16-pipeline='bf16-compensation=unknown' 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: mlir-cim22-opt %s --split-cim-bf16-signs | not mlir-cim22-opt --split-cim-bf16-signs='bf16-compensation=legacy' 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: mlir-cim22-opt %s --cim-bf16-pipeline='stage=packet bf16-compensation=legacy' | FileCheck %s --check-prefix=PACKET

func.func @signed(%x: tensor<64xf32>) -> tensor<1xf32> {
  %w = arith.constant dense<-0.5> : tensor<1x64xf32>
  %z = arith.constant dense<0.0> : tensor<1xf32>
  %r = linalg.matvec ins(%w, %x : tensor<1x64xf32>, tensor<64xf32>) outs(%z : tensor<1xf32>) -> tensor<1xf32>
  return %r : tensor<1xf32>
}
// LEGACY: module attributes {cim.bf16_compensation = "legacy"}
// LEGACY: dense<-5.000000e-01>
// LEGACY-COUNT-1: linalg.matvec
// LEGACY-NOT: cim.bf16_branch
// NONE: module attributes {cim.bf16_compensation = "none"}
// NONE-COUNT-1: linalg.matvec
// NONE-NOT: cim.bf16_branch
// INVALID: BF16 compensation must be sign-split, legacy or none
// CONFLICT: conflicting BF16 compensation mode
// PACKET: module attributes {cim.bf16_compensation = "legacy",
// PACKET: cimframe.control_bf16_packet
