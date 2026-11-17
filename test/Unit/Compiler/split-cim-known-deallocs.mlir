// RUN: mlir-cim22-opt %s --cim-simplify-independent-deallocs | FileCheck %s

func.func @distinct(%cond: i1) {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  // CHECK-LABEL: func.func @distinct
  // CHECK-COUNT-2: bufferization.dealloc
  bufferization.dealloc (%a, %b : memref<4xf32>, memref<4xf32>)
    if (%cond, %cond)
  return
}

func.func @duplicate(%cond: i1) {
  %a = memref.alloc() : memref<4xf32>
  // CHECK-LABEL: func.func @duplicate
  // CHECK: bufferization.dealloc
  // CHECK-SAME: memref<4xf32>, memref<4xf32>) if (%arg0, %arg0)
  bufferization.dealloc (%a, %a : memref<4xf32>, memref<4xf32>)
    if (%cond, %cond)
  return
}

func.func @views(%cond: i1) {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  %av = memref.cast %a : memref<4xf32> to memref<?xf32>
  %bv = memref.cast %b : memref<4xf32> to memref<?xf32>
  // CHECK-LABEL: func.func @views
  // CHECK-COUNT-2: bufferization.dealloc
  bufferization.dealloc (%av, %bv : memref<?xf32>, memref<?xf32>)
    if (%cond, %cond)
  return
}

func.func @mixed_aliases(%cond0: i1, %cond1: i1, %cond2: i1) {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  %av = memref.cast %a : memref<4xf32> to memref<?xf32>
  // CHECK-LABEL: func.func @mixed_aliases
  // CHECK: arith.ori
  // CHECK-COUNT-2: bufferization.dealloc
  bufferization.dealloc (%av, %a, %b : memref<?xf32>, memref<4xf32>, memref<4xf32>)
    if (%cond0, %cond1, %cond2)
  return
}

func.func @unknown(%cond: i1, %a: memref<4xf32>) {
  %b = memref.alloc() : memref<4xf32>
  %unknown = func.call @make_buffer() : () -> memref<4xf32>
  // CHECK-LABEL: func.func @unknown
  // CHECK-NOT: arith.ori
  // CHECK: bufferization.dealloc
  // CHECK-SAME: memref<4xf32>, memref<4xf32>) if (%arg0, %arg0)
  bufferization.dealloc (%b, %unknown : memref<4xf32>, memref<4xf32>)
    if (%cond, %cond)
  return
}

func.func @retained(%cond: i1) {
  %a = memref.alloc() : memref<4xf32>
  %keep = memref.alloc() : memref<4xf32>
  // CHECK-LABEL: func.func @retained
  // CHECK: bufferization.dealloc
  // CHECK-SAME: memref<4xf32>, memref<4xf32>) if (%arg0, %arg0) retain
  // CHECK-SAME: memref<4xf32>)
  bufferization.dealloc (%a, %keep : memref<4xf32>, memref<4xf32>)
    if (%cond, %cond) retain (%keep : memref<4xf32>)
  return
}

func.func @branch(%cond: i1) {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  // CHECK-LABEL: func.func @branch
  // CHECK-COUNT-1: bufferization.dealloc
  scf.if %cond {
    bufferization.dealloc (%a, %b : memref<4xf32>, memref<4xf32>)
      if (%cond, %cond)
    scf.yield
  }
  return
}

func.func @loop(%cond: i1) {
  %a = memref.alloc() : memref<4xf32>
  %b = memref.alloc() : memref<4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-LABEL: func.func @loop
  // CHECK-COUNT-1: bufferization.dealloc
  scf.for %i = %c0 to %c1 step %c1 {
    bufferization.dealloc (%a, %b : memref<4xf32>, memref<4xf32>)
      if (%cond, %cond)
    scf.yield
  }
  return
}

func.func private @make_buffer() -> memref<4xf32>
