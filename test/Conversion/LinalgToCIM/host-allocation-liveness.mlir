// RUN: mlir-cim22-opt %s --split-input-file --cim-host-to-llvm | FileCheck %s

func.func private @dispatch(memref<?xf32>)

func.func private @consume(%buffer: memref<16384xf32>, %value: f32) {
  %zero = arith.constant 0 : index
  memref.store %value, %buffer[%zero] : memref<16384xf32>
  return
}

// CHECK-LABEL: llvm.func @entry
// CHECK: llvm.call @malloc
// CHECK: llvm.call @consume
// CHECK: llvm.call @free
// CHECK: llvm.call @malloc
// CHECK: llvm.call @consume
// CHECK: llvm.call @free
func.func @entry(%value: f32) {
  %stack = memref.alloca() : memref<16xf32>
  %cast = memref.cast %stack : memref<16xf32> to memref<?xf32>
  func.call @dispatch(%cast) : (memref<?xf32>) -> ()
  %first = memref.alloc() : memref<16384xf32>
  func.call @consume(%first, %value) : (memref<16384xf32>, f32) -> ()
  %second = memref.alloc() : memref<16384xf32>
  func.call @consume(%second, %value) : (memref<16384xf32>, f32) -> ()
  return
}

// -----

func.func private @retain(memref<16xf32>)
func.func private @observe()

// An opaque callee can use the buffer again before ownership frees it.
// CHECK-LABEL: llvm.func @opaque
// CHECK: llvm.call @retain
// CHECK-NOT: llvm.call @free
// CHECK: llvm.call @observe
// CHECK: llvm.call @free
func.func @opaque() {
  %buffer = memref.alloc() : memref<16xf32>
  func.call @retain(%buffer) : (memref<16xf32>) -> ()
  func.call @observe() : () -> ()
  return
}

// -----

memref.global "private" @saved : memref<1xindex> = dense<0>

func.func private @save(%buffer: memref<16xf32>) {
  %zero = arith.constant 0 : index
  %address = memref.extract_aligned_pointer_as_index %buffer : memref<16xf32> -> index
  %saved = memref.get_global @saved : memref<1xindex>
  memref.store %address, %saved[%zero] : memref<1xindex>
  return
}

func.func private @read() -> f32 {
  %zero = arith.constant 0 : index
  %saved = memref.get_global @saved : memref<1xindex>
  %address = memref.load %saved[%zero] : memref<1xindex>
  %raw = arith.index_cast %address : index to i64
  %ptr = llvm.inttoptr %raw : i64 to !llvm.ptr
  %value = llvm.load %ptr : !llvm.ptr -> f32
  return %value : f32
}

// CHECK-LABEL: llvm.func @pointer_escape
// CHECK: llvm.call @save
// CHECK-NOT: llvm.call @free
// CHECK: llvm.call @read
// CHECK: llvm.call @free
func.func @pointer_escape() -> f32 {
  %zero = arith.constant 0 : index
  %nine = arith.constant 9.0 : f32
  %buffer = memref.alloc() : memref<16xf32>
  memref.store %nine, %buffer[%zero] : memref<16xf32>
  func.call @save(%buffer) : (memref<16xf32>) -> ()
  %value = func.call @read() : () -> f32
  return %value : f32
}
