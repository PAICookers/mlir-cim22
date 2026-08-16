// RUN: mlir-cim22-opt %s --pass-pipeline='builtin.module(form-cim-program,func.func(materialize-cim-schedule,map-cim-schedule))' > %t
// RUN: %python %S/../../../python/CIM22/schedule_oracle.py %t --m 1 --k 128 --n 320 | FileCheck %s --check-prefix=M5
// RUN: %python %S/../../../python/CIM22/route_oracle.py %t --require-all-slots | FileCheck %s --check-prefix=M4
// RUN: FileCheck %s --check-prefix=MAP < %t

// M5: PASS software-only M=1 K=128 N=320 work=40 groups=20 dtype=int32 shape=(1, 320) seed=2205 weight=random
// M5: boundaries first=0:(0,0,0)/g0 group19=[38:(0,19,0)/g19,39:(0,19,1)/g19] next=NA last=[38:(0,19,0)/g19,39:(0,19,1)/g19]

// M4: PASS software-only profile=cim22-4x5-v1 version=1 work=40 cores=20 macros=[0,1]
// M4: boundaries zero=w0/c0/m0 route=[0,0,0,0,0,0] corner=w38/c19/m0 route=[3,1,0,0,0,0]

// MAP-LABEL: func.func @matmul_integer
// MAP-SAME: attributes {cim.placement_policy = "core-major-dual-macro-v1", cim.route_policy = "lower-left-maximal-xy-v1", cim.target_profile = "cim22-4x5-v1", cim.target_profile_version = 1 : i64}
// MAP: cim.vmm {{.*}} {cim.mapping = {core_coord = array<i64: 0, 0>, destination = array<i64: 0, 0>, ingress = array<i64: 0, 0>, route = array<i64: 0, 0, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 0 : i64, group_id = 0 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 0 : i64, work_id = 0 : i64}
// MAP: cim.vmm {{.*}} {cim.mapping = {core_coord = array<i64: 3, 4>, destination = array<i64: 3, 4>, ingress = array<i64: 0, 0>, route = array<i64: 3, 1, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 19 : i64, group_id = 19 : i64, k_tile = 0 : i64, m_tile = 0 : i64, macro_slot = 0 : i64, n_tile = 19 : i64, work_id = 38 : i64}
// MAP: cim.vmm {{.*}} {cim.mapping = {core_coord = array<i64: 3, 4>, destination = array<i64: 3, 4>, ingress = array<i64: 0, 0>, route = array<i64: 3, 1, 0, 0, 0, 0>, source = array<i64: 0, 0>}, core_slot = 19 : i64, group_id = 19 : i64, k_tile = 1 : i64, m_tile = 0 : i64, macro_slot = 1 : i64, n_tile = 19 : i64, work_id = 39 : i64}
// MAP: arith.extsi {{.*}} tensor<16xi21> to tensor<16xi32>
// MAP: arith.addi {{.*}} : tensor<16xi32>
// MAP-NOT: arith.trunci
// MAP: return {{.*}} : tensor<320x1xi32>

func.func @matmul_integer(%input: tensor<128x1xi8>) -> tensor<320x1xi32> {
  %weight = arith.constant dense<0> : tensor<320x128xi8>
  %zero = arith.constant dense<0> : tensor<320x1xi32>
  %result = linalg.matmul {cim.onnx.matmul_integer}
      ins(%weight, %input : tensor<320x128xi8>, tensor<128x1xi8>)
      outs(%zero : tensor<320x1xi32>) -> tensor<320x1xi32>
  return %result : tensor<320x1xi32>
}
