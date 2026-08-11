//===- CIMFrameCodecTest.cpp - CIMFrame raw-flit tests ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Target/CIMFrameCodec.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace mlir;
using namespace mlir::cimframe;
using mlir::cim22::target::encodeCIMFrameInt8Packets;

namespace {
using Route = std::array<int32_t, 6>;

bool check(bool condition, const char *message) {
  if (condition)
    return true;
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

OwningOpRef<ModuleOp> makeModule(OpBuilder &builder) {
  return ModuleOp::create(builder.getUnknownLoc());
}

DenseI32ArrayAttr routeAttr(OpBuilder &builder, const Route &route) {
  return builder.getDenseI32ArrayAttr(route);
}

ControlInt8PacketOp addControl(OpBuilder &builder, ModuleOp module,
                               const Route &route, int32_t macro) {
  builder.setInsertionPointToEnd(module.getBody());
  return ControlInt8PacketOp::create(builder, builder.getUnknownLoc(),
                                     routeAttr(builder, route),
                                     builder.getI32IntegerAttr(macro));
}

void addWork(OpBuilder &builder, ModuleOp module, const Route &route) {
  builder.setInsertionPointToEnd(module.getBody());
  WorkOncePacketOp::create(builder, builder.getUnknownLoc(),
                           routeAttr(builder, route));
}

void addWeight(OpBuilder &builder, ModuleOp module, const Route &route,
               ArrayRef<int32_t> words) {
  builder.setInsertionPointToEnd(module.getBody());
  CIMInt8WeightPacketOp::create(builder, builder.getUnknownLoc(),
                                routeAttr(builder, route),
                                builder.getDenseI32ArrayAttr(words));
}

bool testControlAndWork(OpBuilder &builder) {
  constexpr Route zeroRoute{};
  auto module = makeModule(builder);
  addControl(builder, *module, zeroRoute, 0);
  addWork(builder, *module, zeroRoute);

  auto flits = encodeCIMFrameInt8Packets(*module);
  return check(succeeded(flits), "control/work encoding succeeds") &&
         check(flits->size() == 2, "control/work emits 2 flits") &&
         check((*flits)[0] == UINT64_C(0x8000000000000000),
               "zero-route Macro 0 control") &&
         check((*flits)[1] == UINT64_C(0x9000000000000001),
               "zero-route once work");
}

bool testMacroAndRouteBoundaries(OpBuilder &builder) {
  constexpr Route zeroRoute{};
  constexpr Route boundaryRoute{-31, 31, -1, 0, 0, 0};
  auto module = makeModule(builder);
  addControl(builder, *module, zeroRoute, 1);
  addWork(builder, *module, zeroRoute);
  addControl(builder, *module, boundaryRoute, 1);
  addWork(builder, *module, boundaryRoute);

  auto flits = encodeCIMFrameInt8Packets(*module);
  return check(succeeded(flits), "boundary route encoding succeeds") &&
         check(flits->size() == 4, "two work pairs emit 4 flits") &&
         check((*flits)[0] == UINT64_C(0x8000000000000001),
               "zero-route Macro 1 control") &&
         check((*flits)[2] == UINT64_C(0x8fdf840000000001),
               "sign-magnitude route boundary");
}

bool testWeightPacket(OpBuilder &builder) {
  constexpr Route route{3, 1, 0, 0, 0, 0};
  std::array<int32_t, 256> words{};
  words.front() = -1;
  words.back() = std::numeric_limits<int32_t>::min();

  auto module = makeModule(builder);
  addControl(builder, *module, route, 0);
  addWeight(builder, *module, route, words);

  auto first = encodeCIMFrameInt8Packets(*module);
  auto second = encodeCIMFrameInt8Packets(*module);
  return check(succeeded(first), "weight encoding succeeds") &&
         check(succeeded(second), "repeated weight encoding succeeds") &&
         check(first->size() == 258, "control/weight emits 258 flits") &&
         check(*first == *second, "weight encoding is deterministic") &&
         check((*first)[0] == UINT64_C(0x80c1000000000000),
               "mapped control flit") &&
         check((*first)[1] == UINT64_C(0x20c1000000000100),
               "mapped weight head") &&
         check((*first)[2] == UINT64_C(0x000000ffffffff00),
               "address 0 all-ones body") &&
         check((*first)[257] == UINT64_C(0x00000080000000ff),
               "address 255 signed-min body");
}

bool testMixedPairOrder(OpBuilder &builder) {
  constexpr Route route{};
  std::array<int32_t, 256> words{};
  auto module = makeModule(builder);
  addControl(builder, *module, route, 0);
  addWork(builder, *module, route);
  addControl(builder, *module, route, 1);
  addWeight(builder, *module, route, words);

  auto flits = encodeCIMFrameInt8Packets(*module);
  return check(succeeded(flits), "mixed pair encoding succeeds") &&
         check(flits->size() == 260, "mixed pairs preserve total length") &&
         check((*flits)[0] == UINT64_C(0x8000000000000000),
               "mixed first control") &&
         check((*flits)[1] == UINT64_C(0x9000000000000001),
               "mixed work follows control") &&
         check((*flits)[2] == UINT64_C(0x8000000000000001),
               "mixed second control") &&
         check((*flits)[3] == UINT64_C(0x2000000000000100),
               "mixed weight head follows control");
}

bool testInvalidStages(OpBuilder &builder) {
  constexpr Route zeroRoute{};
  constexpr Route otherRoute{1, 0, 0, 0, 0, 0};
  constexpr Route copyRoute{0, 0, 0, 1, 0, 0};
  std::array<int32_t, 256> words{};
  ScopedDiagnosticHandler diagnostics(builder.getContext(),
                                      [](Diagnostic &) {});

  auto empty = makeModule(builder);
  if (!check(failed(encodeCIMFrameInt8Packets(*empty)),
             "empty packet stage is rejected"))
    return false;

  auto command = makeModule(builder);
  builder.setInsertionPointToEnd(command->getBody());
  StartInt8OnceOp::create(builder, builder.getUnknownLoc(),
                          routeAttr(builder, zeroRoute),
                          builder.getI32IntegerAttr(0));
  if (!check(failed(encodeCIMFrameInt8Packets(*command)),
             "command stage is rejected"))
    return false;

  auto mixed = makeModule(builder);
  builder.setInsertionPointToEnd(mixed->getBody());
  StartInt8OnceOp::create(builder, builder.getUnknownLoc(),
                          routeAttr(builder, zeroRoute),
                          builder.getI32IntegerAttr(0));
  addControl(builder, *mixed, zeroRoute, 0);
  addWork(builder, *mixed, zeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*mixed)),
             "mixed command/packet stage is rejected"))
    return false;

  auto orphan = makeModule(builder);
  addWork(builder, *orphan, zeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*orphan)),
             "orphan work is rejected"))
    return false;

  auto reversed = makeModule(builder);
  addWork(builder, *reversed, zeroRoute);
  addControl(builder, *reversed, zeroRoute, 0);
  if (!check(failed(encodeCIMFrameInt8Packets(*reversed)),
             "reversed pair is rejected"))
    return false;

  auto separated = makeModule(builder);
  addControl(builder, *separated, zeroRoute, 0);
  builder.setInsertionPointToEnd(separated->getBody());
  ModuleOp separator = ModuleOp::create(builder.getUnknownLoc());
  builder.insert(separator.getOperation());
  addWork(builder, *separated, zeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*separated)),
             "separated pair is rejected"))
    return false;

  auto mismatch = makeModule(builder);
  addControl(builder, *mismatch, zeroRoute, 0);
  addWork(builder, *mismatch, otherRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*mismatch)),
             "route mismatch is rejected"))
    return false;

  auto badMacro = makeModule(builder);
  addControl(builder, *badMacro, zeroRoute, 2);
  addWork(builder, *badMacro, zeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*badMacro)),
             "invalid Macro is rejected"))
    return false;

  auto badRoute = makeModule(builder);
  constexpr Route outOfRange{32, 0, 0, 0, 0, 0};
  addControl(builder, *badRoute, outOfRange, 0);
  addWork(builder, *badRoute, outOfRange);
  if (!check(failed(encodeCIMFrameInt8Packets(*badRoute)),
             "invalid route is rejected"))
    return false;

  auto multicast = makeModule(builder);
  addControl(builder, *multicast, copyRoute, 0);
  addWork(builder, *multicast, copyRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*multicast)),
             "nonzero Copy route is rejected"))
    return false;

  auto badWords = makeModule(builder);
  addControl(builder, *badWords, zeroRoute, 0);
  addWeight(builder, *badWords, zeroRoute, ArrayRef(words).drop_back());
  if (!check(failed(encodeCIMFrameInt8Packets(*badWords)),
             "wrong word count is rejected"))
    return false;

  std::array<int32_t, 257> tooManyWords{};
  auto oversizedWords = makeModule(builder);
  addControl(builder, *oversizedWords, zeroRoute, 0);
  addWeight(builder, *oversizedWords, zeroRoute, tooManyWords);
  if (!check(failed(encodeCIMFrameInt8Packets(*oversizedWords)),
             "oversized word count is rejected"))
    return false;

  auto unexpectedAttribute = makeModule(builder);
  auto control = addControl(builder, *unexpectedAttribute, zeroRoute, 0);
  control->setAttr("type", builder.getI32IntegerAttr(8));
  addWeight(builder, *unexpectedAttribute, zeroRoute, words);
  return check(failed(encodeCIMFrameInt8Packets(*unexpectedAttribute)),
               "unexpected protocol attribute is rejected");
}
} // namespace

int main() {
  MLIRContext context;
  context.loadDialect<CIMFrameDialect>();
  OpBuilder builder(&context);
  return testControlAndWork(builder) && testMacroAndRouteBoundaries(builder) &&
                 testWeightPacket(builder) && testMixedPairOrder(builder) &&
                 testInvalidStages(builder)
             ? 0
             : 1;
}
