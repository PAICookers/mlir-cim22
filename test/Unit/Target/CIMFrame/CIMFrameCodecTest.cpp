//===- CIMFrameCodecTest.cpp - CIMFrame raw-flit tests ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Target/CIMFrameCodec.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace mlir;
using namespace mlir::cimframe;
using mlir::cim22::target::encodeCIMFrameInt8Packets;

using Route = std::array<int32_t, 6>;

static bool check(bool Condition, const char *Message) {
  if (Condition)
    return true;
  std::cerr << "FAIL: " << Message << '\n';
  return false;
}

static OwningOpRef<ModuleOp> makeModule(OpBuilder &Builder) {
  return ModuleOp::create(Builder.getUnknownLoc());
}

static DenseI32ArrayAttr routeAttr(OpBuilder &Builder, const Route &Route) {
  return Builder.getDenseI32ArrayAttr(Route);
}

static ControlInt8PacketOp addControl(OpBuilder &Builder, ModuleOp Module,
                                      const Route &Route, int32_t Macro) {
  Builder.setInsertionPointToEnd(Module.getBody());
  return ControlInt8PacketOp::create(Builder, Builder.getUnknownLoc(),
                                     routeAttr(Builder, Route),
                                     Builder.getI32IntegerAttr(Macro));
}

static void addWork(OpBuilder &Builder, ModuleOp Module, const Route &Route) {
  Builder.setInsertionPointToEnd(Module.getBody());
  WorkOncePacketOp::create(Builder, Builder.getUnknownLoc(),
                           routeAttr(Builder, Route));
}

static void addWeight(OpBuilder &Builder, ModuleOp Module, const Route &Route,
                      ArrayRef<int32_t> Words) {
  Builder.setInsertionPointToEnd(Module.getBody());
  CIMInt8WeightPacketOp::create(Builder, Builder.getUnknownLoc(),
                                routeAttr(Builder, Route),
                                Builder.getDenseI32ArrayAttr(Words));
}

static bool testControlAndWork(OpBuilder &Builder) {
  constexpr Route ZeroRoute{};
  auto Module = makeModule(Builder);
  addControl(Builder, *Module, ZeroRoute, 0);
  addWork(Builder, *Module, ZeroRoute);

  auto Flits = encodeCIMFrameInt8Packets(*Module);
  return check(succeeded(Flits), "control/work encoding succeeds") &&
         check(Flits->size() == 2, "control/work emits 2 flits") &&
         check((*Flits)[0] == UINT64_C(0x8000000000000000),
               "zero-route Macro 0 control") &&
         check((*Flits)[1] == UINT64_C(0x9000000000000001),
               "zero-route once work");
}

static bool testMacroAndRouteBoundaries(OpBuilder &Builder) {
  constexpr Route ZeroRoute{};
  constexpr Route BoundaryRoute{-31, 31, -1, 0, 0, 0};
  auto Module = makeModule(Builder);
  addControl(Builder, *Module, ZeroRoute, 1);
  addWork(Builder, *Module, ZeroRoute);
  addControl(Builder, *Module, BoundaryRoute, 1);
  addWork(Builder, *Module, BoundaryRoute);

  auto Flits = encodeCIMFrameInt8Packets(*Module);
  return check(succeeded(Flits), "boundary route encoding succeeds") &&
         check(Flits->size() == 4, "two work pairs emit 4 flits") &&
         check((*Flits)[0] == UINT64_C(0x8000000000000001),
               "zero-route Macro 1 control") &&
         check((*Flits)[2] == UINT64_C(0x8fdf840000000001),
               "sign-magnitude route boundary");
}

static bool testWeightPacket(OpBuilder &Builder) {
  constexpr Route Route{3, 1, 0, 0, 0, 0};
  std::array<int32_t, 256> Words{};
  Words.front() = -1;
  Words.back() = std::numeric_limits<int32_t>::min();

  auto Module = makeModule(Builder);
  addControl(Builder, *Module, Route, 0);
  addWeight(Builder, *Module, Route, Words);

  auto First = encodeCIMFrameInt8Packets(*Module);
  auto Second = encodeCIMFrameInt8Packets(*Module);
  if (!check(succeeded(First), "weight encoding succeeds") ||
      !check(succeeded(Second), "repeated weight encoding succeeds") ||
      !check(First->size() == 258, "control/weight emits 258 flits") ||
      !check(*First == *Second, "weight encoding is deterministic") ||
      !check((*First)[0] == UINT64_C(0x80c1000000000000),
             "mapped control flit") ||
      !check((*First)[1] == UINT64_C(0x20c1000000000100),
             "mapped weight head") ||
      !check((*First)[2] == UINT64_C(0x000000ffffffff00),
             "address 0 all-ones body") ||
      !check((*First)[257] == UINT64_C(0x00000080000000ff),
             "address 255 signed-min body"))
    return false;

  addControl(Builder, *Module, Route, 0);
  addWeight(Builder, *Module, Route, Words);
  auto Repeated = encodeCIMFrameInt8Packets(*Module);
  return check(succeeded(Repeated), "repeated transaction encoding succeeds") &&
         check(Repeated->size() == First->size() * 2,
               "repeated transaction emits two independent pairs") &&
         check(std::equal(First->begin(), First->end(), Repeated->begin()),
               "first transaction is preserved") &&
         check(std::equal(First->begin(), First->end(),
                          Repeated->begin() + First->size()),
               "second transaction is preserved");
}

static bool testMixedPairOrder(OpBuilder &Builder) {
  constexpr Route Route{};
  std::array<int32_t, 256> Words{};
  auto Module = makeModule(Builder);
  addControl(Builder, *Module, Route, 0);
  addWork(Builder, *Module, Route);
  addControl(Builder, *Module, Route, 1);
  addWeight(Builder, *Module, Route, Words);

  auto Flits = encodeCIMFrameInt8Packets(*Module);
  return check(succeeded(Flits), "mixed pair encoding succeeds") &&
         check(Flits->size() == 260, "mixed pairs preserve total length") &&
         check((*Flits)[0] == UINT64_C(0x8000000000000000),
               "mixed first control") &&
         check((*Flits)[1] == UINT64_C(0x9000000000000001),
               "mixed work follows control") &&
         check((*Flits)[2] == UINT64_C(0x8000000000000001),
               "mixed second control") &&
         check((*Flits)[3] == UINT64_C(0x2000000000000100),
               "mixed weight head follows control");
}

static bool testInvalidStages(OpBuilder &Builder) {
  constexpr Route ZeroRoute{};
  constexpr Route OtherRoute{1, 0, 0, 0, 0, 0};
  constexpr Route CopyRoute{0, 0, 0, 1, 0, 0};
  std::array<int32_t, 256> Words{};
  ScopedDiagnosticHandler Diagnostics(Builder.getContext(),
                                      [](Diagnostic &) {});

  auto Empty = makeModule(Builder);
  if (!check(failed(encodeCIMFrameInt8Packets(*Empty)),
             "empty packet stage is rejected"))
    return false;

  auto Command = makeModule(Builder);
  Builder.setInsertionPointToEnd(Command->getBody());
  StartInt8OnceOp::create(Builder, Builder.getUnknownLoc(),
                          routeAttr(Builder, ZeroRoute),
                          Builder.getI32IntegerAttr(0));
  if (!check(failed(encodeCIMFrameInt8Packets(*Command)),
             "command stage is rejected"))
    return false;

  auto Mixed = makeModule(Builder);
  Builder.setInsertionPointToEnd(Mixed->getBody());
  StartInt8OnceOp::create(Builder, Builder.getUnknownLoc(),
                          routeAttr(Builder, ZeroRoute),
                          Builder.getI32IntegerAttr(0));
  addControl(Builder, *Mixed, ZeroRoute, 0);
  addWork(Builder, *Mixed, ZeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*Mixed)),
             "mixed command/packet stage is rejected"))
    return false;

  auto Orphan = makeModule(Builder);
  addWork(Builder, *Orphan, ZeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*Orphan)),
             "orphan work is rejected"))
    return false;

  auto Reversed = makeModule(Builder);
  addWork(Builder, *Reversed, ZeroRoute);
  addControl(Builder, *Reversed, ZeroRoute, 0);
  if (!check(failed(encodeCIMFrameInt8Packets(*Reversed)),
             "reversed pair is rejected"))
    return false;

  auto Separated = makeModule(Builder);
  addControl(Builder, *Separated, ZeroRoute, 0);
  Builder.setInsertionPointToEnd(Separated->getBody());
  ModuleOp Separator = ModuleOp::create(Builder.getUnknownLoc());
  Builder.insert(Separator.getOperation());
  addWork(Builder, *Separated, ZeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*Separated)),
             "separated pair is rejected"))
    return false;

  auto Mismatch = makeModule(Builder);
  addControl(Builder, *Mismatch, ZeroRoute, 0);
  addWork(Builder, *Mismatch, OtherRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*Mismatch)),
             "route mismatch is rejected"))
    return false;

  auto BadMacro = makeModule(Builder);
  addControl(Builder, *BadMacro, ZeroRoute, 2);
  addWork(Builder, *BadMacro, ZeroRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*BadMacro)),
             "invalid Macro is rejected"))
    return false;

  auto BadRoute = makeModule(Builder);
  constexpr Route OutOfRange{32, 0, 0, 0, 0, 0};
  addControl(Builder, *BadRoute, OutOfRange, 0);
  addWork(Builder, *BadRoute, OutOfRange);
  if (!check(failed(encodeCIMFrameInt8Packets(*BadRoute)),
             "invalid route is rejected"))
    return false;

  auto Multicast = makeModule(Builder);
  addControl(Builder, *Multicast, CopyRoute, 0);
  addWork(Builder, *Multicast, CopyRoute);
  if (!check(failed(encodeCIMFrameInt8Packets(*Multicast)),
             "nonzero Copy route is rejected"))
    return false;

  auto BadWords = makeModule(Builder);
  addControl(Builder, *BadWords, ZeroRoute, 0);
  addWeight(Builder, *BadWords, ZeroRoute, ArrayRef(Words).drop_back());
  if (!check(failed(encodeCIMFrameInt8Packets(*BadWords)),
             "wrong word count is rejected"))
    return false;

  std::array<int32_t, 257> TooManyWords{};
  auto OversizedWords = makeModule(Builder);
  addControl(Builder, *OversizedWords, ZeroRoute, 0);
  addWeight(Builder, *OversizedWords, ZeroRoute, TooManyWords);
  if (!check(failed(encodeCIMFrameInt8Packets(*OversizedWords)),
             "oversized word count is rejected"))
    return false;

  auto UnexpectedAttribute = makeModule(Builder);
  auto Control = addControl(Builder, *UnexpectedAttribute, ZeroRoute, 0);
  Control->setAttr("type", Builder.getI32IntegerAttr(8));
  addWeight(Builder, *UnexpectedAttribute, ZeroRoute, Words);
  return check(failed(encodeCIMFrameInt8Packets(*UnexpectedAttribute)),
               "unexpected protocol attribute is rejected");
}

static bool emitPacketSummary(StringRef Path, MLIRContext &Context) {
  auto Module = parseSourceFile<ModuleOp>(Path, ParserConfig(&Context));
  if (!check(static_cast<bool>(Module), "packet-stage MLIR parses"))
    return false;
  auto Flits = encodeCIMFrameInt8Packets(*Module);
  if (!check(succeeded(Flits), "packet-stage encoding succeeds") ||
      !check(Flits->size() >= 258, "packet-stage has one transaction"))
    return false;
  const size_t Transactions =
      llvm::count_if(Module->getBody()->getOperations(), [](Operation &Op) {
        return isa<ControlInt8PacketOp>(Op);
      });
  const size_t Last = Flits->size() - 258;
  std::cout << "transactions=" << Transactions << " flits=" << Flits->size()
            << std::hex << std::setfill('0') << " first_control=0x"
            << std::setw(16) << (*Flits)[0] << " first_weight_head=0x"
            << std::setw(16) << (*Flits)[1] << " first_body=0x" << std::setw(16)
            << (*Flits)[2] << " last_control=0x" << std::setw(16)
            << (*Flits)[Last] << " last_weight_head=0x" << std::setw(16)
            << (*Flits)[Last + 1] << " last_body=0x" << std::setw(16)
            << Flits->back() << std::dec << '\n';
  return true;
}

int main(int Argc, char **Argv) {
  if (Argc > 2) {
    std::cerr << "usage: " << Argv[0] << " [packet-stage.mlir]\n";
    return 1;
  }
  MLIRContext Context;
  Context.loadDialect<cim::CIMDialect, CIMFrameDialect, arith::ArithDialect,
                      func::FuncDialect, linalg::LinalgDialect,
                      tensor::TensorDialect>();
  if (Argc == 2)
    return emitPacketSummary(Argv[1], Context) ? 0 : 1;
  OpBuilder Builder(&Context);
  return testControlAndWork(Builder) && testMacroAndRouteBoundaries(Builder) &&
                 testWeightPacket(Builder) && testMixedPairOrder(Builder) &&
                 testInvalidStages(Builder)
             ? 0
             : 1;
}
