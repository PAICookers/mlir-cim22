//===- Passes.cpp - CIM22 target passes ------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Target/Passes.h"

#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Target/CIM22Target.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"

#include <array>
#include <optional>

namespace mlir::cim22::target {
#define GEN_PASS_DEF_MAPCIMSCHEDULE
#include "CIM22/Target/Passes.h.inc"

namespace {
constexpr llvm::StringLiteral kTileAttrs[] = {"m_tile", "n_tile", "k_tile"};
constexpr llvm::StringLiteral kScheduleAttrs[] = {"work_id", "group_id"};
constexpr llvm::StringLiteral kMappingAttrs[] = {"core_slot", "macro_slot",
                                                 "cim.mapping"};
constexpr llvm::StringLiteral kFunctionAttrs[] = {
    "cim.target_profile", "cim.target_profile_version", "cim.placement_policy",
    "cim.route_policy"};

struct MappingResult {
  int64_t coreSlot;
  int64_t macroSlot;
  CoreCoordinate coordinate;
  Route route;
};

unsigned countPresent(Operation *operation,
                      ArrayRef<llvm::StringLiteral> names) {
  return llvm::count_if(
      names, [operation](StringRef name) { return operation->hasAttr(name); });
}

std::optional<int64_t> readNonNegativeI64(cim::VMMOp op, StringRef name) {
  auto integer = dyn_cast_or_null<IntegerAttr>(op->getAttr(name));
  if (!integer || !integer.getType().isSignlessInteger(64)) {
    op.emitOpError("map-cim-schedule expects '")
        << name << "' to be an i64 attribute";
    return std::nullopt;
  }
  if (integer.getInt() < 0) {
    op.emitOpError("map-cim-schedule expects '")
        << name << "' to be non-negative";
    return std::nullopt;
  }
  return integer.getInt();
}

LogicalResult verifyFunctionAttributes(func::FuncOp function) {
  auto profile = function->getAttrOfType<StringAttr>("cim.target_profile");
  if (!profile || profile.getValue() != kProfileId)
    return function.emitError(
               "map-cim-schedule requires cim.target_profile = '")
           << kProfileId << "'";

  auto version =
      function->getAttrOfType<IntegerAttr>("cim.target_profile_version");
  if (!version || !version.getType().isSignlessInteger(64) ||
      version.getInt() != kProfileVersion)
    return function.emitError(
               "map-cim-schedule requires cim.target_profile_version = ")
           << kProfileVersion << " : i64";

  auto placement = function->getAttrOfType<StringAttr>("cim.placement_policy");
  if (!placement || placement.getValue() != kPlacementPolicy)
    return function.emitError(
               "map-cim-schedule requires cim.placement_policy = '")
           << kPlacementPolicy << "'";

  auto route = function->getAttrOfType<StringAttr>("cim.route_policy");
  if (!route || route.getValue() != kRoutePolicy)
    return function.emitError("map-cim-schedule requires cim.route_policy = '")
           << kRoutePolicy << "'";
  return success();
}

FailureOr<MappingResult> getExpectedMapping(cim::VMMOp op) {
  if (countPresent(op, kTileAttrs) != std::size(kTileAttrs) ||
      countPresent(op, kScheduleAttrs) != std::size(kScheduleAttrs)) {
    op.emitOpError("map-cim-schedule requires complete tile and logical "
                   "schedule identity");
    return failure();
  }
  for (StringRef name : kTileAttrs)
    if (!readNonNegativeI64(op, name))
      return failure();

  auto workId = readNonNegativeI64(op, "work_id");
  auto groupId = readNonNegativeI64(op, "group_id");
  if (!workId || !groupId)
    return failure();
  if (*groupId != *workId / 2) {
    op.emitOpError("map-cim-schedule expects group_id = ")
        << *workId / 2 << " for work_id = " << *workId;
    return failure();
  }

  const int64_t coreSlot = (*workId / kMacrosPerCore) % kCoreCount;
  const int64_t macroSlot = *workId % kMacrosPerCore;
  FailureOr<CoreCoordinate> coordinate = resolveCoreSlot(coreSlot);
  if (failed(coordinate)) {
    op.emitOpError("map-cim-schedule cannot resolve core_slot ") << coreSlot;
    return failure();
  }
  FailureOr<Route> route = routeFromLowerLeft(*coordinate);
  if (failed(route)) {
    op.emitOpError("map-cim-schedule cannot route to core_slot ") << coreSlot;
    return failure();
  }
  FailureOr<CoreCoordinate> replayed = replayRoute({0, 0}, *route);
  if (failed(replayed) || replayed->row != coordinate->row ||
      replayed->column != coordinate->column) {
    op.emitOpError("map-cim-schedule route does not reach core_slot ")
        << coreSlot;
    return failure();
  }
  return MappingResult{coreSlot, macroSlot, *coordinate, *route};
}

DictionaryAttr buildMapping(Builder &builder, const MappingResult &result) {
  const std::array<int64_t, 2> coordinate{result.coordinate.row,
                                          result.coordinate.column};
  const std::array<int64_t, 2> lowerLeft{0, 0};
  return builder.getDictionaryAttr({
      builder.getNamedAttr("core_coord",
                           builder.getDenseI64ArrayAttr(coordinate)),
      builder.getNamedAttr("ingress", builder.getDenseI64ArrayAttr(lowerLeft)),
      builder.getNamedAttr("source", builder.getDenseI64ArrayAttr(lowerLeft)),
      builder.getNamedAttr("destination",
                           builder.getDenseI64ArrayAttr(coordinate)),
      builder.getNamedAttr(
          "route", builder.getDenseI64ArrayAttr(result.route.distances)),
  });
}

LogicalResult verifyMappedOperation(cim::VMMOp op,
                                    const MappingResult &expected,
                                    DictionaryAttr expectedMapping) {
  auto coreSlot = readNonNegativeI64(op, "core_slot");
  auto macroSlot = readNonNegativeI64(op, "macro_slot");
  if (!coreSlot || !macroSlot)
    return failure();
  if (*coreSlot != expected.coreSlot || *macroSlot != expected.macroSlot)
    return op.emitOpError(
        "map-cim-schedule rejects stale target resource mapping");

  auto mapping = op->getAttrOfType<DictionaryAttr>("cim.mapping");
  if (!mapping || mapping != expectedMapping)
    return op.emitOpError(
        "map-cim-schedule rejects invalid or stale cim.mapping");
  return success();
}

class MapCIMSchedule final : public impl::MapCIMScheduleBase<MapCIMSchedule> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (failed(validateProfile())) {
      function.emitError("map-cim-schedule built-in target profile is invalid");
      return signalPassFailure();
    }

    SmallVector<cim::VMMOp> vmms;
    function.walk([&](cim::VMMOp op) { vmms.push_back(op); });
    const unsigned functionAttrCount = countPresent(function, kFunctionAttrs);
    if (vmms.empty()) {
      if (functionAttrCount != 0) {
        function.emitError(
            "map-cim-schedule rejects target attributes without cim.vmm work");
        signalPassFailure();
      }
      return;
    }

    if (!function.getBody().hasOneBlock() ||
        llvm::any_of(vmms, [&](cim::VMMOp op) {
          return op->getBlock() != &function.getBody().front();
        })) {
      function.emitError(
          "map-cim-schedule requires scheduled cim.vmm in one function block");
      return signalPassFailure();
    }

    const bool anyMapped = llvm::any_of(vmms, [](cim::VMMOp op) {
      return countPresent(op, kMappingAttrs) != 0;
    });
    const bool allMapped = llvm::all_of(vmms, [](cim::VMMOp op) {
      return countPresent(op, kMappingAttrs) == std::size(kMappingAttrs);
    });
    if (anyMapped != allMapped) {
      function.emitError(
          "map-cim-schedule rejects mixed mapped and unmapped cim.vmm work");
      return signalPassFailure();
    }

    if (allMapped) {
      if (functionAttrCount != std::size(kFunctionAttrs)) {
        function.emitError("map-cim-schedule requires all target profile and "
                           "policy attributes together");
        return signalPassFailure();
      }
      if (failed(verifyFunctionAttributes(function)))
        return signalPassFailure();
    } else if (functionAttrCount != 0) {
      function.emitError(
          "map-cim-schedule rejects stale function target attributes");
      return signalPassFailure();
    }

    Builder builder(function.getContext());
    SmallVector<MappingResult> expectedMappings;
    expectedMappings.reserve(vmms.size());
    for (cim::VMMOp vmm : vmms) {
      FailureOr<MappingResult> expected = getExpectedMapping(vmm);
      if (failed(expected))
        return signalPassFailure();
      expectedMappings.push_back(*expected);
      if (allMapped && failed(verifyMappedOperation(
                           vmm, *expected, buildMapping(builder, *expected))))
        return signalPassFailure();
    }
    if (allMapped)
      return;

    function->setAttr("cim.target_profile", builder.getStringAttr(kProfileId));
    function->setAttr("cim.target_profile_version",
                      builder.getI64IntegerAttr(kProfileVersion));
    function->setAttr("cim.placement_policy",
                      builder.getStringAttr(kPlacementPolicy));
    function->setAttr("cim.route_policy", builder.getStringAttr(kRoutePolicy));
    for (auto [vmm, mapping] : llvm::zip_equal(vmms, expectedMappings)) {
      vmm->setAttr("core_slot", builder.getI64IntegerAttr(mapping.coreSlot));
      vmm->setAttr("macro_slot", builder.getI64IntegerAttr(mapping.macroSlot));
      vmm->setAttr("cim.mapping", buildMapping(builder, mapping));
    }
  }
};
} // namespace
} // namespace mlir::cim22::target
