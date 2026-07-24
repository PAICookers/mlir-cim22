//===- CIM22Target.h - CIM22 target facts ----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIM22_TARGET_CIM22TARGET_H
#define CIM22_TARGET_CIM22TARGET_H

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstdint>

namespace mlir::cim22::target {

inline constexpr llvm::StringLiteral kProfileId = "cim22-4x5-v1";
inline constexpr int64_t kProfileVersion = 1;
inline constexpr int64_t kCoreCount = 20;
inline constexpr int64_t kMacrosPerCore = 2;
inline constexpr llvm::StringLiteral kPlacementPolicy =
    "core-major-dual-macro-v1";
inline constexpr llvm::StringLiteral kRoutePolicy = "lower-left-maximal-xy-v1";

struct CoreCoordinate {
  int64_t row;
  int64_t column;
};

struct Route {
  std::array<int64_t, 6> distances;
};

LogicalResult validateProfile();
FailureOr<CoreCoordinate> resolveCoreSlot(int64_t coreSlot);
FailureOr<Route> routeFromLowerLeft(CoreCoordinate destination);
FailureOr<CoreCoordinate> replayRoute(CoreCoordinate source,
                                      const Route &route);

} // namespace mlir::cim22::target

#endif // CIM22_TARGET_CIM22TARGET_H
