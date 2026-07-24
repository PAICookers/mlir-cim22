//===- CIM22Target.cpp - CIM22 target facts --------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Target/CIM22Target.h"

#include <algorithm>

namespace mlir::cim22::target {
namespace {
constexpr int64_t kRows = 4;
constexpr int64_t kColumns = 5;
constexpr int64_t kRouteLimit = 31;
constexpr CoreCoordinate kLowerLeft{0, 0};

constexpr std::array<CoreCoordinate, kCoreCount> kCoreSlots{{
    {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 0}, {1, 1},
    {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1}, {2, 2}, {2, 3},
    {2, 4}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4},
}};

bool isLegal(CoreCoordinate coordinate) {
  return coordinate.row >= 0 && coordinate.row < kRows &&
         coordinate.column >= 0 && coordinate.column < kColumns;
}

bool isEqual(CoreCoordinate lhs, CoreCoordinate rhs) {
  return lhs.row == rhs.row && lhs.column == rhs.column;
}
} // namespace

LogicalResult validateProfile() {
  for (size_t index = 0; index < kCoreSlots.size(); ++index) {
    const CoreCoordinate coordinate = kCoreSlots[index];
    if (!isLegal(coordinate))
      return failure();
    for (size_t other = index + 1; other < kCoreSlots.size(); ++other)
      if (isEqual(coordinate, kCoreSlots[other]))
        return failure();
  }
  return success();
}

FailureOr<CoreCoordinate> resolveCoreSlot(int64_t coreSlot) {
  if (coreSlot < 0 || coreSlot >= static_cast<int64_t>(kCoreSlots.size()))
    return failure();
  return kCoreSlots[coreSlot];
}

FailureOr<Route> routeFromLowerLeft(CoreCoordinate destination) {
  if (!isLegal(destination))
    return failure();
  const int64_t diagonal = std::min(destination.row, destination.column);
  return Route{{diagonal, destination.column - diagonal,
                destination.row - diagonal, 0, 0, 0}};
}

FailureOr<CoreCoordinate> replayRoute(CoreCoordinate source,
                                      const Route &route) {
  if (!isEqual(source, kLowerLeft) ||
      std::any_of(route.distances.begin(), route.distances.end(),
                  [](int64_t distance) {
                    return distance < 0 || distance > kRouteLimit;
                  }) ||
      route.distances[3] != 0 || route.distances[4] != 0 ||
      route.distances[5] != 0)
    return failure();

  CoreCoordinate destination{
      source.row + route.distances[0] + route.distances[2],
      source.column + route.distances[0] + route.distances[1]};
  if (!isLegal(destination))
    return failure();
  return destination;
}

} // namespace mlir::cim22::target
