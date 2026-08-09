//===- ProgramImageTest.cpp - Program image verification -----------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CIM22/Artifact/ProgramImage.h"
#include "CIM22/Artifact/CIM22Image_generated.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using mlir::cim22::artifact::buildProgramImage;
using mlir::cim22::artifact::loadProgramImage;
using mlir::cim22::artifact::LogicalSlotRef;
using mlir::cim22::artifact::OpaqueSectionInput;
using mlir::cim22::artifact::ProgramImage;
using mlir::cim22::artifact::SegmentRef;
using mlir::cim22::artifact::writeProgramImage;

constexpr uint16_t kFormatMajorField = 4;
constexpr uint16_t kTargetProfileVersionField = 10;
constexpr uint16_t kInvocationSchemaVersionField = 12;
constexpr uint16_t kLogicalViewSchemaVersionField = 14;
constexpr uint16_t kLogicalSlotField = 6;
constexpr uint16_t kSectionIdField = 4;
constexpr uint16_t kSectionOffsetField = 6;
constexpr uint16_t kSectionSizeField = 8;
constexpr std::size_t kSizePrefixBytes = 4;
constexpr std::size_t kRootOffsetBytes = 4;
constexpr std::size_t kIdentifierOffset = kSizePrefixBytes + kRootOffsetBytes;
constexpr std::size_t kAlignment = 8;

constexpr std::array<uint8_t, 5> kSection0{0x00, 0x7f, 0x80, 0xff, 0x11};
constexpr std::array<uint8_t, 9> kSection1{0xde, 0xad, 0xbe, 0xef, 0x42,
                                           0x00, 0x01, 0x02, 0x03};
constexpr std::array<uint8_t, 32> kSection0SHA256{
    0x00, 0x06, 0xb3, 0x95, 0xfa, 0xb4, 0x7b, 0x4b, 0x51, 0xc6, 0xbf,
    0x96, 0x5b, 0x48, 0xa4, 0x7d, 0xc2, 0xf0, 0x96, 0x75, 0x31, 0xb9,
    0x12, 0xe3, 0xcb, 0x5c, 0xd6, 0xda, 0xbf, 0x1d, 0x8e, 0xd1};
constexpr std::array<uint8_t, 32> kSection1SHA256{
    0xd3, 0xf2, 0xad, 0x2e, 0x05, 0x65, 0x0b, 0xac, 0x23, 0x58, 0x69,
    0x68, 0x37, 0xb5, 0x16, 0xc8, 0x65, 0x3f, 0x41, 0x8a, 0x60, 0x07,
    0x7d, 0x62, 0x00, 0x1f, 0x10, 0x76, 0xf8, 0x34, 0xe6, 0xb4};

bool check(bool condition, llvm::StringRef message) {
  if (condition)
    return true;
  llvm::errs() << "FAIL: " << message << '\n';
  return false;
}

uint16_t readU16(llvm::ArrayRef<uint8_t> bytes, std::size_t offset) {
  return static_cast<uint16_t>(bytes[offset]) |
         static_cast<uint16_t>(bytes[offset + 1]) << 8;
}

uint32_t readU32(llvm::ArrayRef<uint8_t> bytes, std::size_t offset) {
  uint32_t value = 0;
  for (unsigned byte = 0; byte < 4; ++byte)
    value |= static_cast<uint32_t>(bytes[offset + byte]) << (8 * byte);
  return value;
}

void writeU32(std::vector<uint8_t> &bytes, std::size_t offset, uint32_t value) {
  for (unsigned byte = 0; byte < 4; ++byte)
    bytes[offset + byte] = static_cast<uint8_t>(value >> (8 * byte));
}

void writeU64(std::vector<uint8_t> &bytes, std::size_t offset, uint64_t value) {
  for (unsigned byte = 0; byte < 8; ++byte)
    bytes[offset + byte] = static_cast<uint8_t>(value >> (8 * byte));
}

std::size_t alignUp(std::size_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

std::size_t pointerOffset(llvm::ArrayRef<uint8_t> bytes, const void *pointer) {
  return static_cast<const uint8_t *>(pointer) - bytes.data();
}

std::size_t tableFieldOffset(llvm::ArrayRef<uint8_t> bytes, const void *table,
                             uint16_t field) {
  const std::size_t tableOffset = pointerOffset(bytes, table);
  const uint32_t vtableDistance = readU32(bytes, tableOffset);
  const std::size_t vtableOffset = tableOffset - vtableDistance;
  return tableOffset + readU16(bytes, vtableOffset + field);
}

std::vector<SegmentRef> validSegments() {
  return {SegmentRef{0, std::vector<LogicalSlotRef>{{0, 0}, {1, 1}},
                     std::vector<LogicalSlotRef>{{0, 0}, {1, 1}}}};
}

std::array<OpaqueSectionInput, 2> validSections() {
  return {OpaqueSectionInput{0, kSection0}, OpaqueSectionInput{1, kSection1}};
}

bool rejectsLoad(llvm::ArrayRef<uint8_t> bytes, llvm::StringRef message) {
  auto result = loadProgramImage(bytes);
  if (result)
    return check(false, message);
  llvm::consumeError(result.takeError());
  return true;
}

bool rejectsBuild(llvm::StringRef entryFunction,
                  std::vector<SegmentRef> segments,
                  llvm::ArrayRef<OpaqueSectionInput> sections,
                  llvm::StringRef message) {
  auto result = buildProgramImage(entryFunction, segments, sections);
  if (result)
    return check(false, message);
  llvm::consumeError(result.takeError());
  return true;
}

template <typename Mutator>
bool rejectsMutation(const ProgramImage &image, llvm::StringRef message,
                     Mutator mutate) {
  std::vector<uint8_t> bytes(image.bytes().begin(), image.bytes().end());
  mutate(bytes);
  return rejectsLoad(bytes, message);
}

bool testCanonicalImage() {
  const auto sections = validSections();
  const auto segments = validSegments();
  auto first = buildProgramImage("main", segments, sections);
  if (!first) {
    llvm::logAllUnhandledErrors(first.takeError(), llvm::errs(),
                                "FAIL: build: ");
    return false;
  }
  auto second = buildProgramImage("main", segments, sections);
  if (!second) {
    llvm::logAllUnhandledErrors(second.takeError(), llvm::errs(),
                                "FAIL: rebuild: ");
    return false;
  }

  const auto bytes = first->bytes();
  const auto *metadata = first->metadata();
  const uint32_t metadataSize = readU32(bytes, 0);
  const std::size_t payloadBase = alignUp(kSizePrefixBytes + metadataSize);
  const auto *sectionMetadata = metadata->sections();
  if (!check(first->bytes() == second->bytes(),
             "equal inputs must produce identical bytes") ||
      !check(cim22::image::SizePrefixedProgramImageBufferHasIdentifier(
                 bytes.data()),
             "size-prefixed identifier") ||
      !check(metadata->format_major() == 1 && metadata->format_minor() == 0 &&
                 metadata->target_profile()->str() == "cim22-4x5-v1" &&
                 metadata->target_profile_version() == 1 &&
                 metadata->invocation_schema_version() == 1 &&
                 metadata->logical_view_schema_version() == 1 &&
                 metadata->entry_function()->str() == "main" &&
                 metadata->section_alignment() == 8,
             "metadata compatibility fields") ||
      !check(metadata->segments()->size() == 1 &&
                 metadata->segments()->Get(0)->group_id() == 0 &&
                 metadata->segments()->Get(0)->host_to_cim()->size() == 2 &&
                 metadata->segments()->Get(0)->cim_to_host()->size() == 2,
             "logical segment metadata") ||
      !check(sectionMetadata->size() == 2 &&
                 sectionMetadata->Get(0)->id() == 0 &&
                 sectionMetadata->Get(0)->offset() == 0 &&
                 sectionMetadata->Get(0)->size() == kSection0.size() &&
                 sectionMetadata->Get(1)->id() == 1 &&
                 sectionMetadata->Get(1)->offset() == 8 &&
                 sectionMetadata->Get(1)->size() == kSection1.size(),
             "canonical section metadata") ||
      !check(payloadBase % kAlignment == 0 &&
                 payloadBase + 8 + kSection1.size() == bytes.size(),
             "payload alignment and exact EOF") ||
      !check(std::equal(bytes.begin() + payloadBase,
                        bytes.begin() + payloadBase + kSection0.size(),
                        kSection0.begin()) &&
                 std::all_of(bytes.begin() + payloadBase + kSection0.size(),
                             bytes.begin() + payloadBase + 8,
                             [](uint8_t byte) { return byte == 0; }) &&
                 std::equal(bytes.begin() + payloadBase + 8, bytes.end(),
                            kSection1.begin()),
             "exact sections and zero padding") ||
      !check(std::equal(sectionMetadata->Get(0)->sha256()->begin(),
                        sectionMetadata->Get(0)->sha256()->end(),
                        kSection0SHA256.begin()) &&
                 std::equal(sectionMetadata->Get(1)->sha256()->begin(),
                            sectionMetadata->Get(1)->sha256()->end(),
                            kSection1SHA256.begin()),
             "independent fixed SHA-256 values"))
    return false;

  auto section0 = first->section(0);
  auto section1 = first->section(1);
  auto missing = first->section(2);
  if (!check(section0 && *section0 == llvm::ArrayRef<uint8_t>(kSection0),
             "section 0 access") ||
      !check(section1 && *section1 == llvm::ArrayRef<uint8_t>(kSection1),
             "section 1 access") ||
      !check(!missing, "unknown section access must fail"))
    return false;
  llvm::consumeError(missing.takeError());

  std::string written;
  llvm::raw_string_ostream firstStream(written);
  if (llvm::Error error = writeProgramImage(*first, firstStream)) {
    llvm::logAllUnhandledErrors(std::move(error), llvm::errs(),
                                "FAIL: first write: ");
    return false;
  }
  firstStream.flush();
  const std::vector<uint8_t> writtenBytes(written.begin(), written.end());
  auto loaded = loadProgramImage(writtenBytes);
  if (!loaded) {
    llvm::logAllUnhandledErrors(loaded.takeError(), llvm::errs(),
                                "FAIL: reload: ");
    return false;
  }
  std::string rewritten;
  llvm::raw_string_ostream secondStream(rewritten);
  if (llvm::Error error = writeProgramImage(*loaded, secondStream)) {
    llvm::logAllUnhandledErrors(std::move(error), llvm::errs(),
                                "FAIL: second write: ");
    return false;
  }
  secondStream.flush();
  return check(writtenBytes == std::vector<uint8_t>(first->bytes().begin(),
                                                    first->bytes().end()) &&
                   rewritten == written && loaded->bytes() == first->bytes(),
               "build-write-load-write byte identity");
}

bool testBuildRejections() {
  const auto sections = validSections();
  bool ok = true;

  ok &= rejectsBuild("", validSegments(), sections, "empty entry function");

  auto groupIds = validSegments();
  groupIds[0].groupId = 1;
  ok &= rejectsBuild("main", groupIds, sections, "non-dense group ids");

  auto slots = validSegments();
  slots[0].hostToCIM[1].logicalSlot = 2;
  ok &= rejectsBuild("main", slots, sections, "non-dense logical slots");

  auto pairs = validSegments();
  pairs[0].cimToHost[1].workId = 2;
  ok &= rejectsBuild("main", pairs, sections,
                     "mismatched logical work pairs");

  auto noWorks = validSegments();
  noWorks[0].hostToCIM.clear();
  noWorks[0].cimToHost.clear();
  ok &= rejectsBuild("main", noWorks, sections, "zero-work segment");

  auto threeWorks = validSegments();
  threeWorks[0].hostToCIM.push_back({2, 2});
  threeWorks[0].cimToHost.push_back({2, 2});
  ok &= rejectsBuild("main", threeWorks, sections, "three-work segment");

  ok &= rejectsBuild("main", validSegments(), {}, "absent sections");
  const std::array<uint8_t, 0> emptyBytes{};
  const std::array<OpaqueSectionInput, 1> emptySection{
      OpaqueSectionInput{0, emptyBytes}};
  ok &= rejectsBuild("main", validSegments(), emptySection, "empty section");

  auto badIds = sections;
  badIds[1].id = 2;
  ok &= rejectsBuild("main", validSegments(), badIds,
                     "non-dense section ids");
  return ok;
}

bool testLoadMutations() {
  const auto sections = validSections();
  const auto segments = validSegments();
  auto built = buildProgramImage("main", segments, sections);
  if (!built) {
    llvm::logAllUnhandledErrors(built.takeError(), llvm::errs(),
                                "FAIL: mutation fixture: ");
    return false;
  }
  const auto bytes = built->bytes();
  const auto *metadata = built->metadata();
  const auto *segment = metadata->segments()->Get(0);
  const auto *logicalRef = segment->host_to_cim()->Get(1);
  const auto *section1 = metadata->sections()->Get(1);
  const std::size_t payloadBase = alignUp(kSizePrefixBytes + readU32(bytes, 0));
  bool ok = true;

  ok &= rejectsMutation(*built, "bad identifier", [](auto &mutated) {
    mutated[kIdentifierOffset] ^= 0x01;
  });
  ok &=
      rejectsMutation(*built, "incompatible format major", [&](auto &mutated) {
        mutated[tableFieldOffset(bytes, metadata, kFormatMajorField)] = 2;
      });
  ok &= rejectsMutation(
      *built, "incompatible target profile", [&](auto &mutated) {
        mutated[pointerOffset(bytes, metadata->target_profile()->Data())] = 'x';
      });
  ok &= rejectsMutation(
      *built, "incompatible target profile version", [&](auto &mutated) {
        writeU64(mutated,
                 tableFieldOffset(bytes, metadata, kTargetProfileVersionField),
                 2);
      });
  ok &= rejectsMutation(
      *built, "incompatible invocation schema", [&](auto &mutated) {
        writeU64(
            mutated,
            tableFieldOffset(bytes, metadata, kInvocationSchemaVersionField),
            2);
      });
  ok &= rejectsMutation(
      *built, "incompatible logical-view schema", [&](auto &mutated) {
        writeU64(
            mutated,
            tableFieldOffset(bytes, metadata, kLogicalViewSchemaVersionField),
            2);
      });
  ok &=
      rejectsMutation(*built, "failed FlatBuffers verifier", [](auto &mutated) {
        writeU32(mutated, kSizePrefixBytes,
                 std::numeric_limits<uint32_t>::max());
      });
  ok &= rejectsMutation(*built, "section bounds", [&](auto &mutated) {
    writeU64(mutated, tableFieldOffset(bytes, section1, kSectionSizeField),
             std::numeric_limits<uint64_t>::max());
  });
  ok &= rejectsMutation(*built, "misaligned section", [&](auto &mutated) {
    writeU64(mutated, tableFieldOffset(bytes, section1, kSectionOffsetField),
             7);
  });
  ok &= rejectsMutation(*built, "overlapping section", [&](auto &mutated) {
    writeU64(mutated, tableFieldOffset(bytes, section1, kSectionOffsetField),
             0);
  });
  ok &= rejectsMutation(*built, "nonzero section padding", [&](auto &mutated) {
    mutated[payloadBase + kSection0.size()] = 1;
  });
  ok &= rejectsMutation(*built, "trailing byte",
                        [](auto &mutated) { mutated.push_back(0); });
  ok &= rejectsMutation(*built, "section digest mismatch",
                        [&](auto &mutated) { mutated[payloadBase] ^= 0x80; });
  ok &= rejectsMutation(*built, "invalid digest length", [&](auto &mutated) {
    const auto *digest = section1->sha256();
    writeU32(mutated, pointerOffset(bytes, digest->Data()) - 4, 31);
  });
  ok &= rejectsMutation(
      *built, "non-dense logical reference", [&](auto &mutated) {
        writeU32(mutated,
                 tableFieldOffset(bytes, logicalRef, kLogicalSlotField), 7);
      });
  ok &= rejectsMutation(*built, "non-dense section id", [&](auto &mutated) {
    writeU32(mutated, tableFieldOffset(bytes, section1, kSectionIdField), 7);
  });

  std::vector<uint8_t> truncated(bytes.begin(), bytes.end() - 1);
  ok &= rejectsLoad(truncated, "truncated image");
  std::vector<uint8_t> overflowingPrefix(bytes.begin(), bytes.end());
  writeU32(overflowingPrefix, 0, std::numeric_limits<uint32_t>::max());
  ok &= rejectsLoad(overflowingPrefix, "overflowing metadata prefix");
  return ok;
}

} // namespace

int main() {
  return testCanonicalImage() && testBuildRejections() && testLoadMutations()
             ? 0
             : 1;
}
