#include "CIM22/Artifact/ProgramImage.h"

#include "CIM22/Artifact/CIM22Image_generated.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mlir::cim22::artifact {
namespace {

using ErrorKind = ProgramImageErrorKind;

llvm::Error makeImageError(ErrorKind kind, const llvm::Twine &message) {
  return llvm::make_error<ProgramImageError>(kind, message.str());
}

const char *errorKindName(ErrorKind kind) {
  switch (kind) {
  case ErrorKind::InvalidArgument:
    return "invalid argument";
  case ErrorKind::MalformedBuffer:
    return "malformed buffer";
  case ErrorKind::IncompatibleMetadata:
    return "incompatible metadata";
  case ErrorKind::IntegrityMismatch:
    return "integrity mismatch";
  case ErrorKind::IOFailure:
    return "I/O failure";
  }
  llvm_unreachable("unknown ProgramImage error kind");
}

bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedAlignUp(uint64_t value, uint64_t alignment, uint64_t &result) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0)
    return false;
  uint64_t sum = 0;
  if (!checkedAdd(value, alignment - 1, sum))
    return false;
  result = sum & ~(alignment - 1);
  return true;
}

llvm::Error validateCompatibility(uint16_t formatMajor, uint16_t formatMinor,
                                  llvm::StringRef targetProfile,
                                  uint64_t targetProfileVersion,
                                  uint64_t invocationSchemaVersion,
                                  uint64_t logicalViewSchemaVersion,
                                  uint32_t sectionAlignment) {
  if (formatMajor != kProgramImageFormatMajor)
    return makeImageError(ErrorKind::IncompatibleMetadata,
                          "unsupported format major version");
  if (formatMinor > kProgramImageFormatMinor)
    return makeImageError(ErrorKind::IncompatibleMetadata,
                          "unsupported format minor version");
  if (targetProfile != kProgramImageTargetProfile ||
      targetProfileVersion != kProgramImageTargetProfileVersion)
    return makeImageError(ErrorKind::IncompatibleMetadata,
                          "unsupported target profile");
  if (invocationSchemaVersion != kProgramImageInvocationSchemaVersion)
    return makeImageError(ErrorKind::IncompatibleMetadata,
                          "unsupported invocation schema version");
  if (logicalViewSchemaVersion != kProgramImageLogicalViewSchemaVersion)
    return makeImageError(ErrorKind::IncompatibleMetadata,
                          "unsupported logical-view schema version");
  if (sectionAlignment != kProgramImageSectionAlignment)
    return makeImageError(ErrorKind::IncompatibleMetadata,
                          "unsupported section alignment");
  return llvm::Error::success();
}

template <typename SlotRange>
llvm::Error validateSlots(const SlotRange &slots, llvm::StringRef direction,
                          uint64_t groupId) {
  for (size_t index = 0; index < slots.size(); ++index) {
    if (slots[index].logicalSlot != index)
      return makeImageError(ErrorKind::InvalidArgument,
                            llvm::Twine("group ") + llvm::Twine(groupId) +
                                " has non-dense " + direction +
                                " logical slots");
  }
  return llvm::Error::success();
}

llvm::Error validateBuildInput(llvm::StringRef entryFunction,
                               llvm::ArrayRef<SegmentRef> segments,
                               llvm::ArrayRef<OpaqueSectionInput> sections) {
  if (entryFunction.empty())
    return makeImageError(ErrorKind::InvalidArgument,
                          "entry function must not be empty");

  for (size_t index = 0; index < segments.size(); ++index) {
    const SegmentRef &segment = segments[index];
    if (segment.groupId != index)
      return makeImageError(ErrorKind::InvalidArgument,
                            "segment group IDs must be dense from zero");
    if (segment.hostToCIM.size() != segment.cimToHost.size() ||
        (segment.hostToCIM.size() != 1 && segment.hostToCIM.size() != 2))
      return makeImageError(
          ErrorKind::InvalidArgument,
          "each segment must contain one or two matched works");
    if (segment.hostToCIM.size() == 1 && index + 1 != segments.size())
      return makeImageError(ErrorKind::InvalidArgument,
                            "a one-work segment must be final");
    if (llvm::Error error =
            validateSlots(segment.hostToCIM, "host-to-CIM", segment.groupId))
      return error;
    if (llvm::Error error =
            validateSlots(segment.cimToHost, "CIM-to-host", segment.groupId))
      return error;
    for (size_t slot = 0; slot < segment.hostToCIM.size(); ++slot) {
      if (segment.hostToCIM[slot].workId != segment.cimToHost[slot].workId)
        return makeImageError(ErrorKind::InvalidArgument,
                              "input and result work IDs must match by slot");
      if (slot != 0 &&
          segment.hostToCIM[slot - 1].workId == segment.hostToCIM[slot].workId)
        return makeImageError(ErrorKind::InvalidArgument,
                              "work IDs within a segment must be distinct");
    }
  }

  if (sections.empty())
    return makeImageError(ErrorKind::InvalidArgument,
                          "at least one opaque section is required");
  for (size_t index = 0; index < sections.size(); ++index) {
    if (sections[index].id != index)
      return makeImageError(ErrorKind::InvalidArgument,
                            "opaque section IDs must be dense from zero");
    if (sections[index].bytes.empty())
      return makeImageError(ErrorKind::InvalidArgument,
                            "opaque sections must not be empty");
  }
  return llvm::Error::success();
}

struct ValidatedImage {
  std::vector<std::pair<size_t, size_t>> ranges;
};

llvm::Expected<ValidatedImage>
validateImageBytes(llvm::ArrayRef<uint8_t> bytes) {
  constexpr uint64_t sizePrefixBytes = sizeof(uint32_t);
  if (bytes.size() < sizePrefixBytes)
    return makeImageError(ErrorKind::MalformedBuffer,
                          "truncated FlatBuffer size prefix");

  const uint64_t metadataSize = llvm::support::endian::read32le(bytes.data());
  uint64_t metadataEnd = 0;
  if (!checkedAdd(sizePrefixBytes, metadataSize, metadataEnd) ||
      metadataEnd > bytes.size())
    return makeImageError(ErrorKind::MalformedBuffer,
                          "FlatBuffer metadata size exceeds the image");
  if (metadataSize <
      sizeof(flatbuffers::uoffset_t) + flatbuffers::kFileIdentifierLength)
    return makeImageError(ErrorKind::MalformedBuffer,
                          "FlatBuffer metadata is too short for an identifier");

  if (!::cim22::image::SizePrefixedProgramImageBufferHasIdentifier(
          bytes.data()))
    return makeImageError(ErrorKind::MalformedBuffer,
                          "invalid size-prefixed ProgramImage identifier");
  flatbuffers::Verifier verifier(bytes.data(), metadataEnd);
  if (!::cim22::image::VerifySizePrefixedProgramImageBuffer(verifier))
    return makeImageError(ErrorKind::MalformedBuffer,
                          "FlatBuffer verifier rejected ProgramImage metadata");

  const ::cim22::image::ProgramImage *metadata =
      ::cim22::image::GetSizePrefixedProgramImage(bytes.data());
  if (llvm::Error error = validateCompatibility(
          metadata->format_major(), metadata->format_minor(),
          metadata->target_profile()->string_view(),
          metadata->target_profile_version(),
          metadata->invocation_schema_version(),
          metadata->logical_view_schema_version(),
          metadata->section_alignment()))
    return std::move(error);
  if (metadata->entry_function()->size() == 0)
    return makeImageError(ErrorKind::MalformedBuffer,
                          "entry function must not be empty");

  const auto *segments = metadata->segments();
  for (size_t index = 0; index < segments->size(); ++index) {
    const auto *segment = segments->Get(index);
    if (segment->group_id() != index)
      return makeImageError(ErrorKind::MalformedBuffer,
                            "segment group IDs are not dense from zero");
    const auto *inputs = segment->host_to_cim();
    const auto *results = segment->cim_to_host();
    if (inputs->size() != results->size() ||
        (inputs->size() != 1 && inputs->size() != 2))
      return makeImageError(
          ErrorKind::MalformedBuffer,
          "segment does not contain one or two matched works");
    if (inputs->size() == 1 && index + 1 != segments->size())
      return makeImageError(ErrorKind::MalformedBuffer,
                            "a one-work segment is not final");
    for (size_t slot = 0; slot < inputs->size(); ++slot) {
      if (inputs->Get(slot)->logical_slot() != slot ||
          results->Get(slot)->logical_slot() != slot)
        return makeImageError(
            ErrorKind::MalformedBuffer,
            "segment logical slots are not direction-local and dense");
      if (inputs->Get(slot)->work_id() != results->Get(slot)->work_id())
        return makeImageError(ErrorKind::MalformedBuffer,
                              "input and result work IDs do not match by slot");
      if (slot != 0 &&
          inputs->Get(slot - 1)->work_id() == inputs->Get(slot)->work_id())
        return makeImageError(ErrorKind::MalformedBuffer,
                              "work IDs within a segment are not distinct");
    }
  }

  uint64_t payloadBase = 0;
  if (!checkedAlignUp(metadataEnd, kProgramImageSectionAlignment,
                      payloadBase) ||
      payloadBase > bytes.size())
    return makeImageError(ErrorKind::MalformedBuffer,
                          "payload base overflows the image");
  if (!std::all_of(bytes.begin() + metadataEnd, bytes.begin() + payloadBase,
                   [](uint8_t byte) { return byte == 0; }))
    return makeImageError(ErrorKind::MalformedBuffer,
                          "metadata padding is not zero");

  const auto *sections = metadata->sections();
  if (sections->size() == 0)
    return makeImageError(ErrorKind::MalformedBuffer,
                          "at least one opaque section is required");

  ValidatedImage validated;
  validated.ranges.reserve(sections->size());
  uint64_t expectedOffset = 0;
  uint64_t previousEnd = 0;
  for (size_t index = 0; index < sections->size(); ++index) {
    const auto *section = sections->Get(index);
    if (section->id() != index)
      return makeImageError(ErrorKind::MalformedBuffer,
                            "opaque section IDs are not dense from zero");
    if (section->size() == 0)
      return makeImageError(ErrorKind::MalformedBuffer,
                            "opaque section size must not be zero");
    if (section->offset() != expectedOffset ||
        section->offset() % kProgramImageSectionAlignment != 0)
      return makeImageError(ErrorKind::MalformedBuffer,
                            "opaque section offset is not canonical");
    if (section->sha256()->size() != llvm::SHA256::hash({}).size())
      return makeImageError(ErrorKind::MalformedBuffer,
                            "opaque section digest must contain 32 bytes");

    uint64_t start = 0;
    uint64_t end = 0;
    if (!checkedAdd(payloadBase, section->offset(), start) ||
        !checkedAdd(start, section->size(), end) || end > bytes.size() ||
        start > std::numeric_limits<size_t>::max() ||
        end > std::numeric_limits<size_t>::max())
      return makeImageError(ErrorKind::MalformedBuffer,
                            "opaque section range exceeds the image");
    llvm::ArrayRef<uint8_t> sectionBytes(bytes.data() + start, section->size());
    const std::array<uint8_t, 32> digest = llvm::SHA256::hash(sectionBytes);
    if (!std::equal(digest.begin(), digest.end(), section->sha256()->begin()))
      return makeImageError(ErrorKind::IntegrityMismatch,
                            "opaque section SHA-256 does not match");
    validated.ranges.emplace_back(static_cast<size_t>(start),
                                  static_cast<size_t>(section->size()));

    previousEnd = section->offset() + section->size();
    if (index + 1 < sections->size()) {
      if (!checkedAlignUp(previousEnd, kProgramImageSectionAlignment,
                          expectedOffset))
        return makeImageError(ErrorKind::MalformedBuffer,
                              "opaque section alignment overflows");
      const uint64_t paddingStart = end;
      uint64_t paddingEnd = 0;
      if (!checkedAdd(payloadBase, expectedOffset, paddingEnd) ||
          paddingEnd > bytes.size())
        return makeImageError(ErrorKind::MalformedBuffer,
                              "opaque section padding exceeds the image");
      if (!std::all_of(bytes.begin() + paddingStart, bytes.begin() + paddingEnd,
                       [](uint8_t byte) { return byte == 0; }))
        return makeImageError(ErrorKind::MalformedBuffer,
                              "opaque section padding is not zero");
    }
  }

  uint64_t finalEnd = 0;
  if (!checkedAdd(payloadBase, previousEnd, finalEnd) ||
      finalEnd != bytes.size())
    return makeImageError(ErrorKind::MalformedBuffer,
                          "trailing bytes follow the final opaque section");
  return validated;
}

} // namespace

char ProgramImageError::ID = 0;

void ProgramImageError::log(llvm::raw_ostream &os) const {
  os << errorKindName(errorKind) << ": " << errorMessage;
}

std::error_code ProgramImageError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

const ::cim22::image::ProgramImage *ProgramImage::metadata() const {
  return ::cim22::image::GetSizePrefixedProgramImage(ownedBytes.data());
}

llvm::Expected<llvm::ArrayRef<uint8_t>>
ProgramImage::section(uint32_t id) const {
  if (id >= ranges.size())
    return makeImageError(ErrorKind::InvalidArgument,
                          "opaque section ID is out of range");
  const auto [offset, size] = ranges[id];
  return llvm::ArrayRef<uint8_t>(ownedBytes.data() + offset, size);
}

llvm::Expected<ProgramImage>
buildProgramImage(llvm::StringRef entryFunction,
                  llvm::ArrayRef<SegmentRef> segments,
                  llvm::ArrayRef<OpaqueSectionInput> sections) {
  if (llvm::Error error =
          validateBuildInput(entryFunction, segments, sections))
    return std::move(error);

  std::vector<uint64_t> sectionOffsets;
  std::vector<std::array<uint8_t, 32>> sectionDigests;
  sectionOffsets.reserve(sections.size());
  sectionDigests.reserve(sections.size());
  uint64_t nextOffset = 0;
  for (const OpaqueSectionInput &section : sections) {
    sectionOffsets.push_back(nextOffset);
    sectionDigests.push_back(llvm::SHA256::hash(section.bytes));
    uint64_t end = 0;
    if (!checkedAdd(nextOffset, section.bytes.size(), end) ||
        !checkedAlignUp(end, kProgramImageSectionAlignment, nextOffset))
      return makeImageError(ErrorKind::InvalidArgument,
                            "opaque section layout overflows");
  }

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<::cim22::image::SegmentRef>> segmentOffsets;
  segmentOffsets.reserve(segments.size());
  for (const SegmentRef &segment : segments) {
    std::vector<flatbuffers::Offset<::cim22::image::LogicalSlotRef>> inputs;
    std::vector<flatbuffers::Offset<::cim22::image::LogicalSlotRef>> results;
    inputs.reserve(segment.hostToCIM.size());
    results.reserve(segment.cimToHost.size());
    for (const LogicalSlotRef &slot : segment.hostToCIM)
      inputs.push_back(::cim22::image::CreateLogicalSlotRef(
          builder, slot.workId, slot.logicalSlot));
    for (const LogicalSlotRef &slot : segment.cimToHost)
      results.push_back(::cim22::image::CreateLogicalSlotRef(
          builder, slot.workId, slot.logicalSlot));
    const auto inputVector = builder.CreateVector(inputs);
    const auto resultVector = builder.CreateVector(results);
    segmentOffsets.push_back(::cim22::image::CreateSegmentRef(
        builder, segment.groupId, inputVector, resultVector));
  }

  std::vector<flatbuffers::Offset<::cim22::image::OpaqueSection>>
      opaqueSectionOffsets;
  opaqueSectionOffsets.reserve(sections.size());
  for (size_t index = 0; index < sections.size(); ++index) {
    const auto digest = builder.CreateVector(sectionDigests[index].data(),
                                             sectionDigests[index].size());
    opaqueSectionOffsets.push_back(::cim22::image::CreateOpaqueSection(
        builder, sections[index].id, sectionOffsets[index],
        sections[index].bytes.size(), digest));
  }

  const auto targetProfile = builder.CreateString(kProgramImageTargetProfile);
  const auto entryFunctionOffset =
      builder.CreateString(entryFunction.data(), entryFunction.size());
  const auto segmentVector = builder.CreateVector(segmentOffsets);
  const auto sectionVector = builder.CreateVector(opaqueSectionOffsets);
  const auto root = ::cim22::image::CreateProgramImage(
      builder, kProgramImageFormatMajor, kProgramImageFormatMinor, targetProfile,
      kProgramImageTargetProfileVersion, kProgramImageInvocationSchemaVersion,
      kProgramImageLogicalViewSchemaVersion, entryFunctionOffset, segmentVector,
      kProgramImageSectionAlignment, sectionVector);
  ::cim22::image::FinishSizePrefixedProgramImageBuffer(builder, root);

  std::vector<uint8_t> bytes(builder.GetBufferPointer(),
                             builder.GetBufferPointer() + builder.GetSize());
  uint64_t payloadBase = 0;
  if (!checkedAlignUp(bytes.size(), kProgramImageSectionAlignment,
                      payloadBase) ||
      payloadBase > std::numeric_limits<size_t>::max())
    return makeImageError(ErrorKind::InvalidArgument,
                          "ProgramImage metadata layout overflows");
  bytes.resize(payloadBase, 0);
  for (size_t index = 0; index < sections.size(); ++index) {
    bytes.insert(bytes.end(), sections[index].bytes.begin(),
                 sections[index].bytes.end());
    if (index + 1 < sections.size()) {
      uint64_t alignedSize = 0;
      if (!checkedAlignUp(bytes.size(), kProgramImageSectionAlignment,
                          alignedSize) ||
          alignedSize > std::numeric_limits<size_t>::max())
        return makeImageError(ErrorKind::InvalidArgument,
                              "ProgramImage payload layout overflows");
      bytes.resize(alignedSize, 0);
    }
  }

  llvm::Expected<ValidatedImage> validated = validateImageBytes(bytes);
  if (!validated)
    return validated.takeError();
  return ProgramImage(std::move(bytes), std::move(validated->ranges));
}

llvm::Expected<ProgramImage>
loadProgramImage(llvm::ArrayRef<uint8_t> inputBytes) {
  std::vector<uint8_t> bytes(inputBytes.begin(), inputBytes.end());
  llvm::Expected<ValidatedImage> validated = validateImageBytes(bytes);
  if (!validated)
    return validated.takeError();
  return ProgramImage(std::move(bytes), std::move(validated->ranges));
}

llvm::Error writeProgramImage(const ProgramImage &image,
                              llvm::raw_ostream &output) {
  output.write(reinterpret_cast<const char *>(image.bytes().data()),
               image.bytes().size());
  output.flush();
  llvm::raw_fd_ostream *fileOutput = nullptr;
#if defined(__cpp_rtti) || defined(__GXX_RTTI) || defined(_CPPRTTI)
  fileOutput = dynamic_cast<llvm::raw_fd_ostream *>(&output);
#else
  if (output.get_kind() == llvm::raw_ostream::OStreamKind::OK_FDStream)
    fileOutput = static_cast<llvm::raw_fd_ostream *>(&output);
#endif
  if (fileOutput) {
    if (!fileOutput->has_error())
      return llvm::Error::success();
    const std::string message = fileOutput->error().message();
    fileOutput->clear_error();
    return makeImageError(ErrorKind::IOFailure,
                          llvm::Twine("failed to write ProgramImage: ") +
                              message);
  }
  return llvm::Error::success();
}

} // namespace mlir::cim22::artifact
