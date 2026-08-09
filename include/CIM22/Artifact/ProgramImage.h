#ifndef CIM22_ARTIFACT_PROGRAMIMAGE_H
#define CIM22_ARTIFACT_PROGRAMIMAGE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace cim22::image {
struct ProgramImage;
}

namespace mlir::cim22::artifact {

inline constexpr uint16_t kProgramImageFormatMajor = 1;
inline constexpr uint16_t kProgramImageFormatMinor = 0;
inline constexpr char kProgramImageTargetProfile[] = "cim22-4x5-v1";
inline constexpr uint64_t kProgramImageTargetProfileVersion = 1;
inline constexpr uint64_t kProgramImageInvocationSchemaVersion = 1;
inline constexpr uint64_t kProgramImageLogicalViewSchemaVersion = 1;
inline constexpr uint32_t kProgramImageSectionAlignment = 8;

struct LogicalSlotRef {
  uint64_t workId = 0;
  uint32_t logicalSlot = 0;
};

struct SegmentRef {
  uint64_t groupId = 0;
  std::vector<LogicalSlotRef> hostToCIM;
  std::vector<LogicalSlotRef> cimToHost;
};

struct OpaqueSectionInput {
  uint32_t id = 0;
  llvm::ArrayRef<uint8_t> bytes;
};

enum class ProgramImageErrorKind {
  InvalidArgument,
  MalformedBuffer,
  IncompatibleMetadata,
  IntegrityMismatch,
  IOFailure,
};

class ProgramImageError final : public llvm::ErrorInfo<ProgramImageError> {
public:
  static char ID;

  ProgramImageError(ProgramImageErrorKind kind, std::string message)
      : errorKind(kind), errorMessage(std::move(message)) {}

  ProgramImageErrorKind kind() const { return errorKind; }
  void log(llvm::raw_ostream &os) const override;
  std::error_code convertToErrorCode() const override;

private:
  ProgramImageErrorKind errorKind;
  std::string errorMessage;
};

class ProgramImage {
public:
  llvm::ArrayRef<uint8_t> bytes() const { return ownedBytes; }
  const ::cim22::image::ProgramImage *metadata() const;
  llvm::Expected<llvm::ArrayRef<uint8_t>> section(uint32_t id) const;

private:
  using SectionRange = std::pair<size_t, size_t>;

  ProgramImage(std::vector<uint8_t> bytes,
               std::vector<SectionRange> sectionRanges)
      : ownedBytes(std::move(bytes)), ranges(std::move(sectionRanges)) {}

  std::vector<uint8_t> ownedBytes;
  std::vector<SectionRange> ranges;

  friend llvm::Expected<ProgramImage>
  buildProgramImage(llvm::StringRef entryFunction,
                    llvm::ArrayRef<SegmentRef> segments,
                    llvm::ArrayRef<OpaqueSectionInput> sections);
  friend llvm::Expected<ProgramImage>
  loadProgramImage(llvm::ArrayRef<uint8_t> bytes);
};

llvm::Expected<ProgramImage>
buildProgramImage(llvm::StringRef entryFunction,
                  llvm::ArrayRef<SegmentRef> segments,
                  llvm::ArrayRef<OpaqueSectionInput> sections);

llvm::Expected<ProgramImage> loadProgramImage(llvm::ArrayRef<uint8_t> bytes);

llvm::Error writeProgramImage(const ProgramImage &image,
                              llvm::raw_ostream &output);

} // namespace mlir::cim22::artifact

#endif // CIM22_ARTIFACT_PROGRAMIMAGE_H
