//===- CIMRTLArtifactExporter.cpp - INT8 RTL artifacts --------*- C++ -*-===//

#include "CIM22/Target/Passes.h"

#include "CIM22/Execution/CIMExecutable.h"
#include "CIM22/Target/CIMExecutable.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir::cim22::target {
#define GEN_PASS_DEF_EXPORTCIMRTLARTIFACTS
#include "CIM22/Target/Passes.h.inc"

namespace {
using ::cim22::execution::CIMTransaction;
using ::cim22::execution::CIMWork;
using ::cim22::execution::ReadbackBinding;
using ::cim22::execution::StaticWeightSection;

constexpr int64_t kCoreCount = 20;
constexpr int64_t kCacheRows = 8;
constexpr int64_t kInputElements = 64;
constexpr int64_t kOutputLanes = 16;
constexpr int64_t kInt21Min = -(int64_t{1} << 20);
constexpr int64_t kInt21Max = (int64_t{1} << 20) - 1;

using InputRow = std::array<int8_t, kInputElements>;
using InputRows = std::array<InputRow, kCacheRows>;
using OutputRow = std::array<int64_t, kOutputLanes>;
using OutputRows = std::array<OutputRow, kCacheRows>;
using Route = std::array<int32_t, 6>;
using TestCore = std::array<int32_t, 3>;

// The supplier 4x5 TB uses this fixed multicast expansion from (0,0).
constexpr Route kMulticastRoute{0, 0, 0, 0, 4, 3};

struct WorkArtifact {
  const CIMWork *work = nullptr;
  const StaticWeightSection *weight = nullptr;
  const ReadbackBinding *readback = nullptr;
};

struct TransactionArtifact {
  InputRows inputs{};
  std::vector<WorkArtifact> works;
  const WorkArtifact *macro[2] = {nullptr, nullptr};
  std::array<OutputRows, 2> expected{};
  llvm::SmallVector<uint64_t> configFlits;
  llvm::SmallVector<uint64_t> workFlits;
  llvm::SmallVector<uint64_t> readbackFlits;
};

std::string joinPath(StringRef root, ArrayRef<StringRef> components) {
  llvm::SmallString<256> result(root);
  for (StringRef component : components)
    llvm::sys::path::append(result, component);
  return result.str().str();
}

const StaticWeightSection *findWeight(const CIMTransaction &transaction,
                                      int64_t groupId, int64_t workId) {
  auto it = llvm::find_if(
      transaction.getStaticWeights(), [&](const StaticWeightSection &weight) {
        return weight.groupId == groupId && weight.workId == workId;
      });
  return it == transaction.getStaticWeights().end() ? nullptr : &*it;
}

const ReadbackBinding *findReadback(const CIMTransaction &transaction,
                                    int64_t groupId, int64_t workId,
                                    int64_t macroIdx) {
  auto it = llvm::find_if(
      transaction.getReadbacks(), [&](const ReadbackBinding &readback) {
        return readback.groupId == groupId && readback.workId == workId &&
               readback.macroSlot == macroIdx;
      });
  return it == transaction.getReadbacks().end() ? nullptr : &*it;
}

FailureOr<InputRows> readInputCache(ModuleOp module, StringRef filePath) {
  auto buffer = llvm::MemoryBuffer::getFile(filePath);
  if (!buffer)
    return module.emitError("cannot read input cache file '")
           << filePath << "': " << buffer.getError().message();

  InputRows inputs{};
  std::array<bool, kCacheRows> seen{};
  StringRef remaining = (*buffer)->getBuffer();
  while (!remaining.empty()) {
    auto [line, next] = remaining.split('\n');
    remaining = next;
    line = line.trim();
    if (line.empty())
      continue;
    auto [addressText, bits] = line.split(',');
    if (bits.empty())
      return module.emitError("input cache line must be address,bits: ")
             << line;
    int64_t address = 0;
    if (addressText.trim().getAsInteger(10, address) || address < 0 ||
        address >= kCacheRows || seen[address])
      return module.emitError("input cache address must cover 0..7 once: ")
             << addressText;
    bits = bits.trim();
    if (bits.size() != 1024 || bits.find_first_not_of("01") != StringRef::npos)
      return module.emitError("input cache row must contain exactly 1024 bits");
    for (int64_t element = 0; element < kInputElements; ++element) {
      StringRef slot = bits.substr(element * 16, 16);
      if (slot.substr(0, 8) != "00000000")
        return module.emitError(
            "input cache uses the TB 8-bit value in the low byte of each slot");
      uint64_t value = 0;
      if (slot.substr(8).getAsInteger(2, value))
        return module.emitError("invalid input cache byte");
      inputs[address][element] = static_cast<int8_t>(value);
    }
    seen[address] = true;
  }
  if (llvm::any_of(seen, [](bool value) { return !value; }))
    return module.emitError("input cache file must contain rows 0..7");
  return inputs;
}

FailureOr<OutputRows> computeExpected(ModuleOp module, const InputRows &inputs,
                                      const StaticWeightSection &weight) {
  OutputRows outputs{};
  for (int64_t row = 0; row < kCacheRows; ++row)
    for (int64_t lane = 0; lane < kOutputLanes; ++lane) {
      int64_t sum = 0;
      for (int64_t k = 0; k < kInputElements; ++k)
        sum += static_cast<int64_t>(inputs[row][k]) *
               static_cast<int64_t>(weight.values[lane * kInputElements + k]);
      if (sum < kInt21Min || sum > kInt21Max)
        return module.emitError("INT8 RTL expected output exceeds signed 21 bits");
      outputs[row][lane] = sum;
    }
  return outputs;
}

uint64_t encodeSignedMagnitude6(int32_t value) {
  const uint64_t magnitude = static_cast<uint64_t>(value < 0 ? -value : value);
  return value < 0 ? magnitude | uint64_t{0x20} : magnitude;
}

uint64_t encodeRoute(const Route &route) {
  constexpr unsigned shifts[] = {54, 48, 42, 36, 30, 24};
  uint64_t encoded = 0;
  for (auto [value, shift] : llvm::zip_equal(route, shifts))
    encoded |= encodeSignedMagnitude6(value) << shift;
  return encoded;
}

uint64_t controlFrame(const Route &route, int64_t macro) {
  return (uint64_t{0x8} << 60) | encodeRoute(route) |
         static_cast<uint64_t>(macro);
}

uint64_t workFrame(const Route &route, uint64_t run) {
  return (uint64_t{0x9} << 60) | encodeRoute(route) | run;
}

uint64_t testReturnRouteFrame(const Route &route, const TestCore &testCore) {
  return (uint64_t{0xa} << 60) | encodeRoute(route) |
         (encodeSignedMagnitude6(testCore[0]) << 12) |
         (encodeSignedMagnitude6(testCore[1]) << 6) |
         encodeSignedMagnitude6(testCore[2]);
}

uint64_t inputHeadFrame(int64_t row) {
  return (uint64_t{0x1} << 60) | encodeRoute(kMulticastRoute) |
         ((uint64_t{0x8} | static_cast<uint64_t>(row)) << 14) | 16;
}

uint64_t weightHeadFrame() {
  return (uint64_t{0x2} << 60) | encodeRoute(kMulticastRoute) | 256;
}

uint64_t cacheReadFrame(int64_t address) {
  return (uint64_t{0x5} << 60) | encodeRoute(kMulticastRoute) |
         (uint64_t{1} << 23) | (static_cast<uint64_t>(address) << 14);
}

uint64_t cimReadFrame() {
  return (uint64_t{0x6} << 60) | encodeRoute(kMulticastRoute) |
         (uint64_t{1} << 23);
}

FailureOr<TransactionArtifact>
buildArtifacts(ModuleOp module, const CIMTransaction &transaction,
               StringRef inputCacheFile) {
  if (inputCacheFile.empty())
    return module.emitError("INT8 RTL artifact export requires input-cache-file");
  FailureOr<InputRows> inputs = readInputCache(module, inputCacheFile);
  if (failed(inputs))
    return failure();

  TransactionArtifact result;
  result.inputs = *inputs;
  result.works.reserve(transaction.getReadbacks().size());
  std::array<std::array<bool, 2>, kCoreCount> seen{};
  for (const auto &group : transaction.getGroups()) {
    if (group.works.empty())
      return module.emitError("RTL artifact export rejects an empty group");
    for (const CIMWork &work : group.works) {
      if (work.coreSlot < 0 || work.coreSlot >= kCoreCount ||
          work.macroSlot < 0 || work.macroSlot > 1)
        return module.emitError("RTL artifact export requires core_idx in [0,19] "
                                "and macro_idx in [0,1]");
      const StaticWeightSection *weight =
          findWeight(transaction, group.groupId, work.workId);
      const ReadbackBinding *readback =
          findReadback(transaction, group.groupId, work.workId, work.macroSlot);
      if (!weight || !readback)
        return module.emitError("RTL artifact export lacks binding for work ")
               << work.workId;
      if (weight->words.size() != 256)
        return module.emitError("INT8 RTL artifact requires 256 CIM weight words");
      if (work.route[3] != 0 || work.route[4] != 0 || work.route[5] != 0)
        return module.emitError("RTL artifact export requires onecast plan routes");
      if (seen[work.coreSlot][work.macroSlot])
        return module.emitError("RTL artifact export rejects duplicate core/macro");
      seen[work.coreSlot][work.macroSlot] = true;
      result.works.push_back(WorkArtifact{&work, weight, readback});
      WorkArtifact *current = &result.works.back();
      if (!result.macro[work.macroSlot])
        result.macro[work.macroSlot] = current;
      else if (result.macro[work.macroSlot]->weight->words != weight->words)
        return module.emitError(
            "multicast source mode cannot represent different weights for one Macro");
    }
  }
  if (result.works.empty() || !result.macro[0] || !result.macro[1])
    return module.emitError("INT8 RTL artifact export requires both active Macros");

  for (int64_t macro = 0; macro < 2; ++macro) {
    FailureOr<OutputRows> expected =
        computeExpected(module, result.inputs, *result.macro[macro]->weight);
    if (failed(expected))
      return failure();
    result.expected[macro] = *expected;
  }

  // The supplier TB configures the response route once per core, not once per
  // Macro work.
  std::array<bool, kCoreCount> configuredCore{};
  for (const WorkArtifact &work : result.works) {
    if (configuredCore[work.work->coreSlot])
      continue;
    configuredCore[work.work->coreSlot] = true;
    result.configFlits.push_back(
        testReturnRouteFrame(work.work->route, work.readback->testCore));
  }

  // One multicast payload is shared by every core for each selected Macro.
  for (int64_t macro = 0; macro < 2; ++macro) {
    const WorkArtifact &representative = *result.macro[macro];
    result.configFlits.push_back(controlFrame(kMulticastRoute, macro));
    for (int64_t row = 0; row < kCacheRows; ++row) {
      result.configFlits.push_back(inputHeadFrame(row));
      for (int64_t word = 0; word < 16; ++word) {
        uint64_t packed = 0;
        for (int64_t slot = 0; slot < 4; ++slot)
          packed = (packed << 16) |
                   static_cast<uint64_t>(static_cast<uint8_t>(
                       result.inputs[row][word * 4 + slot]));
        result.configFlits.push_back(packed);
      }
    }
    result.configFlits.push_back(weightHeadFrame());
    for (int32_t word : representative.weight->words)
      result.configFlits.push_back(static_cast<uint64_t>(static_cast<uint32_t>(word)));
  }

  result.workFlits.push_back(workFrame(kMulticastRoute, 1));
  result.workFlits.push_back(workFrame(kMulticastRoute, 0));

  for (int64_t macro = 0; macro < 2; ++macro) {
    result.readbackFlits.push_back(controlFrame(kMulticastRoute, macro));
    for (int64_t address = 0; address < kCacheRows; ++address)
      result.readbackFlits.push_back(cacheReadFrame(address));
    result.readbackFlits.push_back(cimReadFrame());
  }
  return result;
}

LogicalResult writeFile(ModuleOp module, StringRef filePath,
                        const std::function<void(llvm::raw_ostream &)> &write) {
  std::error_code error;
  llvm::raw_fd_ostream stream(filePath, error, llvm::sys::fs::OF_Text);
  if (error)
    return module.emitError("cannot create '")
           << filePath << "': " << error.message();
  write(stream);
  stream.close();
  if (stream.has_error())
    return module.emitError("cannot finish writing '") << filePath << "'";
  return success();
}

LogicalResult writeFlits(ModuleOp module, StringRef filePath,
                         ArrayRef<uint64_t> flits) {
  return writeFile(module, filePath, [&](llvm::raw_ostream &stream) {
    for (uint64_t flit : flits)
      stream << std::bitset<64>(flit).to_string() << '\n';
  });
}

LogicalResult writeWeight(ModuleOp module, StringRef filePath,
                          const StaticWeightSection &weight) {
  return writeFile(module, filePath, [&](llvm::raw_ostream &stream) {
    for (auto [address, word] : llvm::enumerate(weight.words))
      stream << address << ','
             << std::bitset<32>(static_cast<uint32_t>(word)).to_string()
             << '\n';
  });
}

LogicalResult writeInputs(ModuleOp module, StringRef filePath,
                          const InputRows &inputs) {
  return writeFile(module, filePath, [&](llvm::raw_ostream &stream) {
    for (auto [address, row] : llvm::enumerate(inputs)) {
      stream << address << ',';
      for (int8_t value : row)
        stream << "00000000"
               << std::bitset<8>(static_cast<uint8_t>(value)).to_string();
      stream << '\n';
    }
  });
}

LogicalResult writeExpectedOutput(ModuleOp module, StringRef filePath,
                                  const OutputRows &outputs) {
  return writeFile(module, filePath, [&](llvm::raw_ostream &stream) {
    constexpr uint64_t mask = (uint64_t{1} << 21) - 1;
    for (const OutputRow &row : outputs) {
      for (int64_t lane = kOutputLanes - 1; lane >= 0; --lane) {
        if (lane != kOutputLanes - 1)
          stream << ' ';
        stream << std::bitset<21>(static_cast<uint64_t>(row[lane]) & mask)
                      .to_string();
      }
      stream << '\n';
    }
  });
}

LogicalResult writeArtifacts(ModuleOp module, StringRef outputDir,
                             const TransactionArtifact &artifact) {
  std::error_code error = llvm::sys::fs::create_directories(
      joinPath(outputDir, {"sources"}));
  if (error)
    return module.emitError("cannot create INT8 RTL artifact directories: ")
           << error.message();
  error = llvm::sys::fs::create_directories(joinPath(outputDir, {"expected"}));
  if (error)
    return module.emitError("cannot create INT8 RTL expected directory: ")
           << error.message();

  if (failed(writeFlits(module, joinPath(outputDir, {"01_config.frames.txt"}),
                        artifact.configFlits)) ||
      failed(writeFlits(module, joinPath(outputDir, {"02_work.frames.txt"}),
                        artifact.workFlits)) ||
      failed(writeFlits(module, joinPath(outputDir, {"03_readback.frames.txt"}),
                        artifact.readbackFlits)))
    return failure();

  if (failed(writeWeight(module, joinPath(outputDir, {"sources", "cim1_w.txt"}),
                         *artifact.macro[0]->weight)) ||
      failed(writeWeight(module, joinPath(outputDir, {"sources", "cim2_w.txt"}),
                         *artifact.macro[1]->weight)) ||
      failed(writeInputs(module, joinPath(outputDir, {"sources", "cache1_in.txt"}),
                         artifact.inputs)) ||
      failed(writeInputs(module, joinPath(outputDir, {"sources", "cache2_in.txt"}),
                         artifact.inputs)) ||
      failed(writeExpectedOutput(module,
                                 joinPath(outputDir, {"expected", "output_1.txt"}),
                                 artifact.expected[0])) ||
      failed(writeExpectedOutput(module,
                                 joinPath(outputDir, {"expected", "output_2.txt"}),
                                 artifact.expected[1])))
    return failure();
  return success();
}

class ExportCIMRTLArtifacts final
    : public impl::ExportCIMRTLArtifactsBase<ExportCIMRTLArtifacts> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (outputDir.empty()) {
      module.emitError("export-cim-rtl-artifacts requires output-dir");
      return signalPassFailure();
    }
    if (llvm::sys::fs::exists(outputDir)) {
      module.emitError("RTL artifact output directory already exists: ")
          << outputDir;
      return signalPassFailure();
    }
    FailureOr<CIMTransaction> transaction = compileCIMTransaction(module);
    if (failed(transaction))
      return signalPassFailure();
    FailureOr<TransactionArtifact> artifact =
        buildArtifacts(module, *transaction, inputCacheFile);
    if (failed(artifact) ||
        failed(writeArtifacts(module, outputDir, *artifact)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::cim22::target
