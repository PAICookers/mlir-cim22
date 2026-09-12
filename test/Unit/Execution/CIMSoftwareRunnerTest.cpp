//===- CIMSoftwareRunnerTest.cpp - CIM22 software runner tests -*- C++ -*-===//

#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Support/BF16Support.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cim22::execution;

namespace {
enum class FixtureFault {
  None,
  MissingInput,
  MissingWeight,
  MissingReadback,
  RouteMismatch,
  InvalidAddress,
  Overflow
};

struct VectorTrace final : CIMTraceSink {
  std::vector<CIMTraceEvent> events;
  void record(const CIMTraceEvent &event) override { events.push_back(event); }
};

void addPacket(std::vector<CIMFramePacket> &packets, CIMPacketKind kind,
               int64_t group, int64_t work, int64_t macro,
               int64_t address = -1) {
  CIMFramePacket packet{kind};
  packet.groupId = group;
  packet.workId = work;
  packet.macroSlot = macro;
  packet.cacheAddress = address;
  packets.push_back(packet);
}

CIMTransaction makeExecutable(FixtureFault fault = FixtureFault::None) {
  std::vector<CIMGroup> groups;
  for (int64_t groupId : {1, 0}) {
    CIMGroup group;
    group.groupId = groupId;
    for (int64_t macro = 0; macro < 2; ++macro)
      group.works.push_back({groupId * 2 + macro, 0, macro, {}});
    groups.push_back(std::move(group));
  }

  std::vector<StaticWeightSection> weights;
  std::vector<DynamicInputBinding> inputs;
  std::vector<ReadbackBinding> readbacks;
  std::vector<CIMFramePacket> packets;
  for (int64_t groupId = 0; groupId < 2; ++groupId) {
    for (int64_t macro = 0; macro < 2; ++macro) {
      int64_t workId = groupId * 2 + macro;
      StaticWeightSection weight;
      weight.groupId = groupId;
      weight.workId = workId;
      weight.coreSlot = 0;
      weight.macroSlot = macro;
      weight.values.fill(0);
      if (fault == FixtureFault::Overflow) {
        weight.values.fill(-128);
      } else {
        for (size_t row = 0; row < kCIMOutputElements; ++row) {
          weight.values[row * kCIMInputElements] =
              static_cast<int8_t>(groupId + macro + 1);
          weight.values[row * kCIMInputElements + 1] =
              static_cast<int8_t>(-static_cast<int>(row % 3));
        }
      }
      weights.push_back(std::move(weight));

      inputs.push_back({groupId, workId, macro, workId});
      readbacks.push_back({groupId, workId, macro, macro == 0 ? 0 : 7, {}, {}});

      if (fault != FixtureFault::MissingInput || macro != 1 || groupId != 1)
        addPacket(packets, CIMPacketKind::InputCacheWrite, groupId, workId,
                  macro);
      if (fault != FixtureFault::MissingWeight || macro != 0 || groupId != 1)
        addPacket(packets, CIMPacketKind::Weight, groupId, workId, macro);
      if (fault != FixtureFault::MissingReadback || macro != 1 ||
          groupId != 1) {
        addPacket(packets, CIMPacketKind::ReturnRoute, groupId, workId, macro);
        addPacket(packets, CIMPacketKind::Control, groupId, workId, macro);
        addPacket(packets, CIMPacketKind::OutputCacheRead, groupId, workId,
                  macro, macro == 0 ? 0 : 7);
      }
    }
  }
  if (fault == FixtureFault::RouteMismatch)
    readbacks.back().route[0] = 1;
  if (fault == FixtureFault::InvalidAddress)
    readbacks.back().outputCacheAddress = 8;

  return CIMTransaction("cim22-4x5-v1", 1, 1, std::move(groups),
                        std::move(weights), std::move(packets),
                        std::move(inputs), std::move(readbacks), {});
}

CIMTransaction
makeBF16Executable(bool withExponent = true, int8_t weightMantissa = 0,
                   const StaticWeightSection *fixture = nullptr) {
  const int64_t macro = fixture ? fixture->macroSlot : 0;
  CIMGroup group;
  group.groupId = 0;
  group.works.push_back({0, 0, macro, {}});

  StaticWeightSection weight;
  weight.groupId = 0;
  weight.workId = 0;
  weight.coreSlot = 0;
  weight.macroSlot = 0;
  weight.dataType = CIMDataType::BF16;
  weight.values.fill(weightMantissa);
  weight.exponents.fill(0);
  if (fixture)
    weight = *fixture;

  DynamicInputBinding input{0, 0, macro, 0, CIMDataType::BF16};
  ReadbackBinding readback{0, 0, macro, 0, {}, {}, CIMDataType::BF16};
  std::vector<CIMFramePacket> packets;
  auto add = [&](CIMPacketKind kind) {
    CIMFramePacket packet{kind};
    packet.groupId = 0;
    packet.workId = 0;
    packet.macroSlot = macro;
    if (kind == CIMPacketKind::OutputCacheRead)
      packet.cacheAddress = 0;
    packet.dataType = CIMDataType::BF16;
    packets.push_back(packet);
  };
  add(CIMPacketKind::InputCacheWrite);
  if (withExponent)
    add(CIMPacketKind::WeightExponent);
  add(CIMPacketKind::Weight);
  add(CIMPacketKind::ReturnRoute);
  add(CIMPacketKind::Control);
  add(CIMPacketKind::OutputCacheRead);
  return CIMTransaction("cim22-4x5-v1", 1, 1, {std::move(group)},
                        {std::move(weight)}, std::move(packets), {input},
                        {readback}, {});
}

int32_t expectedValue(const StaticWeightSection &weight,
                      llvm::ArrayRef<int8_t> input, size_t row) {
  int32_t sum = 0;
  for (size_t k = 0; k < kCIMInputElements; ++k)
    sum += static_cast<int32_t>(weight.values[row * kCIMInputElements + k]) *
           static_cast<int32_t>(input[k]);
  return sum;
}

// The Python adapter only parses supplier snapshots. Each expected stage is
// independent of the production helper/runner evaluated here.
template <typename T, size_t N>
std::array<T, N> fixtureArray(const llvm::json::Object &object,
                              llvm::StringRef field) {
  std::array<T, N> values{};
  const llvm::json::Array *array = object.getArray(field);
  if (!array || array->size() != N)
    llvm::report_fatal_error("invalid BF16 fixture array: " + field);
  for (size_t index = 0; index < N; ++index) {
    if constexpr (std::is_unsigned_v<T>)
      values[index] = static_cast<T>((*array)[index].getAsUINT64().value());
    else
      values[index] = static_cast<T>((*array)[index].getAsInteger().value());
  }
  return values;
}

bool replayBF16Fixtures() {
  auto buffer = llvm::MemoryBuffer::getSTDIN();
  if (!buffer) {
    llvm::errs() << buffer.getError().message() << '\n';
    return false;
  }
  auto json = llvm::json::parse((*buffer)->getBuffer());
  if (!json) {
    llvm::errs() << llvm::toString(json.takeError()) << '\n';
    return false;
  }
  const llvm::json::Array *rows = json->getAsArray();
  if (!rows || rows->empty())
    return false;
  CIMSoftwareRunner runner;
  for (const llvm::json::Value &value : *rows) {
    const llvm::json::Object &row = *value.getAsObject();
    auto checkStage = [&](bool matches, llvm::StringRef stage) {
      if (!matches)
        llvm::errs() << "FAIL " << row.getString("case").value()
                     << " row=" << row.getInteger("row").value()
                     << " stage=" << stage << '\n';
      return matches;
    };
    StaticWeightSection weight;
    weight.groupId = weight.workId = weight.coreSlot = 0;
    weight.macroSlot = row.getInteger("macro").value();
    weight.dataType = CIMDataType::BF16;
    weight.values = fixtureArray<int8_t, 1024>(row, "weight");
    weight.exponents = fixtureArray<uint8_t, 16>(row, "weight_exponents");
    const auto input = fixtureArray<uint16_t, 64>(row, "input");
    const auto aligned = mlir::cim22::prealignBF16Vector(input);
    if (!checkStage(aligned.exponent == row.getInteger("input_exponent"),
                    "input_exponent") ||
        !checkStage(aligned.mantissas ==
                        fixtureArray<int8_t, 64>(row, "aligned"),
                    "aligned") ||
        !checkStage(mlir::cim22::packBF16InputCacheRow(input) ==
                        fixtureArray<uint64_t, 16>(row, "input_words"),
                    "input_words") ||
        !checkStage(mlir::cim22::packBF16WeightExponents(weight.exponents) ==
                        fixtureArray<uint64_t, 2>(row, "exponent_words"),
                    "exponent_words"))
      return false;

    const auto intermediate = fixtureArray<int32_t, 16>(row, "intermediate");
    const auto expected = fixtureArray<uint16_t, 16>(row, "output");
    for (size_t lane = 0; lane < 16; ++lane) {
      const int32_t sum = expectedValue(weight, aligned.mantissas, lane);
      if (!checkStage(sum == intermediate[lane] && sum >= kCIMI21Min &&
                          sum < kCIMI21MaxExclusive,
                      "intermediate") ||
          !checkStage(mlir::cim22::int21ToBF16(
                          intermediate[lane], weight.exponents[lane],
                          aligned.exponent) == expected[lane],
                      "output"))
        return false;
    }
    if (!checkStage(mlir::cim22::decodeBF16OutputCacheResponse(
                        fixtureArray<uint64_t, 6>(row, "response")) == expected,
                    "response"))
      return false;

    auto executable = makeBF16Executable(true, 0, &weight);
    std::array<uint16_t, 16> actual{};
    CIMInputView inputView{{}, input};
    CIMOutputView outputView{{}, actual};
    CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(&inputView, 1)};
    CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(&outputView, 1)};
    // Reuse the transaction with distinct invocations, exercising both supplier
    // Macro identities without treating the packet list as a hardware relay.
    for (int repeat = 0; repeat < 2; ++repeat) {
      actual.fill(0xdead);
      if (auto error = runner.run(executable, inputs, outputs)) {
        llvm::errs() << llvm::toString(std::move(error)) << '\n';
        return false;
      }
      if (!checkStage(actual == expected, "runner"))
        return false;
    }
  }
  llvm::outs() << "PASS supplier-fixture-match BF16 rows=" << rows->size()
               << " inputs=" << rows->size() * 64
               << " outputs=" << rows->size() * 16 << '\n';
  return true;
}

const StaticWeightSection &weightFor(const CIMTransaction &executable,
                                     int64_t group, int64_t work) {
  for (const StaticWeightSection &weight : executable.getStaticWeights())
    if (weight.groupId == group && weight.workId == work)
      return weight;
  assert(false && "missing test weight");
  return executable.getStaticWeights().front();
}

void expectError(llvm::Error error) {
  assert(error && "expected runner error");
  llvm::consumeError(std::move(error));
}

void checkRun(CIMSoftwareRunner &runner, const CIMTransaction &executable,
              VectorTrace *trace = nullptr) {
  std::array<std::array<int8_t, kCIMInputElements>, 4> inputStorage{};
  std::array<std::array<int32_t, kCIMOutputElements>, 4> outputStorage{};
  for (size_t index = 0; index < inputStorage.size(); ++index)
    for (size_t k = 0; k < kCIMInputElements; ++k)
      inputStorage[index][k] =
          static_cast<int8_t>(static_cast<int>(index * 3 + k) % 15 - 7);

  std::array<CIMInputView, 4> inputViews{};
  std::array<CIMOutputView, 4> outputViews{};
  for (size_t index = 0; index < inputViews.size(); ++index) {
    inputViews[index].values = llvm::ArrayRef<int8_t>(inputStorage[index]);
    outputViews[index].values =
        llvm::MutableArrayRef<int32_t>(outputStorage[index]);
  }
  CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(inputViews)};
  CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(outputViews)};
  assert(!runner.run(executable, inputs, outputs, trace));

  for (size_t index = 0; index < executable.getReadbacks().size(); ++index) {
    const ReadbackBinding &binding = executable.getReadbacks()[index];
    const StaticWeightSection &weight =
        weightFor(executable, binding.groupId, binding.workId);
    size_t inputIndex = static_cast<size_t>(binding.workId);
    for (size_t row = 0; row < kCIMOutputElements; ++row)
      assert(outputStorage[index][row] ==
             expectedValue(weight, inputStorage[inputIndex], row));
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--bf16-fixtures")
    return replayBF16Fixtures() ? 0 : 1;
  CIMTransaction executable = makeExecutable();
  CIMSoftwareRunner runner;
  VectorTrace trace;
  checkRun(runner, executable, &trace);
  assert(trace.events.size() == 26);
  for (size_t index = 0; index < trace.events.size(); ++index)
    assert(trace.events[index].sequence == index);
  assert(trace.events.front().kind == CIMTraceEventKind::BeginRun);
  assert(trace.events.back().kind == CIMTraceEventKind::Complete);
  assert(trace.events[1].kind == CIMTraceEventKind::ConfigureInput);
  assert(trace.events[2].kind == CIMTraceEventKind::ConfigureWeight);
  assert(trace.events[3].kind == CIMTraceEventKind::SelectMacro);
  assert(trace.events[4].kind == CIMTraceEventKind::StartWork);
  assert(trace.events[9].groupId == 0 && trace.events[9].workId == 0);
  assert(trace.events[10].cacheRowOrAddress == 0);
  assert(trace.events[13].groupId == 1 && trace.events[13].workId == 2);

  checkRun(runner, executable);

  std::array<CIMInputView, 0> noInputs{};
  std::array<CIMOutputView, 4> badOutputs{};
  CIMRunInputs badInputRequest{llvm::ArrayRef<CIMInputView>(noInputs)};
  CIMRunOutputs badOutputRequest{
      llvm::MutableArrayRef<CIMOutputView>(badOutputs)};
  expectError(runner.run(executable, badInputRequest, badOutputRequest));

  for (FixtureFault fault :
       {FixtureFault::MissingInput, FixtureFault::MissingWeight,
        FixtureFault::MissingReadback, FixtureFault::RouteMismatch,
        FixtureFault::InvalidAddress, FixtureFault::Overflow}) {
    CIMTransaction faulty = makeExecutable(fault);
    std::array<std::array<int8_t, kCIMInputElements>, 4> inputStorage{};
    std::array<std::array<int32_t, kCIMOutputElements>, 4> outputStorage{};
    std::array<CIMInputView, 4> inputViews{};
    std::array<CIMOutputView, 4> outputViews{};
    for (size_t index = 0; index < 4; ++index) {
      inputViews[index].values = llvm::ArrayRef<int8_t>(inputStorage[index]);
      outputViews[index].values =
          llvm::MutableArrayRef<int32_t>(outputStorage[index]);
    }
    if (fault == FixtureFault::Overflow)
      for (auto &input : inputStorage)
        input.fill(-128);
    CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(inputViews)};
    CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(outputViews)};
    expectError(runner.run(faulty, inputs, outputs));
  }

  CIMTransaction bf16 = makeBF16Executable();
  std::array<uint16_t, kCIMInputElements> bf16InputStorage{};
  bf16InputStorage.fill(0x3f80);
  std::array<uint16_t, kCIMOutputElements> bf16OutputStorage{};
  CIMInputView bf16InputView;
  bf16InputView.bf16Values = llvm::ArrayRef<uint16_t>(bf16InputStorage);
  CIMOutputView bf16OutputView;
  bf16OutputView.bf16Values =
      llvm::MutableArrayRef<uint16_t>(bf16OutputStorage);
  CIMRunInputs bf16Inputs{llvm::ArrayRef<CIMInputView>(&bf16InputView, 1)};
  CIMRunOutputs bf16Outputs{
      llvm::MutableArrayRef<CIMOutputView>(&bf16OutputView, 1)};
  assert(!runner.run(bf16, bf16Inputs, bf16Outputs));
  for (uint16_t value : bf16OutputStorage)
    assert(value == 0x4000);

  // With input mantissa 64, 64 prealigned unit weights sum to 4096.
  // The supplier reconstruction with weight exponent 0 produces 0x4280.
  assert(!runner.run(makeBF16Executable(true, 1), bf16Inputs, bf16Outputs));
  for (uint16_t value : bf16OutputStorage)
    assert(value == 0x4280);
  assert(!runner.run(makeBF16Executable(true, -1), bf16Inputs, bf16Outputs));
  for (uint16_t value : bf16OutputStorage)
    assert(value == 0xc280);

  // Supplier negative zero aligns to -128. Reject the positive INT21
  // overflow corner instead of letting BF16 reconstruction hide it.
  bf16InputStorage.fill(0x8000);
  llvm::Error overflow =
      runner.run(makeBF16Executable(true, -128), bf16Inputs, bf16Outputs);
  assert(overflow && "negative-zero BF16 overflow must fail");
  std::string overflowMessage = llvm::toString(std::move(overflow));
  assert(overflowMessage == "CIM software result exceeds signed i21 range");
  bf16InputStorage.fill(0x3f80);

  bf16InputStorage[0] = 0x7f80;
  expectError(runner.run(bf16, bf16Inputs, bf16Outputs));
  bf16InputStorage[0] = 0x7fc1;
  expectError(runner.run(bf16, bf16Inputs, bf16Outputs));
  bf16InputStorage[0] = 0x3f80;
  expectError(runner.run(makeBF16Executable(false), bf16Inputs, bf16Outputs));

  DynamicInputBinding mixedInput{0, 0, 0, 0, CIMDataType::Int8};
  std::vector<CIMFramePacket> mixedPackets(bf16.getPackets().begin(),
                                           bf16.getPackets().end());
  std::vector<CIMGroup> mixedGroups(bf16.getGroups().begin(),
                                    bf16.getGroups().end());
  CIMTransaction mixed("cim22-4x5-v1", 1, 1, std::move(mixedGroups),
                       {bf16.getStaticWeights().front()},
                       std::move(mixedPackets), {mixedInput},
                       {bf16.getReadbacks().front()}, {});
  bf16InputStorage[0] = 0x3f80;
  expectError(runner.run(mixed, bf16Inputs, bf16Outputs));

  CIMUartRunner uartRunner;
  std::array<std::array<int8_t, kCIMInputElements>, 4> inputStorage{};
  std::array<std::array<int32_t, kCIMOutputElements>, 4> outputStorage{};
  std::array<CIMInputView, 4> inputViews{};
  std::array<CIMOutputView, 4> outputViews{};
  for (size_t index = 0; index < 4; ++index) {
    inputViews[index].values = llvm::ArrayRef<int8_t>(inputStorage[index]);
    outputViews[index].values =
        llvm::MutableArrayRef<int32_t>(outputStorage[index]);
  }
  CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(inputViews)};
  CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(outputViews)};
  expectError(uartRunner.run(executable, inputs, outputs));
  return 0;
}
