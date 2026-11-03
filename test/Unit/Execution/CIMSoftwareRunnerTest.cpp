//===- CIMSoftwareRunnerTest.cpp - CIM22 software runner tests -*- C++ -*-===//

#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Support/BF16Support.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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
  weight.exponents.fill(127);
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
  for (const llvm::json::Value &value : *rows) {
    const llvm::json::Object &row = *value.getAsObject();
    auto checkStage = [&](bool matches, llvm::StringRef stage) {
      if (!matches)
        llvm::errs() << "FAIL " << row.getString("case").value()
                     << " row=" << row.getInteger("row").value()
                     << " stage=" << stage << '\n';
      return matches;
    };
    // Historical supplier arithmetic remains evidence, not the active numeric
    // oracle. Its immutable streams still independently verify frame layout.
    const auto input = fixtureArray<uint16_t, 64>(row, "input");
    const auto exponents = fixtureArray<uint8_t, 16>(row, "weight_exponents");
    if (!checkStage(mlir::cim22::packBF16InputCacheRow(input) ==
                        fixtureArray<uint64_t, 16>(row, "input_words"),
                    "input_words") ||
        !checkStage(mlir::cim22::packBF16WeightExponents(exponents) ==
                        fixtureArray<uint64_t, 2>(row, "exponent_words"),
                    "exponent_words") ||
        !checkStage(mlir::cim22::decodeBF16OutputCacheResponse(
                        fixtureArray<uint64_t, 6>(row, "response")) ==
                        fixtureArray<uint16_t, 16>(row, "output"),
                    "response"))
      return false;
  }
  llvm::outs() << "PASS historical-layout-match BF16 rows=" << rows->size()
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
// Analytic expectations exercise the production runner without compiler or
// Host fallback; the pair sweep does not duplicate its reduction loop.
void checkInt8NumericModel() {
  CIMSoftwareRunner runner;
  std::array<int8_t, kCIMInputElements> input{};
  std::array<int32_t, kCIMOutputElements> output{};
  std::array<int64_t, kCIMOutputElements> expected{};
  std::array<int8_t, 16 * 64> values{};
  CIMInputView inputView{input, {}};
  CIMOutputView outputView{output, {}};
  CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(&inputView, 1)};
  CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(&outputView, 1)};
  auto transaction = [&] {
    CIMGroup group{0, {{0, 0, 0, {}}}};
    StaticWeightSection weight;
    weight.groupId = weight.workId = weight.coreSlot = weight.macroSlot = 0;
    weight.values = values;
    std::vector<CIMFramePacket> packets;
    for (auto kind : {CIMPacketKind::InputCacheWrite, CIMPacketKind::Weight,
                      CIMPacketKind::ReturnRoute, CIMPacketKind::Control,
                      CIMPacketKind::OutputCacheRead})
      addPacket(packets, kind, 0, 0, 0, 0);
    return CIMTransaction("cim22-4x5-v1", 1, 1, {group}, {weight},
                          std::move(packets), {{0, 0, 0, 0}},
                          {{0, 0, 0, 0, {}, {}}}, {});
  };
  auto check = [&](const CIMTransaction &executable) {
    const auto original = input;
    output.fill(77);
    assert(!runner.run(executable, inputs, outputs));
    for (size_t lane = 0; lane < output.size(); ++lane)
      assert(static_cast<int64_t>(output[lane]) == expected[lane]);
    assert(input == original);
  };

  // All 65,536 signed byte products at both ends of each 32-item half.
  for (size_t k : {0, 31, 32, 63}) {
    input.fill(0);
    for (int base = -128; base <= 127; base += 16) {
      values.fill(0);
      for (size_t lane = 0; lane < output.size(); ++lane)
        values[lane * 64 + k] =
            static_cast<int8_t>(base + static_cast<int>(lane));
      const auto executable = transaction();
      for (int a = -128; a <= 127; ++a) {
        input[k] = static_cast<int8_t>(a);
        for (size_t lane = 0; lane < output.size(); ++lane)
          expected[lane] =
              static_cast<int64_t>(a) * (base + static_cast<int>(lane));
        check(executable);
      }
    }
  }
  for (size_t k = 0; k < input.size(); ++k) {
    input.fill(0);
    input[k] = -128;
    values.fill(0);
    for (size_t lane = 0; lane < output.size(); ++lane) {
      values[lane * 64 + k] = static_cast<int8_t>(static_cast<int>(lane) - 8);
      expected[lane] = -128 * (static_cast<int>(lane) - 8);
    }
    check(transaction());
  }

  input.fill(-128);
  values.fill(-128);
  for (size_t lane = 0; lane < output.size(); ++lane)
    values[lane * 64 + 63] = -127;
  expected.fill(1048448); // 63*16384 + 16256, the nearest safe corner.
  check(transaction());
  values.fill(127);
  const auto negative = transaction();
  expected.fill(-1040384); // 64*(-128)*127.
  check(negative);
  input.fill(0);
  expected.fill(0);
  check(negative);
  input.fill(-128);
  expected.fill(-1040384);
  check(negative);

  input.fill(127);
  for (size_t lane = 0; lane < output.size(); ++lane)
    for (size_t k = 0; k < input.size(); ++k)
      values[lane * 64 + k] = k % 2 ? -127 : 127;
  expected.fill(0); // Equal positive/negative terms cancel before range check.
  check(transaction());

  input.fill(-128);
  values.fill(127);
  std::fill_n(values.end() - 64, 64, int8_t{-128});
  output.fill(77);
  VectorTrace trace;
  auto error = runner.run(transaction(), inputs, outputs, &trace);
  assert(error && "64*(-128)*(-128) must not wrap or saturate");
  assert(llvm::toString(std::move(error)).find("signed i21") !=
         std::string::npos);
  for (int32_t value : output)
    assert(value == 77); // No lane from this rejected group was published.
  assert(trace.events.back().kind == CIMTraceEventKind::Error);
  assert(std::none_of(trace.events.begin(), trace.events.end(),
                      [](const auto &event) {
                        return event.kind == CIMTraceEventKind::Complete;
                      }));
  expected.fill(-1040384);
  check(negative); // A failed call does not poison the next run.
  llvm::outs() << "PASS INT8 software model: 262144 pair-position checks, "
                  "1024 lane-position checks, signed dense extremes, "
                  "zero/cancellation/reuse, signed21 overflow rejection\n";
}

void checkBF16LegacyCompensation() {
  using Mode = BF16Compensation;
  std::array<uint16_t, kCIMInputElements> input{}, laneWeights{};
  std::array<uint16_t, kCIMOutputElements> output{};
  CIMInputView inputView{{}, input};
  CIMOutputView outputView{{}, output};
  CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(&inputView, 1)};
  CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(&outputView, 1)};
  auto run = [&](Mode mode) {
    std::array<uint16_t, kCIMOutputElements * kCIMInputElements> bits{};
    for (size_t lane = 0; lane < kCIMOutputElements; ++lane)
      std::copy(laneWeights.begin(), laneWeights.end(),
                bits.begin() + lane * kCIMInputElements);
    auto prepared = mlir::cim22::prealignBF16WeightTile(bits);
    StaticWeightSection weight;
    weight.groupId = weight.workId = weight.coreSlot = weight.macroSlot = 0;
    weight.dataType = CIMDataType::BF16;
    weight.values = prepared.mantissas;
    weight.exponents = prepared.exponents;
    CIMSoftwareRunner runner(mode);
    return runner.run(makeBF16Executable(true, 0, &weight), inputs, outputs);
  };
  auto check = [&](Mode mode, uint16_t expected) {
    const auto original = input;
    assert(!run(mode));
    assert(input == original);
    for (uint16_t value : output)
      assert(value == expected);
  };
  input.fill(0x3f80);
  laneWeights.fill(0x3f00);
  check(Mode::Legacy, 0x4200); // 64 * .5 = 32.
  laneWeights.fill(0x3f08);
  check(Mode::None, 0x7f88);   // Uncompensated NaN is intentional.
  check(Mode::Legacy, 0x4208); // 34.

  input.fill(0);
  laneWeights.fill(0);
  input[0] = input[1] = 0x3f80;
  laneWeights[0] = 0x3f80;
  laneWeights[1] = 0xbf00;
  check(Mode::Legacy, 0xbf00); // Legacy returns -.5 instead of +.5.

  laneWeights[0] = laneWeights[1] = 0x3f00;
  input[1] = 0xbf00;
  check(Mode::Legacy, 0xbe80); // Legacy returns -.25 instead of +.25.
  input[1] = 0xbf80;
  check(Mode::Legacy, 0x3c80); // A zero dot retains the old false 1/64.
  input[0] = 0x3f84;
  check(Mode::Legacy, 0x3c80); // p=128 shares that converter bit pattern.

  input.fill(0);
  laneWeights.fill(0);
  input[0] = 0x7880; // 2^114: delta=1, input exponent becomes 240.
  laneWeights[0] = 0x3f00;
  check(Mode::Legacy, 0x7800); // 2^113.
  input[0] = 1;
  laneWeights[0] = 0x3f80;
  check(Mode::Legacy, 0); // Preserve the old all-subnormal input loss.
  input[0] = 0x7f80;
  expectError(run(Mode::Legacy));
  input[0] = 0x7fc1;
  expectError(run(Mode::Legacy));
  input[0] = input[1] = 0x7f7f;
  laneWeights[0] = laneWeights[1] = 0x3f80;
  expectError(run(Mode::Legacy)); // True BF16 overflow is not saturated.
}

void checkBF16NonnegativeCorrection() {
  CIMSoftwareRunner corrected(BF16Compensation::SignSplit);
  std::array<uint16_t, kCIMInputElements> input{};
  std::array<uint16_t, kCIMOutputElements> output{};
  CIMInputView inputView{{}, input};
  CIMOutputView outputView{{}, output};
  CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(&inputView, 1)};
  CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(&outputView, 1)};
  StaticWeightSection weight;
  weight.groupId = weight.workId = weight.coreSlot = weight.macroSlot = 0;
  weight.dataType = CIMDataType::BF16;
  auto setWeight = [&](int8_t mantissa, uint8_t exponent) {
    weight.values.fill(mantissa);
    weight.exponents.fill(exponent);
  };
  auto run = [&] {
    return corrected.run(makeBF16Executable(true, 0, &weight), inputs, outputs);
  };
  auto check = [&](uint16_t expected) {
    const auto originalInput = input;
    const auto originalWeight = weight.values;
    const auto originalExponents = weight.exponents;
    assert(!run());
    for (uint16_t bits : output)
      assert(bits == expected);
    assert(input == originalInput && weight.values == originalWeight &&
           weight.exponents == originalExponents);
  };

  setWeight(64, 126); // .5.
  input.fill(0x3f80);
  check(0x4200); // 64*.5 = 32; E127 requires no normalization.

  input.fill(0);
  input[0] = 0x3f80;
  input[1] = 0x3f00;
  check(0x3f40); // Sparse .75.

  // Nonzero p=128 has the same raw chip bits as p=0. Support, rather than
  // output bits or the internal integer sum, decides whether to publish zero.
  setWeight(2, 127);
  input[1] = 0;
  check(0x3d00); // q_input=64, q_weight=2: true product 1/32.

  // Orthogonal support, an all-zero weight lane and input lost in ordinary
  // alignment all skip the chip path. The retained lane still computes one.
  setWeight(0, 127);
  weight.values[0] = 64;
  weight.values[64 + 1] = 64;
  weight.values[3 * 64 + 2] = 64;
  input[1] = 0x3b80; // 2^-8, below the retained alignment precision.
  assert(!run());
  assert(output[0] == 0x3f80);
  for (size_t lane = 1; lane < output.size(); ++lane)
    assert(output[lane] == 0);

  setWeight(64, 127); // Unit weights.
  input.fill(0);
  check(0);
  input.fill(0x8000); // Negative zero is accepted and canonicalized locally.
  check(0);

  // Full positive BF16 exponent range, including all-subnormal rows. The
  // smallest subnormal needs a +133 shift before the unmodified prealigner.
  input.fill(0);
  input[0] = 1;
  check(1);
  input[0] = 0x007f;
  check(0x007f);
  input[0] = 0x0080;
  check(0x0080);
  input[0] = 0x3f00;
  check(0x3f00);
  input[0] = 0x7f00; // 2^127 remains finite after compensation.
  check(0x7f00);

  // Mixed normal/subnormal support drops only the unresolvable small input.
  input[0] = 0x3f80;
  input[1] = 1;
  check(0x3f80);

  // Sparse support traversal must reach bit 63 after an earlier unmatched
  // input bit, and must include the final prepared weight byte.
  input.fill(0);
  input[0] = input[63] = 0x3f80;
  setWeight(0, 127);
  weight.values.back() = 64;
  assert(!run());
  for (size_t lane = 0; lane + 1 < output.size(); ++lane)
    assert(output[lane] == 0);
  assert(output.back() == 0x3f80);

  // Final ABI rounding is allowed to underflow a genuinely tiny result.
  input.fill(0);
  input[0] = 1;
  input[1] = 0;
  setWeight(1, 127); // 1/64.
  check(0);

  setWeight(64, 254); // 2^127 * 2 exceeds the true BF16 finite range.
  input[0] = 0x4000;
  expectError(run());
  setWeight(64, 127);
  input[0] = 0x3f80;
  input[1] = 0xbf80;
  expectError(run());
  input[1] = 0x8001; // Negative subnormal also rejected.
  expectError(run());
  for (uint16_t nonfinite :
       {uint16_t{0x7f80}, uint16_t{0x7fc1}, uint16_t{0xff80}}) {
    input[1] = nonfinite;
    expectError(run());
  }
  input.fill(0); // Zero input does not hide an invalid prepared weight.
  weight.values.back() = -1;
  expectError(run());
  setWeight(64, 255);
  expectError(run());
  setWeight(64, 127);
  input[0] = 0x3f80;
  check(0x3f80); // Rejection does not poison the following work.

  llvm::outs() << "PASS nonnegative BF16 correction: normal/sparse, true "
                  "p128, quantized zero support, negative zero, full exponent "
                  "range/subnormal, input/weight rejection, true overflow\n";
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--bf16-layout-fixtures")
    return replayBF16Fixtures() ? 0 : 1;
  checkBF16NonnegativeCorrection();
  checkBF16LegacyCompensation();
  checkInt8NumericModel();
  if (argc == 2 && llvm::StringRef(argv[1]) == "--int8-numeric")
    return 0;
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
    assert(value == 0x7f80);

  // Unit weights/input both align to 64 at E=127. The raw exponent sum is
  // clamped to 255 by the RTL-derived reconstruction model.
  assert(!runner.run(makeBF16Executable(true, 64), bf16Inputs, bf16Outputs));
  for (uint16_t value : bf16OutputStorage)
    assert(value == 0x7f80);
  assert(!runner.run(makeBF16Executable(true, -64), bf16Inputs, bf16Outputs));
  for (uint16_t value : bf16OutputStorage)
    assert(value == 0xff80);

  bf16InputStorage.fill(0x8000);
  // ExpPrealign maps an all-negative-zero row to -128. With -128 weights,
  // 64 products exceed signed INT21 and hit the existing overflow gate.
  expectError(
      runner.run(makeBF16Executable(true, -128), bf16Inputs, bf16Outputs));
  assert(!runner.run(makeBF16Executable(true, -127), bf16Inputs, bf16Outputs));
  for (uint16_t value : bf16OutputStorage)
    assert(value == 0x467e);
  // Complementing 0xff before the shift maps this dense input to -128.
  bf16InputStorage.fill(0xbfff);
  assert(!runner.run(makeBF16Executable(true, -127), bf16Inputs, bf16Outputs));
  for (uint16_t value : bf16OutputStorage)
    assert(value == 0x7ffe); // Direct clamp with mantissa 254.
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

  return 0;
}
