//===- CIMSoftwareRunnerTest.cpp - CIM22 software runner tests -*- C++ -*-===//

#include "CIM22/Execution/CIMRunner.h"

#include <array>
#include <cassert>
#include <cstdint>
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

int32_t expectedValue(const StaticWeightSection &weight,
                      llvm::ArrayRef<int8_t> input, size_t row) {
  int32_t sum = 0;
  for (size_t k = 0; k < kCIMInputElements; ++k)
    sum += static_cast<int32_t>(weight.values[row * kCIMInputElements + k]) *
           static_cast<int32_t>(input[k]);
  return sum;
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

int main() {
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
