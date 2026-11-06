//===- CIMBoardRunner.cpp - CIM22 hardware runner -------------*- C++ -*-===//

#include "CIM22/Runtime/CIMBoardRunner.h"

#include "CIM22/Runtime/CIMPacketCodec.h"
#include "CIM22/Support/BF16Support.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace cim22::runtime {
namespace {
using namespace cim22::execution;

llvm::Error runnerError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

const CIMWork *findWork(const CIMTransaction &executable, int64_t groupId,
                        int64_t workId) {
  for (const CIMGroup &group : executable.getGroups())
    if (group.groupId == groupId)
      for (const CIMWork &work : group.works)
        if (work.workId == workId)
          return &work;
  return nullptr;
}

const StaticWeightSection *findWeight(const CIMTransaction &executable,
                                      int64_t groupId, int64_t workId) {
  auto it = llvm::find_if(
      executable.getStaticWeights(), [&](const StaticWeightSection &weight) {
        return weight.groupId == groupId && weight.workId == workId;
      });
  return it == executable.getStaticWeights().end() ? nullptr : &*it;
}

const DynamicInputBinding *findInput(const CIMTransaction &executable,
                                     int64_t groupId, int64_t workId,
                                     size_t &index) {
  for (auto [candidateIndex, input] :
       llvm::enumerate(executable.getDynamicInputs())) {
    if (input.groupId == groupId && input.workId == workId) {
      index = candidateIndex;
      return &input;
    }
  }
  return nullptr;
}

const ReadbackBinding *findReadback(const CIMTransaction &executable,
                                    int64_t groupId, int64_t workId,
                                    size_t &index) {
  for (auto [candidateIndex, readback] :
       llvm::enumerate(executable.getReadbacks())) {
    if (readback.groupId == groupId && readback.workId == workId) {
      index = candidateIndex;
      return &readback;
    }
  }
  return nullptr;
}

struct WorkBindings {
  const DynamicInputBinding *input = nullptr;
  const StaticWeightSection *weight = nullptr;
  const ReadbackBinding *readback = nullptr;
  size_t inputIndex = 0;
};

llvm::Expected<WorkBindings>
resolveWorkBindings(const CIMTransaction &executable, const CIMGroup &group,
                    const CIMWork &work) {
  WorkBindings bindings;
  size_t readbackIndex = 0;
  bindings.input =
      findInput(executable, group.groupId, work.workId, bindings.inputIndex);
  bindings.weight = findWeight(executable, group.groupId, work.workId);
  bindings.readback =
      findReadback(executable, group.groupId, work.workId, readbackIndex);
  if (!bindings.input || !bindings.weight || !bindings.readback)
    return runnerError("CIM work has incomplete runtime bindings");
  if (work.coreSlot < 0 || work.coreSlot >= 20 || work.macroSlot < 0 ||
      work.macroSlot >= 2 || bindings.input->macroSlot != work.macroSlot ||
      bindings.weight->coreSlot != work.coreSlot ||
      bindings.weight->macroSlot != work.macroSlot ||
      bindings.readback->macroSlot != work.macroSlot)
    return runnerError("CIM work has inconsistent Core or Macro placement");
  if (bindings.input->dataType != bindings.weight->dataType ||
      bindings.input->dataType != bindings.readback->dataType)
    return runnerError("CIM work has inconsistent data mode");
  return bindings;
}

void emitTrace(CIMTraceSink *trace, uint64_t &sequence, CIMTraceEventKind kind,
               int64_t groupId = -1, int64_t workId = -1,
               int64_t macroSlot = -1, int64_t packetIndex = -1,
               int64_t cacheAddress = -1) {
  if (!trace)
    return;
  trace->record({sequence++, kind, groupId, workId, macroSlot, packetIndex,
                 cacheAddress});
}

llvm::Error validateInputs(const CIMTransaction &executable,
                           const CIMRunInputs &inputs) {
  if (executable.getTargetProfile() != "cim22-4x5-v1" ||
      executable.getTargetProfileVersion() != 1 ||
      executable.getExecutionPlanVersion() != 1)
    return runnerError("unsupported CIM executable profile or version");
  if (inputs.values.size() != executable.getDynamicInputs().size())
    return runnerError("CIM input count does not match executable bindings");

  for (const CIMGroup &group : executable.getGroups()) {
    if (group.works.empty())
      return runnerError("CIM group has no work");
    const CIMWork &first = group.works.front();
    if (first.coreSlot < 0 || first.coreSlot >= 20)
      return runnerError("CIM work Core is outside [0, 19]");
    if (llvm::Error error = validateCIMRoute(first.route))
      return error;
    std::set<std::pair<int64_t, int64_t>> placements;
    for (const CIMWork &work : group.works) {
      if (work.coreSlot != first.coreSlot || work.route != first.route)
        return runnerError(
            "CIM group works must share one Core and Once route");
      if (work.macroSlot < 0 || work.macroSlot >= 2)
        return runnerError("CIM work Macro is outside [0, 1]");
      if (!placements.emplace(work.coreSlot, work.macroSlot).second)
        return runnerError("CIM group reuses a physical Macro placement");
    }
  }

  for (auto [index, binding] : llvm::enumerate(executable.getDynamicInputs())) {
    const CIMInputView &input = inputs.values[index];
    if (binding.dataType != CIMDataType::Int8 &&
        binding.dataType != CIMDataType::BF16)
      return runnerError("unsupported CIM input data mode");
    if (binding.inputCacheAddress < 0 || binding.inputCacheAddress >= 8)
      return runnerError("CIM input Cache binding is outside [0, 7]");
    if (binding.dataType == CIMDataType::Int8 &&
        input.values.size() != kCIMInputElements)
      return runnerError("CIM input view must contain exactly 64 INT8 values");
    if (binding.dataType == CIMDataType::BF16 &&
        input.bf16Values.size() != kCIMInputElements)
      return runnerError("CIM input view must contain exactly 64 BF16 values");
    if (binding.dataType == CIMDataType::BF16 &&
        llvm::any_of(input.bf16Values, [](uint16_t value) {
          return ((value >> 7) & 0xff) == 0xff;
        }))
      return runnerError("CIM BF16 input does not support NaN or infinity");
  }
  for (const ReadbackBinding &binding : executable.getReadbacks()) {
    if (binding.dataType != CIMDataType::Int8 &&
        binding.dataType != CIMDataType::BF16)
      return runnerError("unsupported CIM readback data mode");
    if (binding.outputCacheAddress < 0 || binding.outputCacheAddress >= 8)
      return runnerError("CIM output Cache binding is outside [0, 7]");
    if (llvm::Error error = validateCIMRoute(binding.route))
      return error;
    const CIMWork *work = findWork(executable, binding.groupId, binding.workId);
    if (!work || binding.route != work->route)
      return runnerError("CIM readback route differs from Work route");
    if (binding.macroSlot != work->macroSlot)
      return runnerError("CIM readback Macro differs from Work placement");
    for (int32_t value : binding.testCore)
      if (value < -31 || value > 31)
        return runnerError("CIM readback test Core is outside [-31, 31]");
  }
  return llvm::Error::success();
}

llvm::Error validateOutputs(const CIMTransaction &executable,
                            const CIMRunOutputs &outputs) {
  if (outputs.values.size() != executable.getReadbacks().size())
    return runnerError("CIM output count does not match executable bindings");
  for (auto [index, binding] : llvm::enumerate(executable.getReadbacks())) {
    const CIMOutputView &output = outputs.values[index];
    if (binding.dataType != CIMDataType::Int8 &&
        binding.dataType != CIMDataType::BF16)
      return runnerError("unsupported CIM readback data mode");
    if (binding.dataType == CIMDataType::Int8 &&
        output.values.size() != kCIMOutputElements)
      return runnerError(
          "CIM output view must contain exactly 16 INT32 values");
    if (binding.dataType == CIMDataType::BF16 &&
        output.bf16Values.size() != kCIMOutputElements)
      return runnerError("CIM output view must contain exactly 16 BF16 values");
  }
  return llvm::Error::success();
}

// Broadcast only with canonical routes and complete coverage of the relevant
// Core set. Sparse selections and reads still share the Macro's barrier.
llvm::Error appendReadbackBatches(const CIMTransaction &executable,
                                  uint32_t selectionDelayMs,
                                  CIMDeviceProgram &program) {
  constexpr size_t coreCount = 20;
  constexpr size_t cacheRows = 8;
  // Supplier TB full-array route from the lower-left ingress.
  constexpr std::array<int32_t, 6> broadcast{0, 0, 0, 0, 4, 3};
  struct PendingBatch {
    CIMDeviceReadbackBatch batch;
    std::array<const ReadbackBinding *, coreCount> selections{};
    std::array<std::bitset<coreCount>, cacheRows> rowCores;
    bool canonicalRoutes = true;
  };
  std::array<PendingBatch, 2> pending;

  for (auto [index, binding] : llvm::enumerate(executable.getReadbacks())) {
    const CIMWork *work = findWork(executable, binding.groupId, binding.workId);
    auto &entry = pending[binding.macroSlot];
    const size_t core = work->coreSlot;
    const size_t row = binding.outputCacheAddress;
    if (entry.rowCores[row].test(core))
      return runnerError("CIM Macro batch has a duplicate Core/Cache row");
    entry.rowCores[row].set(core);
    const auto *selected = entry.selections[core];
    if (selected && selected->dataType != binding.dataType)
      return runnerError("CIM Macro batch has conflicting readback modes");
    entry.selections[core] = &binding;
    entry.batch.targets.push_back(
        {index, static_cast<uint8_t>(core), static_cast<uint8_t>(row)});

    const int32_t x = core % 5;
    const int32_t y = core / 5;
    const int32_t diagonal = std::min(x, y);
    entry.canonicalRoutes &=
        binding.route ==
        std::array<int32_t, 6>{diagonal, x - diagonal, y - diagonal, 0, 0, 0};
  }

  for (auto [macro, entry] : llvm::enumerate(pending)) {
    auto &batch = entry.batch;
    if (batch.targets.empty())
      continue;
    batch.selection.phase = batch.request.phase = CIMDevicePhase::Readback;
    batch.selection.postDelayMs = selectionDelayMs;
    const auto *first = entry.selections.front();
    if (entry.canonicalRoutes && first &&
        llvm::all_of(entry.selections, [&](const ReadbackBinding *binding) {
          return binding && binding->dataType == first->dataType;
        })) {
      batch.selection.flits.push_back(
          encodeCIMControlFlit(broadcast, macro, first->dataType));
    } else {
      for (const auto *binding : entry.selections)
        if (binding)
          batch.selection.flits.push_back(
              encodeCIMControlFlit(binding->route, macro, binding->dataType));
    }

    std::bitset<cacheRows> broadcastRows;
    for (const auto &target : batch.targets) {
      const auto row = target.cacheAddress;
      if (entry.canonicalRoutes && entry.rowCores[row].all()) {
        if (broadcastRows.test(row))
          continue;
        broadcastRows.set(row);
        batch.request.flits.push_back(
            encodeCIMOutputCacheReadFlit(broadcast, row));
      } else {
        const auto &binding = executable.getReadbacks()[target.outputIndex];
        batch.request.flits.push_back(
            encodeCIMOutputCacheReadFlit(binding.route, row));
      }
    }
    program.readbackBatches.push_back(std::move(batch));
  }
  return llvm::Error::success();
}

llvm::Expected<CIMDeviceProgram>
materializeProgram(const CIMTransaction &executable, const CIMRunInputs &inputs,
                   const CIM22HardwareConfig &hardware, CIMTraceSink *trace,
                   uint64_t &sequence, BF16Compensation compensation,
                   llvm::ArrayRef<CIMWeightIdentity> preparedWeights,
                   bool captureRawReadbacks) {
  CIMDeviceProgram program;
  program.useWeightResidency = !preparedWeights.empty();
  program.captureRawReadbacks = captureRawReadbacks;
  const bool zeroWeightExponent = compensation != BF16Compensation::None;

  // The supplier board flow configures one return path per active Core before
  // selecting or programming any Macro. Multiple Macro works on the same Core
  // therefore share one return-route configuration.
  CIMDeviceOperation returnRoutes;
  std::array<const ReadbackBinding *, 20> configuredRoutes{};
  std::array<int64_t, 20> configuredGroups{};
  configuredGroups.fill(-1);
  for (const CIMGroup &group : executable.getGroups()) {
    for (const CIMWork &work : group.works) {
      auto bindings = resolveWorkBindings(executable, group, work);
      if (!bindings)
        return bindings.takeError();
      const ReadbackBinding *readback = bindings->readback;
      const ReadbackBinding *&configured = configuredRoutes[work.coreSlot];
      if (configured) {
        if (configuredGroups[work.coreSlot] != group.groupId)
          return runnerError("CIM program has multiple groups on one Core");
        if (configured->route != readback->route ||
            configured->testCore != readback->testCore)
          return runnerError(
              "CIM Core has conflicting return-route configurations");
        continue;
      }
      configured = readback;
      configuredGroups[work.coreSlot] = group.groupId;
      returnRoutes.flits.push_back(encodeCIMReturnRouteFlit(
          readback->route, work.coreSlot, readback->testCore));
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureReturnRoute,
                readback->groupId, readback->workId, readback->macroSlot);
    }
  }
  if (returnRoutes.flits.empty())
    return runnerError("CIM program has no active Core return route");
  program.operations.push_back(std::move(returnRoutes));

  for (const CIMGroup &group : executable.getGroups()) {
    for (const CIMWork &work : group.works) {
      auto bindings = resolveWorkBindings(executable, group, work);
      if (!bindings)
        return bindings.takeError();
      const DynamicInputBinding &input = *bindings->input;
      const StaticWeightSection &weight = *bindings->weight;

      CIMDeviceOperation select;
      select.flits.push_back(
          encodeCIMControlFlit(work.route, work.macroSlot, input.dataType));
      if (input.dataType == CIMDataType::BF16) {
        // Zero only wire Ew; retain original Ew for output restoration.
        const std::array<uint8_t, kCIMOutputElements> zeroExponents{};
        const auto exponentWords = mlir::cim22::packBF16WeightExponents(
            zeroWeightExponent ? zeroExponents : weight.exponents);
        if (llvm::Error error = appendCIMWeightExponentPacket(
                weight.route, exponentWords, select.flits))
          return std::move(error);
      } else {
        const std::array<uint64_t, 2> exponentWords{};
        if (llvm::Error error = appendCIMWeightExponentPacket(
                weight.route, exponentWords, select.flits))
          return std::move(error);
      }
      program.operations.push_back(std::move(select));
      emitTrace(trace, sequence, CIMTraceEventKind::SelectMacro, group.groupId,
                work.workId, work.macroSlot);

      std::array<uint64_t, 16> inputWords{};
      if (input.dataType == CIMDataType::BF16) {
        std::array<uint16_t, kCIMInputElements> values{};
        std::copy(inputs.values[bindings->inputIndex].bf16Values.begin(),
                  inputs.values[bindings->inputIndex].bf16Values.end(),
                  values.begin());
        inputWords = mlir::cim22::packBF16InputCacheRow(values);
      } else {
        inputWords = packCIMInt8InputCacheRow(
            inputs.values[bindings->inputIndex].values);
      }
      CIMDeviceOperation inputWrite;
      inputWrite.postDelayMs = hardware.cache_write_delay_ms;
      if (llvm::Error error =
              appendCIMInputCachePacket(work.route, input.inputCacheAddress,
                                        inputWords, inputWrite.flits))
        return std::move(error);
      program.operations.push_back(std::move(inputWrite));
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureInput,
                group.groupId, work.workId, work.macroSlot, -1,
                input.inputCacheAddress);

      CIMDeviceOperation weightWrite;
      weightWrite.postDelayMs = hardware.weight_write_delay_ms;
      if (program.useWeightResidency) {
        const size_t index = &weight - executable.getStaticWeights().data();
        const CIMWeightIdentity &identity = preparedWeights[index];
        if (!identity || identity->words.size() != 256 ||
            identity->dataType != weight.dataType ||
            identity->route != weight.route)
          return runnerError("invalid prepared CIM weight identity");
        weightWrite.weight = CIMDeviceWeightWrite{
            static_cast<size_t>(work.coreSlot * 2 + work.macroSlot), identity,
            compensation};
      } else {
        llvm::SmallVector<uint32_t, 256> weightWords;
        weightWords.reserve(weight.words.size());
        for (int32_t word : weight.words)
          weightWords.push_back(llvm::bit_cast<uint32_t>(word));
        if (llvm::Error error = appendCIMWeightWordsPacket(
                weight.route, weightWords, weightWrite.flits))
          return std::move(error);
      }
      program.operations.push_back(std::move(weightWrite));
      // This trace describes the plan. A device may reuse resident weights.
      emitTrace(trace, sequence, CIMTraceEventKind::ConfigureWeight,
                group.groupId, work.workId, work.macroSlot);
    }
  }

  // The execution-plan contract completes every Core/Macro configuration
  // before triggering work. Keep all Once flits in one batch so the board
  // wait for computation and writeback begins after every active Core starts.
  // Once terminates automatically.
  CIMDeviceOperation startWork;
  startWork.phase = CIMDevicePhase::Work;
  startWork.postDelayMs = hardware.work_delay_ms;
  for (const CIMGroup &group : executable.getGroups()) {
    startWork.flits.push_back(encodeCIMWorkOnceFlit(group.works.front().route));
    emitTrace(trace, sequence, CIMTraceEventKind::StartWork, group.groupId);
  }
  program.operations.push_back(std::move(startWork));

  if (auto error = appendReadbackBatches(
          executable, hardware.readback_select_delay_ms, program))
    return std::move(error);
  return program;
}

llvm::Error validateResponse(const CIMTransaction &executable,
                             const CIMDeviceResponse &response,
                             CIMTraceSink *trace, uint64_t &sequence) {
  if (response.readbacks.size() != executable.getReadbacks().size())
    return runnerError("CIM device returned an unexpected readback count");
  std::vector<bool> seen(executable.getReadbacks().size(), false);

  for (const CIMDeviceReadback &result : response.readbacks) {
    if (result.outputIndex >= executable.getReadbacks().size() ||
        seen[result.outputIndex])
      return runnerError("CIM device returned an invalid readback index");
    seen[result.outputIndex] = true;
    const ReadbackBinding &binding =
        executable.getReadbacks()[result.outputIndex];
    const CIMWork *work = findWork(executable, binding.groupId, binding.workId);
    if (!work)
      return runnerError("CIM readback references unknown work");

    auto key = decodeCIMOutputCacheResponseHeader(result.headerFlit);
    if (!key)
      return key.takeError();
    if (key->coreIndex != work->coreSlot ||
        key->cacheAddress != binding.outputCacheAddress)
      return runnerError(
          "CIM output Cache response head does not match binding");

    emitTrace(trace, sequence, CIMTraceEventKind::ReadOutput, binding.groupId,
              binding.workId, binding.macroSlot, -1,
              binding.outputCacheAddress);
  }
  return llvm::Error::success();
}

llvm::Error decodeResponse(const CIMTransaction &executable,
                           const CIMDeviceResponse &response,
                           CIMRunOutputs &outputs) {
  std::vector<std::array<int32_t, kCIMOutputElements>> decodedInt21(
      outputs.values.size());
  for (const CIMDeviceReadback &result : response.readbacks) {
    auto values = decodeCIMInt21OutputCacheResponse(result.bodyFlits);
    if (!values)
      return values.takeError();
    decodedInt21[result.outputIndex] = *values;
  }
  for (size_t index = 0; index < outputs.values.size(); ++index) {
    if (executable.getReadbacks()[index].dataType == CIMDataType::BF16) {
      // Hardware already converted to BF16. The unsigned cast extracts the
      // slot's low 16 bits, ignoring the upper five even after sign extension.
      std::transform(decodedInt21[index].begin(), decodedInt21[index].end(),
                     outputs.values[index].bf16Values.begin(),
                     [](int32_t slot) { return static_cast<uint16_t>(slot); });
    } else {
      std::copy(decodedInt21[index].begin(), decodedInt21[index].end(),
                outputs.values[index].values.begin());
    }
  }
  return llvm::Error::success();
}
} // namespace

llvm::Error CIMBoardRunner::run(const execution::CIMTransaction &executable,
                                const execution::CIMRunInputs &inputs,
                                execution::CIMRunOutputs &outputs,
                                execution::CIMTraceSink *trace) {
  return run(executable, inputs, outputs, BF16Compensation::None, trace);
}

llvm::Error CIMBoardRunner::run(
    const execution::CIMTransaction &executable,
    const execution::CIMRunInputs &inputs, execution::CIMRunOutputs &outputs,
    execution::BF16Compensation compensation, execution::CIMTraceSink *trace,
    llvm::ArrayRef<execution::CIMWeightIdentity> preparedWeights) {
  uint64_t sequence = 0;
  emitTrace(trace, sequence, execution::CIMTraceEventKind::BeginRun);
  if (llvm::Error error = validateOutputs(executable, outputs)) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return error;
  }
  std::vector<BF16InputPreparation> prepared;
  std::vector<CIMInputView> chipInputs;
  const bool compensate = compensation != BF16Compensation::None;
  auto fail = [&](llvm::Error error) -> llvm::Error {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return error;
  };
  if (compensate) {
    if (auto error = validateInputs(executable, inputs))
      return fail(std::move(error));
    prepared.resize(inputs.values.size());
    chipInputs.assign(inputs.values.begin(), inputs.values.end());
    for (auto [index, binding] : llvm::enumerate(executable.getDynamicInputs()))
      if (binding.dataType == CIMDataType::BF16) {
        auto value =
            prepareBF16Input(inputs.values[index].bf16Values, compensation);
        if (!value)
          return fail(value.takeError());
        prepared[index] = std::move(*value);
        chipInputs[index].bf16Values = prepared[index].chipValues;
      }
    // Validate every used group before the first device operation.
    for (const CIMGroup &group : executable.getGroups())
      for (const CIMWork &work : group.works) {
        auto bindings = resolveWorkBindings(executable, group, work);
        if (!bindings)
          return fail(bindings.takeError());
        if (bindings->weight->dataType == CIMDataType::BF16)
          if (auto error = validateBF16Group(*bindings->weight,
                                             prepared[bindings->inputIndex],
                                             compensation))
            return fail(std::move(error));
      }
  }
  auto response =
      captureImpl(executable, compensate ? CIMRunInputs{chipInputs} : inputs,
                  trace, sequence, compensation, preparedWeights);
  if (!response) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return response.takeError();
  }
  if (llvm::Error error = decodeResponse(executable, *response, outputs)) {
    device_.clearWeightResidency();
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return error;
  }
  if (compensate) {
    for (auto [index, binding] : llvm::enumerate(executable.getReadbacks())) {
      if (binding.dataType != CIMDataType::BF16)
        continue;
      size_t inputIndex = 0;
      const auto *input =
          findInput(executable, binding.groupId, binding.workId, inputIndex);
      const auto *weight =
          findWeight(executable, binding.groupId, binding.workId);
      if (!input || !weight) {
        device_.clearWeightResidency();
        return fail(runnerError("CIM BF16 restoration binding is missing"));
      }
      auto output = outputs.values[index].bf16Values;
      for (size_t lane = 0; lane < kCIMOutputElements; ++lane) {
        const bool hasSupport =
            compensation != BF16Compensation::SignSplit ||
            hasBF16Support(
                llvm::ArrayRef(weight->values)
                    .slice(lane * kCIMInputElements, kCIMInputElements),
                prepared[inputIndex].support);
        auto restored =
            restoreBF16Output(output[lane], weight->exponents[lane], hasSupport,
                              prepared[inputIndex], compensation);
        if (!restored) {
          device_.clearWeightResidency();
          return fail(restored.takeError());
        }
        output[lane] = *restored;
      }
    }
  }
  emitTrace(trace, sequence, execution::CIMTraceEventKind::Complete);
  return llvm::Error::success();
}

llvm::Expected<CIMDeviceResponse>
CIMBoardRunner::capture(const execution::CIMTransaction &executable,
                        const execution::CIMRunInputs &inputs,
                        execution::CIMTraceSink *trace) {
  device_.clearWeightResidency();
  uint64_t sequence = 0;
  emitTrace(trace, sequence, execution::CIMTraceEventKind::BeginRun);
  auto response = captureImpl(executable, inputs, trace, sequence,
                              BF16Compensation::None, {},
                              /*captureRawReadbacks=*/true);
  if (!response) {
    emitTrace(trace, sequence, execution::CIMTraceEventKind::Error);
    return response.takeError();
  }
  emitTrace(trace, sequence, execution::CIMTraceEventKind::Complete);
  return response;
}

llvm::Expected<CIMDeviceResponse> CIMBoardRunner::captureImpl(
    const execution::CIMTransaction &executable,
    const execution::CIMRunInputs &inputs, execution::CIMTraceSink *trace,
    uint64_t &sequence, execution::BF16Compensation compensation,
    llvm::ArrayRef<execution::CIMWeightIdentity> preparedWeights,
    bool captureRawReadbacks) {
  if (llvm::Error error = validateInputs(executable, inputs))
    return std::move(error);
  if (!preparedWeights.empty() &&
      preparedWeights.size() != executable.getStaticWeights().size())
    return runnerError("prepared CIM weight count differs from transaction");
  if (!device_.supportsWeightResidency())
    preparedWeights = {};
  auto program =
      materializeProgram(executable, inputs, hardware_, trace, sequence,
                         compensation, preparedWeights, captureRawReadbacks);
  if (!program)
    return program.takeError();
  CIMDeviceResponse response;
  if (llvm::Error error = device_.execute(*program, response))
    return std::move(error);
  if (llvm::Error error =
          validateResponse(executable, response, trace, sequence)) {
    device_.clearWeightResidency();
    return std::move(error);
  }
  return response;
}

} // namespace cim22::runtime
