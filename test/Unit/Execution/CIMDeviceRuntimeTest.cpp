//===- CIMDeviceRuntimeTest.cpp - CIM22 device runtime tests ---*- C++ -*-===//

#include "CIM22/Runtime/CIMBoardRunner.h"
#include "CIM22/Runtime/CIMPacketCodec.h"
#include "CIM22/Support/BF16Support.h"
#include "CIM22/Support/Int8WeightLayout.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace cim22::execution;
using namespace cim22::runtime;

namespace {
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                   #condition);                                                \
      std::abort();                                                            \
    }                                                                          \
  } while (false)
constexpr std::array<int32_t, 16> kExpected{
    0,   1,      -1,      -1048576, 1048575, 74565, -74565, 42,
    -42, 262144, -262144, 7,        -7,      21930, -21930, 291};
constexpr std::array<uint64_t, 6> kResponse{
    0x00091fea9580ab55ULL, 0xffff900003f00000ULL, 0x80001fffd6000157ULL,
    0xb72ec2468afffff8ULL, 0x00007ffffc000020ULL, 0x0000000000000000ULL};

// Synthetic protocol fixture, not a DUT capture.
// Fixed bit-string fixture: lane15..0 as 21-bit slots, then 48 zero bits.
// Upper slot bits deliberately vary; BF16 payloads include signed zeros,
// subnormals, infinities and NaNs. They are payload bits, not numeric INT21.
constexpr std::array<uint16_t, 16> kBF16Expected{
    0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x7f7f, 0xff7f,
    0x7f80, 0xff80, 0x7fc1, 0xffff, 0x1234, 0xabcd, 0x55aa, 0xaa55};
constexpr std::array<uint64_t, 6> kBF16Response{
    0xfd52afd56abf579bULL, 0xf1234ffffffdff07ULL, 0xfff01f7f80fffbffULL,
    0xdfdfff7f01f3f80fULL, 0xc000fc0007f0001fULL, 0x0000000000000000ULL};

class FakeCIMDevice final : public CIMDevice {
public:
  enum class ResponseFault {
    None,
    Missing,
    InvalidIndex,
    DuplicateIndex,
    Header
  };

  llvm::Error open() override {
    opened = true;
    return llvm::Error::success();
  }

  llvm::Error execute(const CIMDeviceProgram &value,
                      CIMDeviceResponse &response) override {
    if (!opened)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "fake CIM device is closed");
    ++executeCount;
    program = value;
    response.readbacks.clear();
    for (const auto &batch : program.readbackBatches) {
      const bool bf16Batch = std::all_of(
          batch.selection.flits.begin(), batch.selection.flits.end(),
          [](uint64_t flit) { return (flit & 2) != 0; });
      const size_t first = response.readbacks.size();
      for (const auto &target : batch.targets) {
        CIMDeviceReadback readback;
        readback.outputIndex = target.outputIndex;
        readback.headerFlit = (uint64_t{5} << 60) |
                              (uint64_t{target.coreIndex} << 18) |
                              (uint64_t{target.cacheAddress} << 14) | 6;
        readback.bodyFlits = bf16Batch || mixedBF16Output == target.outputIndex
                                 ? bf16Response
                                 : kResponse;
        response.readbacks.push_back(readback);
      }
      if (reverseReadbacks)
        std::reverse(response.readbacks.begin() + first,
                     response.readbacks.end());
    }
    if (corruptResponsePadding)
      response.readbacks.back().bodyFlits[5] = 1;
    switch (responseFault) {
    case ResponseFault::None:
      break;
    case ResponseFault::Missing:
      response.readbacks.pop_back();
      break;
    case ResponseFault::InvalidIndex:
      response.readbacks.front().outputIndex = response.readbacks.size();
      break;
    case ResponseFault::DuplicateIndex:
      response.readbacks.back().outputIndex =
          response.readbacks.front().outputIndex;
      break;
    case ResponseFault::Header:
      response.readbacks.front().headerFlit ^= uint64_t{1} << 18;
      break;
    }
    return llvm::Error::success();
  }

  bool isFaulted() override { return false; }
  void close() override { opened = false; }

  bool opened = false;
  std::array<uint64_t, 6> bf16Response = kBF16Response;
  std::optional<size_t> mixedBF16Output;
  CIMDeviceProgram program;
  size_t executeCount = 0;
  bool corruptResponsePadding = false;
  bool reverseReadbacks = false;
  ResponseFault responseFault = ResponseFault::None;
};

CIMTransaction makeTransaction(CIMDataType dataType = CIMDataType::Int8) {
  const std::array<int32_t, 6> route{};
  const std::array<int32_t, 3> testCore{15, 0, 0};
  CIMGroup group;
  group.groupId = 0;
  group.works.push_back({0, 3, 0, route});

  StaticWeightSection weight;
  weight.groupId = 0;
  weight.workId = 0;
  weight.coreSlot = 3;
  weight.macroSlot = 0;
  weight.words.resize(256, 0);
  weight.dataType = dataType;

  DynamicInputBinding input{0, 0, 0, 0, dataType, 7};
  ReadbackBinding readback{0, 0, 0, 7, route, testCore, dataType};
  return CIMTransaction("cim22-4x5-v1", 1, 1, {std::move(group)},
                        {std::move(weight)}, {}, {input}, {readback}, {});
}

CIMTransaction
makeTwoGroupTransaction(CIMDataType secondType = CIMDataType::Int8) {
  const std::array<int32_t, 6> route{};
  const std::array<int32_t, 3> testCore{};
  std::vector<CIMGroup> groups;
  std::vector<StaticWeightSection> weights;
  std::vector<DynamicInputBinding> inputs;
  std::vector<ReadbackBinding> readbacks;
  for (int64_t groupId = 0; groupId < 2; ++groupId) {
    CIMGroup group;
    group.groupId = groupId;
    group.works.push_back({groupId, 3 + groupId, 0, route});
    groups.push_back(std::move(group));

    StaticWeightSection weight;
    weight.groupId = groupId;
    weight.workId = groupId;
    weight.coreSlot = 3 + groupId;
    weight.macroSlot = 0;
    weight.words.resize(256, 0);
    const CIMDataType mode = groupId == 0 ? CIMDataType::Int8 : secondType;
    weight.dataType = mode;
    weights.push_back(std::move(weight));
    inputs.push_back({groupId, groupId, 0, groupId, mode, 7});
    readbacks.push_back({groupId, groupId, 0, 7, route, testCore, mode});
  }
  return CIMTransaction("cim22-4x5-v1", 1, 1, std::move(groups),
                        std::move(weights), {}, std::move(inputs),
                        std::move(readbacks), {});
}

CIMTransaction makeTwoMacroTransaction() {
  const std::array<int32_t, 6> route{};
  const std::array<int32_t, 3> testCore{};
  CIMGroup group;
  group.groupId = 0;
  group.works.push_back({0, 3, 0, route});
  group.works.push_back({1, 3, 1, route});

  std::vector<StaticWeightSection> weights;
  std::vector<DynamicInputBinding> inputs;
  std::vector<ReadbackBinding> readbacks;
  for (int64_t macro = 0; macro < 2; ++macro) {
    StaticWeightSection weight;
    weight.groupId = 0;
    weight.workId = macro;
    weight.coreSlot = 3;
    weight.macroSlot = macro;
    weight.words.resize(256, 0);
    weights.push_back(std::move(weight));
    inputs.push_back({0, macro, macro, macro, CIMDataType::Int8, macro});
    readbacks.push_back(
        {0, macro, macro, 6 + macro, route, testCore, CIMDataType::Int8});
  }
  return CIMTransaction("cim22-4x5-v1", 1, 1, {std::move(group)},
                        std::move(weights), {}, std::move(inputs),
                        std::move(readbacks), {});
}

CIMTransaction makeBF16Transaction() {
  const std::array<int32_t, 6> route{};
  const std::array<int32_t, 3> testCore{};
  CIMGroup group;
  group.groupId = 0;
  group.works.push_back({0, 3, 0, route});

  StaticWeightSection weight;
  weight.groupId = 0;
  weight.workId = 0;
  weight.coreSlot = 3;
  weight.macroSlot = 0;
  weight.words.resize(256, 0);
  weight.dataType = CIMDataType::BF16;
  for (size_t index = 0; index < weight.exponents.size(); ++index)
    weight.exponents[index] = static_cast<uint8_t>(index);

  DynamicInputBinding input{0, 0, 0, 0, CIMDataType::BF16, 2};
  ReadbackBinding readback{0, 0, 0, 7, route, testCore, CIMDataType::BF16};
  return CIMTransaction("cim22-4x5-v1", 1, 1, {std::move(group)},
                        {std::move(weight)}, {}, {input}, {readback}, {});
}

void expectSuccess(llvm::Error error) { CHECK(!error); }

void testPacketCodec() {
  const std::array<int32_t, 6> route{};
  const std::array<int32_t, 3> testCore{1, -2, 3};
  std::array<int8_t, 64> input{};
  input[0] = 1;
  input[1] = -2;
  input[2] = 127;
  input[3] = -128;
  const auto words = packCIMInt8InputCacheRow(input);
  CHECK(words[0] == 0x000100fe007f0080ULL);
  for (size_t index = 1; index < 16; ++index)
    CHECK(words[index] == 0);

  llvm::SmallVector<uint64_t> flits;
  expectSuccess(appendCIMInputCachePacket(route, 2, words, flits));
  CHECK(flits.size() == 17);
  CHECK(flits.front() == 0x1000000000028010ULL);
  CHECK(encodeCIMWorkOnceFlit(route) == 0x9000000000000001ULL);
  CHECK(encodeCIMWorkIdleFlit(route) == 0x9000000000000000ULL);
  CHECK(encodeCIMReturnRouteFlit(route, 3, testCore) == 0xa0000000000c1883ULL);
  CHECK(encodeCIMOutputCacheReadFlit(route, 7) == 0x500000000081c000ULL);

  auto decoded = decodeCIMInt21OutputCacheResponse(kResponse);
  CHECK(decoded && *decoded == kExpected);
}

void testBoardRunner() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  CIMTransaction executable = makeTransaction();

  std::array<int8_t, 64> inputStorage{};
  inputStorage[0] = 1;
  inputStorage[1] = -2;
  inputStorage[2] = 127;
  inputStorage[3] = -128;
  const std::array<CIMInputView, 1> inputViews{CIMInputView{inputStorage, {}}};
  CIMRunInputs inputs{inputViews};

  std::array<int32_t, 16> outputStorage{};
  std::array<CIMOutputView, 1> outputViews{CIMOutputView{outputStorage, {}}};
  CIMRunOutputs outputs{outputViews};

  expectSuccess(runner.run(executable, inputs, outputs));
  CHECK(outputStorage == kExpected);
  CHECK(device.program.operations.size() == 5);
  CHECK(device.program.operations[0].flits.size() == 1);
  CHECK(device.program.operations[0].flits.front() == 0xa0000000000cf000ULL);
  CHECK(device.program.operations[1].flits.size() == 4);
  CHECK((device.program.operations[1].flits.front() >> 60) == 0x8);
  CHECK(device.program.operations[2].flits.size() == 17);
  CHECK(device.program.operations[2].flits[1] == 0x000100fe007f0080ULL);
  CHECK(device.program.operations[2].flits.back() == 0);
  CHECK(device.program.operations[2].postDelayMs == 1);
  CHECK(device.program.operations[3].flits.size() == 257);
  CHECK(device.program.operations[3].postDelayMs == 2);
  CHECK(device.program.operations[4].flits.size() == 1);
  CHECK(device.program.operations[4].flits.front() == 0x9000000000000001ULL);
  CHECK(device.program.operations[4].postDelayMs == 10);
  CHECK(device.program.operations[4].phase == CIMDevicePhase::Work);
  CHECK(device.program.readbackBatches.size() == 1);
  CHECK(device.program.readbackBatches.front().selection.flits.size() == 1);
  CHECK(device.program.readbackBatches.front().selection.postDelayMs == 40);
  CHECK(device.program.readbackBatches.front().request.flits.size() == 1);
  CHECK(device.program.readbackBatches.front().request.flits.back() ==
        0x500000000081c000ULL);
  CIMBoardRunner tuned(device, {0, 3, 4, 5, 6});
  expectSuccess(tuned.run(executable, inputs, outputs));
  CHECK(device.program.operations[2].postDelayMs == 3);
  CHECK(device.program.operations[3].postDelayMs == 4);
  CHECK(device.program.operations[4].postDelayMs == 5);
  CHECK(device.program.readbackBatches.front().selection.postDelayMs == 6);
  CHECK(outputStorage == kExpected);
  device.close();
}

void testGroupExecutionOrder() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  CIMTransaction executable = makeTwoGroupTransaction();

  std::array<std::array<int8_t, 64>, 2> inputStorage{};
  std::array<CIMInputView, 2> inputViews{CIMInputView{inputStorage[0], {}},
                                         CIMInputView{inputStorage[1], {}}};
  CIMRunInputs inputs{inputViews};
  std::array<std::array<int32_t, 16>, 2> outputStorage{};
  std::array<CIMOutputView, 2> outputViews{CIMOutputView{outputStorage[0], {}},
                                           CIMOutputView{outputStorage[1], {}}};
  CIMRunOutputs outputs{outputViews};

  expectSuccess(runner.run(executable, inputs, outputs));
  CHECK(device.program.operations.size() == 8);
  CHECK(device.program.operations[0].flits.size() == 2);
  CHECK((device.program.operations[0].flits[0] >> 60) == 0xa);
  CHECK((device.program.operations[0].flits[1] >> 60) == 0xa);
  CHECK((device.program.operations[4].flits.front() >> 60) == 0x8);
  CHECK(device.program.operations[7].flits.size() == 2);
  CHECK(device.program.operations[7].postDelayMs == 10);
  CHECK(device.program.operations[7].phase == CIMDevicePhase::Work);
  CHECK(device.program.operations[7].flits[0] == 0x9000000000000001ULL);
  CHECK(device.program.operations[7].flits[1] == 0x9000000000000001ULL);
  CHECK(device.program.readbackBatches.size() == 1);
  const auto &batch = device.program.readbackBatches.front();
  CHECK(batch.selection.flits.size() == 2);
  CHECK(batch.selection.postDelayMs == 40);
  CHECK(batch.request.flits.size() == 2);
  CHECK(batch.targets.size() == 2);
  device.close();
}

void testFullArrayReadbackBatches() {
  FakeCIMDevice device;
  device.reverseReadbacks = true;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  std::vector<CIMGroup> groups;
  std::vector<StaticWeightSection> weights;
  std::vector<DynamicInputBinding> inputs;
  std::vector<ReadbackBinding> readbacks;
  for (int32_t core = 0; core < 20; ++core) {
    const int32_t x = core % 5;
    const int32_t y = core / 5;
    const int32_t diagonal = std::min(x, y);
    const std::array<int32_t, 6> route{
        diagonal, x - diagonal, y - diagonal, 0, 0, 0};
    CIMGroup group{core, {}};
    for (int64_t macro = 0; macro < 2; ++macro) {
      const int64_t id = core * 2 + macro;
      group.works.push_back({id, core, macro, route});
      StaticWeightSection weight;
      weight.groupId = weight.coreSlot = core;
      weight.workId = id;
      weight.macroSlot = macro;
      weight.route = route;
      weight.words.resize(256, 0);
      weights.push_back(std::move(weight));
      inputs.push_back({core, id, macro, id});
      for (int64_t row = 0; row < 8; ++row)
        readbacks.push_back({core, id, macro, row, route, {}});
    }
    groups.push_back(std::move(group));
  }
  std::array<int8_t, 64> data{};
  std::vector<CIMInputView> views(40, {data, {}});

  auto capture = [&](std::vector<ReadbackBinding> bindings) {
    CIMTransaction executable("cim22-4x5-v1", 1, 1, groups, weights, {}, inputs,
                              std::move(bindings), {});
    return runner.capture(executable, {views});
  };
  auto response = capture(readbacks);
  CHECK(response && response->readbacks.size() == 320);
  CHECK(device.program.readbackBatches.size() == 2);
  for (size_t macro = 0; macro < 2; ++macro) {
    const auto &batch = device.program.readbackBatches[macro];
    CHECK(batch.selection.flits.size() == 1);
    CHECK(batch.selection.flits.front() == 0x8000000103000000ULL + macro);
    CHECK(batch.request.flits.size() == 8);
    CHECK(batch.targets.size() == 160);
    for (size_t row = 0; row < 8; ++row)
      CHECK(batch.request.flits[row] == 0x5000000103800000ULL + (row << 14));
  }

  readbacks.pop_back();
  response = capture(readbacks);
  CHECK(response && response->readbacks.size() == 319);
  const auto &sparse = device.program.readbackBatches[1];
  CHECK(sparse.targets.size() == 159);
  CHECK(sparse.request.flits.size() == 7 + 19);
  readbacks.push_back(readbacks.back());
  auto duplicate = capture(readbacks);
  CHECK(!duplicate);
  CHECK(
      llvm::toString(duplicate.takeError()).find("duplicate Core/Cache row") !=
      std::string::npos);
  CHECK(device.executeCount == 2);
}

void testTwoMacrosShareReturnRouteAndOnce() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  const std::array<int32_t, 6> route{};
  CIMTransaction executable = makeTwoMacroTransaction();

  std::array<std::array<int8_t, 64>, 2> inputStorage{};
  std::array<CIMInputView, 2> inputViews{CIMInputView{inputStorage[0], {}},
                                         CIMInputView{inputStorage[1], {}}};
  CIMRunInputs inputs{inputViews};
  std::array<std::array<int32_t, 16>, 2> outputStorage{};
  std::array<CIMOutputView, 2> outputViews{CIMOutputView{outputStorage[0], {}},
                                           CIMOutputView{outputStorage[1], {}}};
  CIMRunOutputs outputs{outputViews};

  expectSuccess(runner.run(executable, inputs, outputs));
  CHECK(device.program.operations.size() == 8);
  CHECK(device.program.operations[0].flits.size() == 1);
  CHECK(device.program.operations[1].flits.size() == 4);
  CHECK(device.program.operations[4].flits.size() == 4);
  CHECK(device.program.operations[1].flits[1] == 0x3000000000000002ULL);
  CHECK(device.program.operations[1].flits[2] == 0);
  CHECK(device.program.operations[1].flits[3] == 0);
  CHECK(device.program.operations[7].flits.size() == 1);
  CHECK(device.program.operations[7].flits.front() == 0x9000000000000001ULL);
  CHECK(device.program.readbackBatches.size() == 2);
  CHECK(device.program.readbackBatches[0].selection.postDelayMs == 40);
  CHECK(device.program.readbackBatches[0].selection.flits.size() == 1);
  CHECK(device.program.readbackBatches[0].request.flits.size() == 1);
  CHECK(device.program.readbackBatches[0].request.flits.back() ==
        encodeCIMOutputCacheReadFlit(route, 6));
  CHECK(device.program.readbackBatches[1].request.flits.back() ==
        encodeCIMOutputCacheReadFlit(route, 7));
  device.close();
}

void testBF16RawCaptureKeepsPrefixAndBody() {
  FakeCIMDevice device;
  device.corruptResponsePadding = true;
  expectSuccess(device.open());
  auto hardware = defaultCIM22HardwareConfig();
  hardware.work_delay_ms = 37;
  CIMBoardRunner runner(device, hardware);
  CIMTransaction executable = makeBF16Transaction();

  std::array<uint16_t, 64> inputStorage{};
  const std::array<CIMInputView, 1> inputViews{CIMInputView{{}, inputStorage}};
  CIMRunInputs inputs{inputViews};
  auto response = runner.capture(executable, inputs);
  CHECK(response);
  CHECK(response->readbacks.size() == 1);
  CHECK(response->readbacks[0].outputIndex == 0);
  CHECK(response->readbacks[0].headerFlit == 0x50000000000dc006ULL);
  auto expectedBody = kBF16Response;
  expectedBody[5] = 1;
  CHECK(response->readbacks[0].bodyFlits == expectedBody);
  CHECK(device.executeCount == 1);
  CHECK(device.program.operations.size() == 6);
  const CIMDeviceOperation &prefix = device.program.operations[1];
  CHECK(prefix.flits.size() == 4);
  CHECK((prefix.flits[0] >> 60) == 0x8);
  CHECK((prefix.flits[1] >> 60) == 0x3);
  CHECK((prefix.flits[1] & 0x3fff) == 2);
  const std::array<int32_t, 6> route{};
  const auto &startWork = device.program.operations[4];
  CHECK(startWork.phase == CIMDevicePhase::Work);
  CHECK(startWork.postDelayMs == hardware.work_delay_ms);
  CHECK(startWork.flits.size() == 1);
  CHECK(startWork.flits.front() == encodeCIMWorkOnceFlit(route));
  CHECK(device.program.operations.back().flits.front() ==
        encodeCIMWorkIdleFlit(route));
  CHECK(device.program.readbackBatches.size() == 1);
  device.close();
}

void testRawCaptureValidatesInputsBeforeDeviceExecution() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  CIMTransaction executable = makeBF16Transaction();
  auto missingInputs = runner.capture(executable, {});
  CHECK(!missingInputs);
  CHECK(llvm::toString(missingInputs.takeError()).find("input count") !=
        std::string::npos);

  std::array<uint16_t, 63> inputStorage{};
  const std::array<CIMInputView, 1> inputViews{CIMInputView{{}, inputStorage}};
  auto shortInput = runner.capture(executable, CIMRunInputs{inputViews});
  CHECK(!shortInput);
  CHECK(llvm::toString(shortInput.takeError()).find("64 BF16") !=
        std::string::npos);

  std::array<uint16_t, 64> fullInput{};
  const std::array<CIMInputView, 1> fullViews{CIMInputView{{}, fullInput}};
  for (uint16_t value : {0x7f80, 0xff80, 0x7fc1, 0xff81}) {
    fullInput.back() = value;
    auto invalidInput = runner.capture(executable, CIMRunInputs{fullViews});
    CHECK(!invalidInput);
    CHECK(llvm::toString(invalidInput.takeError()).find("NaN or infinity") !=
          std::string::npos);
  }
  CHECK(device.executeCount == 0);
  device.close();
}

void testRawCaptureValidatesResponseBindings() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  CIMTransaction executable = makeTwoMacroTransaction();
  std::array<std::array<int8_t, 64>, 2> inputStorage{};
  const std::array<CIMInputView, 2> inputViews{
      CIMInputView{inputStorage[0], {}}, CIMInputView{inputStorage[1], {}}};

  for (FakeCIMDevice::ResponseFault fault :
       {FakeCIMDevice::ResponseFault::Missing,
        FakeCIMDevice::ResponseFault::InvalidIndex,
        FakeCIMDevice::ResponseFault::DuplicateIndex,
        FakeCIMDevice::ResponseFault::Header}) {
    device.responseFault = fault;
    auto response = runner.capture(executable, CIMRunInputs{inputViews});
    CHECK(!response);
    llvm::consumeError(response.takeError());
  }
  CHECK(device.executeCount == 4);
  device.close();
}

void testUnknownDataModeIsRejectedBeforeDeviceExecution() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  CIMTransaction executable = makeTransaction(static_cast<CIMDataType>(-1));
  std::array<int8_t, 64> inputStorage{};
  const std::array<CIMInputView, 1> inputViews{CIMInputView{inputStorage, {}}};
  CIMRunInputs inputs{inputViews};
  auto response = runner.capture(executable, inputs);
  CHECK(!response);
  llvm::consumeError(response.takeError());
  std::array<int32_t, 16> outputStorage{};
  std::array<CIMOutputView, 1> outputViews{CIMOutputView{outputStorage, {}}};
  CIMRunOutputs outputs{outputViews};
  llvm::Error error = runner.run(executable, inputs, outputs);
  CHECK(error);
  llvm::consumeError(std::move(error));
  CHECK(device.executeCount == 0);
  device.close();
}

void testInvalidPlacementAndReadbackRouteRejectBeforeIO() {
  for (bool duplicateMacro : {false, true}) {
    CIMTransaction source = makeTwoMacroTransaction();
    auto groups = source.getGroups().vec();
    auto weights = source.getStaticWeights().vec();
    auto inputs = source.getDynamicInputs().vec();
    auto readbacks = source.getReadbacks().vec();
    if (duplicateMacro) {
      groups[0].works[1].macroSlot = 0;
      weights[1].macroSlot = 0;
      inputs[1].macroSlot = 0;
      readbacks[1].macroSlot = 0;
    } else {
      // A different legal onecast route must not read a different Core.
      readbacks[0].route[0] = 1;
    }
    CIMTransaction invalid("cim22-4x5-v1", 1, 1, std::move(groups),
                           std::move(weights), {}, std::move(inputs),
                           std::move(readbacks), {});
    FakeCIMDevice device;
    expectSuccess(device.open());
    CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
    std::array<std::array<int8_t, 64>, 2> inputStorage{};
    const std::array<CIMInputView, 2> inputViews{
        CIMInputView{inputStorage[0], {}}, CIMInputView{inputStorage[1], {}}};
    const CIMRunInputs runInputs{inputViews};
    auto response = runner.capture(invalid, runInputs);
    CHECK(!response);
    const std::string expected =
        duplicateMacro ? "physical Macro" : "Work route";
    CHECK(llvm::toString(response.takeError()).find(expected) !=
          std::string::npos);
    std::array<std::array<int32_t, 16>, 2> outputStorage{};
    std::array<CIMOutputView, 2> outputViews{
        CIMOutputView{outputStorage[0], {}},
        CIMOutputView{outputStorage[1], {}}};
    CIMRunOutputs outputs{outputViews};
    llvm::Error error = runner.run(invalid, runInputs, outputs);
    CHECK(error);
    CHECK(llvm::toString(std::move(error)).find(expected) != std::string::npos);
    CHECK(device.executeCount == 0);
    device.close();
  }
}

void testBF16BoardReadbackPreservesBits() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  CIMTransaction executable = makeBF16Transaction();
  std::array<uint16_t, 64> inputStorage{};
  const std::array<CIMInputView, 1> inputViews{CIMInputView{{}, inputStorage}};
  CIMRunInputs inputs{inputViews};
  std::array<uint16_t, 16> outputStorage{};
  std::array<CIMOutputView, 1> outputViews{CIMOutputView{{}, outputStorage}};
  CIMRunOutputs outputs{outputViews};

  for (uint16_t inputBits : {uint16_t{0}, uint16_t{0x3f80}}) {
    inputStorage.fill(inputBits);
    expectSuccess(runner.run(executable, inputs, outputs));
    CHECK(outputStorage == kBF16Expected);
  }
  CHECK(device.executeCount == 2);
  CHECK((device.program.readbackBatches[0].selection.flits.front() & 2) != 0);

  // Reject short destinations before touching the device.
  outputViews[0].bf16Values =
      llvm::MutableArrayRef(outputStorage).take_front(15);
  llvm::Error error = runner.run(executable, inputs, outputs);
  CHECK(error);
  CHECK(llvm::toString(std::move(error)).find("16 BF16") != std::string::npos);
  CHECK(device.executeCount == 2);
  device.close();
}

// Synthetic wire fixtures: lane 0 starts at bit 48 of the 384-bit body.
// These canned converter results do not compute or simulate a dot product.
std::array<uint64_t, 6> bf16Response(const std::array<uint16_t, 16> &values) {
  std::array<uint64_t, 6> body{};
  for (size_t lane = 0; lane < values.size(); ++lane)
    for (size_t bit = 0; bit < 16; ++bit) {
      const size_t position = 48 + 21 * lane + bit;
      body[5 - position / 64] |= uint64_t{(values[lane] >> bit) & 1U}
                                 << (position % 64);
    }
  return body;
}

StaticWeightSection compensatedWeight(int8_t mantissa, uint8_t exponent) {
  auto weight = makeBF16Transaction().getStaticWeights().front();
  weight.values.fill(0);
  weight.values[0] = mantissa;
  weight.exponents.fill(exponent);
  std::array<uint8_t, 16 * 64> bytes{};
  std::transform(weight.values.begin(), weight.values.end(), bytes.begin(),
                 [](int8_t value) { return static_cast<uint8_t>(value); });
  const auto words = mlir::cim22::mapInt8WeightTileToCIMWords(bytes);
  weight.words.assign(words.begin(), words.end());
  return weight;
}

CIMTransaction singleBF16(const StaticWeightSection &weight) {
  const auto base = makeBF16Transaction();
  return CIMTransaction("cim22-4x5-v1", 1, 1, {base.getGroups().front()},
                        {weight}, {}, {base.getDynamicInputs().front()},
                        {base.getReadbacks().front()}, {});
}

void testBF16BoardCompensation() {
  struct FinalTrace : CIMTraceSink {
    CIMTraceEventKind last = CIMTraceEventKind::Error;
    void record(const CIMTraceEvent &event) override { last = event.kind; }
  } trace;
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  std::array<uint16_t, 64> input{};
  input[0] = 0x7880; // 2^114.
  std::array<uint16_t, 16> output{}, reply{};
  std::array<CIMInputView, 1> inputViews{CIMInputView{{}, input}};
  std::array<CIMOutputView, 1> outputViews{CIMOutputView{{}, output}};
  CIMRunInputs inputs{inputViews};
  CIMRunOutputs outputs{outputViews};
  auto weight = compensatedWeight(2, 126); // One 1/64 weight in lane 0.
  auto executable = singleBF16(weight);
  auto checkInputPacket = [&](uint16_t first) {
    std::array<uint16_t, 64> expected{};
    expected[0] = first;
    const auto words = mlir::cim22::packBF16InputCacheRow(expected);
    const auto &flits = device.program.operations[2].flits;
    CHECK(flits.size() == 17);
    CHECK(std::equal(words.begin(), words.end(), flits.begin() + 1));
  };
  // p=128 and zero share chip bits; only lane 0 has quantized support.
  reply.fill(0x4000); // C(128,0,127)=2; scale 2^107 gives 2^108.
  device.bf16Response = bf16Response(reply);
  expectSuccess(runner.run(executable, inputs, outputs,
                           BF16Compensation::SignSplit, &trace));
  CHECK(trace.last == CIMTraceEventKind::Complete);
  CHECK(output[0] == 0x7580);
  CHECK(std::all_of(output.begin() + 1, output.end(),
                    [](uint16_t bits) { return bits == 0; }));
  checkInputPacket(0x3f80);
  CHECK(device.program.operations[1].flits[2] == 0);
  CHECK(device.program.operations[1].flits[3] == 0);
  CHECK(executable.getStaticWeights().front().exponents == weight.exponents);
  CHECK(input[0] == 0x7880);

  reply.fill(0x7880); // Legacy p128/zero at E240; restore by 2^-6.
  device.bf16Response = bf16Response(reply);
  expectSuccess(
      runner.run(executable, inputs, outputs, BF16Compensation::Legacy));
  CHECK(output[0] == 0x7580);
  CHECK(output[1] == 0x7580); // Preserve legacy false-zero result 2^108.
  checkInputPacket(0x7800);
  CHECK(device.program.operations[1].flits[2] == 0);

  // A compensated run must not alter subsequent raw run/capture semantics.
  device.bf16Response = kBF16Response;
  expectSuccess(runner.run(executable, inputs, outputs));
  CHECK(output == kBF16Expected);
  const auto exponents = mlir::cim22::packBF16WeightExponents(
      executable.getStaticWeights().front().exponents);
  CHECK(device.program.operations[1].flits[2] == exponents[0]);
  CHECK(device.program.operations[1].flits[3] == exponents[1]);
  auto capture = runner.capture(executable, inputs);
  CHECK(capture && capture->readbacks.front().bodyFlits == kBF16Response);

  const auto before = device.executeCount;
  input[1] = 0xbf80;
  auto negativeInput = runner.run(executable, inputs, outputs,
                                  BF16Compensation::SignSplit, &trace);
  CHECK(negativeInput);
  CHECK(trace.last == CIMTraceEventKind::Error);
  llvm::consumeError(std::move(negativeInput));
  CHECK(device.executeCount == before);
  input[1] = 0;
  for (uint16_t bad : {uint16_t{0x7f80}, uint16_t{0x7fc1}}) {
    input[0] = bad;
    auto error =
        runner.run(executable, inputs, outputs, BF16Compensation::Legacy);
    CHECK(error);
    llvm::consumeError(std::move(error));
    CHECK(device.executeCount == before);
  }
  input.fill(0);
  executable = singleBF16(compensatedWeight(-1, 127));
  auto negative =
      runner.run(executable, inputs, outputs, BF16Compensation::SignSplit);
  CHECK(negative);
  llvm::consumeError(std::move(negative));
  CHECK(device.executeCount == before);

  auto extremeWeight = compensatedWeight(-128, 0);
  extremeWeight.values.fill(-128);
  std::array<uint8_t, 16 * 64> extremeBytes{};
  extremeBytes.fill(128);
  const auto extremeWords =
      mlir::cim22::mapInt8WeightTileToCIMWords(extremeBytes);
  extremeWeight.words.assign(extremeWords.begin(), extremeWords.end());
  for (uint16_t bits : {uint16_t{0x8000}, uint16_t{0xbfff}}) {
    // Both negative zero and finite -1.9921875 prealign to q=-128.
    extremeWeight.exponents.fill(bits == 0x8000 ? 0 : 127);
    executable = singleBF16(extremeWeight);
    input.fill(bits);
    auto range = runner.run(executable, inputs, outputs,
                            BF16Compensation::Legacy, &trace);
    CHECK(range && trace.last == CIMTraceEventKind::Error);
    llvm::consumeError(std::move(range));
    CHECK(device.executeCount == before);
  }

  input.fill(0);
  input[0] = 0x7f7f;
  executable = singleBF16(compensatedWeight(127, 254));
  reply.fill(0x4000);
  reply[0] = 0x437c; // C(127*127,0,127)=252; true restored overflow.
  device.bf16Response = bf16Response(reply);
  auto overflow = runner.run(executable, inputs, outputs,
                             BF16Compensation::SignSplit, &trace);
  CHECK(overflow);
  CHECK(trace.last == CIMTraceEventKind::Error);
  llvm::consumeError(std::move(overflow));
  CHECK(device.executeCount == before + 1);
  device.close();
}

void testBF16CompensationWorkIsolation() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  device.reverseReadbacks = true;
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  const auto base = makeBF16Transaction();
  auto group = base.getGroups().front();
  group.works.push_back({1, 3, 1, {}});
  auto first = compensatedWeight(64, 126);
  auto second = first;
  second.workId = 1;
  second.macroSlot = 1;
  auto input0 = base.getDynamicInputs().front();
  auto input1 = input0;
  input1.workId = input1.macroSlot = input1.inputSlot = 1;
  auto output0 = base.getReadbacks().front();
  auto output1 = output0;
  output1.workId = output1.macroSlot = 1;
  CIMTransaction executable("cim22-4x5-v1", 1, 1, {group}, {first, second}, {},
                            {input0, input1}, {output0, output1}, {});
  std::array<std::array<uint16_t, 64>, 2> input{};
  input[0][0] = 0x7880;
  input[1][0] = 0x3f00;
  std::array<std::array<uint16_t, 16>, 2> output{};
  const std::array<CIMInputView, 2> inputs{CIMInputView{{}, input[0]},
                                           CIMInputView{{}, input[1]}};
  std::array<CIMOutputView, 2> outputs{CIMOutputView{{}, output[0]},
                                       CIMOutputView{{}, output[1]}};
  CIMRunInputs runInputs{inputs};
  CIMRunOutputs runOutputs{outputs};
  std::array<uint16_t, 16> reply{};
  reply.fill(0x4000);
  reply[0] = 0x4280;
  device.bf16Response = bf16Response(reply);
  for (bool swapped : {false, true}) {
    expectSuccess(runner.run(executable, runInputs, runOutputs,
                             BF16Compensation::SignSplit));
    CHECK(output[swapped ? 1 : 0][0] == 0x7800);
    CHECK(output[swapped ? 0 : 1][0] == 0x3e80);
    std::swap(input[0], input[1]);
  }
  CHECK(device.executeCount == 2);
  device.close();
}

void testMixedModeReadbacks() {
  for (bool corruptPadding : {false, true}) {
    FakeCIMDevice device;
    device.mixedBF16Output = 1;
    // Valid replies arrive out of order; malformed replies fail only after
    // one valid reply has decoded, checking publication remains atomic.
    device.reverseReadbacks = !corruptPadding;
    device.corruptResponsePadding = corruptPadding;
    expectSuccess(device.open());
    CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
    CIMTransaction executable = makeTwoGroupTransaction(CIMDataType::BF16);
    std::array<int8_t, 64> intInput{};
    std::array<uint16_t, 64> bf16Input{};
    const std::array<CIMInputView, 2> inputViews{CIMInputView{intInput, {}},
                                                 CIMInputView{{}, bf16Input}};
    std::array<int32_t, 16> intOutput;
    std::array<uint16_t, 16> bf16Output;
    intOutput.fill(77);
    bf16Output.fill(0xdead);
    std::array<CIMOutputView, 2> outputViews{CIMOutputView{intOutput, {}},
                                             CIMOutputView{{}, bf16Output}};
    CIMRunOutputs outputs{outputViews};
    llvm::Error error = runner.run(executable, {inputViews}, outputs);
    if (corruptPadding) {
      CHECK(error);
      CHECK(llvm::toString(std::move(error)).find("non-zero padding") !=
            std::string::npos);
      CHECK(std::all_of(intOutput.begin(), intOutput.end(),
                        [](int32_t value) { return value == 77; }));
      CHECK(std::all_of(bf16Output.begin(), bf16Output.end(),
                        [](uint16_t value) { return value == 0xdead; }));
    } else {
      expectSuccess(std::move(error));
      CHECK(intOutput == kExpected);
      CHECK(bf16Output == kBF16Expected);
    }
    device.close();
  }
}

void testMalformedResponseDoesNotPublishPartialOutputs() {
  FakeCIMDevice device;
  device.corruptResponsePadding = true;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, defaultCIM22HardwareConfig());
  CIMTransaction executable = makeTwoMacroTransaction();
  std::array<std::array<int8_t, 64>, 2> inputStorage{};
  const std::array<CIMInputView, 2> inputViews{
      CIMInputView{inputStorage[0], {}}, CIMInputView{inputStorage[1], {}}};
  CIMRunInputs inputs{inputViews};
  std::array<std::array<int32_t, 16>, 2> outputStorage;
  for (auto &output : outputStorage)
    output.fill(77);
  std::array<CIMOutputView, 2> outputViews{CIMOutputView{outputStorage[0], {}},
                                           CIMOutputView{outputStorage[1], {}}};
  CIMRunOutputs outputs{outputViews};

  llvm::Error error = runner.run(executable, inputs, outputs);
  CHECK(error);
  CHECK(llvm::toString(std::move(error)).find("non-zero padding") !=
        std::string::npos);
  const std::array<int32_t, 16> untouched = [] {
    std::array<int32_t, 16> values{};
    values.fill(77);
    return values;
  }();
  for (const auto &output : outputStorage)
    CHECK(output == untouched);
  device.close();
}
// Independently reconstruct physical Cache/CIM operands using the Windows
// chip_test.cpp cacheInputLane/cimWeightLane formulas, not production inverses.
void testInt8PhysicalDotProduct() {
  std::array<uint8_t, 16 * 64> weights{};
  for (size_t lane = 0; lane < 16; ++lane)
    for (size_t k = 0; k < 64; ++k)
      weights[lane * 64 + k] = static_cast<uint8_t>(3 * k + 5 * lane);
  const auto cim = mlir::cim22::mapInt8WeightTileToCIMWords(weights);
  const auto signedByte = [](uint8_t value) -> int32_t {
    return value < 128 ? value : static_cast<int32_t>(value) - 256;
  };
  for (size_t k = 0; k < 64; ++k) {
    std::array<int8_t, 64> input{};
    input[k] = k % 2 ? -3 : 2;
    const auto cache = packCIMInt8InputCacheRow(input);
    for (size_t lane = 0; lane < 16; ++lane) {
      int32_t dot = 0;
      for (size_t p = 0; p < 64; ++p) {
        const auto x =
            static_cast<uint8_t>(cache[15 - p / 4] >> (16 * (p % 4)));
        uint8_t w = 0;
        for (size_t bit = 0; bit < 8; ++bit)
          w |= ((cim[(p % 32) * 8 + bit] >> (2 * lane + (p >= 32))) & 1U)
               << bit;
        dot += signedByte(x) * signedByte(w);
      }
      CHECK(dot == input[k] * signedByte(weights[lane * 64 + k]));
    }
  }
}

} // namespace

int main() {
  testInt8PhysicalDotProduct();
  testPacketCodec();
  testBoardRunner();
  testGroupExecutionOrder();
  testFullArrayReadbackBatches();
  testTwoMacrosShareReturnRouteAndOnce();
  testBF16RawCaptureKeepsPrefixAndBody();
  testRawCaptureValidatesInputsBeforeDeviceExecution();
  testRawCaptureValidatesResponseBindings();
  testUnknownDataModeIsRejectedBeforeDeviceExecution();
  testInvalidPlacementAndReadbackRouteRejectBeforeIO();
  testBF16BoardReadbackPreservesBits();
  testBF16BoardCompensation();
  testBF16CompensationWorkIsolation();
  testMixedModeReadbacks();
  testMalformedResponseDoesNotPublishPartialOutputs();
  return 0;
}
