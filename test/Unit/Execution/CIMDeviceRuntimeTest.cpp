//===- CIMDeviceRuntimeTest.cpp - CIM22 device runtime tests ---*- C++ -*-===//

#include "CIM22/Runtime/CIMBoardRunner.h"
#include "CIM22/Runtime/CIMPacketCodec.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using namespace cim22::execution;
using namespace cim22::runtime;

namespace {
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition))                                                          \
      std::abort();                                                            \
  } while (false)
constexpr std::array<int32_t, 16> kExpected{
    0,   1,      -1,      -1048576, 1048575, 74565, -74565, 42,
    -42, 262144, -262144, 7,        -7,      21930, -21930, 291};
constexpr std::array<uint64_t, 6> kResponse{
    0x00091fea9580ab55ULL, 0xffff900003f00000ULL, 0x80001fffd6000157ULL,
    0xb72ec2468afffff8ULL, 0x00007ffffc000020ULL, 0x0000000000000000ULL};

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
    for (const CIMDeviceReadbackCommand &command : program.readbacks) {
      CIMDeviceReadback readback;
      readback.outputIndex = command.outputIndex;
      const uint64_t outputCacheRead = command.request.flits.back();
      const int64_t coreSlot =
          outputCoreSlots.empty() ? 3 : outputCoreSlots[command.outputIndex];
      readback.headerFlit = (uint64_t{5} << 60) |
                            (static_cast<uint64_t>(coreSlot) << 18) |
                            (((outputCacheRead >> 14) & 0xf) << 14) | 6;
      readback.bodyFlits = kResponse;
      if (corruptResponsePadding &&
          command.outputIndex + 1 == program.readbacks.size())
        readback.bodyFlits[5] = 1;
      response.readbacks.push_back(readback);
    }
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
  std::vector<int64_t> outputCoreSlots;
  CIMDeviceProgram program;
  size_t executeCount = 0;
  bool corruptResponsePadding = false;
  ResponseFault responseFault = ResponseFault::None;
};

CIMTransaction makeTransaction(CIMDataType dataType = CIMDataType::Int8) {
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
  weight.dataType = dataType;

  DynamicInputBinding input{0, 0, 0, 0, dataType, 7};
  ReadbackBinding readback{0, 0, 0, 7, route, testCore, dataType};
  return CIMTransaction("cim22-4x5-v1", 1, 1, {std::move(group)},
                        {std::move(weight)}, {}, {input}, {readback}, {});
}

CIMTransaction makeTwoGroupTransaction() {
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
    weights.push_back(std::move(weight));
    inputs.push_back({groupId, groupId, 0, groupId, CIMDataType::Int8, 7});
    readbacks.push_back(
        {groupId, groupId, 0, 7, route, testCore, CIMDataType::Int8});
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
  CHECK(words[15] == 0x0080007f00fe0001ULL);
  for (size_t index = 0; index < 15; ++index)
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
  CIMBoardRunner runner(device, {});
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
  CHECK(device.program.operations[0].flits.front() == 0xa0000000000c0000ULL);
  CHECK(device.program.operations[1].flits.size() == 1);
  CHECK((device.program.operations[1].flits.front() >> 60) == 0x8);
  CHECK(device.program.operations[2].flits.size() == 17);
  CHECK(device.program.operations[2].flits.back() == 0x0080007f00fe0001ULL);
  CHECK(device.program.operations[2].postDelayMs == 1);
  CHECK(device.program.operations[3].flits.size() == 257);
  CHECK(device.program.operations[3].postDelayMs == 2);
  CHECK(device.program.operations[4].flits.size() == 1);
  CHECK(device.program.operations[4].flits.front() == 0x9000000000000001ULL);
  CHECK(device.program.operations[4].postDelayMs == 2);
  CHECK(device.program.readbacks.size() == 1);
  CHECK(device.program.readbacks.front().request.flits.size() == 2);
  CHECK(device.program.readbacks.front().request.flits.back() ==
        0x500000000081c000ULL);
  device.close();
}

void testGroupExecutionOrder() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, {});
  CIMTransaction executable = makeTwoGroupTransaction();
  device.outputCoreSlots = {3, 4};

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
  CHECK(device.program.operations[7].postDelayMs == 2);
  CHECK(device.program.operations[7].flits[0] == 0x9000000000000001ULL);
  CHECK(device.program.operations[7].flits[1] == 0x9000000000000001ULL);
  device.close();
}

void testTwoMacrosShareReturnRouteAndOnce() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, {});
  const std::array<int32_t, 6> route{};
  CIMTransaction executable = makeTwoMacroTransaction();
  device.outputCoreSlots = {3, 3};

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
  CHECK(device.program.operations[1].flits.size() == 1);
  CHECK(device.program.operations[4].flits.size() == 1);
  CHECK(device.program.operations[7].flits.size() == 1);
  CHECK(device.program.operations[7].flits.front() == 0x9000000000000001ULL);
  CHECK(device.program.readbacks.size() == 2);
  CHECK(device.program.readbacks[0].request.flits.back() ==
        encodeCIMOutputCacheReadFlit(route, 6));
  CHECK(device.program.readbacks[1].request.flits.back() ==
        encodeCIMOutputCacheReadFlit(route, 7));
  device.close();
}

void testBF16RawCaptureKeepsPrefixAndBody() {
  FakeCIMDevice device;
  device.corruptResponsePadding = true;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, {});
  CIMTransaction executable = makeBF16Transaction();

  std::array<uint16_t, 64> inputStorage{};
  const std::array<CIMInputView, 1> inputViews{CIMInputView{{}, inputStorage}};
  CIMRunInputs inputs{inputViews};
  auto response = runner.capture(executable, inputs);
  CHECK(response);
  CHECK(response->readbacks.size() == 1);
  CHECK(response->readbacks[0].outputIndex == 0);
  CHECK(response->readbacks[0].headerFlit == 0x50000000000dc006ULL);
  auto expectedBody = kResponse;
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
  CHECK(device.program.operations[5].flits.front() ==
        encodeCIMWorkIdleFlit(route));
  device.close();
}

void testRawCaptureValidatesInputsBeforeDeviceExecution() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, {});
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
  CHECK(device.executeCount == 0);
  device.close();
}

void testRawCaptureValidatesResponseBindings() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, {});
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
  CIMBoardRunner runner(device, {});
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
    CIMBoardRunner runner(device, {});
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

void testBF16IsRejectedBeforeDeviceExecution() {
  FakeCIMDevice device;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, {});
  CIMTransaction executable = makeBF16Transaction();
  std::array<uint16_t, 64> inputStorage{};
  const std::array<CIMInputView, 1> inputViews{CIMInputView{{}, inputStorage}};
  CIMRunInputs inputs{inputViews};
  std::array<uint16_t, 16> outputStorage{};
  std::array<CIMOutputView, 1> outputViews{CIMOutputView{{}, outputStorage}};
  CIMRunOutputs outputs{outputViews};

  llvm::Error error = runner.run(executable, inputs, outputs);
  CHECK(error);
  CHECK(llvm::toString(std::move(error)).find("BF16 readback is disabled") !=
        std::string::npos);
  CHECK(device.executeCount == 0);
  device.close();
}

void testMalformedResponseDoesNotPublishPartialOutputs() {
  FakeCIMDevice device;
  device.corruptResponsePadding = true;
  expectSuccess(device.open());
  CIMBoardRunner runner(device, {});
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
  const std::array<int32_t, 16> untouched = [] {
    std::array<int32_t, 16> values{};
    values.fill(77);
    return values;
  }();
  for (const auto &output : outputStorage)
    CHECK(output == untouched);
  device.close();
}
} // namespace

int main() {
  testPacketCodec();
  testBoardRunner();
  testGroupExecutionOrder();
  testTwoMacrosShareReturnRouteAndOnce();
  testBF16RawCaptureKeepsPrefixAndBody();
  testRawCaptureValidatesInputsBeforeDeviceExecution();
  testRawCaptureValidatesResponseBindings();
  testUnknownDataModeIsRejectedBeforeDeviceExecution();
  testInvalidPlacementAndReadbackRouteRejectBeforeIO();
  testBF16IsRejectedBeforeDeviceExecution();
  testMalformedResponseDoesNotPublishPartialOutputs();
  return 0;
}
