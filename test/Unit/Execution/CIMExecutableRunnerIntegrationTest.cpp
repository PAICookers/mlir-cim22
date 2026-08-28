//===- CIMExecutableRunnerIntegrationTest.cpp - CIM22 pipeline test -*- C++
//-*-===//

#include "CIM22/Conversion/LinalgToCIM/ExecutionPlanVerifier.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Dialect/CIMFrame/IR/CIMFrameDialect.h"
#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Target/CIMExecutable.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace mlir;
using namespace ::cim22::execution;

namespace {
struct Trace final : CIMTraceSink {
  std::vector<CIMTraceEvent> events;
  void record(const CIMTraceEvent &event) override { events.push_back(event); }
};

const DynamicInputBinding *findInput(const CIMExecutable &executable,
                                     int64_t groupId, int64_t workId) {
  for (const DynamicInputBinding &input : executable.getDynamicInputs())
    if (input.groupId == groupId && input.workId == workId)
      return &input;
  return nullptr;
}

const StaticWeightSection *findWeight(const CIMExecutable &executable,
                                      int64_t groupId, int64_t workId) {
  for (const StaticWeightSection &weight : executable.getStaticWeights())
    if (weight.groupId == groupId && weight.workId == workId)
      return &weight;
  return nullptr;
}

size_t inputIndex(const CIMExecutable &executable,
                  const DynamicInputBinding *target) {
  for (auto [index, input] : llvm::enumerate(executable.getDynamicInputs()))
    if (&input == target)
      return index;
  assert(false && "missing dynamic input binding");
  return 0;
}
} // namespace

int main(int argc, char **argv) {
  const bool batch2 = argc == 3 && std::string(argv[2]) == "batch2";
  if (argc != 2 && !batch2) {
    std::cerr << "usage: mlir-cim22-cim-executable-runner-test <plan.mlir> "
                 "[batch2]\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<cim::CIMDialect, cimframe::CIMFrameDialect,
                      arith::ArithDialect, func::FuncDialect,
                      linalg::LinalgDialect, tensor::TensorDialect>();
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module) {
    std::cerr << "FAIL: plan must parse\n";
    return 1;
  }

  size_t readbacksBefore = 0;
  Attribute firstMapping;
  module->walk([&](cim::ReadbackOp readback) {
    if (readbacksBefore++ == 0)
      firstMapping = readback->getAttr("cim.mapping");
  });

  auto executable = mlir::cim22::target::compileCIMExecutable(*module);
  if (failed(executable)) {
    std::cerr << "FAIL: CIM executable construction\n";
    return 1;
  }
  const CIMExecutable &value = *executable;
  const size_t expectedGroups = batch2 ? 4 : 20;
  const size_t expectedWorks = batch2 ? 8 : 40;
  if (value.getGroups().size() != expectedGroups ||
      value.getStaticWeights().size() != expectedWorks ||
      value.getDynamicInputs().size() != expectedWorks ||
      value.getReadbacks().size() != expectedWorks)
    return 1;

  std::vector<std::array<int8_t, kCIMInputElements>> inputStorage(
      expectedWorks);
  std::vector<CIMInputView> inputViews(expectedWorks);
  for (size_t index = 0; index < inputStorage.size(); ++index) {
    for (size_t k = 0; k < kCIMInputElements; ++k)
      inputStorage[index][k] =
          static_cast<int8_t>(static_cast<int>(index + k) % 7 - 3);
    inputViews[index].values = llvm::ArrayRef<int8_t>(inputStorage[index]);
  }
  std::vector<std::array<int32_t, kCIMOutputElements>> outputStorage(
      expectedWorks);
  std::vector<CIMOutputView> outputViews(expectedWorks);
  for (size_t index = 0; index < outputStorage.size(); ++index)
    outputViews[index].values =
        llvm::MutableArrayRef<int32_t>(outputStorage[index]);

  CIMRunInputs inputs{llvm::ArrayRef<CIMInputView>(inputViews)};
  CIMRunOutputs outputs{llvm::MutableArrayRef<CIMOutputView>(outputViews)};
  Trace trace;
  CIMSoftwareRunner runner;
  if (llvm::Error error = runner.run(value, inputs, outputs, &trace)) {
    llvm::logAllUnhandledErrors(std::move(error), llvm::errs(), "FAIL: ");
    return 1;
  }

  bool sawAddress0 = false;
  bool sawAddress1 = false;
  bool sawAddress7 = false;
  for (size_t index = 0; index < value.getReadbacks().size(); ++index) {
    const ReadbackBinding &binding = value.getReadbacks()[index];
    const DynamicInputBinding *input =
        findInput(value, binding.groupId, binding.workId);
    const StaticWeightSection *weight =
        findWeight(value, binding.groupId, binding.workId);
    if (!input || !weight)
      return 1;
    sawAddress0 |= binding.outputCacheAddress == 0;
    sawAddress1 |= binding.outputCacheAddress == 1;
    sawAddress7 |= binding.outputCacheAddress == 7;
    size_t inputPosition = inputIndex(value, input);
    for (size_t row = 0; row < kCIMOutputElements; ++row) {
      int32_t expected = 0;
      for (size_t k = 0; k < kCIMInputElements; ++k)
        expected +=
            static_cast<int32_t>(weight->values[row * kCIMInputElements + k]) *
            static_cast<int32_t>(inputStorage[inputPosition][k]);
      if (outputStorage[index][row] != expected)
        return 1;
    }
  }

  size_t readbacksAfter = 0;
  Attribute mappingAfter;
  module->walk([&](cim::ReadbackOp readback) {
    if (readbacksAfter++ == 0)
      mappingAfter = readback->getAttr("cim.mapping");
  });
  const size_t expectedTraceEvents = batch2 ? 50 : 242;
  const bool addressesValid =
      batch2 ? (sawAddress0 && sawAddress1) : (sawAddress0 && sawAddress7);
  if (readbacksBefore != readbacksAfter || firstMapping != mappingAfter ||
      !addressesValid || trace.events.size() != expectedTraceEvents ||
      trace.events.front().kind != CIMTraceEventKind::BeginRun ||
      trace.events.back().kind != CIMTraceEventKind::Complete)
    return 1;

  std::cout << "PASS CIM runner integration profile="
            << value.getTargetProfile().str()
            << " groups=" << value.getGroups().size()
            << " works=" << value.getDynamicInputs().size()
            << " readbacks=" << value.getReadbacks().size()
            << " trace_events=" << trace.events.size()
            << " output_addresses=" << (batch2 ? "0,1" : "0,7") << '\n';
  return 0;
}
