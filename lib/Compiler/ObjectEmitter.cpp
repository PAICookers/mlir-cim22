//===- ObjectEmitter.cpp - AOT Host code and embedded CIM model data
//--------===//
#include "CIM22/Compiler/ObjectEmitter.h"
#include "CIM22/Compiler/RegisterAll.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Runtime/ModelArtifact.h"
#include "CIM22/Target/CIMExecutable.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>

using namespace mlir;
namespace {
struct CompileState {
  ::cim22::runtime::ModelArtifact model;
  std::string error;
};
// Keep the existing descriptor calling convention. Embed a single versioned
// model descriptor and binary data in the PIC object; no runtime compiler or
// sidecar file is needed to find the entry, weights, or tensor signature.
void addModelDescriptor(llvm::Module &module, StringRef entryName,
                        StringRef bytes) {
  auto &context = module.getContext();
  llvm::IRBuilder<> builder(context);
  auto *pointer = builder.getPtrTy();
  auto *entry = module.getFunction(("_mlir_ciface_" + entryName).str());
  auto *invoke = llvm::Function::Create(
      llvm::FunctionType::get(builder.getVoidTy(), {pointer}, false),
      llvm::GlobalValue::InternalLinkage, "__cim22_invoke", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", invoke));
  SmallVector<llvm::Value *> arguments;
  for (size_t i = 0; i < entry->arg_size(); ++i) {
    auto *slot = builder.CreateConstGEP1_64(pointer, invoke->getArg(0), i);
    auto *address = builder.CreateLoad(pointer, slot);
    arguments.push_back(builder.CreateLoad(pointer, address));
  }
  builder.CreateCall(entry, arguments);
  builder.CreateRetVoid();
  auto *payload = llvm::ConstantDataArray::getString(context, bytes, false);
  auto *data = new llvm::GlobalVariable(module, payload->getType(), true,
                                        llvm::GlobalValue::PrivateLinkage,
                                        payload, "__cim22_data");
  auto *type = llvm::StructType::get(builder.getInt64Ty(), pointer,
                                     builder.getInt64Ty(), pointer);
  auto *descriptor = new llvm::GlobalVariable(
      module, type, true, llvm::GlobalValue::PrivateLinkage,
      llvm::ConstantStruct::get(type, builder.getInt64(1), data,
                                builder.getInt64(bytes.size()), invoke),
      "__cim22_descriptor");
  auto *get = llvm::Function::Create(
      llvm::FunctionType::get(pointer, {}, false),
      llvm::GlobalValue::ExternalLinkage, "cim22_model_v1", module);
  builder.SetInsertPoint(llvm::BasicBlock::Create(context, "entry", get));
  builder.CreateRet(descriptor);
}

// Tensor packing is ordinary Host IR. Stack buffers live only for the native
// call; returned tensors are materialized before the wrapper returns.
LogicalResult
defineDispatchWrapper(func::FuncOp stub, int64_t id, func::FuncOp dispatchFunc,
                      const ::cim22::execution::CIMTransaction &transaction) {
  Type inputType =
      cast<MemRefType>(dispatchFunc.getArgumentTypes()[1]).getElementType();
  Type outputType =
      cast<MemRefType>(dispatchFunc.getArgumentTypes()[2]).getElementType();
  bool int8 = inputType.isInteger(8);
  auto dataType = int8 ? ::cim22::execution::CIMDataType::Int8
                       : ::cim22::execution::CIMDataType::BF16;
  if (llvm::any_of(
          transaction.getDynamicInputs(),
          [&](const auto &binding) { return binding.dataType != dataType; }) ||
      llvm::any_of(transaction.getReadbacks(), [&](const auto &binding) {
        return binding.dataType != dataType;
      }))
    return stub.emitError("runtime transaction has incompatible data types");
  for (Type type : stub.getArgumentTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getShape() != ArrayRef<int64_t>{64} ||
        tensor.getEncoding() || tensor.getElementType() != inputType)
      return stub.emitError("runtime transaction input type mismatch");
  }
  for (Type type : stub.getResultTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getShape() != ArrayRef<int64_t>{16} ||
        tensor.getEncoding() ||
        (int8 ? !tensor.getElementType().isInteger(21)
              : tensor.getElementType() != outputType))
      return stub.emitError("runtime transaction output type mismatch");
  }
  if (stub.getNumResults() != transaction.getReadbacks().size())
    return stub.emitError("runtime readback count differs from Host results");
  Block *block = stub.addEntryBlock();
  // The wrapper packs inputs by index, so a strided view needs no prior copy.
  auto inputLayout = StridedLayoutAttr::get(
      stub.getContext(), ShapedType::kDynamic, {ShapedType::kDynamic});
  for (unsigned index = 0; index < stub.getNumArguments(); ++index)
    stub.setArgAttr(index,
                    bufferization::BufferizationDialect::kBufferLayoutAttrName,
                    inputLayout);
  OpBuilder builder(block, block->begin());
  Location loc = stub.getLoc();
  Value input = memref::AllocaOp::create(
      builder, loc,
      MemRefType::get(
          {static_cast<int64_t>(transaction.getDynamicInputs().size()), 64},
          inputType));
  Value output = memref::AllocaOp::create(
      builder, loc,
      MemRefType::get(
          {static_cast<int64_t>(transaction.getReadbacks().size()), 16},
          outputType));
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value bound = arith::ConstantIndexOp::create(builder, loc, 64);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  for (auto [index, binding] :
       llvm::enumerate(transaction.getDynamicInputs())) {
    if (binding.inputSlot < 0 ||
        binding.inputSlot >= static_cast<int64_t>(block->getNumArguments()))
      return stub.emitError("runtime input binding is outside Host arguments");
    Value argument = block->getArgument(binding.inputSlot);
    Value row = arith::ConstantIndexOp::create(builder, loc, index);
    scf::ForOp::create(
        builder, loc, zero, bound, one, ValueRange{},
        [&](OpBuilder &nested, Location at, Value i, ValueRange) {
          Value value =
              tensor::ExtractOp::create(nested, at, argument, ValueRange{i});
          memref::StoreOp::create(nested, at, value, input, ValueRange{row, i});
          scf::YieldOp::create(nested, at);
        });
  }
  Value inputCast = memref::CastOp::create(
      builder, loc, dispatchFunc.getArgumentTypes()[1], input);
  Value outputCast = memref::CastOp::create(
      builder, loc, dispatchFunc.getArgumentTypes()[2], output);
  Value kernel = arith::ConstantIntOp::create(builder, loc, id, 64);
  func::CallOp::create(builder, loc, dispatchFunc,
                       ValueRange{kernel, inputCast, outputCast});
  SmallVector<Value> results;
  for (auto [index, type] : llvm::enumerate(stub.getResultTypes())) {
    Value row = arith::ConstantIndexOp::create(builder, loc, index);
    results.push_back(tensor::GenerateOp::create(
        builder, loc, type, ValueRange{},
        [&](OpBuilder &nested, Location at, ValueRange indices) {
          Value value = memref::LoadOp::create(nested, at, output,
                                               ValueRange{row, indices[0]});
          // Runners provide signed i21 tile values in i32 storage.
          if (int8)
            value = arith::TruncIOp::create(
                nested, at, cast<RankedTensorType>(type).getElementType(),
                value);
          tensor::YieldOp::create(nested, at, value);
        }));
  }
  func::ReturnOp::create(builder, loc, results);
  stub->removeAttr("cim.kernel");
  return success();
}

// Validate split packet IR, build native dispatch bindings and emit an object.
LogicalResult load(CompileState &state, StringRef path, StringRef entryName,
                   StringRef objectPath) {
  DialectRegistry registry;
  registerCIM22Dialects(registry);
  registerCIM22Extensions(registry);
  registerCIM22Passes();
  registerCIM22Pipelines();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  // Limit live rewrite state during native lowering of large Host graphs.
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  // Dispatch wrappers introduce SCF and MemRef even when the parsed artifact
  // contains neither. Load them before constructing operations, not later when
  // the pass manager initializes dependencies.
  context.loadAllAvailableDialects();
  ScopedDiagnosticHandler diagnostics(&context, [&](Diagnostic &diag) {
    llvm::raw_string_ostream stream(state.error);
    diag.print(stream);
    stream << '\n';
    return success();
  });
  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module)
    return failure();
  if ((*module)->hasAttr("cim.bf16_corrected"))
    return module->emitError(
        "obsolete cim.bf16_corrected marker; recompile the model");
  StringRef compensation = "none";
  if (Attribute attr = (*module)->getAttr("cim.bf16_compensation")) {
    auto mode = dyn_cast<StringAttr>(attr);
    if (!mode || (mode.getValue() != "sign-split" &&
                  mode.getValue() != "legacy" && mode.getValue() != "none"))
      return module->emitError(
          "BF16 compensation must be sign-split, legacy or none");
    compensation = mode.getValue();
  }
  auto parts = getCIMSplitParts(*module);
  if (failed(parts))
    return failure();
  auto [host, device] = *parts;
  auto entry = host.lookupSymbol<func::FuncOp>(entryName);
  if (!entry || entry.isDeclaration())
    return host.emitError("runtime entry function is missing");
  state.model.numInputs = entry.getNumArguments();
  SmallVector<Type> boundary(entry.getArgumentTypes());
  llvm::append_range(boundary, entry.getResultTypes());
  for (Type type : boundary) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getEncoding() ||
        !(tensor.getElementType().isF32() ||
          tensor.getElementType().isSignlessInteger(8) ||
          tensor.getElementType().isSignlessInteger(32)) ||
        llvm::any_of(tensor.getShape(), [](int64_t dim) { return dim <= 0; }))
      return entry.emitError(
          "runtime boundary requires static FP32, i8 or i32 tensors");
    state.model.shapes.emplace_back(tensor.getShape());
    state.model.types.push_back(
        tensor.getElementType().isF32()
            ? ::cim22::runtime::BoundaryDataType::FP32
        : tensor.getElementType().isInteger(8)
            ? ::cim22::runtime::BoundaryDataType::Int8
            : ::cim22::runtime::BoundaryDataType::Int32);
  }
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(host.getBody());
  if (host.lookupSymbol("cim22_dispatch") ||
      host.lookupSymbol("cim22_dispatch_int8") ||
      host.lookupSymbol("cim22_dispatch_legacy") ||
      host.lookupSymbol("cim22_dispatch_sign_split"))
    return host.emitError("reserved runtime dispatch symbol already exists");
  auto dispatchFunc = func::FuncOp::create(
      builder, host.getLoc(),
      compensation == "sign-split" ? "cim22_dispatch_sign_split"
      : compensation == "legacy"   ? "cim22_dispatch_legacy"
                                   : "cim22_dispatch",
      builder.getFunctionType(
          {builder.getI64Type(),
           MemRefType::get({ShapedType::kDynamic, 64}, builder.getBF16Type()),
           MemRefType::get({ShapedType::kDynamic, 16}, builder.getBF16Type())},
          {}));
  dispatchFunc.setPrivate();
  dispatchFunc->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto dispatchInt8 = func::FuncOp::create(
      builder, host.getLoc(), "cim22_dispatch_int8",
      builder.getFunctionType(
          {builder.getI64Type(),
           MemRefType::get({ShapedType::kDynamic, 64}, builder.getI8Type()),
           MemRefType::get({ShapedType::kDynamic, 16}, builder.getI32Type())},
          {}));
  dispatchInt8.setPrivate();
  dispatchInt8->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  // Index the large device module once. Keep ordinals because packet
  // verification relies on the original adjacency within each transaction.
  using IndexedOp = std::pair<size_t, Operation *>;
  llvm::DenseMap<StringAttr, IndexedOp> weights;
  llvm::DenseMap<StringAttr, SmallVector<IndexedOp>> packets;
  SmallVector<IndexedOp> shared;
  for (auto [index, op] : llvm::enumerate(device.getBody()->getOperations())) {
    if (isa<func::FuncOp>(op))
      continue;
    IndexedOp entry{index, &op};
    auto binding = op.getAttrOfType<DictionaryAttr>("cim.plan_binding");
    auto function = binding ? binding.getAs<FlatSymbolRefAttr>("function")
                            : FlatSymbolRefAttr();
    if (auto weight = dyn_cast<cim::StaticWeightOp>(op)) {
      weights.try_emplace(weight.getSymNameAttr(), entry);
    } else if (function) {
      packets[function.getAttr()].push_back(entry);
    } else if (!binding) {
      shared.push_back(entry);
    }
  }
  for (auto kernel : device.getOps<func::FuncOp>()) {
    auto stub = host.lookupSymbol<func::FuncOp>(kernel.getSymName());
    if (!stub || !stub.isDeclaration() ||
        stub.getFunctionType() != kernel.getFunctionType())
      return kernel.emitError("missing Host transaction declaration");
    SmallVector<IndexedOp> operations(shared);
    StringAttr function = kernel.getSymNameAttr();
    if (auto it = packets.find(function); it != packets.end())
      llvm::append_range(operations, it->second);
    llvm::SmallDenseSet<StringAttr> resources;
    kernel.walk([&](cim::ConfigureWeightOp configuration) {
      auto ref = dyn_cast<FlatSymbolRefAttr>(configuration.getResourceAttr());
      if (!ref)
        return;
      StringAttr resource = ref.getAttr();
      if (!resources.insert(resource).second)
        return;
      auto it = weights.find(resource);
      if (it == weights.end())
        return;
      auto binding =
          it->second.second->getAttrOfType<DictionaryAttr>("cim.plan_binding");
      if (binding) {
        auto owner = binding.getAs<FlatSymbolRefAttr>("function");
        if (!owner || owner.getAttr() != function)
          return;
      }
      operations.push_back(it->second);
    });
    llvm::sort(operations, [](const IndexedOp &a, const IndexedOp &b) {
      return a.first < b.first;
    });
    OwningOpRef<ModuleOp> single(ModuleOp::create(kernel.getLoc()));
    for (const IndexedOp &entry : operations)
      single->getBody()->push_back(entry.second->clone());
    single->getBody()->push_back(kernel->clone());
    auto transaction = mlir::cim22::target::compileCIMTransaction(*single);
    if (failed(transaction))
      return failure();
    bool int8 = !transaction->getReadbacks().empty() &&
                transaction->getReadbacks().front().dataType ==
                    ::cim22::execution::CIMDataType::Int8;
    if (failed(defineDispatchWrapper(stub, state.model.transactions.size(),
                                     int8 ? dispatchInt8 : dispatchFunc,
                                     *transaction)))
      return failure();
    state.model.transactions.push_back(std::move(*transaction));
    state.model.kernelNames.push_back(kernel.getSymName().str());
  }
  // Transactions own the decoded data, so device IR can be released before
  // Host bufferization allocates its analysis state.
  device.erase();
  entry->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  if (failed(verify(host)))
    return failure();
  PassManager pm(&context);
  if (failed(parsePassPipeline("canonicalize,cse,cim-host-to-llvm", pm)) ||
      failed(pm.run(host)))
    return failure();
  auto interface =
      host.lookupSymbol<LLVM::LLVMFuncOp>(("_mlir_ciface_" + entryName).str());
  if (!interface ||
      !isa<LLVM::LLVMVoidType>(interface.getFunctionType().getReturnType()) ||
      interface.getFunctionType().getParams().size() !=
          state.model.shapes.size())
    return host.emitError("runtime requires input/output descriptor arguments");
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
  auto machineBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!machineBuilder) {
    state.error = llvm::toString(machineBuilder.takeError());
    return failure();
  }
  machineBuilder->setRelocationModel(llvm::Reloc::PIC_)
      .setCodeGenOptLevel(llvm::CodeGenOptLevel::None)
      .setCodeModel(llvm::CodeModel::Small)
      .setCPU("generic")
      .setFeatures("");
  auto machine = machineBuilder->createTargetMachine();
  if (!machine) {
    state.error = llvm::toString(machine.takeError());
    return failure();
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(host, llvmContext);
  if (!llvmModule)
    return failure();
  module = nullptr;
  auto data = ::cim22::runtime::encodeModelArtifact(state.model);
  if (!data) {
    state.error = llvm::toString(data.takeError());
    return failure();
  }
  addModelDescriptor(*llvmModule, entryName, *data);
  ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                  machine->get());
  llvm::orc::SimpleCompiler compiler(**machine);
  auto object = compiler(*llvmModule);
  if (!object) {
    state.error = llvm::toString(object.takeError());
    return failure();
  }
  int fd;
  SmallString<256> temporary;
  if (auto error = llvm::sys::fs::createUniqueFile(objectPath + ".tmp-%%%%%%",
                                                   fd, temporary)) {
    state.error = "cannot create object file: " + error.message();
    return failure();
  }
  llvm::ToolOutputFile output(temporary, fd);
  output.os() << (*object)->getBuffer();
  output.os().close();
  if (output.os().has_error()) {
    state.error = "cannot write object file: " + output.os().error().message();
    output.os().clear_error();
    return failure();
  }
  if (auto error = llvm::sys::fs::rename(temporary, objectPath)) {
    state.error = "cannot publish object file: " + error.message();
    return failure();
  }
  output.keep();
  return success();
}
} // namespace

namespace cim22::compiler {
bool emitObject(llvm::StringRef path, llvm::StringRef objectPath,
                llvm::StringRef backend, std::string &error) {
  CompileState state;
  if (backend == "simulator")
    state.model.backend = ::cim22::runtime::RuntimeBackend::Software;
  else if (backend == "usb")
    state.model.backend = ::cim22::runtime::RuntimeBackend::USB;
  else {
    error = "unsupported CIM22 runtime backend";
    return false;
  }
  if (path.empty() || objectPath.empty() ||
      failed(load(state, path, "main", objectPath))) {
    error = state.error.empty() ? "invalid compile arguments" : state.error;
    return false;
  }
  return true;
}
} // namespace cim22::compiler
