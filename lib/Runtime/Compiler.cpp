//===- Compiler.cpp - Native Host code and embedded CIM model data --------===//
#include "CIM22/Compiler/Compiler.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "CIM22/Execution/CIMRunner.h"
#include "CIM22/Execution/Runtime.h"
#include "CIM22/Target/CIMExecutable.h"
#include "Session.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

using namespace mlir;
using ::cim22::runtime::Session;
namespace {
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
  for (Type type : stub.getArgumentTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getShape() != ArrayRef<int64_t>{64} ||
        !tensor.getElementType().isBF16())
      return stub.emitError("runtime supports BF16 transaction inputs only");
  }
  for (Type type : stub.getResultTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getShape() != ArrayRef<int64_t>{16} ||
        !tensor.getElementType().isBF16())
      return stub.emitError("runtime supports BF16 transaction outputs only");
  }
  if (stub.getNumResults() != transaction.getReadbacks().size())
    return stub.emitError("runtime readback count differs from Host results");
  Block *block = stub.addEntryBlock();
  OpBuilder builder(block, block->begin());
  Location loc = stub.getLoc();
  Value input = memref::AllocaOp::create(
      builder, loc,
      MemRefType::get(
          {static_cast<int64_t>(transaction.getDynamicInputs().size()), 64},
          builder.getBF16Type()));
  Value output = memref::AllocaOp::create(
      builder, loc,
      MemRefType::get(
          {static_cast<int64_t>(transaction.getReadbacks().size()), 16},
          builder.getBF16Type()));
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
          tensor::YieldOp::create(nested, at, value);
        }));
  }
  func::ReturnOp::create(builder, loc, results);
  stub->removeAttr("cim.kernel");
  return success();
}

LogicalResult load(Session &session, const char *path, StringRef entryName,
                   StringRef objectPath = {}) {
  DialectRegistry registry;
  registerCIM22Compiler(registry);
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);
  // Dispatch wrappers introduce SCF and MemRef even when the parsed artifact
  // contains neither. Load them before constructing operations, not later when
  // the pass manager initializes dependencies.
  context.loadAllAvailableDialects();
  ScopedDiagnosticHandler diagnostics(&context, [&](Diagnostic &diag) {
    llvm::raw_string_ostream stream(session.error);
    diag.print(stream);
    stream << '\n';
    return success();
  });
  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module)
    return failure();
  auto parts = getCIMSplitParts(*module);
  if (failed(parts))
    return failure();
  auto [host, device] = *parts;
  auto entry = host.lookupSymbol<func::FuncOp>(entryName);
  if (!entry || entry.isDeclaration())
    return host.emitError("runtime entry function is missing");
  session.model.numInputs = entry.getNumArguments();
  SmallVector<Type> boundary(entry.getArgumentTypes());
  llvm::append_range(boundary, entry.getResultTypes());
  for (Type type : boundary) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getEncoding() ||
        !tensor.getElementType().isF32() ||
        llvm::any_of(tensor.getShape(), [](int64_t dim) { return dim <= 0; }))
      return entry.emitError("runtime boundary requires static FP32 tensors");
    session.model.shapes.emplace_back(tensor.getShape());
  }
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(host.getBody());
  if (host.lookupSymbol("cim22_dispatch"))
    return host.emitError(
        "reserved runtime symbol cim22_dispatch already exists");
  auto dispatchFunc = func::FuncOp::create(
      builder, host.getLoc(), "cim22_dispatch",
      builder.getFunctionType(
          {builder.getI64Type(),
           MemRefType::get({ShapedType::kDynamic, 64}, builder.getBF16Type()),
           MemRefType::get({ShapedType::kDynamic, 16}, builder.getBF16Type())},
          {}));
  dispatchFunc.setPrivate();
  dispatchFunc->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  for (auto kernel : device.getOps<func::FuncOp>()) {
    auto stub = host.lookupSymbol<func::FuncOp>(kernel.getSymName());
    if (!stub || !stub.isDeclaration() ||
        stub.getFunctionType() != kernel.getFunctionType())
      return kernel.emitError("missing Host transaction declaration");
    llvm::SmallDenseSet<Attribute> resources;
    kernel.walk([&](cim::ConfigureWeightOp weight) {
      resources.insert(weight.getResourceAttr());
    });
    OwningOpRef<ModuleOp> single(ModuleOp::create(kernel.getLoc()));
    for (Operation &op : device.getBody()->getOperations()) {
      if (isa<func::FuncOp>(op))
        continue;
      // Unrelated static weights need not be cloned and verified per kernel.
      if (auto weight = dyn_cast<cim::StaticWeightOp>(op);
          weight && !resources.contains(FlatSymbolRefAttr::get(weight)))
        continue;
      auto binding = op.getAttrOfType<DictionaryAttr>("cim.plan_binding");
      if (binding) {
        auto symbol = binding.getAs<FlatSymbolRefAttr>("function");
        if (!symbol || symbol.getValue() != kernel.getSymName())
          continue;
      }
      single->getBody()->push_back(op.clone());
    }
    single->getBody()->push_back(kernel->clone());
    auto transaction = mlir::cim22::target::compileCIMTransaction(*single);
    if (failed(transaction) ||
        failed(defineDispatchWrapper(stub, session.model.transactions.size(),
                                     dispatchFunc, *transaction)))
      return failure();
    session.model.transactions.push_back(std::move(*transaction));
    session.model.kernelNames.push_back(kernel.getSymName().str());
  }
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
          session.model.shapes.size())
    return host.emitError("runtime requires input/output descriptor arguments");
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  });
  ExecutionEngineOptions options;
  // Static time unrolling can produce very large Host functions. LLVM's O2
  // loop rotation/MemorySSA updates have excessive memory cost on these
  // graphs. Keep the first software runtime at O0; numerical behavior and
  // bounded compilation take priority over Host performance here.
  options.jitCodeGenOptLevel = llvm::CodeGenOptLevel::None;
  // Library loading should not create a process-global perf dump directory.
  options.enablePerfNotificationListener = false;
  auto machineBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!machineBuilder) {
    session.error = llvm::toString(machineBuilder.takeError());
    return failure();
  }
  machineBuilder->setRelocationModel(llvm::Reloc::PIC_)
      .setCodeGenOptLevel(llvm::CodeGenOptLevel::None);
  // Keep the target's JIT code model: Windows JIT helpers and their state can
  // be farther apart than the small model permits. AOT DLLs use small code.
  if (!objectPath.empty())
    machineBuilder->setCodeModel(llvm::CodeModel::Small)
        .setCPU("generic")
        .setFeatures("");
  auto machine = machineBuilder->createTargetMachine();
  if (!machine) {
    session.error = llvm::toString(machine.takeError());
    return failure();
  }
  if (!objectPath.empty()) {
    // AOT emits bytes directly: ExecutionEngine::dumpToObjectFile has no error
    // result and cannot report an open/write failure to the native caller.
    llvm::LLVMContext llvmContext;
    auto module = translateModuleToLLVMIR(host, llvmContext);
    if (!module)
      return failure();
    auto data = ::cim22::runtime::encodeModelData(session.model);
    if (!data) {
      session.error = llvm::toString(data.takeError());
      return failure();
    }
    addModelDescriptor(*module, entryName, *data);
    ExecutionEngine::setupTargetTripleAndDataLayout(module.get(),
                                                    machine->get());
    llvm::orc::SimpleCompiler compiler(**machine);
    auto object = compiler(*module);
    if (!object) {
      session.error = llvm::toString(object.takeError());
      return failure();
    }
    // Publish only a complete object; a failed write leaves any prior output
    // intact. A sibling temporary also makes the final rename atomic.
    int fd;
    SmallString<256> temporary;
    if (auto error = llvm::sys::fs::createUniqueFile(objectPath + ".tmp-%%%%%%",
                                                     fd, temporary)) {
      session.error = "cannot create object file: " + error.message();
      return failure();
    }
    llvm::ToolOutputFile output(temporary, fd);
    output.os() << (*object)->getBuffer();
    output.os().close();
    if (output.os().has_error()) {
      session.error =
          "cannot write object file: " + output.os().error().message();
      output.os().clear_error();
      return failure();
    }
    if (auto error = llvm::sys::fs::rename(temporary, objectPath)) {
      session.error = "cannot publish object file: " + error.message();
      return failure();
    }
    output.keep();
    return success();
  }
  auto engine = ExecutionEngine::create(host, options, std::move(*machine));
  if (!engine) {
    session.error += llvm::toString(engine.takeError());
    return failure();
  }
  (*engine)->registerSymbols([](llvm::orc::MangleAndInterner mangle) {
    return llvm::orc::SymbolMap{
        {mangle("_mlir_ciface_cim22_dispatch"),
         llvm::orc::ExecutorSymbolDef(
             llvm::orc::ExecutorAddr::fromPtr(&_mlir_ciface_cim22_dispatch),
             llvm::JITSymbolFlags::Exported)}};
  });
  auto invoke = (*engine)->lookupPacked(("_mlir_ciface_" + entryName).str());
  if (!invoke) {
    session.error = llvm::toString(invoke.takeError());
    return failure();
  }
  session.invoke = *invoke;
  session.code = std::move(*engine);
  return success();
}
} // namespace

extern "C" void *cim22_session_load_packet(const char *path, const char *entry,
                                           char *error, size_t capacity) {
  auto session = std::make_unique<Session>();
  if (!path || !entry || failed(load(*session, path, entry))) {
    if (session->error.empty())
      session->error = "invalid runtime load arguments";
    ::cim22::runtime::copyError(session->error, error, capacity);
    return nullptr;
  }
  session->runner = std::make_unique<::cim22::execution::CIMSoftwareRunner>();
  return session.release();
}

extern "C" int cim22_compile_object(const char *path, const char *objectPath,
                                    const char *backend, char *error,
                                    size_t capacity) {
  Session session;
  if (backend && llvm::StringRef(backend) == "software")
    session.model.backend = ::cim22::runtime::RuntimeBackend::Software;
  else if (backend && llvm::StringRef(backend) == "usb")
    session.model.backend = ::cim22::runtime::RuntimeBackend::USB;
  else {
    ::cim22::runtime::copyError("unsupported CIM22 runtime backend", error,
                                capacity);
    return 1;
  }
  if (!path || !objectPath || !*objectPath ||
      failed(load(session, path, "main", objectPath))) {
    ::cim22::runtime::copyError(
        session.error.empty() ? "invalid compile arguments" : session.error,
        error, capacity);
    return 1;
  }
  return 0;
}
