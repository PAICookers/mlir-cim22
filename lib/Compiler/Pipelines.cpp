//===- Pipelines.cpp - Mixed BF16 CIM and Host pipelines ------*- C++ -*-===//

#include "CIM22/Compiler/RegisterAll.h"
#include "CIM22/Conversion/LinalgToCIM/Passes.h"
#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include <mutex>

using namespace mlir;

FailureOr<std::pair<ModuleOp, ModuleOp>> getCIMSplitParts(ModuleOp root) {
  ModuleOp host, device;
  for (auto child : root.getOps<ModuleOp>()) {
    auto part = child->getAttrOfType<StringAttr>("cim.pipeline_part");
    if (part && part.getValue() == "host" && !host)
      host = child;
    else if (part && part.getValue() == "device" && !device)
      device = child;
    else
      return root.emitError("expected exactly one Host and one device module");
  }
  if (!root->hasAttr("cim.split_program") || !host || !device ||
      std::distance(root.getBody()->begin(), root.getBody()->end()) != 2)
    return root.emitError("expected an outline-cim-bf16 split program");
  return std::make_pair(host, device);
}

namespace {
// Reuse upstream pipelines for both the optimizer and the native runtime.
constexpr StringLiteral prepareLinear =
    "canonicalize,cse,normalize-cim-linear{allow-f32=true}";
constexpr StringLiteral prepareConv = "normalize-cim-conv{allow-f32=true "
                                      "allow-bias-epilogue=true}";
constexpr StringLiteral vmmPipeline = "partition-cim-program,form-cim-program";
constexpr StringLiteral packetPipeline =
    "partition-cim-program,form-cim-program,"
    "func.func(materialize-cim-schedule,map-cim-schedule),"
    "materialize-cim-execution-plan,materialize-cim-static-weight-section,"
    "lower-cimframe-commands-to-packets,verify-cimframe";
constexpr StringLiteral hostBufferization =
    "convert-elementwise-to-linalg,"
    "one-shot-bufferize{bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map "
    "analysis-heuristic=top-down},"
    "buffer-results-to-out-params{modify-public-functions=true},"
    "buffer-deallocation-pipeline,"
    "convert-bufferization-to-memref,lower-cim-host-copies";
constexpr StringLiteral hostLLVMLowering =
    "convert-linalg-to-loops,expand-strided-metadata,"
    // Fold static descriptors and loop bounds before expanding control flow.
    "canonicalize,cse,"
    "lower-affine,"
    // Bound SCF conversion bookkeeping to one Host function and discard
    // rollback state on large, statically expanded graphs.
    "func.func(convert-scf-to-cf{allow-pattern-rollback=false}),"
    "arith-expand{include-bf16=true},"
    "convert-math-to-llvm,convert-arith-to-llvm,"
    "convert-cf-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
    "reconcile-unrealized-casts";

void appendPipeline(OpPassManager &pm, StringRef pipeline) {
  if (failed(parsePassPipeline(pipeline, pm)))
    llvm::report_fatal_error("invalid built-in CIM22 pipeline");
}

class MarkResidencyLinear final
    : public PassWrapper<MarkResidencyLinear, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkResidencyLinear)
  void runOnOperation() override {
    Builder builder(&getContext());
    int64_t next = 0;
    // Mark before convolution normalization also creates MatMuls.
    getOperation().walk([&](linalg::LinalgOp op) {
      if (isa<linalg::MatmulOp, linalg::MatvecOp>(op.getOperation()) &&
          op.hasPureTensorSemantics() && op->getNumResults() == 1 &&
          cast<ShapedType>(op->getResult(0).getType()).hasStaticShape())
        op->setAttr(cim::kResidencyLinearAttr,
                    builder.getI64IntegerAttr(next++));
    });
  }
};

using WaveSignatures = llvm::DenseMap<StringAttr, ArrayAttr>;

// Compare the ordered physical placements and exact typed tile values.
// Equal tiles can use distinct resource symbols in different branches.
ArrayAttr weightSignature(cim::TransactionOp transaction,
                          const SymbolTable &symbols) {
  Builder builder(transaction.getContext());
  SmallVector<Attribute> entries;
  for (auto write :
       transaction.getBody().front().getOps<cim::ConfigureWeightOp>()) {
    auto resource = dyn_cast<FlatSymbolRefAttr>(write.getResourceAttr());
    auto weight = resource ? dyn_cast_or_null<cim::StaticWeightOp>(
                                 symbols.lookup(resource.getAttr()))
                           : cim::StaticWeightOp();
    Attribute core = write->getAttr("core_idx");
    Attribute macro = write->getAttr("macro_idx");
    Attribute mapping = write->getAttr("cim.mapping");
    if (!weight || !core || !macro || !mapping)
      return {};
    entries.push_back(
        builder.getArrayAttr({core, macro, mapping, weight.getValue()}));
  }
  return entries.empty() ? ArrayAttr() : builder.getArrayAttr(entries);
}

// Preparations may move only within a straight-line window. A dependency on
// a dispatch result, an unrelated call or a side effect closes the window.
bool movablePreparation(ArrayRef<func::CallOp> calls,
                        SmallVectorImpl<Operation *> &preparation) {
  Operation *first = calls.front();
  Operation *last = calls.back();
  Block *block = first->getBlock();
  if (llvm::any_of(calls, [&](auto call) { return call->getBlock() != block; }))
    return false;
  llvm::SmallDenseSet<Operation *> targets;
  for (auto call : calls)
    targets.insert(call);
  for (Operation *op = first; op != last; op = op->getNextNode()) {
    if (!targets.contains(op) &&
        (!isMemoryEffectFree(op) ||
         (op->getNumRegions() && !isa<tensor::PadOp>(op))))
      return false;
  }
  llvm::SmallDenseSet<Operation *> needed;
  SmallVector<Value> values;
  for (auto call : calls)
    llvm::append_range(values, call.getOperands());
  while (!values.empty()) {
    Operation *op = values.pop_back_val().getDefiningOp();
    if (!op || op->getBlock() != block || op->isBeforeInBlock(first) ||
        !needed.insert(op).second)
      continue;
    if (targets.contains(op) || !isMemoryEffectFree(op))
      return false;
    llvm::append_range(values, op->getOperands());
    llvm::SetVector<Value> captures;
    getUsedValuesDefinedAbove(op->getRegions(), captures);
    llvm::append_range(values, captures);
  }
  llvm::append_range(preparation, needed);
  llvm::sort(preparation, [](Operation *left, Operation *right) {
    return left->isBeforeInBlock(right);
  });
  return true;
}

SmallVector<func::CallOp> groupWeightCalls(ArrayRef<func::CallOp> calls,
                                           const WaveSignatures &signatures) {
  // Preserve first-seen signature order and call order within each signature.
  llvm::MapVector<ArrayAttr, SmallVector<func::CallOp>> groups;
  for (auto call : calls) {
    auto found = signatures.find(call.getCalleeAttr().getAttr());
    if (found == signatures.end())
      return {};
    groups[found->second].push_back(call);
  }
  SmallVector<func::CallOp> grouped;
  for (auto &entry : groups)
    llvm::append_range(grouped, entry.second);
  return grouped;
}

void scheduleWeightCalls(ArrayRef<func::CallOp> calls,
                         const WaveSignatures &signatures) {
  auto grouped = groupWeightCalls(calls, signatures);
  if (grouped.empty() || llvm::equal(grouped, calls))
    return;
  SmallVector<Operation *> preparation;
  if (!movablePreparation(calls, preparation))
    return;
  for (Operation *op : preparation)
    op->moveBefore(calls.front());
  Operation *previous = calls.front();
  for (auto call : ArrayRef(grouped).drop_front()) {
    call->moveAfter(previous);
    previous = call;
  }
}

// Collect compatible sign branches of one Linear in a straight-line Host
// window. Call results keep their original consumers and arithmetic order.
void scheduleLinearBranches(ModuleOp host, const WaveSignatures &signatures) {
  auto root = cast<ModuleOp>(host->getParentOp());
  auto compensation = root->getAttrOfType<StringAttr>("cim.bf16_compensation");
  if (!compensation || compensation.getValue() != "sign-split")
    return;

  SymbolTable symbols(host);
  llvm::SmallDenseSet<Operation *> replaced;
  SmallVector<func::FuncOp> functions(host.getOps<func::FuncOp>());
  for (auto function : functions) {
    for (Block &block : function.getBody()) {
      llvm::MapVector<Attribute, SmallVector<func::CallOp>> linearCalls;
      for (auto call : block.getOps<func::CallOp>()) {
        auto callee = dyn_cast_or_null<func::FuncOp>(
            symbols.lookup(call.getCalleeAttr().getAttr()));
        if (!callee || callee.isDeclaration() || !callee.isPrivate() ||
            !callee.getBody().hasOneBlock())
          continue;
        if (Attribute origin = callee->getAttr(cim::kResidencyLinearAttr))
          linearCalls[origin].push_back(call);
      }

      for (auto &entry : linearCalls) {
        auto &calls = entry.second;
        if (calls.size() < 2)
          continue;
        SmallVector<Operation *> preparation;
        if (!movablePreparation(calls, preparation))
          continue;
        SmallVector<func::CallOp> waves;
        SmallVector<func::FuncOp> callees;
        bool valid = true;
        for (auto call : calls) {
          auto callee = cast<func::FuncOp>(
              symbols.lookup(call.getCalleeAttr().getAttr()));
          callees.push_back(callee);
          for (Operation &op : callee.getBody().front().without_terminator()) {
            if (auto wave = dyn_cast<func::CallOp>(op)) {
              if (!signatures.contains(wave.getCalleeAttr().getAttr()))
                valid = false;
              waves.push_back(wave);
            } else if (!isMemoryEffectFree(&op) ||
                       (op.getNumRegions() && !isa<tensor::PadOp>(op))) {
              valid = false;
            }
          }
        }
        if (!valid || waves.empty() ||
            llvm::equal(waves, groupWeightCalls(waves, signatures)))
          continue;

        // Keep the paired waves out of the caller so bufferization analyzes
        // their dispatch results and reductions within one local function.
        for (Operation *op : preparation)
          op->moveBefore(calls.front());
        IRRewriter rewriter(host.getContext());
        SmallVector<Value> operands;
        SmallVector<Type> operandTypes, resultTypes;
        for (auto call : calls) {
          llvm::append_range(operands, call.getOperands());
          llvm::append_range(operandTypes, call.getOperandTypes());
          llvm::append_range(resultTypes, call.getResultTypes());
        }
        Location loc = calls.front().getLoc();
        auto helper = func::FuncOp::create(
            loc, "__cim_resident_linear",
            rewriter.getFunctionType(operandTypes, resultTypes));
        helper.setPrivate();
        symbols.insert(helper);
        rewriter.setInsertionPointToStart(helper.addEntryBlock());
        SmallVector<func::CallOp> helperWaves;
        SmallVector<Value> results;
        unsigned argumentOffset = 0;
        for (auto [call, callee] : llvm::zip_equal(calls, callees)) {
          IRMapping mapping;
          mapping.map(callee.getArguments(),
                      helper.getArguments().slice(argumentOffset,
                                                  call.getNumOperands()));
          argumentOffset += call.getNumOperands();
          for (Operation &op : callee.getBody().front().without_terminator()) {
            if (auto wave = dyn_cast<func::CallOp>(rewriter.clone(op, mapping)))
              helperWaves.push_back(wave);
          }
          auto result =
              cast<func::ReturnOp>(callee.getBody().front().getTerminator());
          for (Value value : result.getOperands())
            results.push_back(mapping.lookup(value));
          replaced.insert(callee);
        }
        func::ReturnOp::create(rewriter, loc, results);
        scheduleWeightCalls(helperWaves, signatures);

        rewriter.setInsertionPoint(calls.front());
        auto groupedCall =
            func::CallOp::create(rewriter, loc, helper, operands);
        unsigned resultOffset = 0;
        for (auto call : calls) {
          unsigned count = call.getNumResults();
          rewriter.replaceOp(
              call, groupedCall.getResults().slice(resultOffset, count));
          resultOffset += count;
        }
      }
    }
  }
  for (Operation *callee : replaced)
    if (SymbolTable::symbolKnownUseEmpty(callee, host))
      symbols.erase(callee);
}

// Upstream MemRef-to-LLVM leaves strided copies as calls to memrefCopy. The
// packaged C runner DSO pulls in MLIR through its float helpers. Express ranked
// copies as upstream Linalg copies so the existing loop lowering makes models
// independent of that DSO, without duplicating its descriptor-copy algorithm.
class LowerHostCopiesPass
    : public PassWrapper<LowerHostCopiesPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHostCopiesPass)
  StringRef getArgument() const final { return "lower-cim-host-copies"; }
  StringRef getDescription() const final {
    return "Lower ranked Host copies through upstream Linalg loops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    SmallVector<memref::CopyOp> copies;
    getOperation()->walk([&](memref::CopyOp op) { copies.push_back(op); });
    for (auto op : copies) {
      if (!isa<MemRefType>(op.getSource().getType()) ||
          !isa<MemRefType>(op.getTarget().getType())) {
        op.emitError("CIM22 Host requires ranked copy operands");
        return signalPassFailure();
      }
      OpBuilder builder(op);
      linalg::CopyOp::create(builder, op.getLoc(), ValueRange{op.getSource()},
                             ValueRange{op.getTarget()});
      op.erase();
    }
  }
};

// Move tile preparation and cross-K accumulation back to Host. Outline each
// complete device wave behind a typed stub so dispatch order can change while
// each wave keeps its input, output, and packet bindings.
LogicalResult outlineTransactions(ModuleOp host, ModuleOp device,
                                  bool scheduleResidency) {
  SymbolTable hostSymbols(host);
  SymbolTable deviceSymbols(device);
  // Keep resource ownership indexed while moving each wave to its own symbol.
  // Updating a binding in place preserves packet order in the device module.
  using PacketOwner = std::pair<StringAttr, Attribute>;
  llvm::DenseMap<PacketOwner, SmallVector<Operation *, 1>> packets;
  for (Operation &packet : device.getBody()->getOperations()) {
    auto binding = packet.getAttrOfType<DictionaryAttr>("cim.plan_binding");
    if (auto owner = binding ? binding.getAs<FlatSymbolRefAttr>("function")
                             : FlatSymbolRefAttr())
      packets[{owner.getAttr(), binding.get("resource")}].push_back(&packet);
  }
  WaveSignatures signatures;
  SmallVector<func::FuncOp> wrappers(device.getOps<func::FuncOp>());
  for (func::FuncOp wrapper : wrappers) {
    StringAttr owner = wrapper.getSymNameAttr();
    auto declaration =
        dyn_cast_or_null<func::FuncOp>(hostSymbols.lookup(owner));
    if (!declaration || !declaration.isDeclaration() ||
        declaration.getFunctionType() != wrapper.getFunctionType())
      return wrapper.emitError("missing matching Host kernel declaration");
    SmallVector<cim::TransactionOp> transactions;
    wrapper.walk([&](cim::TransactionOp op) { transactions.push_back(op); });
    SmallVector<func::CallOp> calls;
    IRRewriter rewriter(host.getContext());
    unsigned suffix = 0;
    for (cim::TransactionOp transaction : transactions) {
      StringAttr name;
      do {
        name = rewriter.getStringAttr(
            (wrapper.getSymName() + "_transaction_" + Twine(suffix++)).str());
      } while (hostSymbols.lookup(name) || deviceSymbols.lookup(name));
      auto type = rewriter.getFunctionType(transaction->getOperandTypes(),
                                           transaction->getResultTypes());
      auto kernel =
          func::FuncOp::create(transaction.getLoc(), name.getValue(), type);
      deviceSymbols.insert(kernel);
      kernel->setAttr("cim.kernel", rewriter.getUnitAttr());
      for (StringRef attribute :
           {"cim.target_profile", "cim.target_profile_version",
            "cim.placement_policy", "cim.route_policy",
            "cim.execution_plan_schema_version"})
        if (Attribute value = wrapper->getAttr(attribute))
          kernel->setAttr(attribute, value);
      Block *entry = kernel.addEntryBlock();
      rewriter.setInsertionPointToStart(entry);
      IRMapping mapping;
      mapping.map(transaction.getInputs(), entry->getArguments());
      Operation *clone = rewriter.clone(*transaction, mapping);
      clone->setAttr(cim::CIMDialect::getTransactionIdxAttrName(),
                     rewriter.getI64IntegerAttr(0));
      // Packet resources follow the outlined function. Match resource symbols
      // rather than work ids, which may be reused by separate transactions.
      auto newOwner = FlatSymbolRefAttr::get(kernel);
      transaction.walk([&](cim::ConfigureWeightOp weight) {
        auto found = packets.find({owner, weight->getAttr("resource")});
        if (found == packets.end())
          return;
        for (Operation *packet : found->second) {
          NamedAttrList binding(
              packet->getAttrOfType<DictionaryAttr>("cim.plan_binding"));
          binding.set("function", newOwner);
          packet->setAttr("cim.plan_binding",
                          binding.getDictionary(host.getContext()));
        }
        // An already moved resource belonged to the first matching wave.
        packets.erase(found);
      });
      func::ReturnOp::create(rewriter, transaction.getLoc(),
                             clone->getResults());

      auto stub =
          func::FuncOp::create(transaction.getLoc(), name.getValue(), type);
      hostSymbols.insert(stub);
      stub.setPrivate();
      stub->setAttr("cim.kernel", rewriter.getUnitAttr());
      rewriter.setInsertionPoint(transaction);
      if (scheduleResidency && wrapper->hasAttr(cim::kResidencyLinearAttr))
        if (auto signature = weightSignature(transaction, deviceSymbols))
          signatures.try_emplace(stub.getSymNameAttr(), signature);
      calls.push_back(rewriter.replaceOpWithNewOp<func::CallOp>(
          transaction, stub, transaction.getInputs()));
    }
    hostSymbols.erase(declaration);
    deviceSymbols.remove(wrapper);
    wrapper->moveBefore(host.getBody(), host.getBody()->end());
    hostSymbols.insert(wrapper);
    wrapper.setPrivate();
    wrapper->removeAttr("cim.execution_plan_schema_version");
    wrapper->removeAttr("cim.kernel");
    if (calls.size() > 1 && wrapper->hasAttr(cim::kResidencyLinearAttr))
      scheduleWeightCalls(calls, signatures);
  }
  if (!signatures.empty())
    scheduleLinearBranches(host, signatures);
  for (auto function : host.getOps<func::FuncOp>())
    function->removeAttr(cim::kResidencyLinearAttr);
  return success();
}

class LowerCIMHostToLLVM final
    : public PassWrapper<LowerCIMHostToLLVM, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCIMHostToLLVM)
  StringRef getArgument() const final { return "lower-cim-host-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower a Host module through upstream bufferization and LLVM passes";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    OpPassManager pm(ModuleOp::getOperationName());
    appendHostPipeline(pm, /*earlyFree=*/true);
    pm.getDependentDialects(registry);
  }
  void runOnOperation() final {
    ModuleOp host = getOperation();
    if (host->hasAttr("cim.split_program")) {
      auto parts = getCIMSplitParts(host);
      if (failed(parts))
        return signalPassFailure();
      host = parts->first;
    }
    bool hasDeviceOps = false;
    host.walk([&](Operation *op) {
      StringRef dialect = op->getName().getDialectNamespace();
      hasDeviceOps |= dialect == "cim" || dialect == "cimframe";
    });
    if (hasDeviceOps) {
      host.emitError("Host LLVM lowering requires outlined CIM kernels");
      return signalPassFailure();
    }
    // Inference inputs belong to the caller, including private C-interface
    // entries. Let One-Shot Bufferize copy before a helper writes to them.
    for (func::FuncOp function : host.getOps<func::FuncOp>()) {
      if (function.isDeclaration() ||
          (function.isPrivate() &&
           !function->hasAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName())))
        continue;
      for (auto [index, type] : llvm::enumerate(function.getArgumentTypes()))
        if (isa<TensorType>(type))
          function.setArgAttr(
              index, bufferization::BufferizationDialect::kWritableAttrName,
              BoolAttr::get(host.getContext(), false));
    }
    OpPassManager pm(ModuleOp::getOperationName());
    appendHostPipeline(pm, canReleaseAfterLastUse(host));
    if (failed(runPipeline(pm, host)))
      return signalPassFailure();
    // Do not silently claim LLVM output when an unsupported Host op survives.
    bool illegal = false;
    host.walk([&](Operation *op) {
      if (op != host && op->getName().getDialectNamespace() != "llvm") {
        op->emitError("unsupported operation remains after Host LLVM lowering");
        illegal = true;
      }
    });
    if (illegal)
      signalPassFailure();
  }

private:
  static void appendHostPipeline(OpPassManager &pm, bool earlyFree) {
    appendPipeline(pm, hostBufferization);
    if (earlyFree)
      appendPipeline(pm, "func.func(optimize-allocation-liveness)");
    appendPipeline(pm, hostLLVMLowering);
  }

  // The upstream pass tracks SSA aliases, not addresses kept by opaque code.
  // Prove that source IR has no such escapes before bufferization. Native CIM
  // wrappers may pass numeric stack buffers to their synchronous dispatch.
  static bool canReleaseAfterLastUse(ModuleOp host) {
    SymbolTableCollection symbols;
    auto numericBuffer = [](Type type) {
      auto buffer = dyn_cast<BaseMemRefType>(type);
      return !buffer ||
             isa<IntegerType, FloatType, IndexType>(buffer.getElementType());
    };
    return !host.walk([&](Operation *op) {
                  if (auto call = dyn_cast<func::CallOp>(op)) {
                    auto callee = symbols.lookupNearestSymbolFrom<func::FuncOp>(
                        call, call.getCalleeAttr());
                    if (callee && !callee.isDeclaration())
                      return WalkResult::advance();
                    for (Value operand : call.getOperands()) {
                      if (isa<IntegerType, FloatType, IndexType>(
                              operand.getType()))
                        continue;
                      while (auto cast =
                                 operand.getDefiningOp<memref::CastOp>())
                        operand = cast.getSource();
                      if (!operand.getDefiningOp<memref::AllocaOp>())
                        return WalkResult::interrupt();
                    }
                    return llvm::all_of(
                               call.getResultTypes(),
                               [](Type type) {
                                 return isa<IntegerType, FloatType, IndexType>(
                                     type);
                               })
                               ? WalkResult::advance()
                               : WalkResult::interrupt();
                  }
                  if (isa<ModuleOp, func::FuncOp, func::ReturnOp>(op))
                    return WalkResult::advance();
                  if (isa<memref::AllocOp, memref::AllocaOp, memref::CastOp,
                          memref::CopyOp, memref::DeallocOp, memref::DimOp,
                          memref::LoadOp, memref::StoreOp>(op) &&
                      llvm::all_of(op->getOperandTypes(), numericBuffer) &&
                      llvm::all_of(op->getResultTypes(), numericBuffer))
                    return WalkResult::advance();
                  StringRef dialect = op->getName().getDialectNamespace();
                  return llvm::StringSwitch<WalkResult>(dialect)
                      .Cases({"arith", "cf", "linalg", "math", "scf", "tensor"},
                             WalkResult::advance())
                      .Default(WalkResult::interrupt());
                })
                .wasInterrupted();
  }
};

class CompileCIMParts final
    : public PassWrapper<CompileCIMParts, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompileCIMParts)
  CompileCIMParts() = default;
  CompileCIMParts(const CompileCIMParts &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "compile-cim-parts"; }
  StringRef getDescription() const final {
    return "Compile the outlined device module and optionally lower Host to "
           "LLVM";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    // Dynamic pipelines must declare their dependencies on the enclosing pass
    // before execution, while the context can still load dialect interfaces.
    OpPassManager pm(ModuleOp::getOperationName());
    appendPipeline(pm, packetPipeline);
    pm.addPass(std::make_unique<LowerCIMHostToLLVM>());
    pm.getDependentDialects(registry);
  }
  Option<std::string> stage{*this, "stage", llvm::cl::init("packet"),
                            llvm::cl::desc("split, vmm, packet or llvm")};
  Option<bool> requireCIM{
      *this, "require-cim", llvm::cl::init(false),
      llvm::cl::desc("Require offload and reject Host named contractions")};
  // This switch controls compiler ordering. Execution decides weight hits
  // against the current session's physical slots.
  Option<bool> weightResidencySchedule{*this, "weight-residency-schedule",
                                       llvm::cl::init(true)};
  void runOnOperation() final {
    if (stage != "split" && stage != "vmm" && stage != "packet" &&
        stage != "llvm") {
      getOperation().emitError("stage must be split, vmm, packet or llvm");
      return signalPassFailure();
    }
    auto parts = getCIMSplitParts(getOperation());
    if (failed(parts))
      return signalPassFailure();
    if (requireCIM) {
      bool missing = parts->second.getOps<func::FuncOp>().empty();
      parts->first.walk([&](linalg::LinalgOp op) {
        if (isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::MatvecOp>(
                op.getOperation()) ||
            op->getName().getStringRef().contains("conv")) {
          op.emitError("required CIM contraction remained on Host");
          missing = true;
        }
      });
      if (missing) {
        getOperation().emitError("required CIM offload was not completed");
        return signalPassFailure();
      }
    }
    if (stage == "split")
      return;
    OpPassManager devicePM(ModuleOp::getOperationName());
    appendPipeline(devicePM, stage == "vmm" ? vmmPipeline : packetPipeline);
    if (failed(runPipeline(devicePM, parts->second)))
      return signalPassFailure();
    if (stage != "vmm") {
      if (failed(outlineTransactions(parts->first, parts->second,
                                     weightResidencySchedule)))
        return signalPassFailure();
      OpPassManager verifyPM(ModuleOp::getOperationName());
      appendPipeline(verifyPM,
                     "func.func(verify-cim-execution-plan),verify-cimframe");
      if (failed(runPipeline(verifyPM, parts->second)))
        return signalPassFailure();
    }
    if (stage == "llvm") {
      OpPassManager hostPM(ModuleOp::getOperationName());
      hostPM.addPass(std::make_unique<LowerCIMHostToLLVM>());
      if (failed(runPipeline(hostPM, parts->first)))
        return signalPassFailure();
    }
  }
};

struct BF16PipelineOptions : PassPipelineOptions<BF16PipelineOptions> {
  Option<std::string> stage{*this, "stage", llvm::cl::init("packet"),
                            llvm::cl::desc("split, vmm, packet or llvm")};
  Option<std::string> maxKernelColumns{
      *this, "max-kernel-columns", llvm::cl::init("auto"),
      llvm::cl::desc("Fixed column width or auto for wave-aware blocking")};
  Option<bool> requireCIM{*this, "require-cim", llvm::cl::init(false)};
  Option<bool> weightResidencySchedule{*this, "weight-residency-schedule",
                                       llvm::cl::init(true)};
  Option<std::string> bf16Compensation{
      *this, "bf16-compensation", llvm::cl::init("sign-split"),
      llvm::cl::desc("Internal BF16 lowering: sign-split, legacy or none")};
};
} // namespace

void registerCIM22Pipelines() {
  static std::once_flag registration;
  std::call_once(registration, [] {
    PassRegistration<LowerHostCopiesPass>();
    PassRegistration<LowerCIMHostToLLVM>();
    PassRegistration<CompileCIMParts>();
    PassPipelineRegistration<>(
        "cim-bf16-prepare",
        "Normalize floating geometry and insert BF16 boundaries",
        [](OpPassManager &pm) {
          appendPipeline(pm, prepareLinear);
          appendPipeline(pm, prepareConv);
          appendPipeline(pm, "prepare-cim-bf16");
        });
    PassPipelineRegistration<BF16PipelineOptions>(
        "cim-bf16-pipeline",
        "Compile mixed Linalg into separate CIM and Host modules",
        [](OpPassManager &pm, const BF16PipelineOptions &options) {
          appendPipeline(pm, prepareLinear);
          if (options.weightResidencySchedule)
            pm.addPass(std::make_unique<MarkResidencyLinear>());
          appendPipeline(pm, prepareConv);
          cim::SplitCIMBF16SignsOptions splitOptions;
          splitOptions.compensation = options.bf16Compensation;
          pm.addPass(cim::createSplitCIMBF16Signs(splitOptions));
          appendPipeline(pm, "prepare-cim-bf16");
          cim::OutlineCIMBF16Options outlineOptions;
          outlineOptions.maxKernelColumns = options.maxKernelColumns;
          pm.addPass(cim::createOutlineCIMBF16(outlineOptions));
          auto compile = std::make_unique<CompileCIMParts>();
          compile->stage = options.stage;
          compile->requireCIM = options.requireCIM;
          compile->weightResidencySchedule = options.weightResidencySchedule;
          pm.addPass(std::move(compile));
        });
    PassPipelineRegistration<>(
        "cim-host-to-llvm", "Lower Host Linalg/Tensor/Arith to LLVM",
        [](OpPassManager &pm) {
          pm.addPass(std::make_unique<LowerCIMHostToLLVM>());
        });
  });
}
