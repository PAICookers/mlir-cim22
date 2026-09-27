//===- Pipelines.cpp - Mixed BF16 CIM and Host pipelines ------*- C++ -*-===//

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace {
// Reuse the registered upstream pipelines. This file belongs to the optimizer
// driver, which registers all upstream and project passes before these aliases.
constexpr StringLiteral preparePipeline =
    "canonicalize,cse,normalize-cim-linear{allow-f32=true},"
    "normalize-cim-conv{allow-f32=true "
    "allow-bias-epilogue=true},prepare-cim-bf16";
constexpr StringLiteral vmmPipeline = "partition-cim-program,form-cim-program";
constexpr StringLiteral packetPipeline =
    "partition-cim-program,form-cim-program,"
    "func.func(materialize-cim-schedule,map-cim-schedule),"
    "materialize-cim-execution-plan,materialize-cim-static-weight-section,"
    "lower-cimframe-commands-to-packets,verify-cimframe";
constexpr StringLiteral hostPipeline =
    "convert-elementwise-to-linalg,"
    "one-shot-bufferize{bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map},"
    "buffer-results-to-out-params,buffer-deallocation-pipeline,"
    "convert-bufferization-to-memref,"
    "convert-linalg-to-loops,expand-strided-metadata,lower-affine,"
    "convert-scf-to-cf,convert-math-to-llvm,convert-arith-to-llvm,"
    "convert-cf-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,"
    "reconcile-unrealized-casts";

void appendPipeline(OpPassManager &pm, StringRef pipeline) {
  if (failed(parsePassPipeline(pipeline, pm)))
    llvm::report_fatal_error("invalid built-in CIM22 pipeline");
}

FailureOr<std::pair<ModuleOp, ModuleOp>> getSplitParts(ModuleOp root) {
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

// Tiling creates Host pad/slice/concat and cross-K accumulation around the
// transactions. Move those wrappers back to Host and outline only transactions
// as device entry points. Thus no tensor arithmetic is delegated to runtime.
LogicalResult outlineTransactions(ModuleOp host, ModuleOp device) {
  SmallVector<func::FuncOp> wrappers(device.getOps<func::FuncOp>());
  for (func::FuncOp wrapper : wrappers) {
    auto declaration = host.lookupSymbol<func::FuncOp>(wrapper.getSymName());
    if (!declaration || !declaration.isDeclaration() ||
        declaration.getFunctionType() != wrapper.getFunctionType())
      return wrapper.emitError("missing matching Host kernel declaration");
    SmallVector<cim::TransactionOp> transactions;
    wrapper.walk([&](cim::TransactionOp op) { transactions.push_back(op); });
    IRRewriter rewriter(host.getContext());
    unsigned suffix = 0;
    for (cim::TransactionOp transaction : transactions) {
      std::string name;
      do {
        name = (wrapper.getSymName() + "_transaction_" + Twine(suffix++)).str();
      } while (SymbolTable::lookupSymbolIn(host, name) ||
               SymbolTable::lookupSymbolIn(device, name));
      auto type = rewriter.getFunctionType(transaction->getOperandTypes(),
                                           transaction->getResultTypes());
      rewriter.setInsertionPointToEnd(device.getBody());
      auto kernel =
          func::FuncOp::create(rewriter, transaction.getLoc(), name, type);
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
      func::ReturnOp::create(rewriter, transaction.getLoc(),
                             clone->getResults());

      rewriter.setInsertionPointToEnd(host.getBody());
      auto stub =
          func::FuncOp::create(rewriter, transaction.getLoc(), name, type);
      stub.setPrivate();
      stub->setAttr("cim.kernel", rewriter.getUnitAttr());
      rewriter.setInsertionPoint(transaction);
      rewriter.replaceOpWithNewOp<func::CallOp>(transaction, stub,
                                                transaction.getInputs());
    }
    declaration.erase();
    wrapper->moveBefore(host.getBody(), host.getBody()->end());
    wrapper.setPrivate();
    wrapper->removeAttr("cim.execution_plan_schema_version");
    wrapper->removeAttr("cim.kernel");
  }
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
    appendPipeline(pm, hostPipeline);
    pm.getDependentDialects(registry);
  }
  void runOnOperation() final {
    ModuleOp host = getOperation();
    if (host->hasAttr("cim.split_program")) {
      auto parts = getSplitParts(host);
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
    OpPassManager pm(ModuleOp::getOperationName());
    appendPipeline(pm, hostPipeline);
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
    appendPipeline(pm, hostPipeline);
    pm.getDependentDialects(registry);
  }
  Option<std::string> stage{*this, "stage", llvm::cl::init("packet"),
                            llvm::cl::desc("split, vmm, packet or llvm")};
  Option<bool> requireCIM{
      *this, "require-cim", llvm::cl::init(false),
      llvm::cl::desc("Require offload and reject Host named contractions")};
  void runOnOperation() final {
    if (stage != "split" && stage != "vmm" && stage != "packet" &&
        stage != "llvm") {
      getOperation().emitError("stage must be split, vmm, packet or llvm");
      return signalPassFailure();
    }
    auto parts = getSplitParts(getOperation());
    if (failed(parts))
      return signalPassFailure();
    if (requireCIM) {
      bool missing = parts->second.getOps<func::FuncOp>().empty();
      parts->first.walk([&](linalg::LinalgOp op) {
        if (isa<linalg::MatmulOp, linalg::MatvecOp>(op.getOperation()) ||
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
      if (failed(outlineTransactions(parts->first, parts->second)))
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
  Option<int64_t> maxKernelColumns{*this, "max-kernel-columns",
                                   llvm::cl::init(32)};
  Option<bool> requireCIM{*this, "require-cim", llvm::cl::init(false)};
};
} // namespace

void registerCIM22Pipelines() {
  PassRegistration<LowerCIMHostToLLVM>();
  PassRegistration<CompileCIMParts>();
  PassPipelineRegistration<>(
      "cim-bf16-prepare",
      "Normalize floating geometry and insert BF16 boundaries",
      [](OpPassManager &pm) { appendPipeline(pm, preparePipeline); });
  PassPipelineRegistration<BF16PipelineOptions>(
      "cim-bf16-pipeline",
      "Compile mixed Linalg into separate CIM and Host modules",
      [](OpPassManager &pm, const BF16PipelineOptions &options) {
        appendPipeline(pm, preparePipeline);
        appendPipeline(pm, "outline-cim-bf16{max-kernel-columns=" +
                               std::to_string(options.maxKernelColumns) + "}");
        auto compile = std::make_unique<CompileCIMParts>();
        compile->stage = options.stage;
        compile->requireCIM = options.requireCIM;
        pm.addPass(std::move(compile));
      });
  PassPipelineRegistration<>(
      "cim-host-to-llvm", "Lower Host Linalg/Tensor/Arith to LLVM",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<LowerCIMHostToLLVM>());
      });
}
