//===- BindParameters.cpp - Explicit entry parameter specialization
//--------===//
#include "CIM22/Conversion/LinalgToCIM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::cim {
#define GEN_PASS_DEF_BINDCIMPARAMETERS
#include "CIM22/Conversion/LinalgToCIM/Passes.h.inc"
namespace {
class BindCIMParameters final
    : public impl::BindCIMParametersBase<BindCIMParameters> {
public:
  using Base::Base;
  void runOnOperation() override {
    auto fail = [&](StringRef message) {
      getOperation().emitError(message);
      signalPassFailure();
    };
    auto manifest = parseSourceFile<ModuleOp>(parameterFile, &getContext());
    if (!manifest)
      return fail("cannot parse parameter manifest");
    auto functions = manifest->getOperation()->getAttrOfType<DictionaryAttr>(
        "cim.parameters");
    if (!functions || functions.empty())
      return fail("expected nonempty cim.parameters dictionary");
    struct Binding {
      func::FuncOp function;
      SmallVector<std::pair<unsigned, DenseElementsAttr>> values;
    };
    SmallVector<Binding> bindings;
    // Validate every name/type before changing any function ABI. This pass is
    // for separately compiled entry points; existing callers cannot be updated
    // without knowing whether they intended the same parameter specialization.
    for (NamedAttribute named : functions) {
      auto function =
          getOperation().lookupSymbol<func::FuncOp>(named.getName());
      auto args = dyn_cast<DictionaryAttr>(named.getValue());
      if (!function || function.isDeclaration() || !args || args.empty())
        return fail(
            "parameter manifest must name defined functions and arguments");
      auto uses = SymbolTable::getSymbolUses(function, getOperation());
      if (!uses || !uses->empty())
        return fail(
            "parameter specialization requires an entry without symbol uses");
      Binding binding{function, {}};
      for (NamedAttribute arg : args) {
        StringRef name = arg.getName().getValue();
        unsigned index;
        auto value = dyn_cast<DenseElementsAttr>(arg.getValue());
        if (!name.consume_front("arg") || name.getAsInteger(10, index) ||
            index >= function.getNumArguments() || !value ||
            value.getType() != function.getArgument(index).getType())
          return fail("parameter argument index or dense tensor type mismatch");
        if (llvm::any_of(binding.values,
                         [&](const auto &v) { return v.first == index; }))
          return fail("duplicate parameter argument index");
        binding.values.emplace_back(index, value);
      }
      bindings.push_back(std::move(binding));
    }
    OpBuilder builder(&getContext());
    for (auto &binding : bindings) {
      builder.setInsertionPointToStart(&binding.function.getBody().front());
      llvm::BitVector erase(binding.function.getNumArguments());
      for (auto [index, value] : binding.values) {
        auto constant = arith::ConstantOp::create(
            builder, binding.function.getLoc(), value);
        binding.function.getArgument(index).replaceAllUsesWith(constant);
        erase.set(index);
      }
      if (failed(binding.function.eraseArguments(erase)))
        return fail("cannot erase specialized entry arguments");
    }
  }
};
} // namespace
} // namespace mlir::cim
