//===- CIMSegments.cpp - CIM transaction analysis --------------*- C++ -*-===//

#include "CIM22/Conversion/LinalgToCIM/CIMSegments.h"

#include "CIM22/Dialect/CIM/IR/CIMDialect.h"
#include "CIM22/Dialect/CIM/IR/CIMOps.h"

namespace mlir::cim {
bool isExecutionPlanOp(Operation *op) {
  return isa<TransactionOp, ConfigureInputOp, ConfigureWeightOp, DispatchOp,
             OnceOp, ReadbackOp, GroupBarrierOp>(op);
}

SmallVector<CIMTransactionInfo>
analyzeCIMTransactions(func::FuncOp function) {
  SmallVector<CIMTransactionInfo> transactions;
  for (Operation &op : function.getBody().front()) {
    auto transaction = dyn_cast<TransactionOp>(op);
    if (!transaction)
      continue;
    const int64_t transactionIdx =
        cast<IntegerAttr>(op.getAttr(CIMDialect::getTransactionIdxAttrName()))
            .getInt();
    CIMTransactionInfo info{transactionIdx, {}, {}};
    info.inputs.assign(transaction.getInputs().begin(),
                       transaction.getInputs().end());
    info.outputs.assign(transaction.getResults().begin(),
                        transaction.getResults().end());
    transactions.push_back(std::move(info));
  }
  return transactions;
}

} // namespace mlir::cim
