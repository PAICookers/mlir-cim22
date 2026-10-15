// Serializable model contract, independent of live execution state.
#ifndef CIM22_RUNTIME_MODELDATA_H
#define CIM22_RUNTIME_MODELDATA_H

#include "CIM22/Execution/CIMExecutable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace cim22::runtime {
struct ModelData {
  llvm::SmallVector<llvm::SmallVector<int64_t>> shapes;
  std::vector<execution::CIMTransaction> transactions;
  std::vector<std::string> kernelNames;
  std::string outputSchema;
  int64_t numInputs = 0;
};

// Protobuf persists owned static data; live buffers/devices remain in Session.
llvm::Expected<std::string> encodeModelData(const ModelData &model);
llvm::Error decodeModelData(llvm::StringRef data, ModelData &model);
} // namespace cim22::runtime

#endif
