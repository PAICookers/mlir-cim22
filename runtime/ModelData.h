// Serializable model contract, independent of live execution state.
#ifndef CIM22_RUNTIME_MODELDATA_H
#define CIM22_RUNTIME_MODELDATA_H

#include "CIM22/Execution/CIMExecutable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace cim22::runtime {
enum class RuntimeBackend { Software, USB };
// Numeric values match the artifact DataType and runtime dtype query ABI.
enum class BoundaryDataType : int64_t { Int8 = 1, FP32 = 3, Int32 = 4 };

struct ModelData {
  llvm::SmallVector<llvm::SmallVector<int64_t>> shapes;
  // Parallel to shapes; every boundary tensor has an explicit element type.
  llvm::SmallVector<BoundaryDataType> types;
  std::vector<execution::CIMTransaction> transactions;
  std::vector<std::string> kernelNames;
  int64_t numInputs = 0;
  RuntimeBackend backend = RuntimeBackend::Software;
};

// Protobuf persists owned static data; live buffers/devices remain in Session.
llvm::Expected<std::string> encodeModelData(const ModelData &model);
llvm::Error decodeModelData(llvm::StringRef data, ModelData &model);
} // namespace cim22::runtime

#endif
