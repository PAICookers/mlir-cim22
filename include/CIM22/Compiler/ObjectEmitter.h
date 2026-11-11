#ifndef CIM22_COMPILER_OBJECTEMITTER_H
#define CIM22_COMPILER_OBJECTEMITTER_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace cim22::compiler {

/// Compile packet-stage MLIR to one host object file.
bool emitObject(llvm::StringRef packetPath, llvm::StringRef objectPath,
                llvm::StringRef backend, std::string &error);

} // namespace cim22::compiler

#endif
