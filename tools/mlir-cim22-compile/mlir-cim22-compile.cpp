#include "CIM22/Compiler/ObjectEmitter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using llvm::cl::opt;

namespace {
opt<std::string> input(llvm::cl::Positional, llvm::cl::desc("<packet.mlir>"),
                        llvm::cl::Required);
opt<std::string> output("o", llvm::cl::desc("Object output path"),
                        llvm::cl::value_desc("path"), llvm::cl::Required);
opt<std::string> backend(
    "backend", llvm::cl::desc("Runtime backend"),
    llvm::cl::value_desc("simulator|usb"),
    llvm::cl::init("usb"));
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Compile CIM22 packet-stage MLIR\n");
  std::string error;
  if (!cim22::compiler::emitObject(input, output, backend, error)) {
    llvm::errs() << "mlir-cim22-compile: " << error << '\n';
    return 1;
  }
  return 0;
}
