import os

import lit.formats
from lit.llvm import llvm_config


config.name = "MLIR-CIM22"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir", ".test"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.mlir_cim22_obj_root, "test")
config.excludes = ["CMakeLists.txt", "Inputs", "README.txt"]

llvm_config.use_default_substitutions()

tool_dirs = [
    os.path.join(config.mlir_cim22_obj_root, "bin"),
    config.llvm_tools_dir,
]
llvm_config.add_tool_substitutions(
    [
        "FileCheck",
        "mlir-cim22-f0-importer-test",
        "mlir-cim22-int8-weight-layout-test",
        "mlir-cim22-onnx-import",
        "mlir-cim22-opt",
    ],
    tool_dirs,
)
