"""Compile exported TF ONNX using the released torch-mlir wheel in isolation."""

import argparse
from pathlib import Path

import onnx
from torch_mlir.compiler_utils import (
    OutputType,
    lower_mlir_module,
    run_pipeline_with_repro_report,
)
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.extras import onnx_importer
from torch_mlir.ir import Context

from .compiler import _compile_linalg


def main() -> None:
    """Import ONNX, reuse native compilation, and leave an owned model library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--runtime", choices=("software", "usb"), required=True)
    parser.add_argument("--debug-dir", type=Path)
    args = parser.parse_args()
    model = onnx.load_model(args.source)
    with Context() as context:
        torch_dialect.register_dialect(context)
        info = onnx_importer.ModelInfo(model)
        module = info.create_module(context=context)
        importer = onnx_importer.NodeImporter.define_function(
            info.main_graph, module.operation
        )
        importer.import_all()
        run_pipeline_with_repro_report(
            module,
            "builtin.module(torch-onnx-to-torch-backend-pipeline)",
            "Importing TensorFlow ONNX into Torch backend IR",
        )
        lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, module)
        linalg_ir = str(module)
    with _compile_linalg(
        linalg_ir,
        artifact_path=args.destination,
        debug_dir=args.debug_dir,
        runtime=args.runtime,
    ):
        pass


if __name__ == "__main__":
    main()
