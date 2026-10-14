"""Isolate torch-mlir import from Dynamo's active FakeTensor capture context."""

import sys
from pathlib import Path

import torch
from torch.export.graph_signature import OutputKind, TensorArgument
from torch_mlir import fx


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError(
            "Expected an ExportedProgram path and output MLIR path"
        )
    program = torch.export.load(sys.argv[1]).run_decompositions({})
    if any(
        spec.kind != OutputKind.USER_OUTPUT
        or not isinstance(spec.arg, TensorArgument)
        for spec in program.graph_signature.output_specs
    ):
        raise ValueError(
            "CIM22 requires tensor outputs without input/buffer mutation"
        )
    module = fx.export_and_import(
        program, output_type="linalg-on-tensors", decomposition_table={}
    )
    Path(sys.argv[2]).write_text(str(module), encoding="utf-8")


if __name__ == "__main__":
    main()
