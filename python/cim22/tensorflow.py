"""Static TensorFlow inference through released ONNX and torch-mlir frontends."""

from pathlib import Path

import onnx
import tensorflow as tf
import tf2onnx
from torch_mlir.compiler_utils import (
    OutputType,
    lower_mlir_module,
    run_pipeline_with_repro_report,
)
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir.extras import onnx_importer
from torch_mlir.ir import Context

from .compiler import RuntimeBackend, _compile_linalg
from .runtime import Session


def compile(
    function: tf.types.experimental.PolymorphicFunction,
    *,
    artifact_path: Path | None = None,
    debug_dir: Path | None = None,
    runtime: RuntimeBackend = "usb",
) -> Session:
    """Freeze stateless inference and compile static FP32 or signed integer IR.

    Args:
        function: A ``tf.function`` with a flat, fully static
            ``input_signature`` (FP32, INT8 or INT32). Wrap Keras calls with
            ``training=False``.
            Captured weights are frozen; subsequent updates require compilation.
        artifact_path: Optional native model library (.dll on Windows, .so on
            Linux). Existing files are never overwritten.
        debug_dir: Optional Linalg/packet IR and object-file diagnostics.
        runtime: ``software`` for the existing CIM reference, ``usb`` for the
            board runtime. CIM-supported FP32 contractions use BF16 internally.

    Returns:
        An owned Session accepting contiguous NumPy arrays of the declared
        dtype in signature order, returning a flat tuple in exported output
        order. Use a context manager or close it explicitly. Stateful updates
        and training are excluded.

    Raises:
        TypeError: The input is not a tf.function.
        ValueError: The signature is not flat and static, or its input/output
            dtypes are outside FP32, INT8 and INT32.
    """
    if not isinstance(function, tf.types.experimental.PolymorphicFunction):
        raise TypeError("Wrap inference in tf.function with input_signature")
    signature = function.input_signature
    if not signature or any(
        not isinstance(spec, tf.TensorSpec)
        or spec.dtype not in (tf.float32, tf.int8, tf.int32)
        or not spec.shape.is_fully_defined()
        or any(dimension <= 0 for dimension in spec.shape)
        for spec in signature
    ):
        raise ValueError(
            "TensorFlow inputs require flat, static positive FP32/INT8/INT32 shapes"
        )

    # Freeze resource variables and delegate TF operator conversion to tf2onnx.
    # No PyTorch model is constructed or executed in this path.
    model, _ = tf2onnx.convert.from_function(
        function,
        input_signature=signature,
        opset=18,
    )
    model.graph.name = "main"
    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(model)
    for output in model.graph.output:
        tensor = output.type.tensor_type
        if (
            tensor.elem_type
            not in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.INT8,
                onnx.TensorProto.INT32,
            )
            or not tensor.HasField("shape")
            or any(
                not dim.HasField("dim_value") or dim.dim_value <= 0
                for dim in tensor.shape.dim
            )
        ):
            raise ValueError(
                "TensorFlow outputs require static positive FP32/INT8/INT32 shapes"
            )

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
    return _compile_linalg(
        linalg_ir,
        artifact_path=artifact_path,
        debug_dir=debug_dir,
        runtime=runtime,
    )
