"""Compile static framework exports through the isolated ONNX worker."""

import shutil
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import onnx

from .compiler import BackendOptions, _run
from .runtime import MODEL_SUFFIX, Session, _initialize_session


def compile_onnx(
    model: onnx.ModelProto, options: BackendOptions, framework: str
) -> Session:
    """Validate an exported graph and load its native model into an owned Session."""
    runtime = options.get("runtime", "usb")
    usb_config = options.get("usb_config")
    artifact_dir = options.get("artifact_dir")
    debug_dir = options.get("debug_dir")
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
                f"{framework} outputs require static positive FP32/INT8/INT32 shapes"
            )

    graph_name = f"graph-{uuid4().hex}"
    # Keep the tracing framework and the compiler's LLVM in separate processes. The child
    # publishes a native library; only the execution runtime is loaded here.
    with ExitStack() as cleanup:
        folder = Path(
            cleanup.enter_context(
                tempfile.TemporaryDirectory(
                    prefix=f".cim22-{framework.lower()}-"
                )
            )
        )
        source = folder / "model.onnx"
        shared = (
            (Path(artifact_dir) / f"{graph_name}{MODEL_SUFFIX}").absolute()
            if artifact_dir is not None
            else folder / f"model{MODEL_SUFFIX}"
        )
        onnx.save_model(model, source)
        command = [
            sys.executable,
            # Keep the working directory from shadowing the installed package.
            "-P",
            "-m",
            "cim22._onnx_compile",
            str(source),
            str(shared),
            "--runtime",
            runtime,
            "--bf16-compensation",
            options.get("bf16_compensation", "sign-split"),
        ]
        if debug_dir is not None:
            command.extend(
                ["--debug-dir", str((Path(debug_dir) / graph_name).absolute())]
            )
        _run(command, f"{framework} compilation")
        if debug_dir is not None:
            shutil.copyfile(source, Path(debug_dir) / graph_name / "model.onnx")
        return _initialize_session(
            Session(shared),
            usb_config,
            cleanup.pop_all() if artifact_dir is None else None,
            usb_session=options.get("usb_session"),
        )
