"""Static TensorFlow inference through released ONNX and torch-mlir frontends."""

from types import TracebackType
from typing import Self

import numpy as np
import tensorflow as tf
import tf2onnx

from ._onnx_frontend import compile_onnx
from .compiler import BackendOptions, _validate_options
from .runtime import Session

type TensorOutputs = (
    tf.Tensor
    | tuple[TensorOutputs, ...]
    | list[TensorOutputs]
    | dict[str, TensorOutputs]
)


class CompiledModel:
    """Owned eager TensorFlow callable backed by a native inference Session.

    ``session`` exposes the low-level NumPy interface and runtime diagnostics.
    Use a context manager or close explicitly to release native resources.
    """

    def __init__(self, session: Session, output_specs: object) -> None:
        """Take ownership of a loaded Session and its TensorFlow output spec.

        Args:
            session: Loaded native Session, closed with this model.
            output_specs: Validated nested TensorSpec structure used to
                reconstruct outputs from the Session's flat result order.
        """
        self.session = session
        self._outputs = output_specs

    def __call__(self, *inputs: tf.Tensor | tf.Variable) -> TensorOutputs:
        """Run eager inference using the native Session.

        Args:
            *inputs: Dense eager tensors or variables matching the compiled
                positional input order, shapes and dtypes.

        Returns:
            TensorFlow tensors arranged in the compiled output structure.

        Raises:
            TypeError: An input is not a dense tensor or variable, or its
                dtype does not match the signature.
            ValueError: Execution is outside eager mode or an input does
                not match the compiled signature.
            RuntimeError: Native execution fails or the model is closed.
        """
        if not tf.executing_eagerly():
            raise ValueError(
                "CIM22 TensorFlow inference requires eager tensors"
            )
        if any(
            not isinstance(value, (tf.Tensor, tf.Variable)) for value in inputs
        ):
            raise TypeError(
                "CIM22 TensorFlow inputs must be dense tf.Tensor or tf.Variable objects"
            )
        outputs = self.session(*(np.asarray(value.numpy()) for value in inputs))
        tensors = [tf.convert_to_tensor(value) for value in outputs]
        return tf.nest.pack_sequence_as(self._outputs, tensors)

    def close(self) -> None:
        """Close the native Session; previously returned tensors remain valid."""
        self.session.close()

    def __enter__(self) -> Self:
        self.session.__enter__()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def compile(
    function: tf.types.experimental.PolymorphicFunction,
    *,
    options: BackendOptions | None = None,
) -> CompiledModel:
    """Freeze stateless inference and compile static FP32 or signed integer IR.

    Args:
        function: A ``tf.function`` with a flat, fully static
            ``input_signature`` (FP32, INT8 or INT32). Wrap Keras calls with
            ``training=False``.
            Captured weights are frozen; subsequent updates require compilation.
        options: Native runtime (USB by default, or simulator), BF16
            compensation, USB configuration, and artifact/debug directories.
            Directories accept strings or Paths. Each compilation writes a
            unique ``graph-<id>.so/.dll`` and matching debug subdirectory.
            USB configuration is applied to the owned Session. CIM-supported
            FP32 contractions use BF16 internally.

    Returns:
        An owned eager callable accepting positional TF tensors in signature
        order and returning TF tensors in the function's output structure.
        Use a context manager or close explicitly. Stateful updates, training,
        gradients and calls inside tf.function are excluded.

    Raises:
        TypeError: The input is not a tf.function.
        ValueError: Invalid options, or the signature is not flat and static
            with FP32, INT8 and INT32 boundary types.
    """
    options = options or {}
    _validate_options(options)
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

    outputs = function.get_concrete_function(*signature).structured_outputs
    flat_outputs = tf.nest.flatten(outputs)
    if not flat_outputs or any(
        not isinstance(value, tf.Tensor) for value in flat_outputs
    ):
        raise ValueError("TensorFlow outputs must be dense tensors")
    output_specs = tf.nest.map_structure(tf.type_spec_from_value, outputs)

    # Export in tf.nest order, including nested dicts that tf2onnx cannot name
    # directly. The callable restores the original structure after execution.
    @tf.function(input_signature=signature)
    def flattened(*inputs: tf.Tensor) -> list[tf.Tensor]:
        return tf.nest.flatten(function(*inputs))

    # Freeze resource variables and delegate TF operator conversion to tf2onnx.
    # No PyTorch model is constructed or executed in this path.
    model, _ = tf2onnx.convert.from_function(
        flattened,
        input_signature=signature,
        opset=18,
    )
    return CompiledModel(
        compile_onnx(model, options, "TensorFlow"), output_specs
    )
