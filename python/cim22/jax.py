"""Static JAX inference through ONNX and the native CIM22 runtime."""

from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Self

import jax
import jax.numpy as jnp
import numpy as np
from jax2onnx import to_onnx

from ._onnx_frontend import compile_onnx
from .compiler import BackendOptions, _validate_options
from .runtime import Session


class CompiledModel:
    """Eager JAX callable owning a native Session and its output tree."""

    def __init__(
        self, session: Session, outputs: jax.tree_util.PyTreeDef
    ) -> None:
        self.session = session
        self._outputs = outputs

    def __call__(self, *inputs: jax.Array) -> Any:
        """Run positional arrays and restore the JAX output pytree."""
        if any(not isinstance(value, jax.Array) for value in inputs):
            raise TypeError("CIM22 JAX inputs must be eager jax.Array objects")
        values = self.session(*(np.asarray(value) for value in inputs))
        return self._outputs.unflatten([jnp.asarray(value) for value in values])

    def close(self) -> None:
        """Release the owned native model."""
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
    function: Callable[..., Any],
    input_signature: Sequence[jax.ShapeDtypeStruct],
    *,
    options: BackendOptions | None = None,
) -> CompiledModel:
    """Compile a pure JAX function with closed-over weights for eager inference.

    Args:
        function: Callable returning a nonempty pytree of dense arrays.
        input_signature: Flat positional specs with positive static dimensions
            and FP32, INT8 or INT32 dtype.
        options: Native runtime (USB by default), board configuration, BF16
            compensation and optional artifact and debug directories.

    Returns:
        A callable accepting JAX arrays. Close it or use a context manager.
    """
    options = options or {}
    _validate_options(options)
    if not callable(function):
        raise TypeError("JAX compilation requires a callable")
    if not input_signature or any(
        not isinstance(spec, jax.ShapeDtypeStruct)
        or spec.dtype not in (np.float32, np.int8, np.int32)
        or any(type(dim) is not int or dim <= 0 for dim in spec.shape)
        for spec in input_signature
    ):
        raise ValueError(
            "JAX inputs require flat, static positive FP32/INT8/INT32 shapes"
        )

    outputs = jax.eval_shape(function, *input_signature)
    specs, tree = jax.tree.flatten(outputs)
    if not specs or any(
        spec.dtype not in (np.float32, np.int8, np.int32) for spec in specs
    ):
        raise ValueError("JAX outputs require FP32/INT8/INT32 arrays")

    def flattened(*inputs: jax.Array) -> tuple[jax.Array, ...]:
        return tuple(jax.tree.leaves(function(*inputs)))

    model = to_onnx(
        flattened, inputs=input_signature, model_name="main", opset=18
    )
    return CompiledModel(compile_onnx(model, options, "JAX"), tree)
