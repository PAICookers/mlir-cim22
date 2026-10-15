"""Load frozen Linux models with Torch and the native execution runtime."""

import sys
from pathlib import Path
from types import TracebackType

import torch
from torch.utils import _pytree

from cim22._schema import decode_outputs
from cim22.runtime import Session, UsbConfig


class CompiledProgram:
    """Frozen inference callable; FP32 CPU inputs and independently owned outputs."""

    def __init__(self, session: Session, output_spec: _pytree.TreeSpec) -> None:
        if output_spec.num_leaves != len(session.output_shapes):
            raise ValueError("Output structure differs from native signature")
        self.session = session
        self._output_spec = output_spec

    def __call__(self, *inputs: torch.Tensor) -> object:
        arrays = []
        for tensor in inputs:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("CIM22 accepts tensor inputs only")
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                raise TypeError("CIM22 boundary requires CPU FP32 tensors")
            if tensor.requires_grad:
                raise ValueError("CIM22 supports inference only; detach inputs")
            if not tensor.is_contiguous():
                raise ValueError("CIM22 boundary requires contiguous tensors")
            arrays.append(tensor.numpy())
        results = self.session(*arrays)
        return _pytree.tree_unflatten(
            [torch.from_numpy(array) for array in results], self._output_spec
        )

    def close(self) -> None:
        """Release native resources without invalidating previously returned tensors."""
        self.session.close()

    def __enter__(self) -> "CompiledProgram":
        self.session.__enter__()
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def load_compiled(
    artifact: Path,
    *,
    library: Path | None = None,
    usb_config: UsbConfig | None = None,
) -> CompiledProgram:
    """Load a Linux model.so using only the native execution runtime.

    Code, weights, transactions and the tensor signature live inside the .so.
    There is no sidecar manifest, compiler subprocess, MLIR parsing or JIT.
    The artifact must be trusted and match the runtime ABI and host architecture.
    """
    if sys.platform != "linux" or artifact.suffix != ".so":
        raise ValueError("CIM22 deployment artifacts must be Linux .so files")
    session = Session(artifact, library)
    try:
        if usb_config is not None:
            session.configure_usb(usb_config)
        return CompiledProgram(session, decode_outputs(session.output_schema))
    except BaseException:
        session.close()
        raise
