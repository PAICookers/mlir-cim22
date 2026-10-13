"""Static FP32-boundary inference through LLVM and the CIM22 BF16 runner."""

from cim22.runtime import Session

__all__ = ["CompiledProgram", "Session", "compile_exported", "load_compiled"]


def __getattr__(name: str) -> object:
    # Session needs NumPy only; the tensor callable additionally needs Torch.
    if name in ("CompiledProgram", "load_compiled"):
        from cim22 import model

        return getattr(model, name)
    if name == "compile_exported":
        from cim22.compiler import compile_exported

        return compile_exported
    raise AttributeError(name)
