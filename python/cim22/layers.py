"""Model-independent PyTorch leaf selection and CIM mixed inference."""

import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote

import torch

from .backend import backend
from .runtime import MODEL_SUFFIX, Session, UsbConfig


class CimLayerInfo(TypedDict):
    """A selectable leaf module in a PyTorch model."""

    name: str
    type: str
    weight_shape: list[int]
    input_shape: list[int]
    output_shape: list[int]


def discover_cim_layers(
    model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...] | None = None,
) -> list[CimLayerInfo]:
    """List Conv2d/Linear leaves; optional inputs run once to collect shapes.

    Shape discovery can update model state. The caller owns state reset;
    temporary hooks are removed even when the model raises.
    """
    result: list[CimLayerInfo] = []
    handles = []

    def record_shape(info: CimLayerInfo, _module, args, output) -> None:
        info["input_shape"] = list(args[0].shape)
        info["output_shape"] = list(output.shape)

    for name, module in model.named_modules(remove_duplicate=False):
        if not name or list(module.children()):
            continue
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        info: CimLayerInfo = {
            "name": name,
            "type": type(module).__name__,
            "weight_shape": list(module.weight.shape),
            "input_shape": [],
            "output_shape": [],
        }
        result.append(info)
        handles.append(
            module.register_forward_hook(partial(record_shape, info))
        )
    try:
        with torch.inference_mode():
            if example_inputs is not None:
                model(*example_inputs)
    finally:
        for handle in handles:
            handle.remove()
    return result


def validate_cim_layers(
    model: torch.nn.Module, selected: Sequence[str] | None
) -> tuple[str, ...]:
    """Validate exact, distinct selectable leaf-module names."""
    if selected is None:
        return ()
    if isinstance(selected, str) or any(
        not isinstance(name, str) or not name for name in selected
    ):
        raise ValueError(
            "CIM layer names must be a sequence of non-empty strings"
        )
    names = tuple(selected)
    if not names:
        raise ValueError("Select at least one CIM layer")
    if len(set(names)) != len(names):
        raise ValueError("CIM layer names must be unique")
    modules = dict(model.named_modules(remove_duplicate=False))
    aliases: dict[int, list[str]] = {}
    for name, module in modules.items():
        aliases.setdefault(id(module), []).append(name)
    validated: list[str] = []
    for name in names:
        module = modules.get(name)
        if module is None:
            raise ValueError(f"Unknown CIM layer: {name}")
        if list(module.children()):
            raise ValueError(f"CIM layer must be a leaf module: {name}")
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            raise ValueError(f"Unsupported CIM layer type: {name}")
        if len(aliases[id(module)]) > 1:
            raise ValueError(
                f"CIM layer has multiple names for the same module: {name}"
            )
        validated.append(name)
    return tuple(validated)


class LayerOffload:
    """Compile selected leaves in place and run the outer model on CPU.

    The model must be an evaluated CPU model with fixed weights and static
    input signatures supported by the CIM backend. Construction warms each
    selected layer under inference_mode; pass reset(model) for stateful
    networks. Callers reset between independent inference sequences.
    Use a separate model instance for an uncompiled reference.
    USB layers share one connection, receive loop and clock controller on
    Linux and Windows; each layer retains its own compiled model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        selected: Sequence[str],
        example_inputs: tuple[torch.Tensor, ...],
        *,
        artifact_dir: str | Path,
        runtime: str = "usb",
        usb_config: UsbConfig | None = None,
        reset: Callable[[torch.nn.Module], None] | None = None,
    ) -> None:
        self.cim_layers = validate_cim_layers(model, selected)
        if not self.cim_layers:
            raise ValueError("Select at least one CIM layer")
        if runtime not in ("simulator", "usb"):
            raise ValueError(
                "Layer offload requires runtime='simulator' or 'usb'"
            )
        if any(module.training for module in model.modules()):
            raise ValueError("Layer offload requires a model in eval mode")
        if any(
            value.device.type != "cpu"
            for value in (
                *model.parameters(),
                *model.buffers(),
                *example_inputs,
            )
        ):
            raise ValueError(
                "Layer offload requires a CPU model and CPU inputs"
            )
        self.model = model
        self.directory = Path(artifact_dir)
        self.layer_artifacts: dict[str, list[str]] = {}
        self.layer_compile_seconds: dict[str, float] = {}
        self._sessions: dict[str, list[Session]] = {
            name: [] for name in self.cim_layers
        }
        compiled_calls = {}
        shared_session = None
        try:
            with torch.inference_mode():
                try:
                    for name, sessions in self._sessions.items():
                        directory = self.directory / quote(name, safe="")
                        directory.mkdir(parents=True, exist_ok=True)
                        module = model.get_submodule(name)
                        compiled_calls[module] = module._compiled_call_impl
                        started = time.perf_counter()
                        module.compile(
                            backend=partial(
                                backend, _on_session=sessions.append
                            ),
                            fullgraph=True,
                            dynamic=False,
                            options={
                                "runtime": runtime,
                                **(
                                    {"usb_session": shared_session}
                                    if shared_session is not None
                                    else {"usb_config": usb_config}
                                ),
                                "artifact_dir": directory,
                                "debug_dir": directory / "ir",
                            },
                        )
                        if reset is not None:
                            reset(model)
                        previous = self._invocations()
                        model(*example_inputs)
                        if not sessions:
                            raise RuntimeError(
                                f"Missing runtime session for CIM layer: {name}"
                            )
                        if runtime == "usb" and shared_session is None:
                            shared_session = sessions[0]
                        self.layer_compile_seconds[name] = (
                            time.perf_counter() - started
                        )
                        artifacts = sorted(directory.glob(f"*{MODEL_SUFFIX}"))
                        if not artifacts:
                            raise RuntimeError(
                                f"No compiled artifact for CIM layer: {name}"
                            )
                        self.layer_artifacts[name] = [str(p) for p in artifacts]
                    self._dispatch_count = self._count_dispatches(previous)
                finally:
                    if reset is not None:
                        reset(model)
        except BaseException as error:
            for module, compiled_call in compiled_calls.items():
                module._compiled_call_impl = compiled_call
            for sessions in self._sessions.values():
                for session in sessions:
                    try:
                        session.close()
                    except Exception as cleanup_error:
                        error.add_note(
                            f"Failed to close CIM session: {cleanup_error}"
                        )
            raise
        self.artifacts = set(self.directory.rglob(f"*{MODEL_SUFFIX}"))

    def _invocations(self) -> dict[Session, int]:
        return {
            session: session._invocation_count
            for sessions in self._sessions.values()
            for session in sessions
        }

    def _count_dispatches(self, previous: dict[Session, int]) -> int:
        total = 0
        for name, sessions in self._sessions.items():
            count = sum(
                session.dispatch_count
                for session in sessions
                if session._invocation_count > previous.get(session, 0)
            )
            if not count:
                raise RuntimeError(
                    f"No CIM operations executed for layer: {name}"
                )
            total += count
        return total

    @property
    def dispatch_count(self) -> int:
        """Sum dispatches from each selected session's latest call."""
        return self._dispatch_count

    def __call__(self, *inputs: torch.Tensor) -> Any:
        """Preserve model outputs and reject extra compilation or fallback."""
        self._dispatch_count = 0
        previous = self._invocations()
        with torch.inference_mode():
            output = self.model(*inputs)
        if self._invocations().keys() != previous.keys():
            raise RuntimeError(
                "Input triggered additional compilation; recompile the model"
            )
        self._dispatch_count = self._count_dispatches(previous)
        return output
