"""Model-independent PyTorch leaf selection and CIM mixed inference."""

import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import quote

import torch

from .backend import session_for_artifact
from .runtime import MODEL_SUFFIX, UsbConfig


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
        raise ValueError("CIM 层名必须是非空字符串列表")
    names = tuple(selected)
    if not names:
        raise ValueError("指定 CIM 层时至少需要一个层名")
    if len(set(names)) != len(names):
        raise ValueError("CIM 层名不能重复")
    modules = dict(model.named_modules(remove_duplicate=False))
    aliases: dict[int, list[str]] = {}
    for name, module in modules.items():
        aliases.setdefault(id(module), []).append(name)
    validated: list[str] = []
    for name in names:
        module = modules.get(name)
        if module is None:
            raise ValueError(f"未知 CIM 层：{name}")
        if list(module.children()):
            raise ValueError(f"CIM 层必须是叶子模块：{name}")
        if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            raise ValueError(f"CIM 层类型不支持：{name}")
        if len(aliases[id(module)]) > 1:
            raise ValueError(f"CIM 层名称指向同一模块实例：{name}")
        validated.append(name)
    return tuple(validated)


class LayerOffload:
    """Compile selected leaves in place and run the outer model on CPU.

    The model must be an evaluated CPU model with fixed weights and static
    input signatures supported by the CIM backend. Construction warms each
    selected layer under inference_mode; pass reset(model) for stateful
    networks. Callers reset between independent inference sequences.
    Use a separate model instance for an uncompiled reference.
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
            raise ValueError("指定 CIM 层时至少需要一个层名")
        if runtime not in ("simulator", "usb"):
            raise ValueError("分层执行仅支持 simulator 或 usb")
        if any(module.training for module in model.modules()):
            raise ValueError("分层执行需要 eval 模型")
        if any(
            value.device.type != "cpu"
            for value in (
                *model.parameters(),
                *model.buffers(),
                *example_inputs,
            )
        ):
            raise ValueError("分层执行需要 CPU 模型和输入")
        self.model = model
        self.directory = Path(artifact_dir)
        self.layer_artifacts: dict[str, list[str]] = {}
        self.layer_compile_seconds: dict[str, float] = {}
        self._directories = {
            name: self.directory / quote(name, safe="")
            for name in self.cim_layers
        }
        with torch.inference_mode():
            try:
                for name, directory in self._directories.items():
                    directory.mkdir(parents=True, exist_ok=True)
                    module = model.get_submodule(name)
                    started = time.perf_counter()
                    module.compile(
                        backend="cim22",
                        fullgraph=True,
                        dynamic=False,
                        options={
                            "runtime": runtime,
                            "usb_config": usb_config,
                            "artifact_dir": directory,
                            "debug_dir": directory / "ir",
                        },
                    )
                    if reset is not None:
                        reset(model)
                    model(*example_inputs)
                    self.layer_compile_seconds[name] = (
                        time.perf_counter() - started
                    )
                    artifacts = sorted(directory.glob(f"*{MODEL_SUFFIX}"))
                    if not artifacts:
                        raise RuntimeError(f"选定 CIM 层未生成模型产物：{name}")
                    self.layer_artifacts[name] = [str(p) for p in artifacts]
                _ = self.dispatch_count
            finally:
                if reset is not None:
                    reset(model)
        self.artifacts = self._artifact_paths()

    def _artifact_paths(self) -> set[Path]:
        return set(self.directory.rglob(f"*{MODEL_SUFFIX}"))

    @property
    def dispatch_count(self) -> int:
        """Sum actual CIM dispatches from the latest selected-layer calls."""
        total = 0
        for name, directory in self._directories.items():
            session = session_for_artifact(directory)
            if session is None:
                raise RuntimeError(f"选定 CIM 层 Session 丢失：{name}")
            if session.dispatch_count <= 0:
                raise RuntimeError(f"选定 CIM 层未产生 CIM dispatch：{name}")
            total += session.dispatch_count
        return total

    def __call__(self, *inputs: torch.Tensor) -> Any:
        """Preserve model outputs and reject extra compilation or fallback."""
        with torch.inference_mode():
            output = self.model(*inputs)
        if self._artifact_paths() != self.artifacts:
            raise RuntimeError("样本触发额外编译，未计入结果；请重新编译")
        _ = self.dispatch_count
        return output
