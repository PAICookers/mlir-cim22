"""Candidate enumeration and selection for the DVS application."""

from collections.abc import Sequence
from functools import partial
from typing import TypedDict

import torch


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
