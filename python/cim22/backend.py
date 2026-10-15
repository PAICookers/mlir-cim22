"""torch.compile backend: freeze module tensors, compile the captured full graph.

Use fullgraph=True and dynamic=False to enforce capture without eager graph
breaks. This adapter uses Dynamo source metadata from the pinned PyTorch wheel;
it does not classify CIM operators. MLIR owns that decision.
"""

import copy
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

import torch
from torch._dynamo.source import NNModuleSource, UnspecializedParamBufferSource

from cim22.compiler import compile_exported
from cim22.model import CompiledProgram


def _module_tensor(node: torch.fx.Node) -> bool:
    argument = node.meta.get("grapharg")
    if argument is None:
        raise RuntimeError("Pinned Dynamo source metadata is missing")
    source = argument.source
    while source is not None:
        if isinstance(source, (NNModuleSource, UnspecializedParamBufferSource)):
            return True
        source = getattr(source, "base", None)
    return False


def backend(
    graph: torch.fx.GraphModule,
    example_inputs: Sequence[torch.Tensor],
    *,
    options: dict[str, object] | None = None,
) -> Callable[..., object]:
    """Build a lazy inference callable; compile on first real-valued invocation.

    Frozen parameters are cloned into a private FX graph. Shape and parameter
    guards reject changes instead of silently executing an old specialization.
    Original model classes and SpikingJelly reset methods are untouched.
    """
    options = options or {}
    unknown = options.keys() - {"artifact_dir", "debug_dir", "runtime"}
    if unknown:
        raise ValueError(f"Unknown CIM22 backend options: {sorted(unknown)}")
    directories: dict[str, Path] = {}
    for key, value in options.items():
        if key == "runtime":
            if value not in ("software", "usb"):
                raise ValueError("CIM22 runtime must be 'software' or 'usb'")
            continue
        if not isinstance(value, (str, Path)):
            raise TypeError(f"CIM22 {key} must be a path")
        directories[key] = Path(value)
    graph_name = f"graph-{uuid4().hex}"
    placeholders = [n for n in graph.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(example_inputs):
        raise ValueError("Dynamo input metadata differs from graph signature")
    frozen_indices = {
        i for i, n in enumerate(placeholders) if _module_tensor(n)
    }
    dynamic_indices = [
        i for i in range(len(placeholders)) if i not in frozen_indices
    ]
    compiled: CompiledProgram | None = None
    frozen: dict[int, tuple[torch.Tensor, int]] = {}
    lock = threading.Lock()

    def invoke(*args: torch.Tensor) -> object:
        nonlocal compiled
        if torch.is_grad_enabled():
            raise ValueError(
                "Use torch.no_grad() or torch.inference_mode() with CIM22"
            )
        if len(args) != len(placeholders) or not all(
            isinstance(arg, torch.Tensor) for arg in args
        ):
            raise TypeError("CIM22 backend requires static tensor arguments")
        with lock:
            if compiled is None:
                # Dynamo may hand out a lazy GraphModule. Materialize an
                # ordinary FX module before re-exporting its frozen signature.
                specialized = torch.fx.GraphModule(
                    copy.deepcopy(graph), copy.deepcopy(graph.graph)
                )
                # Dynamo spells Tensor.data as a C helper which its second
                # trace cannot capture. In inference, detach preserves the
                # same storage and values while remaining an ATen operation.
                for node in specialized.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch._C._autograd._get_data_attr
                    ):
                        node.target = torch.ops.aten.detach.default
                nodes = [
                    n for n in specialized.graph.nodes if n.op == "placeholder"
                ]
                for index in frozen_indices:
                    value = args[index]
                    if value.device.type != "cpu":
                        raise TypeError(
                            "Frozen CIM22 parameters must be on CPU"
                        )
                    name = f"_cim22_frozen_{index}"
                    specialized.register_buffer(name, value.detach().clone())
                    with specialized.graph.inserting_before(nodes[index]):
                        constant = specialized.graph.get_attr(name)
                    nodes[index].replace_all_uses_with(constant)
                    specialized.graph.erase_node(nodes[index])
                    frozen[index] = (value, value._version)
                specialized.graph.lint()
                specialized.recompile()
                specialized.eval()
                program = torch.export.export(
                    specialized,
                    tuple(args[i] for i in dynamic_indices),
                    strict=False,
                )
                artifact_dir = directories.get("artifact_dir")
                debug_dir = directories.get("debug_dir")
                compiled = compile_exported(
                    program,
                    artifact_path=(artifact_dir / f"{graph_name}.so")
                    if artifact_dir is not None
                    else None,
                    debug_dir=debug_dir / graph_name
                    if debug_dir is not None
                    else None,
                    runtime=options.get("runtime", "usb"),
                )
            for index, (value, version) in frozen.items():
                if args[index] is not value or value._version != version:
                    raise ValueError(
                        "Frozen CIM22 parameter/state changed; recompile"
                    )
            return compiled(*(args[i] for i in dynamic_indices))

    return invoke
