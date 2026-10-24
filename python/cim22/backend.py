"""torch.compile backend: functionalize inference, freeze read-only parameters.

Use fullgraph=True and dynamic=False to enforce capture without eager graph
breaks. This adapter uses Dynamo source metadata from the pinned PyTorch wheel;
it does not classify CIM operators. MLIR owns that decision.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.source import NNModuleSource, UnspecializedParamBufferSource
from torch._dynamo.utils import defake
from torch._functorch._aot_autograd.descriptors import PlainAOTInput
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._guards import TracingContext, tracing
from torch.utils._python_dispatch import _disable_current_modes
from torch_mlir.extras.fx_decomp_util import get_decomposition_table

from .compiler import RuntimeBackend, _compile
from .runtime import MODEL_SUFFIX, UsbConfig

type TensorCallable = Callable[..., Sequence[torch.Tensor]]


class BackendOptions(TypedDict, total=False):
    """Options accepted by torch.compile(..., backend="cim22")."""

    runtime: RuntimeBackend
    usb_config: UsbConfig
    artifact_dir: str | Path
    debug_dir: str | Path


def _is_module_input(node: torch.fx.Node) -> bool:
    source = node.meta["grapharg"].source
    while source is not None:
        if isinstance(source, (NNModuleSource, UnspecializedParamBufferSource)):
            return True
        source = getattr(source, "base", None)
    return False


def backend(
    graph: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: BackendOptions | None = None,
) -> TensorCallable:
    """Compile sequential inference; Dynamo caches the returned AOT callable.

    AOTAutograd preserves mutations and input aliases. Read-only parameters are
    cloned into constants and must remain fixed for this specialization.
    """
    options = options or {}
    unknown = options.keys() - {
        "artifact_dir",
        "debug_dir",
        "runtime",
        "usb_config",
    }
    if unknown:
        raise ValueError(f"Unknown CIM22 backend options: {sorted(unknown)}")
    runtime = options.get("runtime", "usb")
    if runtime not in ("software", "usb"):
        raise ValueError("CIM22 runtime must be 'software' or 'usb'")
    usb_config = options.get("usb_config")
    if usb_config is not None and runtime != "usb":
        raise ValueError("CIM22 usb_config requires runtime='usb'")
    if torch.is_grad_enabled():
        raise ValueError(
            "Use torch.no_grad() or torch.inference_mode() with CIM22"
        )
    artifact_dir = options.get("artifact_dir")
    debug_dir = options.get("debug_dir")
    parameter_indices = {
        i
        for i, node in enumerate(graph.graph.find_nodes(op="placeholder"))
        if isinstance(example_inputs[i], torch.nn.Parameter)
        and _is_module_input(node)
    }
    # Failed device setup may retain artifacts; a retry uses a fresh name.
    graph_name = f"graph-{uuid4().hex}"

    def _compiler(
        gm: torch.fx.GraphModule, fake_inputs: Sequence[torch.Tensor]
    ) -> Callable[[list[torch.Tensor]], tuple[torch.Tensor, ...]]:
        metadata = TracingContext.get().fw_metadata
        aliased = {output.base_idx for output in metadata.output_info}
        input_indices: list[int] = []
        # The independent torch-mlir export needs its own context.
        with _disable_current_modes(), tracing(None):
            for i, node in enumerate(gm.graph.find_nodes(op="placeholder")):
                source = node.meta.get("desc")
                mutation = metadata.input_info[i]
                # Freeze read-only weights, preserving mutations and aliases.
                if (
                    not isinstance(source, PlainAOTInput)
                    or source.idx not in parameter_indices
                    or mutation.mutates_data
                    or mutation.mutates_metadata
                    or i in aliased
                ):
                    input_indices.append(i)
                    continue
                value = example_inputs[source.idx]
                name = f"_cim22_frozen_{source.idx}"
                gm.register_buffer(name, value.detach().clone())
                with gm.graph.inserting_before(node):
                    constant = gm.graph.get_attr(name)
                node.replace_all_uses_with(constant)
                gm.graph.erase_node(node)
            gm.recompile()
            session = _compile(
                gm,
                *(defake(fake_inputs[i]) for i in input_indices),
                artifact_path=(
                    Path(artifact_dir) / f"{graph_name}{MODEL_SUFFIX}"
                )
                if artifact_dir is not None
                else None,
                debug_dir=Path(debug_dir) / graph_name
                if debug_dir is not None
                else None,
                runtime=runtime,
            )
        try:
            if usb_config is not None:
                session.configure_usb(usb_config)
        except BaseException:
            session.close()
            raise

        def run(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return tuple(
                torch.from_numpy(array)
                for array in session(
                    *(values[i].detach().numpy() for i in input_indices)
                )
            )

        return make_boxed_func(run)

    # Decompose during AOT's existing trace instead of tracing the graph again.
    return aot_autograd(
        fw_compiler=_compiler, decompositions=get_decomposition_table()
    )(graph, example_inputs)
