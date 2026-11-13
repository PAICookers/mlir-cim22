"""torch.compile backend: functionalize inference, freeze read-only parameters.

Use fullgraph=True and dynamic=False to enforce capture without eager graph
breaks. This adapter uses Dynamo source metadata from the pinned PyTorch wheel;
it does not classify CIM operators. MLIR owns that decision.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4
from weakref import WeakValueDictionary

import torch
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.guards import GuardBuilder, install_guard
from torch._dynamo.source import NNModuleSource, UnspecializedParamBufferSource
from torch._dynamo.utils import defake
from torch._functorch._aot_autograd.descriptors import PlainAOTInput
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._guards import TracingContext, tracing
from torch.utils._python_dispatch import _disable_current_modes

from .compiler import (
    BackendOptions,
    _check_torch_mlir_compatibility,
    _compile,
    _validate_options,
)
from .runtime import MODEL_SUFFIX, Session, _initialize_session

type TensorCallable = Callable[..., Sequence[torch.Tensor]]

_SESSIONS_BY_ARTIFACT: WeakValueDictionary[Path, Session] = (
    WeakValueDictionary()
)


def session_for_artifact(artifact_dir: Path) -> Session | None:
    """Return the most recently captured native session in an artifact dir."""
    matches = [
        session
        for path, session in _SESSIONS_BY_ARTIFACT.items()
        if path.parent == artifact_dir.resolve()
    ]
    return matches[-1] if matches else None


def _is_module_input(node: torch.fx.Node) -> bool:
    source = node.meta["grapharg"].source
    while source is not None:
        if isinstance(source, (NNModuleSource, UnspecializedParamBufferSource)):
            return True
        source = getattr(source, "base", None)
    return False


def _aot_decompositions() -> dict[torch._ops.OpOverload, Callable[..., object]]:
    """Keep static select-scatter updates local to the overwritten slice."""
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table

    # torch-mlir caches its table across backends.
    decompositions = dict(get_decomposition_table())
    select_scatter = torch.ops.aten.select_scatter.default
    original = decompositions.get(select_scatter)
    if original is None:
        return decompositions

    def select_scatter_as_slice(
        tensor: torch.Tensor,
        source: torch.Tensor,
        dim: int | torch.SymInt,
        index: int | torch.SymInt,
    ) -> torch.Tensor:
        if (
            isinstance(dim, int)
            and isinstance(index, int)
            and -tensor.ndim <= dim < tensor.ndim
            and tensor.dtype == source.dtype
            and all(
                isinstance(size, int) for size in (*tensor.shape, *source.shape)
            )
        ):
            axis = dim % tensor.ndim
            extent = tensor.shape[axis]
            if (
                -extent <= index < extent
                and source.shape
                == tensor.shape[:axis] + tensor.shape[axis + 1 :]
            ):
                start = index % extent
                # torch-mlir lowers slice_scatter to tensor.insert_slice.
                return torch.ops.aten.slice_scatter.default(
                    tensor, source.unsqueeze(axis), axis, start, start + 1, 1
                )
        return original(tensor, source, dim, index)

    decompositions[select_scatter] = select_scatter_as_slice
    return decompositions


def backend(
    graph: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: BackendOptions | None = None,
) -> TensorCallable:
    """Compile a static CPU inference graph for torch.compile.

    Reference executes the captured graph with original CPU PyTorch operators.
    Native modes use AOTAutograd to preserve mutations and input aliases.
    Frozen read-only parameters must remain fixed for this specialization.

    Args:
        graph: FX graph supplied by Dynamo.
        example_inputs: Tensors that define the static specialization.
        options: BackendOptions for runtime selection, compensation, device
            configuration and optional output directories.

    Returns:
        A callable that Dynamo caches and uses for inference.

    Raises:
        ValueError: Gradients are enabled, options are unsupported, or
            reference mode received non-CPU tensors.
    """
    options = options or {}
    _validate_options(options, allow_reference=True)
    _check_torch_mlir_compatibility()
    runtime = options.get("runtime", "usb")
    usb_config = options.get("usb_config")
    bf16_compensation = options.get("bf16_compensation", "sign-split")
    if torch.is_grad_enabled():
        raise ValueError(
            "Use torch.no_grad() or torch.inference_mode() with CIM22"
        )
    artifact_dir = options.get("artifact_dir")
    debug_dir = options.get("debug_dir")
    if runtime == "reference":
        if any(
            value.device.type != "cpu"
            for value in example_inputs
            if isinstance(value, torch.Tensor)
        ):
            raise ValueError("CIM22 reference requires CPU tensors")
        if artifact_dir is not None or debug_dir is not None:
            raise ValueError("CIM22 reference does not emit native artifacts")
        return graph.forward

    parameter_sources = {
        i: node.meta["grapharg"].source
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
        frozen_parameters: set[int] = set()
        # The independent torch-mlir export needs its own context.
        with _disable_current_modes(), tracing(None):
            for i, node in enumerate(gm.graph.find_nodes(op="placeholder")):
                source = node.meta.get("desc")
                mutation = metadata.input_info[i]
                # Freeze read-only weights, preserving mutations and aliases.
                if (
                    not isinstance(source, PlainAOTInput)
                    or source.idx not in parameter_sources
                    or mutation.mutates_data
                    or mutation.mutates_metadata
                    or i in aliased
                ):
                    input_indices.append(i)
                    continue
                value = example_inputs[source.idx]
                frozen_parameters.add(source.idx)
                name = f"_cim22_frozen_{source.idx}"
                gm.register_buffer(name, value.detach().clone())
                with gm.graph.inserting_before(node):
                    constant = gm.graph.get_attr(name)
                node.replace_all_uses_with(constant)
                gm.graph.erase_node(node)
            gm.recompile()
            artifact_path = (
                Path(artifact_dir) / f"{graph_name}{MODEL_SUFFIX}"
                if artifact_dir is not None
                else None
            )
            session = _compile(
                gm,
                *(defake(fake_inputs[i]) for i in input_indices),
                artifact_path=artifact_path,
                debug_dir=Path(debug_dir) / graph_name
                if debug_dir is not None
                else None,
                runtime=runtime,
                bf16_compensation=bf16_compensation,
            )
        # Frozen values cannot be reused for a different model's parameters.
        for index in frozen_parameters:
            install_guard(
                parameter_sources[index].make_guard(GuardBuilder.ID_MATCH)
            )
        session = _initialize_session(session, usb_config)
        if artifact_path is not None:
            _SESSIONS_BY_ARTIFACT[artifact_path.resolve()] = session

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
        fw_compiler=_compiler, decompositions=_aot_decompositions()
    )(graph, example_inputs)
