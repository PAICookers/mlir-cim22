"""Partition static AOT forward graphs by the original PyTorch module paths."""

import logging
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch._dynamo import config
from torch._dynamo.source import AttrSource, LocalSource
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch.fx.passes.split_module import split_module

from .compiler import BackendOptions, _compile
from .runtime import MODEL_SUFFIX, Session, _initialize_session

logger = logging.getLogger(__name__)


@dataclass
class CompilationReport:
    """Artifacts and logical CIM calls for one compiled FX specialization."""

    artifacts: tuple[str, ...] = ()
    compile_seconds: float = 0.0
    layer_artifacts: dict[str, list[str]] = field(default_factory=dict)
    layer_compile_seconds: dict[str, float] = field(default_factory=dict)
    layer_dispatch_counts: dict[str, int] = field(default_factory=dict)
    dispatch_count: int = 0


def _stage_directory(
    destination: Path | None, resources: ExitStack
) -> Path | None:
    """Stage on the destination filesystem so publication can rename files."""
    if destination is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        resources.enter_context(
            TemporaryDirectory(prefix=".cim22-", dir=destination.parent)
        )
    )


def _publish_directory(
    pending: Path | None, final: Path | None, rollback: ExitStack
) -> None:
    """Publish a staged directory and register cleanup on failure."""
    if final is not None:
        if final.exists(follow_symlinks=False):
            raise FileExistsError(f"Compilation output already exists: {final}")
        pending.rename(final)
        rollback.callback(shutil.rmtree, final)


def _owners(
    graph: torch.fx.GraphModule, paths: dict[str, str]
) -> dict[torch.fx.Node, str]:
    owners = {}
    for node in graph.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        for path, _ in node.meta.get("nn_module_stack", {}).values():
            if path in paths:
                owners[node] = paths[path]
                break
    return owners


def _selection(
    graph: torch.fx.GraphModule, options: BackendOptions
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Validate options against the installed Dynamo capture contract."""
    strictness = options.get("compile_strictness", "strict")
    if strictness not in ("strict", "allow_host"):
        raise ValueError("compile_strictness must be 'strict' or 'allow_host'")
    if "cim_layers" not in options:
        if "compile_strictness" in options:
            raise ValueError("compile_strictness requires cim_layers")
        return (), {}
    raw = options["cim_layers"]
    if isinstance(raw, str):
        raw = [raw]
    if (
        not isinstance(raw, (list, tuple))
        or not raw
        or any(not isinstance(name, str) or not name for name in raw)
    ):
        raise ValueError(
            "cim_layers requires one or more non-empty module paths"
        )
    selected = tuple(raw)
    if len(set(selected)) != len(selected):
        raise ValueError("cim_layers contains duplicate module paths")
    if options.get("runtime") == "reference":
        raise ValueError("cim_layers requires 'simulator' or 'usb'")

    # The registered backend receives one captured graph, not the full model.
    capture = InstructionTranslator.current_tx()
    if not capture.one_graph:
        raise ValueError("cim_layers requires torch.compile(fullgraph=True)")
    if config.automatic_dynamic_shapes or not config.assume_static_by_default:
        raise ValueError("cim_layers requires torch.compile(dynamic=False)")
    receiver = (
        capture.f_code.co_varnames[0] if capture.f_code.co_argcount else "fn"
    )
    if not isinstance(capture.f_locals.get(receiver), torch.nn.Module):
        receiver = "fn"
    model = capture.f_locals.get(receiver)
    if not isinstance(model, torch.nn.Module) or model.training:
        raise ValueError("cim_layers requires an nn.Module in eval mode")
    modules = dict(model.named_modules(remove_duplicate=False))
    counts = Counter(map(id, modules.values()))
    for path in selected:
        if path not in modules:
            raise ValueError(f"Unknown CIM module path: {path}")
        if any(path.startswith(other + ".") for other in selected):
            raise ValueError(f"Overlapping CIM module paths at: {path}")
        if any(
            counts[id(module)] > 1
            for name, module in modules.items()
            if name == path or name.startswith(path + ".")
        ):
            raise ValueError(f"Ambiguous shared CIM module path: {path}")
    root = LocalSource(receiver)
    sources = {"": root}
    scopes: dict[str, str] = {}
    paths = {}
    for name in modules:
        if not name:
            continue
        parent, _, member = name.rpartition(".")
        sources[name] = AttrSource(sources[parent], member)
        owner = name if name in selected else scopes.get(parent)
        if owner is not None:
            scopes[name] = owner
            # Dynamo also normalizes _modules lookups to dotted names.
            paths[f"{root.name}.{name}"] = owner
            paths[sources[name].name] = owner
    captured = set(_owners(graph, paths).values())
    for path in selected:
        if path not in captured:
            raise ValueError(f"CIM module was not captured: {path}")
    return selected, paths


def _partition(
    graph: torch.fx.GraphModule, module_paths: dict[str, str]
) -> tuple[torch.fx.GraphModule, dict[str, str]]:
    owners = _owners(graph, module_paths)
    groups: dict[torch.fx.Node, int] = {}
    paths: dict[str, str] = {}
    last_owner = None
    group = -1
    # Native inputs are packed, so storage-relative addressing and layout
    # queries must observe the original Torch tensors.
    storage_ops = {
        "aten::as_strided",
        "aten::as_strided_copy",
        "aten::as_strided_scatter",
        "aten::storage_offset",
        "aten::stride",
        "aten::sym_storage_offset",
        "aten::sym_stride",
    }
    # A native result owns its storage. Preserve externally visible views
    # and their view chains in Torch so aliasing survives the partition.
    for node in reversed(list(graph.graph.nodes)):
        owner = owners.get(node)
        schema = getattr(node.target, "_schema", None)
        if (
            owner is not None
            and schema
            and (
                schema.is_mutable
                or schema.name in storage_ops
                or (
                    any(ret.alias_info for ret in schema.returns)
                    and any(
                        arg.op != "get_attr" for arg in node.all_input_nodes
                    )
                    and any(owners.get(user) != owner for user in node.users)
                )
            )
        ):
            owners.pop(node)
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        owner = owners.get(node)
        if not groups or owner != last_owner:
            group += 1
        if owner is not None:
            paths[f"submod_{group}"] = owner
        groups[node] = group
        last_owner = owner
    return (
        split_module(
            graph,
            graph,
            groups.__getitem__,
            keep_original_order=True,
            tuple_return=True,
        ),
        paths,
    )


def _tensor_boundary(node: torch.fx.Node) -> torch.Tensor:
    value = node.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or not all(type(size) is int for size in value.shape)
    ):
        raise ValueError(
            f"CIM boundary requires a static CPU tensor at {node.name}"
        )
    return value


def _compile_partition(
    module: torch.fx.GraphModule,
    partition: torch.fx.GraphModule,
    call: torch.fx.Node,
    *,
    artifact_path: Path | None,
    debug_dir: Path | None,
    options: BackendOptions,
) -> tuple[Session, list[int], list[tuple[int, ...]]]:
    inputs = []
    input_indices = []
    placeholders = list(partition.graph.find_nodes(op="placeholder"))
    for index, (node, source) in enumerate(
        zip(placeholders, call.args, strict=True)
    ):
        if not isinstance(source, torch.fx.Node):
            raise ValueError(
                "CIM partition input has no captured tensor signature"
            )
        if source.op == "get_attr":
            name = f"_cim22_constant_{index}"
            partition.register_buffer(name, getattr(module, source.target))
            with partition.graph.inserting_before(node):
                constant = partition.graph.get_attr(name)
            node.replace_all_uses_with(constant)
            partition.graph.erase_node(node)
        else:
            value = _tensor_boundary(node)
            inputs.append(torch.zeros(value.shape, dtype=value.dtype))
            input_indices.append(index)
    (output,) = partition.graph.find_nodes(op="output")
    strides = [
        tuple(_tensor_boundary(node).stride()) for node in output.args[0]
    ]
    partition.recompile()
    session = _compile(
        partition,
        *inputs,
        artifact_path=artifact_path,
        debug_dir=debug_dir,
        runtime=options.get("runtime", "usb"),
        bf16_compensation=options.get("bf16_compensation", "sign-split"),
    )
    return session, input_indices, strides


class _NativePartition(torch.nn.Module):
    def __init__(
        self,
        session: Session,
        indices: list[int],
        strides: list[tuple[int, ...]],
    ) -> None:
        super().__init__()
        self.session = session
        self.indices = indices
        self.strides = strides

    def forward(self, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        arrays = self.session(
            *(values[i].contiguous().numpy(force=True) for i in self.indices)
        )
        result = []
        for array, stride in zip(arrays, self.strides, strict=True):
            value = torch.from_numpy(array)
            if value.stride() != stride:
                value = torch.empty_strided(
                    value.shape, stride, dtype=value.dtype
                ).copy_(value)
            result.append(value)
        return tuple(result)


def compile_selected(
    graph: torch.fx.GraphModule,
    selected: tuple[str, ...],
    module_paths: dict[str, str],
    options: BackendOptions,
    graph_name: str,
    report: CompilationReport,
) -> Callable[..., Sequence[torch.Tensor]]:
    """Compile complete partitions before exposing a callable to AOTAutograd."""
    module, paths = _partition(graph, module_paths)
    mapped = set()
    partition_counts = Counter(paths.values())
    report.layer_artifacts = {path: [] for path in selected}
    report.layer_compile_seconds = dict.fromkeys(selected, 0.0)
    sessions: list[tuple[str, Session]] = []

    with ExitStack() as staging, ExitStack() as rollback:
        artifact_dir = options.get("artifact_dir")
        debug_dir = options.get("debug_dir")
        final_artifacts = (
            Path(artifact_dir).resolve() / graph_name
            if artifact_dir is not None
            else None
        )
        final_debug = (
            Path(debug_dir).resolve() / graph_name
            if debug_dir is not None
            else None
        )
        artifacts = _stage_directory(final_artifacts, staging)
        if artifacts is not None and final_artifacts == final_debug:
            debug, final_debug = artifacts / "ir", None
        else:
            debug = _stage_directory(final_debug, staging)
        for call in module.graph.nodes:
            path = paths.get(call.target) if call.op == "call_module" else None
            if path is None:
                continue
            name = str(call.target)
            partition = module.get_submodule(name)
            preserve_host_graph = (
                options.get("compile_strictness", "strict") == "allow_host"
                or partition_counts[path] > 1
            )
            native_graph = (
                torch.fx.GraphModule(partition, deepcopy(partition.graph))
                if preserve_host_graph
                else partition
            )
            artifact = (
                artifacts / f"{name}{MODEL_SUFFIX}" if artifacts else None
            )
            started = perf_counter()
            try:
                session, indices, strides = _compile_partition(
                    module,
                    native_graph,
                    call,
                    artifact_path=artifact,
                    debug_dir=debug / name if debug else None,
                    options=options,
                )
            except (RuntimeError, ValueError, TypeError, OSError) as error:
                raise RuntimeError(
                    f"CIM module {path} failed to compile: {error}"
                ) from error
            rollback.callback(session.close)
            report.layer_compile_seconds[path] += perf_counter() - started
            if session.num_transactions:
                mapped.add(path)
                module.set_submodule(
                    name, _NativePartition(session, indices, strides)
                )
                sessions.append((path, session))
                if artifact is not None:
                    report.layer_artifacts[path].append(
                        str(final_artifacts / artifact.name)
                    )
            else:
                session.close()
                if artifact is not None:
                    artifact.unlink()
        for path in selected:
            if path in mapped:
                continue
            if options.get("compile_strictness", "strict") == "strict":
                raise ValueError(f"CIM module has zero transactions: {path}")
            logger.warning(
                "CIM module %s will execute on Host (zero transactions)", path
            )
        usb_session = options.get("usb_session")
        if options.get("runtime", "usb") == "usb":
            usb_config = options.get("usb_config")
            if sessions and usb_session is None and usb_config is None:
                raise ValueError(
                    "CIM USB execution requires usb_config or usb_session"
                )
            for path, session in sessions:
                try:
                    _initialize_session(
                        session,
                        usb_config if usb_session is None else None,
                        usb_session=usb_session,
                    )
                except (RuntimeError, ValueError, TypeError, OSError) as error:
                    raise RuntimeError(
                        f"CIM module {path} failed to initialize USB: {error}"
                    ) from error
                if usb_session is None:
                    usb_session = session
        _publish_directory(artifacts, final_artifacts, rollback)
        _publish_directory(debug, final_debug, rollback)
        rollback.pop_all()
    report.artifacts = tuple(
        artifact
        for paths in report.layer_artifacts.values()
        for artifact in paths
    )

    def run(*values: torch.Tensor) -> Sequence[torch.Tensor]:
        result = module(*values)
        counts = dict.fromkeys(selected, 0)
        for path, session in sessions:
            counts[path] += session.dispatch_count
        report.layer_dispatch_counts = counts
        report.dispatch_count = sum(counts.values())
        return result

    return run
