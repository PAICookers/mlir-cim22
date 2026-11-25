"""Selective compilation through the registered PyTorch backend."""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path

import pytest
import torch
from torch import nn
from torch._dynamo.exc import BackendCompilerFailed, Unsupported
from torch.nn import functional as F

from cim22 import backend
from cim22.runtime import MODEL_SUFFIX, Session, UsbConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("native_toolchain"),
]


@pytest.fixture(autouse=True)
def reset_compiler() -> Iterator[None]:
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def make_linear() -> nn.Linear:
    layer = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        layer.weight.fill_(0.25)
    return layer


class Projection(nn.Module):
    """Call a functional contraction directly in this module's forward."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4) * 0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight)


class MixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(Projection(), nn.ReLU())
        self.middle = make_linear()
        self.head = make_linear()
        self.unused = make_linear()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.middle(self.encoder(value)))


def compile_selected(
    model: nn.Module,
    path: Path,
    selected: str | list[str],
    *,
    strictness: str = "strict",
    fullgraph: bool = True,
    runtime: str = "simulator",
    dynamic: bool | None = False,
    usb_config: UsbConfig | None = None,
) -> nn.Module:
    return torch.compile(
        model.eval(),
        backend="cim22",
        fullgraph=fullgraph,
        dynamic=dynamic,
        options={
            "runtime": runtime,
            "cim_layers": selected,
            "compile_strictness": strictness,
            "artifact_dir": path,
            "debug_dir": path / "debug",
            **({"usb_config": usb_config} if usb_config is not None else {}),
        },
    )


@pytest.mark.parametrize(
    ("selected", "names"),
    [
        ("encoder.0", {"encoder.0"}),
        (["encoder.0", "head"], {"encoder.0", "head"}),
        ("encoder", {"encoder"}),
    ],
    ids=["functional-leaf", "nonadjacent", "parent-module"],
)
def test_reports_only_selected_regions(
    tmp_path: Path, selected: str | list[str], names: set[str]
) -> None:
    model = MixedModel().eval()
    first = torch.tensor([[-1.0, 2.0, 3.0, 4.0]])
    second = first * 2
    compiled = compile_selected(model, tmp_path, selected)
    assert backend.compilation_report(tmp_path) is None
    with torch.inference_mode():
        for value in (first, second):
            torch.testing.assert_close(compiled(value), model(value))
            report = backend.compilation_report(tmp_path)
            assert report is not None
            assert set(report.layer_artifacts) == names
            assert set(report.layer_compile_seconds) == names
            assert set(report.layer_dispatch_counts) == names
            assert all(
                count > 0 for count in report.layer_dispatch_counts.values()
            )
            assert report.dispatch_count == sum(
                report.layer_dispatch_counts.values()
            )
    paths = {
        Path(path)
        for group in report.layer_artifacts.values()
        for path in group
    }
    assert paths == set(tmp_path.rglob(f"*{MODEL_SUFFIX}"))


def test_strict_single_partition_does_not_copy_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cim22._module_offload as offload

    def unexpected_copy(graph: torch.fx.Graph) -> torch.fx.Graph:
        pytest.fail("strict single-partition compilation copied its graph")

    monkeypatch.setattr(offload, "deepcopy", unexpected_copy)
    model = nn.Sequential(make_linear()).eval()
    compiled = compile_selected(model, tmp_path, "0")
    value = torch.ones(1, 4)
    with torch.inference_mode():
        torch.testing.assert_close(compiled(value), model(value))


def test_strict_repeated_module_partitions_preserve_graph_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cim22._module_offload as offload

    copy_graph = offload.deepcopy
    copied_graphs = 0

    def record_copy(graph: torch.fx.Graph) -> torch.fx.Graph:
        nonlocal copied_graphs
        copied_graphs += 1
        return copy_graph(graph)

    class RepeatedBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential(make_linear(), make_linear())

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.block(value).relu() + self.block(value)

    monkeypatch.setattr(offload, "deepcopy", record_copy)
    model = RepeatedBlock().eval()
    value = torch.ones(1, 4)
    compiled = compile_selected(model, tmp_path, "block")
    with torch.inference_mode():
        torch.testing.assert_close(compiled(value), model(value))
    assert copied_graphs == 2


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ([], "at least one|empty"),
        (["head", "head"], "unique|duplicate"),
        (["encoder", "encoder.0"], "overlap"),
        (["head", "typo"], "typo"),
        (
            "unused",
            "unused.*(captur|execut)|(not captured|not executed).*unused",
        ),
    ],
)
def test_invalid_or_uncaptured_path_rejects_compilation(
    tmp_path: Path, selected: str | list[str], message: str
) -> None:
    compiled = compile_selected(MixedModel(), tmp_path, selected)
    with (
        torch.inference_mode(),
        pytest.raises(BackendCompilerFailed, match=f"(?i){message}"),
    ):
        compiled(torch.ones(1, 4))
    assert backend.compilation_report(tmp_path) is None
    assert not list(tmp_path.rglob(f"*{MODEL_SUFFIX}"))


@pytest.mark.parametrize("selected", ["block", "block.projection layer"])
def test_module_paths_follow_dynamo_attribute_spelling(
    tmp_path: Path, selected: str
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential()
            self.block.add_module("projection layer", make_linear())

        def forward(this, value: torch.Tensor) -> torch.Tensor:
            return getattr(this.block, "projection layer")(value)

    model = Model().eval()
    compiled = compile_selected(model, tmp_path, selected)
    value = torch.ones(1, 4)
    with torch.inference_mode():
        torch.testing.assert_close(compiled(value), model(value))
    report = backend.compilation_report(tmp_path)
    assert report is not None
    assert report.layer_dispatch_counts[selected] > 0


def test_unselected_weights_stay_live_on_host(tmp_path: Path) -> None:
    model = MixedModel().eval()
    compiled = compile_selected(model, tmp_path, "head")
    value = torch.ones(1, 4)
    with torch.inference_mode():
        before = compiled(value)
        first = backend.compilation_report(tmp_path)
        model.middle.weight.mul_(2)
        after = compiled(value)
        torch.testing.assert_close(after, before * 2)
        torch.testing.assert_close(after, model(value))
    second = backend.compilation_report(tmp_path)
    assert first is not None and second is not None
    assert first.artifacts == second.artifacts
    assert second.layer_dispatch_counts["head"] > 0
    second.layer_artifacts["head"].clear()
    assert (
        backend.compilation_report(tmp_path).layer_artifacts
        == first.layer_artifacts
    )


def test_rejects_shared_module_path(tmp_path: Path) -> None:
    model = MixedModel()
    model.alias = model.head
    compiled = compile_selected(model, tmp_path, "head")
    with (
        torch.inference_mode(),
        pytest.raises(
            BackendCompilerFailed, match="(?i)alias|ambiguous|same module"
        ),
    ):
        compiled(torch.ones(1, 4))
    assert backend.compilation_report(tmp_path) is None


def test_selecting_layers_requires_fullgraph(tmp_path: Path) -> None:
    compiled = compile_selected(MixedModel(), tmp_path, "head", fullgraph=False)
    with (
        torch.inference_mode(),
        pytest.raises((Unsupported, BackendCompilerFailed), match="fullgraph"),
    ):
        compiled(torch.ones(1, 4))
    assert backend.compilation_report(tmp_path) is None


@pytest.mark.parametrize("dynamic", [None, True])
def test_selecting_layers_requires_explicit_static_shapes(
    tmp_path: Path, dynamic: bool | None
) -> None:
    compiled = compile_selected(MixedModel(), tmp_path, "head", dynamic=dynamic)
    with (
        torch.inference_mode(),
        pytest.raises(BackendCompilerFailed, match="dynamic=False"),
    ):
        compiled(torch.ones(2, 4))
    assert backend.compilation_report(tmp_path) is None
    assert not list(tmp_path.rglob(f"*{MODEL_SUFFIX}"))


def test_graph_break_prevents_selected_execution(tmp_path: Path) -> None:
    class ModelWithBreak(MixedModel):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            projected = self.encoder(value)
            torch._dynamo.graph_break()
            return self.head(projected)

    compiled = compile_selected(ModelWithBreak(), tmp_path, ["encoder", "head"])
    with (
        torch.inference_mode(),
        pytest.raises(Unsupported, match="graph_break"),
    ):
        compiled(torch.ones(1, 4))
    assert backend.compilation_report(tmp_path) is None
    assert not list(tmp_path.rglob(f"*{MODEL_SUFFIX}"))


def test_strict_rejects_host_only_region_and_cleans_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cim22._module_offload as offload

    original_compile = offload._compile
    sessions: list[Session] = []

    def capture_session(*args: object, **kwargs: object) -> Session:
        session = original_compile(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(offload, "_compile", capture_session)
    original_file = tmp_path / "keep.txt"
    original_file.write_text("user data", encoding="utf-8")
    compiled = compile_selected(
        MixedModel(), tmp_path, ["encoder.0", "encoder.1"]
    )
    with (
        torch.inference_mode(),
        pytest.raises(BackendCompilerFailed, match=r"encoder\.1"),
    ):
        compiled(torch.ones(1, 4))
    assert len(sessions) == 2
    assert all(session._invocation_count == 0 for session in sessions)
    for session in sessions:
        with pytest.raises(RuntimeError, match="session is closed"):
            _ = session.dispatch_count
    assert backend.compilation_report(tmp_path) is None
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == [
        original_file
    ]


def test_later_partition_failure_closes_earlier_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cim22._module_offload as offload

    original_compile = offload._compile
    sessions = []

    def fail_second_partition(*args: object, **kwargs: object):
        if sessions:
            raise RuntimeError("injected second partition failure")
        session = original_compile(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(offload, "_compile", fail_second_partition)
    compiled = compile_selected(MixedModel(), tmp_path, ["encoder.0", "head"])
    with (
        torch.inference_mode(),
        pytest.raises(BackendCompilerFailed, match="second partition failure"),
    ):
        compiled(torch.ones(1, 4))
    assert len(sessions) == 1
    with pytest.raises(RuntimeError, match="session is closed"):
        _ = sessions[0].dispatch_count
    assert backend.compilation_report(tmp_path) is None
    assert not list(tmp_path.rglob(f"*{MODEL_SUFFIX}"))


def test_allow_host_preserves_relu_and_accelerated_projection(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = MixedModel().eval()
    value = torch.tensor([[-4.0, 1.0, -2.0, 1.0]])
    compiled = compile_selected(
        model, tmp_path, ["encoder.0", "encoder.1"], strictness="allow_host"
    )
    with torch.inference_mode():
        torch.testing.assert_close(compiled(value), model(value))
    report = backend.compilation_report(tmp_path)
    assert report is not None
    assert report.layer_dispatch_counts["encoder.0"] > 0
    assert report.layer_dispatch_counts.get("encoder.1", 0) == 0
    assert report.dispatch_count == report.layer_dispatch_counts["encoder.0"]
    assert "encoder.1" in caplog.text


@pytest.mark.parametrize("selected", [["0"], ["0", "1"], ["0", "2"]])
def test_selected_convolutions_preserve_unselected_torch_layers(
    tmp_path: Path, selected: list[str]
) -> None:
    model = nn.Sequential(
        nn.Conv2d(2, 2, 1),
        nn.Conv2d(2, 2, 1),
        nn.Conv2d(2, 3, 1),
    ).eval()
    with torch.no_grad():
        for layer in model:
            layer.weight.fill_(0.5)
            layer.bias.fill_(0.5)
    compiled = compile_selected(model, tmp_path, selected)
    value = torch.ones(1, 2, 2, 2)
    with torch.inference_mode():
        for sample in (value, value * 2):
            torch.testing.assert_close(compiled(sample), model(sample))
            report = backend.compilation_report(tmp_path)
            assert report is not None
            assert set(report.layer_dispatch_counts) == set(selected)
            assert all(
                count > 0 for count in report.layer_dispatch_counts.values()
            )
            assert report.dispatch_count == sum(
                report.layer_dispatch_counts.values()
            )


@pytest.mark.parametrize("fail_later", [False, True])
def test_usb_partitions_share_first_session_and_roll_back_setup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_later: bool,
) -> None:
    import cim22._module_offload as offload

    original_compile = offload._compile
    sessions: list[Session] = []
    initialization: list[tuple[Session, UsbConfig | None, Session | None]] = []
    usb_config = UsbConfig(0x1234, 0x5678, 0x81, 0x01)

    def compile_for_simulator(*args: object, **kwargs: object) -> Session:
        assert kwargs["runtime"] == "usb"
        kwargs["runtime"] = "simulator"
        session = original_compile(*args, **kwargs)
        sessions.append(session)
        return session

    def initialize(
        session: Session,
        config: UsbConfig | None = None,
        *,
        usb_session: Session | None = None,
    ) -> Session:
        assert len(sessions) == 2
        assert all(native._invocation_count == 0 for native in sessions)
        initialization.append((session, config, usb_session))
        if fail_later and len(initialization) == 2:
            raise RuntimeError("injected USB initialization failure")
        return session

    monkeypatch.setattr(offload, "_compile", compile_for_simulator)
    monkeypatch.setattr(offload, "_initialize_session", initialize)
    original_file = tmp_path / "keep.txt"
    original_file.write_text("user data", encoding="utf-8")
    debug_file = tmp_path / "debug" / "keep.txt"
    debug_file.parent.mkdir()
    debug_file.write_text("debug user data", encoding="utf-8")
    model = MixedModel().eval()
    compiled = compile_selected(
        model,
        tmp_path,
        ["encoder.0", "head"],
        runtime="usb",
        usb_config=usb_config,
    )
    with torch.inference_mode():
        if fail_later:
            with pytest.raises(
                BackendCompilerFailed, match="USB initialization failure"
            ):
                compiled(torch.ones(1, 4))
        else:
            for value in (torch.ones(1, 4), torch.ones(1, 4) * 2):
                torch.testing.assert_close(compiled(value), model(value))
    assert initialization == [
        (sessions[0], usb_config, None),
        (sessions[1], None, sessions[0]),
    ]
    if fail_later:
        assert all(session._invocation_count == 0 for session in sessions)
        for session in sessions:
            with pytest.raises(RuntimeError, match="session is closed"):
                _ = session.dispatch_count
        assert backend.compilation_report(tmp_path) is None
        assert set(tmp_path.rglob("*")) == {
            original_file,
            debug_file.parent,
            debug_file,
        }
    else:
        report = backend.compilation_report(tmp_path)
        assert report is not None
        assert set(report.layer_dispatch_counts) == {"encoder.0", "head"}
        assert all(count > 0 for count in report.layer_dispatch_counts.values())
        assert len(report.artifacts) == 2
        assert all(session._invocation_count == 2 for session in sessions)
        for session in sessions:
            session.close()


def test_repeated_layer_dispatches_accumulate_per_inference(
    tmp_path: Path,
) -> None:
    class RepeatedLayer(nn.Module):
        def __init__(self, repeat: bool) -> None:
            super().__init__()
            self.repeat = repeat
            self.head = make_linear()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            result = self.head(value)
            if self.repeat:
                return self.head(result.relu())
            return result

    value = torch.ones(1, 4)
    single_dir = tmp_path / "single"
    repeat_dir = tmp_path / "repeat"
    single = compile_selected(RepeatedLayer(False), single_dir, "head")
    eager = RepeatedLayer(True).eval()
    repeat = compile_selected(eager, repeat_dir, "head")
    with torch.inference_mode():
        single(value)
        report = backend.compilation_report(single_dir)
        assert report is not None
        single_count = report.layer_dispatch_counts["head"]
        assert single_count > 0
        for _ in range(2):
            torch.testing.assert_close(repeat(value), eager(value))
            report = backend.compilation_report(repeat_dir)
            assert report is not None
            assert report.layer_dispatch_counts == {"head": 2 * single_count}
            assert report.dispatch_count == 2 * single_count


def test_selected_as_strided_preserves_noncontiguous_input_storage(
    tmp_path: Path,
) -> None:
    class StridedProjection(Projection):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            view = torch.as_strided(value, (1, 4), (4, 1), 0)
            return F.linear(view, self.weight)

    model = nn.Sequential(StridedProjection()).eval()
    compiled = compile_selected(model, tmp_path, "0")
    value = torch.arange(16, dtype=torch.float32).reshape(4, 4).t()
    with torch.inference_mode():
        torch.testing.assert_close(compiled(value), model(value))
    report = backend.compilation_report(tmp_path)
    assert report is not None
    assert report.layer_dispatch_counts["0"] > 0


def test_selected_linear_accepts_input_with_negative_bit(
    tmp_path: Path,
) -> None:
    model = nn.Sequential(nn.Linear(1, 1, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.fill_(1)
    compiled = compile_selected(model, tmp_path, "0")
    value = torch.tensor([[1 + 2j]], dtype=torch.complex64).conj().imag
    assert value.is_neg()
    with torch.inference_mode():
        torch.testing.assert_close(compiled(value), model(value))
    report = backend.compilation_report(tmp_path)
    assert report is not None
    assert report.layer_dispatch_counts["0"] > 0


def test_host_writeback_and_output_aliases_cross_selected_region(
    tmp_path: Path, inference_context: AbstractContextManager[None]
) -> None:
    class UpdateState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = make_linear()

        def forward(
            self, value: torch.Tensor, state: torch.Tensor
        ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
            activation = self.head(value)
            state.add_(activation)
            return {"outputs": (activation, state[:, 1:]), "state": state}

    model = UpdateState().eval()
    compiled = compile_selected(model, tmp_path, "head")
    value, expected_state, state = (torch.ones(1, 4) for _ in range(3))
    with inference_context:
        expected = model(value, expected_state)
        actual = compiled(value, state)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(state, expected_state)
        assert actual["state"] is state
        _, alias = actual["outputs"]
        alias.fill_(0)
        torch.testing.assert_close(state, torch.tensor([[2.0, 0.0, 0.0, 0.0]]))
    report = backend.compilation_report(tmp_path)
    assert report is not None
    assert report.layer_dispatch_counts["head"] > 0
