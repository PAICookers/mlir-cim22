"""Model-independent layer offload using ordinary PyTorch networks."""

import copy

import pytest
import torch

from cim22.layers import LayerOffload, discover_cim_layers, validate_cim_layers


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


class SharedLeafModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.second = torch.nn.Linear(4, 2)
        self.block = torch.nn.Sequential(torch.nn.Linear(2, 2))


def test_single_and_multiple_distinct_leaf_layers() -> None:
    model = SharedLeafModel()
    assert validate_cim_layers(model, ["first"]) == ("first",)
    assert validate_cim_layers(model, ["first", "second"]) == (
        "first",
        "second",
    )


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        (["first", "first"], "不能重复"),
        (["missing"], "未知"),
        (["block"], "叶子"),
        ([], "至少需要一个"),
    ],
)
def test_invalid_layer_selection(selected: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_cim_layers(SharedLeafModel(), selected)


def test_none_preserves_full_model_mode() -> None:
    assert validate_cim_layers(SharedLeafModel(), None) == ()


def test_alias_module_names_are_rejected() -> None:
    model = SharedLeafModel()
    model.alias = model.first
    with pytest.raises(ValueError, match="同一模块"):
        validate_cim_layers(model, ["alias"])


def test_each_selected_layer_must_dispatch(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from cim22 import layers

    instance = layers.LayerOffload.__new__(layers.LayerOffload)
    instance._directories = {
        name: tmp_path / name for name in ("first", "second")
    }
    monkeypatch.setattr(
        layers,
        "session_for_artifact",
        lambda path: SimpleNamespace(
            dispatch_count=3 if path.name == "first" else 0
        ),
    )
    with pytest.raises(RuntimeError, match="second"):
        _ = instance.dispatch_count


def test_session_registry_does_not_own_sessions(tmp_path) -> None:
    import gc
    import weakref

    from cim22 import backend

    class FakeSession:
        pass

    session = FakeSession()
    ref = weakref.ref(session)
    backend._SESSIONS_BY_ARTIFACT[tmp_path / "graph.so"] = session
    assert backend.session_for_artifact(tmp_path) is session
    del session
    gc.collect()
    assert ref() is None
    assert backend.session_for_artifact(tmp_path) is None


@pytest.mark.parametrize("names", [("0",), ("0", "1"), ("0", "2")])
@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_other_networks_dispatch_and_agree(
    tmp_path, native_toolchain, names, kind
):
    torch.manual_seed(42)
    if kind == "linear":
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 4),
            torch.nn.Linear(4, 2),
        ).eval()
        sample = torch.ones(1, 4)
    else:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 2, 1),
            torch.nn.Conv2d(2, 2, 1),
            torch.nn.Conv2d(2, 3, 1),
        ).eval()
        sample = torch.ones(1, 2, 2, 2)
    original = copy.deepcopy(model)
    info = discover_cim_layers(model, (sample,))
    assert [item["name"] for item in info] == ["0", "1", "2"]
    assert info[0]["input_shape"] == list(sample.shape)
    assert not model[0]._forward_hooks
    compiled = LayerOffload(
        model, names, (sample,), artifact_dir=tmp_path, runtime="simulator"
    )
    assert set(compiled.layer_artifacts) == set(names)
    artifacts = compiled.artifacts.copy()
    for value in (sample, sample * 0.5, sample):
        with torch.inference_mode():
            expected = original(value)
        torch.testing.assert_close(
            compiled(value), expected, atol=0.05, rtol=0.05
        )
        assert compiled.dispatch_count > 0
        assert compiled.artifacts == artifacts
    for name, module in model.named_children():
        assert (module._compiled_call_impl is not None) == (name in names)


def test_discovery_without_execution_and_unsupported_type():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU())
    assert [item["name"] for item in discover_cim_layers(model)] == ["0"]
    with pytest.raises(ValueError, match="类型不支持"):
        validate_cim_layers(model, ["1"])


@pytest.mark.parametrize("selection", ["first", [None], [""]])
def test_invalid_name_types(selection):
    with pytest.raises(ValueError, match="非空字符串"):
        validate_cim_layers(SharedLeafModel(), selection)


@pytest.mark.parametrize(
    ("model", "options", "message"),
    [
        (torch.nn.Sequential(torch.nn.Linear(2, 2)), {}, "eval"),
        (
            torch.nn.Sequential(torch.nn.Linear(2, 2)).eval(),
            {"runtime": "reference"},
            "simulator",
        ),
        (
            torch.nn.Sequential(torch.nn.Linear(2, 2, device="meta")).eval(),
            {},
            "CPU",
        ),
    ],
)
def test_compile_preflight(model, options, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        LayerOffload(
            model, ["0"], (torch.ones(1, 2),), artifact_dir=tmp_path, **options
        )
    assert model[0]._compiled_call_impl is None


def test_discovery_removes_hooks_on_model_error():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2)).eval()
    with pytest.raises(RuntimeError):
        discover_cim_layers(model, (torch.ones(1, 3),))
    assert not model[0]._forward_hooks
