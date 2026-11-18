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
        (["first", "first"], "must be unique"),
        (["missing"], "Unknown CIM layer"),
        (["block"], "leaf module"),
        ([], "at least one"),
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
    with pytest.raises(ValueError, match="same module"):
        validate_cim_layers(model, ["alias"])


def test_each_selected_layer_must_dispatch(tmp_path, native_toolchain) -> None:
    class OptionalLayer(torch.nn.Sequential):
        use_second = True

        def forward(self, inputs):
            output = self[0](inputs)
            return self[1](output) if self.use_second else output

    model = OptionalLayer(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
    sample = torch.ones(1, 4)
    offload = LayerOffload(
        model, ["0", "1"], (sample,), artifact_dir=tmp_path, runtime="simulator"
    )
    model.use_second = False
    with pytest.raises(
        RuntimeError, match="No CIM operations executed for layer: 1"
    ):
        offload(sample)
    assert offload.dispatch_count == 0


def test_failed_layer_warmup_restores_model(tmp_path, native_toolchain) -> None:
    from cim22.backend import session_for_artifact

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    ).eval()
    sample = torch.ones(1, 4)
    captured = []

    def fail_second_warmup(_):
        if model[1]._compiled_call_impl is not None and not captured:
            captured.append(session_for_artifact(tmp_path / "0"))
            raise RuntimeError("second warmup failed")

    with pytest.raises(RuntimeError, match="second warmup failed"):
        LayerOffload(
            model,
            ["0", "1"],
            (sample,),
            artifact_dir=tmp_path,
            runtime="simulator",
            reset=fail_second_warmup,
        )
    assert all(layer._compiled_call_impl is None for layer in model)
    with pytest.raises(RuntimeError, match="session is closed"):
        _ = captured[0].dispatch_count
    retry = LayerOffload(
        model, ["0", "1"], (sample,), artifact_dir=tmp_path, runtime="simulator"
    )
    assert retry.dispatch_count > 0


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
    with pytest.raises(ValueError, match="Unsupported CIM layer type"):
        validate_cim_layers(model, ["1"])


@pytest.mark.parametrize("selection", ["first", [None], [""]])
def test_invalid_name_types(selection):
    with pytest.raises(ValueError, match="non-empty strings"):
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


@pytest.mark.parametrize("names", [("0", "1"), ("0", "2")])
def test_usb_layers_attach_to_first_connection(monkeypatch, tmp_path, names):
    from cim22.runtime import MODEL_SUFFIX, UsbConfig

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 2)
    ).eval()
    usb = UsbConfig(1204, 241, 129, 1)
    sessions = {}
    captured = []

    class FakeSession:
        dispatch_count = 2
        _invocation_count = 0

    def compile_leaf(self, **kwargs):
        options = kwargs["options"]
        directory = options["artifact_dir"]
        captured.append(options)
        session = sessions[directory] = FakeSession()
        kwargs["backend"].keywords["_on_session"](session)
        self.register_forward_hook(
            lambda *_: setattr(
                session, "_invocation_count", session._invocation_count + 1
            )
        )
        (directory / ("graph" + MODEL_SUFFIX)).touch()

    monkeypatch.setattr(torch.nn.Module, "compile", compile_leaf)
    compiled = LayerOffload(
        model,
        names,
        (torch.ones(1, 4),),
        artifact_dir=tmp_path,
        runtime="usb",
        usb_config=usb,
    )
    assert captured[0]["usb_config"] is usb
    assert "usb_session" not in captured[0]
    assert captured[1]["usb_session"] is sessions[tmp_path / names[0]]
    assert "usb_config" not in captured[1]
    assert compiled.dispatch_count == 4
