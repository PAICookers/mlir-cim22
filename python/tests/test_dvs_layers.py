"""Validation rules for DVS leaf-layer selection."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def test_discovery_reports_actual_multi_step_shapes(monkeypatch, tmp_path):
    from cim22.web import engine

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Sequential(torch.nn.Linear(2, 3))

        def forward(self, value):
            return self.head(value.mean((-1, -2)))

    model = TinyModel()
    monkeypatch.setattr(engine, "_load_model_module", lambda _: None)
    monkeypatch.setattr(engine, "_new_model", lambda _: model)
    layers = engine.discover_cim_layers(tmp_path)
    assert layers == [
        {
            "name": "head.0",
            "type": "Linear",
            "weight_shape": [3, 2],
            "input_shape": [32, 1, 2],
            "output_shape": [32, 1, 3],
        }
    ]
    assert not model.head[0]._forward_hooks


@pytest.mark.parametrize("selection", [None, []])
def test_engine_preserves_full_model_and_rejects_explicit_empty(
    monkeypatch, tmp_path, selection
):
    from cim22.web import engine

    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(engine, "digest", lambda _: "fixture")
    monkeypatch.setattr(engine, "runtime_library", lambda: tmp_path)
    monkeypatch.setattr(engine, "compile_tool", lambda: tmp_path)
    monkeypatch.setattr(engine, "_load_model_module", lambda _: None)
    monkeypatch.setattr(engine, "_new_model", lambda *args: torch.nn.Identity())
    monkeypatch.setattr(torch, "load", lambda *args, **kw: {"state_dict": {}})
    config = {
        "model_dir": str(tmp_path),
        "weights": "fixture",
        "cim_layers": selection,
    }
    if selection is None:
        instance = engine.DvsEngine(config, tmp_path)
        assert instance.cim_layers == ()
        assert instance.metadata()["execution_mode"] == "full_model"
    else:
        with pytest.raises(ValueError, match="至少需要一个"):
            engine.DvsEngine(config, tmp_path)


@pytest.mark.parametrize(
    "names",
    [("0",), ("0", "1"), ("0", "2")],
    ids=["single", "adjacent", "separated"],
)
def test_multistep_leaf_offload_dispatch_and_numeric_agreement(
    tmp_path, native_toolchain, names
):
    import copy
    import ctypes

    from spikingjelly.activation_based import functional, layer

    from cim22.backend import session_for_artifact
    from cim22.runtime import MODEL_SUFFIX
    from cim22.web.engine import DvsEngine

    torch.manual_seed(42)
    instance = DvsEngine.__new__(DvsEngine)
    instance.cim = torch.nn.Sequential(
        layer.Linear(4, 4), layer.Linear(4, 4), layer.Linear(4, 11)
    ).eval()
    functional.set_step_mode(instance.cim, "m")
    instance.original = copy.deepcopy(instance.cim)
    instance.directory = tmp_path
    instance.config = {"runtime": "simulator"}
    instance.cim_layers = names
    instance.layer_artifacts = {}
    instance.layer_compile_seconds = {}
    instance.last_outputs = {}
    instance._libc = ctypes.CDLL(None)
    sample = torch.ones(32, 1, 4)
    instance.load = lambda _: sample
    instance.compile(lambda _: None)
    assert set(instance.layer_artifacts) == set(names)
    assert instance.dispatch_count > 0

    modules = dict(instance.cim.named_modules())
    for name in ("0", "1", "2"):
        assert (modules[name]._compiled_call_impl is not None) == (
            name in names
        )
    artifacts = instance.artifacts.copy()
    for value in (sample, sample * 0.5, sample):
        instance.infer("original", value)
        instance.infer("cim", value)
        torch.testing.assert_close(
            instance.last_outputs["cim"],
            instance.last_outputs["original"],
            rtol=0.05,
            atol=0.05,
        )
        for name in names:
            layer_dir = tmp_path / "artifacts" / name
            assert len(list(layer_dir.glob("*" + MODEL_SUFFIX))) == 1
            assert session_for_artifact(layer_dir).dispatch_count > 0
        assert instance._artifact_paths() == artifacts


def test_compile_without_selection_uses_original_whole_model_path(
    monkeypatch, tmp_path
):
    from cim22.web import engine

    instance = engine.DvsEngine.__new__(engine.DvsEngine)
    instance.directory = tmp_path
    instance.cim_layers = ()
    instance.config = {"runtime": "simulator"}
    instance.cim = torch.nn.Identity()
    sample = torch.zeros(32, 1, 11)
    instance.load = lambda _: sample
    instance.trim = lambda: None
    captured = {}

    def compile_model(model, **options):
        captured.update(options)
        path = options["options"]["artifact_dir"] / (
            "graph" + engine.MODEL_SUFFIX
        )
        path.touch()
        return model

    monkeypatch.setattr(torch, "compile", compile_model)
    instance.compile(lambda _: None)
    assert captured["backend"] == "cim22"
    assert captured["fullgraph"] is True
    assert instance.compiled is instance.cim
    assert instance.artifacts
