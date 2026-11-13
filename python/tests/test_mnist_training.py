"""Unit tests for the fixed MNIST model and configuration contract."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cim22.mnist_training import MnistConfig, build_model


def test_model_presets_have_fixed_shapes() -> None:
    for name, hidden in (("small", 64), ("standard", 128)):
        model = build_model(name)
        output = model(torch.zeros(3, 1, 28, 28))
        assert output.shape == (3, 10)
        assert model.layers[0].out_features == hidden


def test_configuration_rejects_invalid_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="训练轮数"):
        MnistConfig(tmp_path, tmp_path, epochs=0).validate()
    with pytest.raises(ValueError, match="batch size"):
        MnistConfig(tmp_path, tmp_path, batch_size=0).validate()
    with pytest.raises(ValueError, match="学习率"):
        MnistConfig(tmp_path, tmp_path, learning_rate=0).validate()


def test_training_artifacts_and_metrics(tmp_path: Path, monkeypatch) -> None:
    import json

    import cim22.mnist_training as training

    dataset = torch.utils.data.TensorDataset(
        torch.zeros(8, 1, 28, 28), torch.zeros(8, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    monkeypatch.setattr(training, "_load_data", lambda *args: (loader, loader))
    events = []
    summary = training.train_mnist(
        MnistConfig(tmp_path / "data", tmp_path, epochs=2), events.append
    )
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    model = build_model(checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    loss, accuracy = training._evaluate(model, loader, lambda: False)
    assert accuracy == summary["test_accuracy"]
    assert loss == pytest.approx(summary["test_loss"])
    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 2
    assert all(row["epoch_seconds"] > 0 for row in metrics)
    assert summary["config"]["epochs"] == 2
    assert "evaluating" in [event["stage"] for event in events]
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
