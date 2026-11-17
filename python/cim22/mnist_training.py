"""Small, reproducible MNIST training flow used by the local Web UI."""

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

ModelPreset = Literal["small", "standard"]
Progress = Callable[[dict[str, object]], None]
Cancelled = Callable[[], bool]

HIDDEN_SIZES: dict[ModelPreset, int] = {"small": 64, "standard": 128}


@dataclass(frozen=True)
class MnistConfig:
    """Validated Web training settings."""

    data_dir: Path
    output_dir: Path
    model: ModelPreset = "standard"
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.001
    seed: int = 42

    def validate(self) -> None:
        if self.model not in HIDDEN_SIZES:
            raise ValueError("模型必须是 small 或 standard")
        if not 1 <= self.epochs <= 100:
            raise ValueError("训练轮数必须在 1 到 100 之间")
        if not 1 <= self.batch_size <= 4096:
            raise ValueError("batch size 必须在 1 到 4096 之间")
        if (
            not math.isfinite(self.learning_rate)
            or not 1e-6 <= self.learning_rate <= 1
        ):
            raise ValueError("学习率必须在 0.000001 到 1 之间")
        if self.seed < 0:
            raise ValueError("随机种子不能为负数")

    def json_value(self) -> dict[str, object]:
        value = asdict(self)
        value["data_dir"] = str(self.data_dir)
        value["output_dir"] = str(self.output_dir)
        value["hidden_size"] = HIDDEN_SIZES[self.model]
        return value


class MnistNetwork(nn.Module):
    """Fixed two-layer classifier used for the acceptance demonstration."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(self.flatten(inputs))


def build_model(model: ModelPreset) -> MnistNetwork:
    """Construct one of the supported fixed model presets."""
    if model not in HIDDEN_SIZES:
        raise ValueError("模型必须是 small 或 standard")
    return MnistNetwork(HIDDEN_SIZES[model])


def _save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_data(
    config: MnistConfig,
    progress: Progress,
) -> tuple[DataLoader[object], DataLoader[object]]:
    try:
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError(
            "Web 训练需要安装 torchvision，请重新安装 web extra"
        ) from error

    config.data_dir.mkdir(parents=True, exist_ok=True)
    progress({"stage": "downloading", "message": "正在检查并下载 MNIST 数据"})
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        str(config.data_dir), train=True, download=True, transform=transform
    )
    progress({"stage": "loading", "message": "正在加载 MNIST 测试集"})
    test_set = datasets.MNIST(
        str(config.data_dir), train=False, download=True, transform=transform
    )
    progress({"stage": "loading", "message": "MNIST 数据已就绪，正在加载"})
    generator = torch.Generator().manual_seed(config.seed)
    return (
        DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
        ),
        DataLoader(
            test_set, batch_size=config.batch_size, shuffle=False, num_workers=0
        ),
    )


def _evaluate(
    model: MnistNetwork,
    loader: DataLoader[object],
    cancelled: Cancelled,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for inputs, labels in loader:
            if cancelled():
                raise InterruptedError
            outputs = model(inputs)
            batch = labels.shape[0]
            total_loss += float(criterion(outputs, labels).item()) * batch
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += batch
    return total_loss / total, correct / total


def train_mnist(
    config: MnistConfig,
    progress: Progress,
    cancelled: Cancelled = lambda: False,
) -> dict[str, object]:
    """Download, train, evaluate and persist one MNIST run."""
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(config.output_dir / "config.json", config.json_value())
    torch.manual_seed(config.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    started = time.perf_counter()
    try:
        train_loader, test_loader = _load_data(config, progress)
        model = build_model(config.model)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        metrics_path = config.output_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")
        progress({"stage": "training", "message": "开始训练"})
        final: dict[str, object] = {}
        for epoch in range(1, config.epochs + 1):
            if cancelled():
                raise InterruptedError
            epoch_started = time.perf_counter()
            progress(
                {
                    "stage": "training",
                    "message": f"训练第 {epoch}/{config.epochs} 轮",
                }
            )
            model.train()
            loss_sum = 0.0
            samples = 0
            for inputs, labels in train_loader:
                if cancelled():
                    raise InterruptedError
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                batch = labels.shape[0]
                loss_sum += float(loss.item()) * batch
                samples += batch
            progress(
                {
                    "stage": "evaluating",
                    "message": f"测试评估第 {epoch}/{config.epochs} 轮",
                }
            )
            test_loss, accuracy = _evaluate(model, test_loader, cancelled)
            elapsed = time.perf_counter() - started
            row = {
                "epoch": epoch,
                "epochs": config.epochs,
                "train_loss": loss_sum / samples,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "elapsed_seconds": elapsed,
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            torch.save(
                {
                    "model": config.model,
                    "config": config.json_value(),
                    "state_dict": model.state_dict(),
                },
                config.output_dir / "model.pt.tmp",
            )
            (config.output_dir / "model.pt.tmp").replace(
                config.output_dir / "model.pt"
            )
            progress({"stage": "training", "message": row})
            final = row
        summary = {
            "status": "completed",
            "model": config.model,
            "config": config.json_value(),
            "hidden_size": HIDDEN_SIZES[config.model],
            "train_samples": len(train_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "total_seconds": time.perf_counter() - started,
            **final,
        }
        _save_json(config.output_dir / "summary.json", summary)
        progress({"stage": "completed", "message": summary})
        return summary
    except InterruptedError:
        progress(
            {
                "stage": "cancelled",
                "message": "训练已取消，已保留当前日志和产物",
            }
        )
        raise
