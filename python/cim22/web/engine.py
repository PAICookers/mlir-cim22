"""Fixed DVS adapter and optional leaf-layer CIM offload."""

import ctypes
import importlib.util
import sys
import time
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron, surrogate

from cim22.layers import (
    CimLayerInfo,
    LayerOffload,
    validate_cim_layers,
)
from cim22.layers import (
    discover_cim_layers as discover_model_layers,
)
from cim22.runtime import MODEL_SUFFIX, compile_tool, runtime_library

from .storage import Document, digest, read_json, save_json


def _load_model_module(model_dir: Path) -> object:
    spec = importlib.util.spec_from_file_location(
        "cim22_local_dvs_model",
        model_dir / "model_mobilenet_snn_deploy.py",
    )
    if spec is None or spec.loader is None:
        raise ValueError("无法读取固定 DVS 模型定义")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _new_model(module: object, dropout: float = 0.2) -> torch.nn.Module:
    model = (
        module.DVSGestureMobileNetSNN_Deploy(
            channels=16,
            expansion=2,
            spiking_neuron=neuron.LIFNode,
            dropout=dropout,
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )
        .cpu()
        .eval()
    )
    functional.set_step_mode(model, "m")
    return model


def discover_cim_layers(model_dir: Path) -> list[CimLayerInfo]:
    """Discover the fixed DVS model using the shared SDK layer API."""
    model = _new_model(_load_model_module(model_dir))
    try:
        return discover_model_layers(model, (torch.zeros(32, 1, 2, 128, 128),))
    finally:
        functional.reset_net(model)


def validate_cim_layer_names(
    model_dir: Path, selected: Sequence[str] | None
) -> tuple[str, ...]:
    """Validate names against a fresh fixed DVS model before task creation."""
    return validate_cim_layers(
        _new_model(_load_model_module(model_dir)), selected
    )


class DvsEngine:
    """Own eager and hybrid compiled models for reset-per-sample inference."""

    def __init__(self, config: Document, directory: Path) -> None:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        self.config = config
        self.directory = directory
        self.cim_layers = tuple(config.get("cim_layers") or ())
        identity = {
            "torch": torch.__version__,
            "torch_mlir": version("torch-mlir"),
            "spikingjelly": version("spikingjelly"),
            "runtime": digest(runtime_library()),
            "compile_tool": digest(compile_tool()),
            "adapter": digest(Path(__file__)),
            "python_backend": {
                name: digest(Path(__file__).parents[1] / name)
                for name in (
                    "backend.py",
                    "compiler.py",
                    "runtime.py",
                    "layers.py",
                )
            },
        }
        identity_path = directory / "environment.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("运行环境已变化，请重新导入为新任务")
        save_json(identity_path, identity)
        module = _load_model_module(Path(config["model_dir"]))
        checkpoint = torch.load(
            config["weights"], map_location="cpu", weights_only=True
        )
        for key, expected in {
            "channels": 16,
            "expansion": 2,
            "T": 32,
            "tau": 2.0,
            "neuron": "lif",
        }.items():
            if checkpoint.get(key, expected) != expected:
                raise ValueError(f"固定模型权重配置不符：{key}")
        state = {}
        for key, value in checkpoint["state_dict"].items():
            name = key.replace("classifier.fc1.", "classifier.head.2.")
            name = name.replace("classifier.fc2.", "classifier.head.5.")
            tensor = value.float()
            if not value.is_floating_point():
                tensor = (
                    tensor
                    * torch.as_tensor(
                        checkpoint["scales"].get(key, 1.0)
                    ).float()
                )
            state[name] = tensor

        def build() -> torch.nn.Module:
            model = _new_model(module, checkpoint.get("dropout", 0.2))
            model.load_state_dict(state, strict=True)
            return model

        self.original = build()
        self.cim = build()
        self.compiled: torch.nn.Module | LayerOffload | None = None
        self.layer_artifacts: dict[str, list[str]] = {}
        self.layer_compile_seconds: dict[str, float] = {}
        self.dispatch_count: int | None = None
        self.last_outputs: dict[str, torch.Tensor] = {}
        self._libc = ctypes.CDLL(None)
        validate_cim_layers(self.cim, config.get("cim_layers"))

    def load(self, index: int) -> torch.Tensor:
        """Return contiguous ordinary FP32 [32,1,2,128,128] CPU input."""
        entry = self.config["samples"][index]
        path = Path(self.config["dataset"]) / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"样本内容已变化：{entry['path']}")
        with np.load(path, allow_pickle=False) as archive:
            frames = archive["frames"].astype(np.float32)
        if frames.shape != (32, 2, 128, 128) or not np.isfinite(frames).all():
            raise ValueError(f"样本尺寸或数值无效：{entry['path']}")
        return torch.from_numpy(frames).unsqueeze(1).contiguous()

    def _artifact_paths(self) -> set[Path]:
        return set((self.directory / "artifacts").rglob(f"*{MODEL_SUFFIX}"))

    def _compile_selected(self, sample: torch.Tensor) -> None:
        offload = LayerOffload(
            self.cim,
            self.cim_layers,
            (sample,),
            artifact_dir=self.directory / "artifacts",
            runtime=self.config.get("runtime", "simulator"),
            reset=functional.reset_net,
        )
        self.layer_artifacts = offload.layer_artifacts
        self.layer_compile_seconds = offload.layer_compile_seconds
        self.dispatch_count = offload.dispatch_count
        self.compiled = offload

    def compile(self, phase: Callable[[str], None]) -> float:
        sample = self.load(0)
        artifact_dir = self.directory / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        with torch.inference_mode():
            functional.reset_net(self.cim)
            phase("compiling")
            started = time.perf_counter()
            if self.cim_layers:
                self._compile_selected(sample)
                functional.reset_net(self.cim)
                self.prediction(self.compiled(sample))
            else:
                self.compiled = torch.compile(
                    self.cim,
                    backend="cim22",
                    fullgraph=True,
                    dynamic=False,
                    options={
                        "runtime": self.config.get("runtime", "simulator"),
                        "artifact_dir": artifact_dir,
                    },
                )
                output = self.compiled(sample)
                self.prediction(output)
            elapsed = time.perf_counter() - started
            functional.reset_net(self.cim)
        self.trim()
        self.artifacts = self._artifact_paths()
        if not self.artifacts:
            raise RuntimeError(f"编译未生成模型 {MODEL_SUFFIX}")
        return elapsed

    @staticmethod
    def prediction(output: torch.Tensor) -> int:
        if output.shape != (32, 1, 11) or not torch.isfinite(output).all():
            raise ValueError("模型输出尺寸错误或包含非有限值")
        return int(output.sum(0).argmax(1).item())

    def infer(self, side: str, sample: torch.Tensor) -> tuple[int, float]:
        model = self.original if side == "original" else self.compiled
        if model is None:
            raise RuntimeError("请先编译模型")
        with torch.inference_mode():
            functional.reset_net(
                self.original if side == "original" else self.cim
            )
            started = time.perf_counter()
            output = model(sample)
            elapsed = time.perf_counter() - started
            if not hasattr(self, "last_outputs"):
                self.last_outputs = {}
            self.last_outputs[side] = output.detach().float().cpu()
            if side == "cim":
                if self._artifact_paths() != self.artifacts:
                    raise RuntimeError(
                        "样本触发额外编译，未计入结果；请重新导入"
                    )
                if getattr(self, "cim_layers", ()):
                    self.dispatch_count = self.compiled.dispatch_count
            prediction = self.prediction(output)
        self.trim()
        return prediction, elapsed

    def max_abs_error(self) -> float | None:
        """Return the latest reference/CIM output error when both exist."""
        if set(self.last_outputs) != {"original", "cim"}:
            return None
        return float(
            (self.last_outputs["original"] - self.last_outputs["cim"])
            .abs()
            .max()
            .item()
        )

    def metadata(self) -> Document:
        return {
            "cim_layers": list(self.cim_layers) or None,
            "execution_mode": "hybrid" if self.cim_layers else "full_model",
            "layer_artifacts": self.layer_artifacts,
            "layer_compile_seconds": self.layer_compile_seconds,
            "dispatch_count": self.dispatch_count,
        }

    def trim(self) -> None:
        # Return freed glibc arenas between calls, outside measured inference.
        trim = getattr(self._libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
