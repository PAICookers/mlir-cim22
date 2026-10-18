"""Fixed DVS adapter; only the spawned compute process imports PyTorch."""

import ctypes
import importlib.util
import sys
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from spikingjelly.activation_based import functional, neuron, surrogate

from cim22.runtime import compiler_library, runtime_library

from .storage import Document, digest, read_json, save_json


class DvsEngine:
    """Own eager and compiled models for sequential reset-per-sample calls."""

    def __init__(self, config: Document, directory: Path) -> None:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        self.config = config
        self.directory = directory
        # Saved rows are only comparable under the same software/runtime.
        identity = {
            "torch": torch.__version__,
            "torch_mlir": version("torch-mlir"),
            "spikingjelly": version("spikingjelly"),
            "runtime": digest(runtime_library()),
            "compiler": digest(compiler_library(runtime_library())),
            "adapter": digest(Path(__file__)),
            "python_backend": {
                name: digest(Path(__file__).parents[1] / name)
                for name in ("backend.py", "compiler.py", "runtime.py")
            },
        }
        identity_path = directory / "environment.json"
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("运行环境已变化，请重新导入为新任务")
        save_json(identity_path, identity)
        spec = importlib.util.spec_from_file_location(
            "cim22_local_dvs_model",
            Path(config["model_dir"]) / "model_mobilenet_snn_deploy.py",
        )
        if spec is None or spec.loader is None:
            raise ValueError("无法读取固定 DVS 模型定义")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
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
            model = (
                module.DVSGestureMobileNetSNN_Deploy(
                    channels=16,
                    expansion=2,
                    spiking_neuron=neuron.LIFNode,
                    dropout=checkpoint.get("dropout", 0.2),
                    tau=2.0,
                    detach_reset=True,
                    surrogate_function=surrogate.ATan(),
                )
                .cpu()
                .eval()
            )
            functional.set_step_mode(model, "m")
            model.load_state_dict(state, strict=True)
            return model

        # Parameters are ordinary tensors, created outside inference_mode.
        self.original = build()
        self.cim = build()
        self.compiled: torch.nn.Module | None = None
        self._libc = ctypes.CDLL(None)

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

    def compile(self, phase: Callable[[str], None]) -> float:
        sample = self.load(0)
        artifact_dir = self.directory / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        with torch.inference_mode():
            functional.reset_net(self.cim)
            phase("compiling")
            started = time.perf_counter()
            self.compiled = torch.compile(
                self.cim,
                backend="cim22",
                fullgraph=True,
                dynamic=False,
                options={"runtime": "software", "artifact_dir": artifact_dir},
            )
            # Actual capture/lowering/first execution is entirely separate
            # from measured samples, not just the lazy torch.compile wrapper.
            output = self.compiled(sample)
            elapsed = time.perf_counter() - started
            self.prediction(output)
            functional.reset_net(self.cim)
        self.trim()
        self.artifacts = set(artifact_dir.glob("*.so"))
        if not self.artifacts:
            raise RuntimeError("编译未生成模型 .so")
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
            if (
                side == "cim"
                and set((self.directory / "artifacts").glob("*.so"))
                != self.artifacts
            ):
                raise RuntimeError("样本触发额外编译，未计入结果；请重新导入")
            prediction = self.prediction(output)
        self.trim()
        return prediction, elapsed

    def trim(self) -> None:
        # The verified Linux runner has large transient Host buffers. Return
        # freed glibc arenas between calls, outside measured inference time.
        trim = getattr(self._libc, "malloc_trim", None)
        if trim is not None:
            trim(0)
