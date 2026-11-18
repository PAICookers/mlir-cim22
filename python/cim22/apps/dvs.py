"""Headless CLI for the same fixed DVS application used by the Web UI."""

import argparse
import json
import logging
import shutil
import threading
from contextlib import ExitStack
from pathlib import Path

from cim22.web.engine import (
    DvsEngine,
    discover_cim_layers,
    validate_cim_layer_names,
)
from cim22.web.storage import (
    Document,
    _storage_lock,
    make_config,
    read_rows,
    save_json,
    totals,
)
from cim22.web.worker import run_samples

logger = logging.getLogger(__name__)


def _prepare_output(directory: Path, inputs: tuple[Path, ...]) -> None:
    artifacts = directory / "artifacts"
    state = directory / "state.json"
    files = [
        directory / name
        for name in (
            "config.json",
            "state.json",
            "environment.json",
            "results.csv",
            "summary.json",
            ".storage.lock",
        )
    ]
    paths = [
        artifacts,
        *files,
        *(p.with_suffix(p.suffix + ".tmp") for p in files),
    ]
    for path in paths:
        if (
            path.is_symlink()
            or path.is_junction()
            or any(source.is_relative_to(path) for source in inputs)
        ):
            raise ValueError(f"Output would overwrite a link or input: {path}")
        if path.exists() and (
            not path.is_dir() if path == artifacts else not path.is_file()
        ):
            raise ValueError(f"Unexpected output path type: {path}")
    first_write_files = {
        directory / ".storage.lock",
        directory / "state.json.tmp",
    }
    if any(path.exists() for path in paths if path not in first_write_files):
        # Web state carries a task ID. Only prior CLI runs can be replaced here.
        previous: Document = (
            json.loads(state.read_text(encoding="utf-8"))
            if state.exists()
            else {}
        )
        if (
            not isinstance(previous, dict)
            or previous.get("runtime") != "usb"
            or "compiled" not in previous
            or "id" in previous
        ):
            raise ValueError(
                f"Output conflicts with existing files: {directory}"
            )
    if artifacts.is_dir():
        shutil.rmtree(artifacts)
    # CSV controls resume, and the environment identity is for this run only.
    for name in ("results.csv", "environment.json", "summary.json"):
        (directory / name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    """Compile and optionally execute fixed DVS inputs on a CIM board."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        help="Default: MODEL_DIR/weights/dvsgesture_mobilenet_snn_qat_int8.pt",
    )
    parser.add_argument(
        "--dataset", type=Path, help="Default: MODEL_DIR/data/test"
    )
    parser.add_argument(
        "--usb-config", type=Path, help="USB/PLL configuration JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dvs-results"),
        help="Output directory; replaces DVS CLI results (default: %(default)s)",
    )
    parser.add_argument(
        "--cim-layers",
        nargs="+",
        help="Selected Conv2d/Linear leaves; omit for the full model",
    )
    parser.add_argument("--list-cim-layers", action="store_true")
    parser.add_argument(
        "--action",
        choices=("compile", "single", "full"),
        default="single",
        help="Compile/warm up, run one sample or the full dataset (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    if args.list_cim_layers:
        print(
            json.dumps(
                discover_cim_layers(model_dir), ensure_ascii=False, indent=2
            )
        )
        return
    if args.usb_config is None:
        parser.error("--usb-config is required for board execution")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    validate_cim_layer_names(model_dir, args.cim_layers)
    with args.usb_config.expanduser().open(encoding="utf-8") as stream:
        usb_config = json.load(stream)
    config = make_config(
        model_dir,
        (
            args.weights
            or model_dir / "weights/dvsgesture_mobilenet_snn_qat_int8.pt"
        )
        .expanduser()
        .resolve(),
        (args.dataset or model_dir / "data/test").expanduser().resolve(),
        args.cim_layers,
        usb_config=usb_config,
    )
    directory = args.output.expanduser()
    if directory.is_symlink() or directory.is_junction():
        raise ValueError(f"Output directory is a link: {directory}")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    inputs = (
        (model_dir / "model_mobilenet_snn_deploy.py").resolve(),
        Path(config["weights"]),
        Path(config["dataset"]),
        args.usb_config.expanduser().resolve(),
    )
    lock_path = directory / ".dvs-run.lock"
    if lock_path in inputs:
        raise ValueError(f"Output would overwrite an input: {lock_path}")
    with ExitStack() as stack:
        try:
            stack.enter_context(_storage_lock(lock_path, blocking=False))
        except OSError as error:
            raise RuntimeError(
                f"Cannot lock DVS output directory: {directory}"
            ) from error
        _prepare_output(directory, inputs)
        _run(directory, config, args.action)


def _run(directory: Path, config: Document, action: str) -> None:
    state = {"runtime": "usb", "status": "loading", "compiled": False}
    save_json(directory / "state.json", state)
    save_json(directory / "config.json", config)

    def update(**changes: object) -> None:
        state.update(changes)
        save_json(directory / "state.json", state)
        logger.info(
            "DVS %s; completed=%s", state["status"], state.get("completed", 0)
        )

    try:
        engine = DvsEngine(config, directory)
        seconds = engine.compile(lambda phase: update(status=phase))
        update(
            status="ready",
            compiled=True,
            compile_seconds=seconds,
            **engine.metadata(),
        )
        if action != "compile":
            run_samples(
                directory,
                config,
                engine,
                threading.Event(),
                update,
                action,
            )
        save_json(
            directory / "summary.json",
            {
                "runtime": "usb",
                "totals": totals(read_rows(directory)),
                **engine.metadata(),
            },
        )
        logger.info("Results saved to %s", directory)
    except BaseException as error:
        update(status="interrupted", compiled=False, error=str(error))
        raise


if __name__ == "__main__":
    main()
