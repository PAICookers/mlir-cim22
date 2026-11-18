"""CLI entry point for the local CIM22 Web interface."""

import argparse
import json
from pathlib import Path

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the CIM22 local web interface."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="DVS model resource directory (read-only)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("cim22-web"),
        help=(
            "Output directory for task state, models, logs, results and MNIST "
            "training/cache data; created if missing (default: ./cim22-web)"
        ),
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "examples",
        help="TensorFlow/PyTorch scripts (default: examples beside the cim22 package)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Local server port (default: %(default)s)",
    )
    parser.add_argument(
        "--cim-layers",
        nargs="+",
        help="DVS leaf layers to offload to CIM by default",
    )
    parser.add_argument(
        "--list-cim-layers",
        action="store_true",
        help="List offloadable DVS Conv2d/Linear leaf layers and exit",
    )
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if args.list_cim_layers:
        from .engine import discover_cim_layers

        print(
            json.dumps(
                discover_cim_layers(model_dir), ensure_ascii=False, indent=2
            )
        )
        return
    uvicorn.run(
        create_app(
            args.data_dir.expanduser().resolve(),
            model_dir,
            args.examples_dir.expanduser().resolve()
            if args.examples_dir is not None
            else None,
            args.cim_layers,
        ),
        host="127.0.0.1",
        port=args.port,
        workers=1,
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
