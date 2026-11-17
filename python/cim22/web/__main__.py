"""Start the local application with one command; no Node runtime required."""

import argparse
import json
from pathlib import Path

import uvicorn

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="现有固定 DVS 模型包目录（保持只读）",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path.home() / ".local/share/cim22-web"
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        help="固定 TensorFlow/PyTorch 示例目录（默认使用源码树 python/examples）",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cim-layers",
        nargs="+",
        help="服务默认下放到 CIM 的 DVS 叶子层名",
    )
    parser.add_argument(
        "--list-cim-layers",
        action="store_true",
        help="列出固定 DVS 模型可下放的 Conv2d/Linear 叶子层并退出",
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
            args.data_dir,
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
