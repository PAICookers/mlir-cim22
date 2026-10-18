"""Start the Linux application with one command; no Node runtime required."""

import argparse
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
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.data_dir, args.model_dir.expanduser().resolve()),
        host="127.0.0.1",
        port=args.port,
        workers=1,
        timeout_graceful_shutdown=3,
    )


if __name__ == "__main__":
    main()
