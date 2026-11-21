"""Command-line entry point for the CIM22 Python SDK."""

import argparse
import sys
from importlib.metadata import version

import cim22


def main() -> None:
    """Show SDK usage or report the active Python installation."""
    parser = argparse.ArgumentParser(
        prog="cim22",
        description="Inspect the CIM22 Python SDK installation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""PyTorch example (model and inputs are provided by the application):
  compiled = torch.compile(
      model, backend="cim22", fullgraph=True, dynamic=False,
      options={"runtime": "simulator", "artifact_dir": "artifacts"},
  )
  output = compiled(inputs)

Selected Python backend options:
  runtime       "simulator" for software simulation, "usb" for a board
                (default; requires usb_config or usb_session)
  artifact_dir  Directory for compiled model libraries
  debug_dir     Directory for intermediate IR and object files
""",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--version", action="store_true", help="show the installed SDK version"
    )
    actions.add_argument(
        "--info",
        action="store_true",
        help="show the SDK and Python environment",
    )
    args = parser.parse_args()
    if not (args.version or args.info):
        parser.print_help()
        return

    print(f"mlir-cim22 {version('mlir-cim22')}")
    if args.info:
        print(f"Python executable: {sys.executable}")
        venv = sys.prefix if sys.prefix != sys.base_prefix else "none"
        print(f"Virtual environment: {venv}")
        print(f"Imported module: {cim22.__file__}")


if __name__ == "__main__":
    main()
