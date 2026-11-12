"""Run the platform-neutral installed-wheel validation sequence."""

import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PYTEST_FILES = (
    "tests/test_web.py",
    "tests/test_model_format.py",
    "tests/test_artifacts.py",
    "tests/test_usb_runtime.py",
    "tests/test_pll.py",
    "tests/test_pll_serial.py",
    "tests/test_compile_tool.py",
    "tests/test_backend.py",
    "tests/test_convolution.py",
    "tests/test_tensorflow.py",
)


def _run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        check=True,
        env=os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )


def _check_web_ui() -> None:
    cli = (
        Path(sys.prefix)
        / ("Scripts" if os.name == "nt" else "bin")
        / ("cim22-web.exe" if os.name == "nt" else "cim22-web")
    )
    if not cli.is_file():
        raise RuntimeError("The installed wheel did not provide cim22-web")
    subprocess.run(
        [str(cli), "--help"],
        check=True,
        env=os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )

    from cim22.web.app import create_app
    from fastapi.testclient import TestClient

    with TemporaryDirectory() as directory:
        root = Path(directory)
        with TestClient(
            create_app(root / "data", root / "model"),
            base_url="http://127.0.0.1",
        ) as client:
            response = client.get("/")
            response.raise_for_status()
            asset = re.search(r'(?:src|href)="(/assets/[^"]+)"', response.text)
            if asset is None:
                raise RuntimeError("The wheel does not contain frontend assets")
            client.get(asset.group(1)).raise_for_status()
            client.get("/api/state").raise_for_status()


def main() -> None:
    """Run framework, pytest, and Web UI checks from the temporary workdir."""
    import cim22
    import serial
    from cim22.runtime import _load_library, runtime_library

    prefix = Path(sys.prefix).resolve()
    package = Path(cim22.__file__).resolve()
    serial_package = Path(serial.__file__).resolve()
    for path in (package, serial_package):
        if not path.is_relative_to(prefix):
            raise RuntimeError(
                f"Expected installed dependency, imported {path}"
            )
    library_path = runtime_library().resolve()
    if not library_path.is_relative_to(package.parent):
        raise RuntimeError(f"Expected wheel runtime, loaded {library_path}")
    library = _load_library(library_path)
    if not hasattr(library, "cim22_session_configure_usb_with_clock"):
        raise RuntimeError("Installed runtime lacks the clock callback ABI")
    for name in (
        "cim22_session_configure_usb_with_pll",
        "cim22_session_configure_usb_with_hardware",
    ):
        if hasattr(library, name):
            raise RuntimeError(f"Installed runtime exports retired ABI: {name}")
    print(f"Installed package: {package}; pyserial: {serial_package}")

    _run("examples/torch_compile.py")
    _run("examples/tensorflow_compile.py")
    _run("-m", "pytest", "--import-mode=importlib", "-q", *PYTEST_FILES)
    _check_web_ui()
    print("Installed wheel framework, pytest, and Web UI validation: PASS")


if __name__ == "__main__":
    main()
