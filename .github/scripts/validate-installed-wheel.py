"""Run the platform-neutral installed-wheel validation sequence."""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PYTEST_FILES = (
    "tests/test_web.py",
    "tests/test_model_format.py",
    "tests/test_artifacts.py",
    "tests/test_usb_runtime.py",
    "tests/test_native_compiler.py",
    "tests/test_backend.py",
    "tests/test_convolution.py",
    "tests/test_tensorflow.py",
)


def _run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], check=True)


def _check_web_ui() -> None:
    cli = shutil.which("cim22-web")
    if cli is None:
        raise RuntimeError("The installed wheel did not provide cim22-web")
    subprocess.run([cli, "--help"], check=True)

    from cim22.web.app import create_app
    from fastapi.testclient import TestClient

    with TemporaryDirectory() as directory:
        root = Path(directory)
        with TestClient(create_app(root / "data", root / "model")) as client:
            response = client.get("/")
            response.raise_for_status()
            assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', response.text)
            if not assets:
                raise RuntimeError("The wheel does not contain frontend assets")
            for url in assets:
                client.get(url).raise_for_status()
            client.get("/api/state").raise_for_status()


def main() -> None:
    """Run framework, pytest, and Web UI checks from the temporary workdir."""
    import cim22

    package = Path(cim22.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Expected installed wheel, imported {package}")
    print(f"Installed package: {package}")

    _run("torch_compile.py")
    _run("tensorflow_compile.py")
    _run("-m", "pytest", "--import-mode=importlib", "-q", *PYTEST_FILES)
    _check_web_ui()
    print("Installed wheel framework, pytest, and Web UI validation: PASS")


if __name__ == "__main__":
    main()
