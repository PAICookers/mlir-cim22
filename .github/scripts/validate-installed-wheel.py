"""Install and validate a Linux or Windows wheel outside the checkout."""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

PYTEST_FILES = (
    "tests/test_web.py",
    "tests/test_model_format.py",
    "tests/test_artifacts.py",
    "tests/test_usb_runtime.py",
    "tests/test_layers.py",
    "tests/test_pll.py",
    "tests/test_pll_serial.py",
    "tests/test_compile_tool.py",
    "tests/test_backend.py",
    "tests/test_module_offload.py",
    "tests/test_convolution.py",
    "tests/test_tensorflow.py",
    "tests/test_jax.py",
)


def _checked_wheel(dist_dir: Path) -> Path:
    wheels = list(dist_dir.resolve().glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one wheel in {dist_dir}")
    wheel = wheels[0]
    checksum = wheel.with_suffix(".whl.sha256").read_text(encoding="utf-8")
    fields = checksum.rstrip("\r\n").split("  ", 1)
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if len(fields) != 2 or fields[0].lower() != digest or fields[1] != wheel.name:
        raise ValueError(f"Wheel checksum mismatch for {wheel.name}")
    return wheel


def _install_and_validate(dist_dir: Path) -> None:
    wheel = _checked_wheel(dist_dir)
    script = Path(__file__).resolve()
    source = script.parents[2] / "python"
    env = os.environ.copy()
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "CIM22_OPT",
        "CIM22_COMPILE_TOOL",
        "CIM22_RUNTIME_LIBRARY",
        "CIM22_CLOCK_TEST_LIBRARY",
        "CIM22_CC",
        "LD_LIBRARY_PATH",
    ):
        env.pop(name, None)
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    torch_mlir_platform = (
        "win_amd64"
        if os.name == "nt"
        else "manylinux_2_27_x86_64.manylinux_2_28_x86_64"
    )
    torch_mlir = (
        "https://github.com/llvm/torch-mlir/releases/download/v20261101/"
        f"torch_mlir-20261101-cp312-abi3-{torch_mlir_platform}.whl"
    )

    with TemporaryDirectory(prefix="cim22 test-") as directory:
        work = Path(directory)
        venv = work / "venv"

        def run(*command: str) -> None:
            subprocess.run(command, cwd=work, env=env, check=True)

        run(sys.executable, "-I", "-m", "venv", str(venv))
        python = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        run(
            python,
            "-m",
            "pip",
            "install",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.14.1+cpu",
        )
        run(
            python,
            "-m",
            "pip",
            "install",
            torch_mlir,
            f"{wheel}[web,tensorflow,jax]",
            "ml_dtypes==0.6.0",
            "pytest>=8.3,<10",
            "httpx>=0.28,<1",
            "protobuf>=6.31.1",
        )
        run(python, "-m", "pip", "check")
        for folder in ("tests", "examples"):
            shutil.copytree(
                source / folder,
                work / folder,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
        validation = shutil.copyfile(script, work / script.name)
        run(python, "-I", str(validation))


def _run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        check=True,
        env=os.environ | {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
    )


def _installed_cli(name: str) -> Path:
    return (
        Path(sys.prefix)
        / ("Scripts" if os.name == "nt" else "bin")
        / (f"{name}.exe" if os.name == "nt" else name)
    )


def _check_sdk_cli() -> None:
    result = subprocess.run(
        [str(_installed_cli("cim22")), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != f"mlir-cim22 {version('mlir-cim22')}":
        raise RuntimeError(f"Unexpected CIM22 SDK version: {result.stdout!r}")
    _run("-I", "-m", "cim22", "--info")


def _check_web_ui() -> None:
    subprocess.run(
        [str(_installed_cli("cim22-web")), "--help"],
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


def _validate_installed() -> None:
    """Run framework, pytest, and Web UI checks from the temporary workdir."""
    import cim22
    import serial
    from cim22.runtime import _load_library, runtime_library

    prefix = Path(sys.prefix).resolve()
    package = Path(cim22.__file__).resolve()
    serial_package = Path(serial.__file__).resolve()
    for path in (package, serial_package):
        if not path.is_relative_to(prefix):
            raise RuntimeError(f"Expected installed dependency, imported {path}")
    library_path = runtime_library().resolve()
    if not library_path.is_relative_to(package.parent):
        raise RuntimeError(f"Expected wheel runtime, loaded {library_path}")
    library = _load_library(library_path)
    if not hasattr(library, "cim22_session_configure_usb_with_clock"):
        raise RuntimeError("Installed runtime lacks the clock callback ABI")
    if not hasattr(library, "cim22_session_share_usb"):
        raise RuntimeError("Installed runtime lacks the shared USB session ABI")
    for name in (
        "cim22_session_configure_usb_with_pll",
        "cim22_session_configure_usb_with_hardware",
    ):
        if hasattr(library, name):
            raise RuntimeError(f"Installed runtime exports retired ABI: {name}")
    print(f"Installed package: {package}; pyserial: {serial_package}")

    _check_sdk_cli()
    _run("examples/torch_compile.py")
    _run("examples/tensorflow_compile.py")
    _run("examples/jax_compile.py")
    _run("-m", "pytest", "--import-mode=importlib", "-q", *PYTEST_FILES)
    _check_web_ui()
    print("Installed wheel framework, pytest, and Web UI validation: PASS")


def main() -> None:
    """Check an artifact directory, or run the copied tests in the test venv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dist_dir", type=Path, nargs="?", help="wheel artifact directory"
    )
    args = parser.parse_args()
    if args.dist_dir is None:
        _validate_installed()
    else:
        _install_and_validate(args.dist_dir)


if __name__ == "__main__":
    main()
