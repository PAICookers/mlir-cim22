#!/usr/bin/env bash
# Install and test the artifact without source or native-library overrides.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
dist_dir=$(realpath -- "${1:?Pass the wheel artifact directory}")
shopt -s nullglob
wheels=("$dist_dir"/*.whl)
if (( ${#wheels[@]} != 1 )); then
  echo "Expected exactly one project wheel in $dist_dir" >&2
  exit 1
fi
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
unset PYTHONPATH CIM22_OPT CIM22_COMPILER_LIBRARY CIM22_RUNTIME_LIBRARY LD_LIBRARY_PATH

python -m venv "$work_dir/venv"
py="$work_dir/venv/bin/python"
"$py" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.14.1+cpu'
torch_mlir_wheel=torch_mlir-20261001-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
curl --fail --location --retry 3 --output "$work_dir/$torch_mlir_wheel" \
  https://github.com/llvm/torch-mlir/releases/download/v20261001/torch_mlir-20261001-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
"$py" -m pip install "$work_dir"/torch_mlir-*.whl "${wheels[0]}[web]" \
  'pytest>=8.3,<10' 'httpx>=0.28,<1' 'protobuf>=6.31.1'
"$py" -m pip check
cp -R "$repo_dir/python/tests" "$work_dir/tests"
cp "$repo_dir/python/examples/torch_compile.py" "$work_dir/torch_compile.py"
cd "$work_dir"
"$py" torch_compile.py
"$work_dir/venv/bin/cim22-web" --help
"$py" -m pytest --import-mode=importlib -q \
  tests/test_web.py tests/test_model_format.py tests/test_artifacts.py \
  tests/test_usb_runtime.py tests/test_native_compiler.py tests/test_backend.py
"$py" - <<'PY'
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import cim22
from cim22.web.app import create_app

print(f"Installed package: {cim22.__file__}")
with TemporaryDirectory() as directory:
    root = Path(directory)
    app = create_app(root / "data", root / "model")
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/")
        response.raise_for_status()
        assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', response.text)
        if not assets:
            raise RuntimeError("The wheel does not contain frontend assets")
        for url in assets:
            client.get(url).raise_for_status()
        client.get("/api/state").raise_for_status()
PY
{
  echo "Installed software inference, focused pytest and Web assets/API: PASS"
  uname -sm
  grep '^PRETTY_NAME=' /etc/os-release
  "$py" --version
  "$py" -m pip list --format=freeze
  echo "No physical-board validation performed."
} > "$dist_dir/VALIDATION.txt"
