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
torch_mlir_wheel=torch_mlir-20261101-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
curl --fail --location --retry 3 --output "$work_dir/$torch_mlir_wheel" \
  https://github.com/llvm/torch-mlir/releases/download/v20261101/torch_mlir-20261101-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
"$py" -m pip install "$work_dir"/torch_mlir-*.whl "${wheels[0]}[web,tensorflow]" \
  'ml_dtypes==0.6.0' 'pytest>=8.3,<10' 'httpx>=0.28,<1' 'protobuf>=6.31.1'
"$py" -m pip check
cp -R "$repo_dir/python/tests" "$work_dir/tests"
cp "$repo_dir/python/examples"/{torch_compile,tensorflow_compile}.py "$work_dir"
cp "$repo_dir/.github/scripts/validate-installed-wheel.py" "$work_dir"
cd "$work_dir"
"$py" validate-installed-wheel.py
