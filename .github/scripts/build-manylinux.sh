#!/usr/bin/env bash
# Run only inside the manylinux build container.
set -euo pipefail
export PATH="/opt/python/cp312-cp312/bin:$PATH"
mkdir -p /work/tmp "$HOME" /out

# This job consumes the SDK artifact; only the upstream SDK job builds LLVM.
export MLIR_DIR=/opt/llvm/lib/cmake/mlir
if [[ ! -f "$MLIR_DIR/MLIRConfig.cmake" ]] ||
   [[ "$(/opt/llvm/bin/llvm-config --version)" != "$LLVM_VERSION" ]] ||
   [[ "$(/opt/llvm/bin/llvm-config --shared-mode)" != static ]]; then
  echo "Expected the pinned static LLVM/MLIR development SDK" >&2
  exit 1
fi

# Fresh extraction avoids stale objects from normalized sdist timestamps.
stage_dir=$(mktemp -d /work/release.XXXXXX)
trap 'rm -rf -- "$stage_dir"' EXIT
uv build --python /opt/python/cp312-cp312/bin/python /io/python \
  --sdist --out-dir "$stage_dir/sdist"
mkdir "$stage_dir/source"
tar -xzf "$stage_dir"/sdist/*.tar.gz --strip-components=1 -C "$stage_dir/source"
uv build --python /opt/python/cp312-cp312/bin/python "$stage_dir/source" \
  --wheel --out-dir "$stage_dir/wheel" \
  -Ccmake.define.CMAKE_JOB_POOLS=link=1 \
  -Ccmake.define.CMAKE_JOB_POOL_LINK=link
auditwheel show "$stage_dir"/wheel/*.whl
auditwheel repair --plat manylinux_2_28_x86_64 --wheel-dir /out \
  "$stage_dir"/wheel/*.whl
auditwheel show /out/*.whl > /out/AUDITWHEEL.txt
cp "$stage_dir"/sdist/*.tar.gz /out/

# Check native loading on glibc 2.28 without downloading the Python ML stack.
uv pip install --python /opt/python/cp312-cp312/bin/python --no-deps \
  --target "$stage_dir/installed" /out/*.whl
readelf -d "$stage_dir/installed/cim22/bin/mlir-cim22-opt" \
  "$stage_dir/installed/cim22/lib/"*.so > "$stage_dir/native-dynamic.txt"
if grep -E 'NEEDED.*lib(LLVM|MLIR)' "$stage_dir/native-dynamic.txt"; then
  echo "Expected LLVM/MLIR to be linked statically into the wheel" >&2
  exit 1
fi
"$stage_dir/installed/cim22/bin/mlir-cim22-opt" --version
python - "$stage_dir/installed/cim22/lib" <<'PY'
import ctypes
import sys
from pathlib import Path

libraries = Path(sys.argv[1])
for name in ("libcim22_runtime.so", "libcim22_compiler.so"):
    ctypes.CDLL(str(libraries / name))
    print(f"Loaded: {name}")
PY
echo "glibc 2.28: installed native CLI and library loading PASS" \
  > /out/BASELINE-VALIDATION.txt

mkdir -p /out/licenses
cp /io/LICENSE /out/licenses/PROJECT.txt
cp /opt/llvm/share/cim22-licenses/*.txt /out/licenses/
cp "$stage_dir/source/build/_deps/protobuf-src/LICENSE" /out/licenses/PROTOBUF.txt
cp "$stage_dir/source/build/_deps/protobuf-src/third_party/utf8_range/LICENSE" \
  /out/licenses/UTF8_RANGE.txt
cp "$stage_dir/source/build/_deps/absl-src/LICENSE" /out/licenses/ABSEIL.txt
rpm -ql libusb1 systemd-libs | grep '/licenses/' > "$stage_dir/license-files.txt"
while IFS= read -r license_file; do
  if [[ -f "$license_file" ]]; then
    cp --parents "$license_file" /out/licenses/
  fi
done < "$stage_dir/license-files.txt"
{
  echo "Project commit: $PROJECT_COMMIT"
  echo "LLVM version: $LLVM_VERSION"
  echo "LLVM commit: $LLVM_COMMIT"
  echo "LLVM SDK SHA256: $LLVM_SDK_SHA256"
  echo "Build runner: $BUILD_RUNNER"
  echo "Target: manylinux_2_28_x86_64; USB ON; tests/ONNX OFF"
  python --version
  cmake --version | head -n 1
  cc --version | head -n 1
  ldd --version | head -n 1
} > /out/BUILD-INFO.txt
cd /out
sha256sum ./*.whl ./*.tar.gz > SHA256SUMS
