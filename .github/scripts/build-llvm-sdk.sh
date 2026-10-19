#!/usr/bin/env bash
# Build the SDK once in the same manylinux image used for project wheels.
set -euo pipefail
export PATH="/opt/python/cp312-cp312/bin:$PATH"
export CCACHE_DIR=/ccache CCACHE_MAXSIZE=2G CCACHE_COMPILERCHECK=content
mkdir -p /work/tmp "$HOME" /out
work_dir=/work/llvm
mkdir -p "$work_dir"
trap 'rm -rf -- "$work_dir"' EXIT

# The workflow checks out the pinned official commit at this fixed path.
export CCACHE_BASEDIR=/
cmake -G Ninja -S /llvm-source/llvm -B "$work_dir/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/llvm \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=X86 \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF -DMLIR_LINK_MLIR_DYLIB=OFF \
  -DLLVM_ENABLE_PIC=ON -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_ENABLE_EH=OFF -DLLVM_ENABLE_RTTI=OFF \
  -DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF -DMLIR_INCLUDE_DOCS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
  -DMLIR_INSTALL_AGGREGATE_OBJECTS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DLLVM_DISTRIBUTION_COMPONENTS='llvm-libraries;mlir-libraries;llvm-headers;mlir-headers;cmake-exports;mlir-cmake-exports;llvm-config;llvm-tblgen;mlir-tblgen'
# Let LLVM select and install SDK components without building unrelated tools.
DESTDIR="$work_dir/install" cmake --build "$work_dir/build" \
  --target install-distribution-stripped \
  --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
sdk_dir="$work_dir/install/opt/llvm"
[[ "$("$sdk_dir/bin/llvm-config" --version)" == "$LLVM_VERSION" ]]
[[ "$("$sdk_dir/bin/llvm-config" --shared-mode)" == static ]]
[[ "$("$sdk_dir/bin/llvm-config" --build-mode)" == Release ]]
[[ "$("$sdk_dir/bin/llvm-config" --assertion-mode)" == OFF ]]
"$sdk_dir/bin/mlir-tblgen" --version
mkdir -p "$sdk_dir/share/cim22-licenses"
cp "/llvm-source/LICENSE.TXT" "$sdk_dir/share/cim22-licenses/LLVM.txt"
cp "/llvm-source/llvm/lib/Support/BLAKE3/LICENSE" \
  "$sdk_dir/share/cim22-licenses/BLAKE3.txt"
sed -n '1,/^ \*\//p' "/llvm-source/llvm/lib/Support/xxhash.cpp" \
  > "$sdk_dir/share/cim22-licenses/XXHASH.txt"
printf 'LLVM version: %s\nLLVM commit: %s\nBuild: Release; assertions OFF; X86 static PIC SDK distribution\n' \
  "$LLVM_VERSION" "$LLVM_COMMIT" > "$sdk_dir/SDK-INFO.txt"
ccache --show-stats
# Tar preserves executable permissions when the SDK crosses artifact jobs.
tar --use-compress-program='zstd -T2 -3' -cf /out/llvm-sdk.tar.zst -C "$sdk_dir" .
cd /out
sha256sum llvm-sdk.tar.zst > SHA256SUMS
