# Run from an MSVC x64 developer shell.
param(
    [Parameter(Mandatory)][string]$SourceDir,
    [Parameter(Mandatory)][string]$SdkDir
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

$SourceDir = (Resolve-Path $SourceDir).Path
$SdkDir = [IO.Path]::GetFullPath($SdkDir)
$buildDir = Join-Path (Split-Path $SdkDir) 'llvm-build'
cmake -G Ninja -S "$SourceDir/llvm" -B $buildDir `
    "-DCMAKE_INSTALL_PREFIX=$SdkDir" -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_C_FLAGS=/utf-8 -DCMAKE_CXX_FLAGS=/utf-8 `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DLLVM_USE_CRT_RELEASE=MD `
    -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=X86 `
    -DBUILD_SHARED_LIBS=OFF -DLLVM_BUILD_LLVM_DYLIB=OFF `
    -DLLVM_LINK_LLVM_DYLIB=OFF -DMLIR_LINK_MLIR_DYLIB=OFF `
    -DLLVM_ENABLE_PIC=ON -DLLVM_ENABLE_ASSERTIONS=OFF `
    -DLLVM_ENABLE_EH=OFF -DLLVM_ENABLE_RTTI=OFF `
    -DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF `
    -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF `
    -DLLVM_INCLUDE_DOCS=OFF -DMLIR_INCLUDE_DOCS=OFF `
    -DMLIR_ENABLE_BINDINGS_PYTHON=OFF -DMLIR_ENABLE_EXECUTION_ENGINE=ON `
    -DMLIR_INSTALL_AGGREGATE_OBJECTS=OFF -DLLVM_ENABLE_DIA_SDK=OFF `
    -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF `
    -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF `
    -DLLVM_PARALLEL_LINK_JOBS=1 `
    '-DLLVM_DISTRIBUTION_COMPONENTS=llvm-libraries;mlir-libraries;llvm-headers;mlir-headers;cmake-exports;mlir-cmake-exports;llvm-config;llvm-tblgen;mlir-tblgen'
cmake --build $buildDir --target install-distribution --parallel 2
if ((& "$SdkDir/bin/llvm-config.exe" --version) -ne $env:LLVM_VERSION -or
    (& "$SdkDir/bin/llvm-config.exe" --shared-mode) -ne 'static' -or
    (& "$SdkDir/bin/llvm-config.exe" --build-mode) -ne 'Release' -or
    (& "$SdkDir/bin/llvm-config.exe" --assertion-mode) -ne 'OFF') {
    throw 'Expected the pinned static Release LLVM/MLIR SDK with assertions OFF'
}
& "$SdkDir/bin/mlir-tblgen.exe" --version
$licenses = New-Item -ItemType Directory -Force "$SdkDir/share/cim22-licenses"
Copy-Item "$SourceDir/LICENSE.TXT" "$licenses/LLVM.txt"
Copy-Item "$SourceDir/llvm/lib/Support/BLAKE3/LICENSE" "$licenses/BLAKE3.txt"
$xxhash = Get-Content -Raw "$SourceDir/llvm/lib/Support/xxhash.cpp"
$end = $xxhash.IndexOf(' */')
if ($end -lt 0) { throw 'Cannot locate the xxhash license header' }
$xxhash.Substring(0, $end + 3) | Set-Content -Encoding utf8 "$licenses/XXHASH.txt"
"LLVM version: $env:LLVM_VERSION`nLLVM commit: $env:LLVM_COMMIT`nRelease; X86 static; /MD; assertions OFF" |
    Set-Content -Encoding utf8 "$SdkDir/SDK-INFO.txt"
