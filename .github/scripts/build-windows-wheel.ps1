# Build from a fresh sdist with the reviewed, vendored Windows USB SDK.
param(
    [Parameter(Mandatory)][string]$SdkDir,
    [Parameter(Mandatory)][string]$CyapiRoot,
    [Parameter(Mandatory)][string]$DistDir
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

$repoDir = (Resolve-Path "$PSScriptRoot/../..").Path
$SdkDir = (Resolve-Path $SdkDir).Path
$DistDir = [IO.Path]::GetFullPath($DistDir)
$env:MLIR_DIR = "$SdkDir/lib/cmake/mlir"
$env:CMAKE_GENERATOR = 'Ninja'
if (!(Test-Path "$env:MLIR_DIR/MLIRConfig.cmake") -or
    (& "$SdkDir/bin/llvm-config.exe" --version) -ne $env:LLVM_VERSION -or
    (& "$SdkDir/bin/llvm-config.exe" --shared-mode) -ne 'static') {
    throw 'Expected the pinned static LLVM/MLIR development SDK'
}
$CyapiRoot = (Resolve-Path $CyapiRoot).Path
$cyapiFiles = @(
    'library/cpp/inc/CyAPI.h',
    'library/cpp/inc/CyUSB30_def.h',
    'library/cpp/lib/x64/CyAPI.lib',
    'license/license.rtf',
    'license/license.txt'
)
foreach ($name in $cyapiFiles) {
    if (!(Test-Path "$CyapiRoot/$name" -PathType Leaf)) {
        throw "Provisioned CyAPI SDK is missing $name"
    }
}
foreach ($entry in Get-Content "$CyapiRoot/SHA256SUMS") {
    $expected, $relativePath = $entry -split '  ', 2
    if ((Get-FileHash "$CyapiRoot/$relativePath" -Algorithm SHA256).Hash -ne $expected) {
        throw "Vendored CyAPI checksum mismatch: $relativePath"
    }
}
$stageDir = Join-Path ([IO.Path]::GetTempPath()) "cim22-wheel-$([Guid]::NewGuid())"
New-Item -ItemType Directory -Force $stageDir, $DistDir | Out-Null
try {
    uv build --python 3.12 "$repoDir/python" --sdist --out-dir "$stageDir/sdist"
    $source = New-Item -ItemType Directory "$stageDir/source"
    $sdists = @(Get-ChildItem "$stageDir/sdist/*.tar.gz")
    if ($sdists.Count -ne 1) { throw 'Expected exactly one source archive' }
    tar -xzf $sdists[0].FullName --strip-components=1 -C $source.FullName
    uv build --python 3.12 $source.FullName --wheel --out-dir "$stageDir/wheel" `
        -Ccmake.build-type=Release "-Ccmake.define.MLIR_DIR=$env:MLIR_DIR" `
        -Ccmake.define.CIM22_ENABLE_USB=ON "-Ccmake.define.CYAPI_ROOT=$CyapiRoot" `
        -Ccmake.define.CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL `
        -Ccmake.define.CMAKE_JOB_POOLS=link=1 -Ccmake.define.CMAKE_JOB_POOL_LINK=link
    $wheels = @(Get-ChildItem "$stageDir/wheel/*.whl")
    if ($wheels.Count -ne 1) { throw 'Expected exactly one Windows wheel' }
    python -m wheel unpack $wheels[0].FullName --dest "$stageDir/unpacked"
    $unpacked = @(Get-ChildItem "$stageDir/unpacked" -Directory)
    $metadata = @(Get-ChildItem $unpacked[0].FullName -Directory -Filter '*.dist-info')
    if ($metadata.Count -ne 1) { throw 'Expected one wheel metadata directory' }
    $licenses = New-Item -ItemType Directory -Force "$($metadata[0].FullName)/licenses/third-party"
    Copy-Item "$SdkDir/share/cim22-licenses/*.txt" $licenses.FullName
    Copy-Item "$source/build/_deps/protobuf-src/LICENSE" "$licenses/PROTOBUF.txt"
    Copy-Item "$source/build/_deps/protobuf-src/third_party/utf8_range/LICENSE" "$licenses/UTF8_RANGE.txt"
    Copy-Item "$source/build/_deps/absl-src/LICENSE" "$licenses/ABSEIL.txt"
    $cyapiLicenses = New-Item -ItemType Directory "$licenses/cyapi"
    Copy-Item "$CyapiRoot/license/license.rtf", "$CyapiRoot/license/license.txt" $cyapiLicenses.FullName
    Copy-Item "$CyapiRoot/README.md" "$cyapiLicenses/NOTICE.md"
    # wheel pack recomputes RECORD after adding the required license notices.
    python -m wheel pack $unpacked[0].FullName --dest-dir $DistDir
    $wheels = @(Get-ChildItem "$DistDir/*.whl")
    if ($wheels.Count -ne 1 -or $wheels[0].Name -notlike '*-py3-none-win_amd64.whl') {
        throw 'Expected exactly one py3-none-win_amd64 wheel'
    }
    @'
import sys
import tomllib
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

path = Path(sys.argv[1])
project = tomllib.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))[
    "project"
]
if path.name != f"mlir_cim22-{project['version']}-py3-none-win_amd64.whl":
    raise RuntimeError("Wheel filename does not match the project version")
with ZipFile(path) as wheel:
    metadata = next(
        n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")
    )
    fields = BytesParser().parsebytes(wheel.read(metadata))
    if fields["Name"] != project["name"] or fields["Version"] != project["version"]:
        raise RuntimeError("Wheel metadata version does not match the filename")
    tags = BytesParser().parsebytes(
        wheel.read(metadata.removesuffix("METADATA") + "WHEEL")
    )
    if tags.get_all("Tag") != ["py3-none-win_amd64"]:
        raise RuntimeError("Wheel metadata has unexpected compatibility tags")
    for name in (
        "cim22/bin/mlir-cim22-opt.exe",
        "cim22/lib/cim22_runtime.dll",
        "cim22/lib/cim22_runtime.lib",
        "cim22/lib/cim22_compiler.dll",
        "cim22/lib/cim22_compiler.lib",
        "cim22/share/licenses/cyapi/license.rtf",
    ):
        if name not in wheel.namelist():
            raise RuntimeError(f"Wheel is missing {name}")
    license_prefix = metadata.removesuffix("METADATA") + "licenses/third-party/cyapi/"
    cyapi = Path(sys.argv[3])
    for name in ("license.rtf", "license.txt"):
        if wheel.read(license_prefix + name) != (cyapi / "license" / name).read_bytes():
            raise RuntimeError(f"Wheel did not preserve the CyAPI license: {name}")
    if wheel.read(license_prefix + "NOTICE.md") != (cyapi / "README.md").read_bytes():
        raise RuntimeError("Wheel did not preserve the CyAPI notice")
'@ | python - $wheels[0].FullName "$repoDir/python/pyproject.toml" $CyapiRoot
    uv pip install --python 3.12 --no-deps --target "$stageDir/installed" $wheels[0].FullName
    Push-Location $stageDir
    try {
        & "$stageDir/installed/cim22/bin/mlir-cim22-opt.exe" --version
        @'
import ctypes
import os
import sys
from pathlib import Path

libraries = Path(sys.argv[1])
with os.add_dll_directory(str(libraries)):
    for name in ("cim22_runtime.dll", "cim22_compiler.dll"):
        ctypes.CDLL(str(libraries / name))
        print(f"Loaded: {name}")
'@ | python - "$stageDir/installed/cim22/lib"
    } finally {
        Pop-Location
    }
    $hash = (Get-FileHash $wheels[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($wheels[0].Name)" | Set-Content -Encoding ascii "$DistDir/SHA256SUMS-windows"
    Write-Host 'Windows native CLI and installed library loading PASS; USB ON; no board validation'
} finally {
    Remove-Item -Recurse -Force $stageDir
}
