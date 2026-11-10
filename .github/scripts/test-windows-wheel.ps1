# Validate with MSVC available, but without checkout imports or library overrides.
param([Parameter(Mandatory)][string]$DistDir)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

$repoDir = (Resolve-Path "$PSScriptRoot/../..").Path
$DistDir = (Resolve-Path $DistDir).Path
$wheels = @(Get-ChildItem "$DistDir/*.whl")
if ($wheels.Count -ne 1) { throw 'Expected exactly one Windows wheel' }
$checksum = (Get-Content -Raw "$DistDir/SHA256SUMS-windows").Trim() -split '  ', 2
if ($checksum.Count -ne 2 -or $checksum[0] -notmatch '^[0-9a-fA-F]{64}$' -or
    $checksum[1] -ne $wheels[0].Name -or
    (Get-FileHash $wheels[0].FullName -Algorithm SHA256).Hash -ne $checksum[0]) {
    throw 'Windows wheel checksum mismatch'
}
foreach ($name in @('PYTHONPATH', 'CIM22_OPT', 'CIM22_COMPILER_LIBRARY', 'CIM22_RUNTIME_LIBRARY', 'CIM22_CC', 'LD_LIBRARY_PATH')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$workDir = Join-Path ([IO.Path]::GetTempPath()) "cim22-test-$([Guid]::NewGuid())"
New-Item -ItemType Directory $workDir | Out-Null
try {
    python -m venv "$workDir/venv"
    $py = "$workDir/venv/Scripts/python.exe"
    & $py -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
    $torchMlir = 'https://github.com/llvm/torch-mlir-release/releases/download/dev-wheels/torch_mlir-20260923.595-cp312-abi3-win_amd64.whl'
    & $py -m pip install $torchMlir $wheels[0].FullName 'pytest>=8.3,<10' 'protobuf>=6.31.1'
    & $py -m pip check
    Copy-Item -Recurse "$repoDir/python/tests" "$workDir/tests"
    Copy-Item "$repoDir/python/examples/torch_compile.py" "$workDir/torch_compile.py"
    Push-Location $workDir
    try {
        @'
import sys
from pathlib import Path

import cim22

package = Path(cim22.__file__).resolve()
if not package.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError(f"Expected installed wheel, imported {package}")
print(f"Installed package: {package}")
'@ | & $py -
        & $py torch_compile.py
        & $py -m pytest --import-mode=importlib -q `
            tests/test_artifacts.py tests/test_backend.py tests/test_convolution.py `
            tests/test_model_format.py tests/test_native_compiler.py tests/test_usb_runtime.py
        & $py -m pip list --format=freeze
        Write-Host 'Windows installed software inference and focused pytest PASS; no board validation'
    } finally {
        Pop-Location
    }
} finally {
    Remove-Item -Recurse -Force $workDir
}
