# Validate with MSVC available, but without checkout imports or library overrides.
param([Parameter(Mandatory)][string]$DistDir)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

$repoDir = (Resolve-Path "$PSScriptRoot/../..").Path
$DistDir = (Resolve-Path $DistDir).Path
$wheels = @(Get-ChildItem "$DistDir/*.whl")
if ($wheels.Count -ne 1) { throw 'Expected exactly one Windows wheel' }
$checksum = (Get-Content -Raw "$DistDir/$($wheels[0].Name).sha256").Trim() -split '  ', 2
if ($checksum.Count -ne 2 -or $checksum[0] -notmatch '^[0-9a-fA-F]{64}$' -or
    $checksum[1] -ne $wheels[0].Name -or
    (Get-FileHash $wheels[0].FullName -Algorithm SHA256).Hash -ne $checksum[0]) {
    throw 'Windows wheel checksum mismatch'
}
foreach ($name in @('PYTHONPATH', 'CIM22_OPT', 'CIM22_COMPILE_TOOL', 'CIM22_RUNTIME_LIBRARY', 'CIM22_CLOCK_TEST_LIBRARY', 'CIM22_CC', 'LD_LIBRARY_PATH')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$workDir = Join-Path ([IO.Path]::GetTempPath()) "cim22 test-$([Guid]::NewGuid())"
New-Item -ItemType Directory $workDir | Out-Null
try {
    python -m venv "$workDir/venv"
    $py = "$workDir/venv/Scripts/python.exe"
    & $py -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
    $torchMlir = 'https://github.com/llvm/torch-mlir/releases/download/v20261101/torch_mlir-20261101-cp312-abi3-win_amd64.whl'
    $projectWheel = "$($wheels[0].FullName)[web,tensorflow]"
    & $py -m pip install $torchMlir $projectWheel 'ml_dtypes==0.6.0' 'pytest>=8.3,<10' 'httpx>=0.28,<1' 'protobuf>=6.31.1'
    & $py -m pip check
    Copy-Item -Recurse "$repoDir/python/tests" "$workDir/tests"
    Copy-Item -Recurse "$repoDir/python/examples" "$workDir/examples"
    Copy-Item "$repoDir/.github/scripts/validate-installed-wheel.py" "$workDir/validate-installed-wheel.py"
    Push-Location $workDir
    try {
        & $py validate-installed-wheel.py
    } finally {
        Pop-Location
    }
} finally {
    Remove-Item -Recurse -Force $workDir
}
