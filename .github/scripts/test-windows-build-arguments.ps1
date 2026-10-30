# Capture the production wheel command's native argv without building LLVM.
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
Set-StrictMode -Version Latest

$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    "$PSScriptRoot/build-windows-wheel.ps1", [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Windows wheel script has syntax errors' }
$commands = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.CommandAst] -and
        $node.GetCommandName() -eq 'uv' -and
        @($node.CommandElements | Where-Object { $_.Extent.Text -eq '--wheel' }).Count -gt 0
}, $true))
if ($commands.Count -ne 1) { throw 'Expected one production uv wheel command' }

$source = [pscustomobject]@{ FullName = 'C:\Source tree\unpacked' }
$stageDir = 'C:\Build stage'
$CyapiRoot = 'C:\CyAPI SDK'
$previousMlirDir = $env:MLIR_DIR
$probe = [IO.Path]::GetTempFileName()
try {
    $env:MLIR_DIR = 'C:\LLVM SDK\lib\cmake\mlir'
    'import json, sys; print(json.dumps(sys.argv[1:]))' |
        Set-Content -Encoding utf8 $probe
    # Only replace uv with an argv recorder; execute the actual argument text.
    $command = $commands[0].Extent.Text -replace '^uv\s+', "python `"$probe`" "
    $actual = @(& ([scriptblock]::Create($command)) | ConvertFrom-Json)
    $expected = @(
        'build', '--python', '3.12', $source.FullName, '--wheel', '--out-dir', "$stageDir/wheel",
        '-Ccmake.build-type=Release', "-Ccmake.define.MLIR_DIR=$env:MLIR_DIR",
        '-Ccmake.define.CIM22_ENABLE_USB=ON', "-Ccmake.define.CYAPI_ROOT=$CyapiRoot",
        '-Ccmake.define.CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL',
        '-Ccmake.define.CMAKE_JOB_POOLS=link=1', '-Ccmake.define.CMAKE_JOB_POOL_LINK=link'
    )
    if (($actual | ConvertTo-Json -Compress) -cne ($expected | ConvertTo-Json -Compress)) {
        throw "Windows wheel arguments were changed or split: $($actual | ConvertTo-Json -Compress)"
    }
    Write-Host 'Production Windows wheel argv, including spaced paths: PASS'
} finally {
    $env:MLIR_DIR = $previousMlirDir
    Remove-Item $probe
}
