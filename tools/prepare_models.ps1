param(
    [string]$Revision = "b261bac6fd2cf515557d5d0707481eafa0485ec2",
    [string]$PythonExe = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $PythonExe) {
    $bundled = Join-Path $Root "runtime\python\python.exe"
    $dev = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $bundled) { $PythonExe = $bundled }
    elseif (Test-Path -LiteralPath $dev) { $PythonExe = $dev }
    else { $PythonExe = "python" }
}
& $PythonExe (Join-Path $PSScriptRoot "download_models.py") --root $Root --revision $Revision
if ($LASTEXITCODE -ne 0) { throw "Model preparation failed with exit code $LASTEXITCODE" }
